"""OAS metric add-on to unchanged 448+672 joint feature fusion."""
import time
import numpy as np
from sklearn.covariance import OAS
from src.part2.inspector import AnomalyDINOInspector
from src.part2.gpu_index import ExactGPUIndex
from src.part3.multiscale import attach_multiscale


class OASIndex:
    def __init__(self, bank, device):
        self.ntotal, self.d = bank.shape
        self.mean, self.transform, self.shrinkage = fit_oas_metric(bank)
        self.index = ExactGPUIndex(apply_metric(bank, self.mean, self.transform), device)

    def search(self, queries, k=1):
        return self.index.search(apply_metric(queries, self.mean, self.transform), k)


class OASInspector(AnomalyDINOInspector):
    """Keep equal-weight fusion/common patch search; change only reference metric."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, resolution=672, **kwargs)
        if self.knn_backend != 'torch-gpu':
            raise ValueError('OASInspector requires the exact torch-gpu backend')
        attach_multiscale(self.backbone)
        self.backbone.model.requires_grad_(False)
        self.fit_details = {}

    def fit_references(self, paths, seed=0):
        self.synchronize()
        start = time.perf_counter()
        super().fit_references(paths, seed)
        bank = self.index.bank.detach().cpu().numpy()
        self.index = OASIndex(bank, self.device)
        self.fit_details = {'oas': True, 'oas_shrinkage': self.index.shrinkage,
                            'metric_dim': self.index.d}
        self.synchronize()
        self.bank_ms = (time.perf_counter() - start) * 1000

def fit_oas_metric(bank):
    """Deterministic automatic OAS shrinkage; no dimension/regularizer sweep.

    Divide whitened features by sqrt(d), making mean squared distances per
    feature dimension explicit. Do NOT renormalize whitened vectors to unit norm.
    """
    x = np.asarray(bank, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or (not np.isfinite(x).all()):
        raise ValueError('Require finite normal reference features')
    model = OAS(store_precision=False).fit(x)
    values, vectors = np.linalg.eigh(model.covariance_)
    if values.max() <= 0:
        raise ValueError('Degenerate normal covariance')
    values = np.maximum(values, np.finfo(np.float64).eps * values.max())
    transform = vectors * (1 / np.sqrt(values * x.shape[1])) @ vectors.T
    return (model.location_.astype(np.float32), transform.astype(np.float32), float(model.shrinkage_))

def apply_metric(x, mean, transform):
    out = (np.asarray(x, dtype=np.float32) - mean) @ transform
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite transformed features')
    return np.ascontiguousarray(out, dtype=np.float32)
