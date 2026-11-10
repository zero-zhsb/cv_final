"""Reuse official AnomalyDINO transforms, features, PCA masks and map rendering.

Upstream: dammsi/AnomalyDINO (Apache-2.0), commit b9d1c2648e3a5247437d4d953d907a8f3d994457.
This adapter adds explicit reference lists, local weights, calibration and exports.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch

UPSTREAM_COMMIT = "b9d1c2648e3a5247437d4d953d907a8f3d994457"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_score(patch_distances: np.ndarray) -> float:
    """Match upstream mean_top1p on the flattened, background-zeroed distances."""
    values = np.asarray(patch_distances).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Empty or non-finite patch distances")
    count = max(1, int(len(values) * 0.01))
    return float(np.sort(values)[-count:].mean())


class AnomalyDINOInspector:
    def __init__(self, upstream: Path, dinov2_repo: Path, weights: Path,
                 device: str = "cpu", resolution: int = 448, threads: int = 2):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.set_num_threads(threads)
        cv2.setNumThreads(threads)
        faiss.omp_set_num_threads(threads)
        base = load_module("_course_anomalydino_backbones", upstream / "src/backbones.py")
        self.utils = load_module("_course_anomalydino_utils", upstream / "src/utils.py")
        # Only model sourcing changes. Inherited prepare_image, extract_features and
        # compute_background_mask remain the pinned official implementations.
        class LocalBackbone(base.DINOv2Wrapper):
            def load_model(inner):
                model = torch.hub.load(str(dinov2_repo), "dinov2_vits14",
                                       source="local", pretrained=False)
                state = torch.load(weights, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                model.eval()
                inner.transform = base.transforms.Compose([
                    base.transforms.Resize(size=inner.smaller_edge_size,
                        interpolation=base.transforms.InterpolationMode.BICUBIC, antialias=True),
                    base.transforms.ToTensor(),
                    base.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                return model.to(inner.device)
        self.backbone = LocalBackbone("dinov2_vits14", device, smaller_edge_size=resolution)
        self.device = device
        self.index = None
        self.bank_ms = None

    def synchronize(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)

    @staticmethod
    def read_rgb(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read RGB image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def fit_references(self, paths: list[Path], seed: int = 0):
        if not paths:
            raise ValueError("At least one reference is required")
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.synchronize()
        start = time.perf_counter()
        reference_features = []
        for path in paths:
            image = self.read_rgb(path)
            # VisA agnostic: official rotation=True, mask_ref_images=False.
            for rotated in self.utils.augment_image(image):
                tensor, grid = self.backbone.prepare_image(rotated)
                features = self.backbone.extract_features(tensor)
                mask = self.backbone.compute_background_mask(features, grid, threshold=10, masking_type=False)
                reference_features.append(features[mask])
        bank = np.concatenate(reference_features).astype(np.float32)
        faiss.normalize_L2(bank)
        self.index = faiss.IndexFlatL2(bank.shape[1])
        self.index.add(bank)
        self.synchronize()
        self.bank_ms = (time.perf_counter() - start) * 1000

    def predict(self, path: Path) -> dict:
        if self.index is None:
            raise RuntimeError("Build the reference bank before prediction")
        self.synchronize()
        start = time.perf_counter()
        image = self.read_rgb(path)
        tensor, grid = self.backbone.prepare_image(image)
        features = self.backbone.extract_features(tensor)
        mask = self.backbone.compute_background_mask(features, grid, threshold=10, masking_type=True)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if not mask.any():
            raise RuntimeError(f"Official PCA foreground mask is empty: {path}")
        foreground = np.ascontiguousarray(features[mask], dtype=np.float32)
        faiss.normalize_L2(foreground)
        distances, _ = self.index.search(foreground, k=1)
        output = np.zeros(len(mask), dtype=float)
        output[mask] = (distances / 2).reshape(-1)
        self.synchronize()
        inference_ms = (time.perf_counter() - start) * 1000
        patch_map = output.reshape(grid)
        # Match the upstream rendering path: resize to original shape, sigma=4.
        anomaly_map = self.utils.dists2map(patch_map, image.shape).astype(np.float32)
        return {"score": image_score(output), "anomaly_map": anomaly_map,
                "patch_map": patch_map.astype(np.float32), "inference_ms": inference_ms,
                "spatial_shape": list(tensor.shape[1:])}
