"""Equal-weight concatenation preserves matching to one common reference patch."""
import numpy as np


def fuse_features(*layers):
    if not layers or any(x.ndim != 2 for x in layers):
        raise ValueError('Expected nonempty layers of patch features')
    if len({x.shape[0] for x in layers}) != 1:
        raise ValueError('Patch grids differ')
    normalized=[]
    for x in layers:
        x=np.asarray(x,dtype=np.float32)
        norms=np.linalg.norm(x,axis=1,keepdims=True)
        if not np.isfinite(x).all() or np.any(norms == 0):
            raise ValueError('Nonfinite or zero-norm features')
        normalized.append(x/norms)
    return np.ascontiguousarray(np.concatenate(normalized,axis=1)/np.sqrt(np.float32(len(layers))),dtype=np.float32)
