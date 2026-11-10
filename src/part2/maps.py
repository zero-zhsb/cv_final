"""Compact patch maps with the pinned official dists2map reconstruction semantics."""
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def reconstruct_map(patch_map: np.ndarray, image_shape) -> np.ndarray:
    height, width = map(int, image_shape[:2])
    if height <= 0 or width <= 0 or patch_map.ndim != 2:
        raise ValueError("Invalid image or patch shape")
    # Upstream fills its patch map in float64, renders, then our export casts to f32.
    resized = cv2.resize(patch_map.astype(np.float64), (width, height), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(resized, sigma=4).astype(np.float32)


def load_anomaly_map(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "anomaly_map" in data:
            result = data["anomaly_map"].astype(np.float32)
        elif "patch_map" in data and "image_shape" in data:
            result = reconstruct_map(data["patch_map"], data["image_shape"])
        else:
            raise ValueError(f"Unsupported map archive: {path}")
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"Invalid anomaly map: {path}")
    return result
