"""RGB template baseline with the pinned AnomalyDINO input geometry.

Geometry follows DINOv2Wrapper.prepare_image at AnomalyDINO commit
b9d1c2648e3a5247437d4d953d907a8f3d994457: PIL bicubic short-edge resize,
then crop the right/bottom to a multiple of 14. RGB stays in [0, 1].
"""
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def prepare_rgb(image: Image.Image, resolution=448, patch_size=14):
    image = image.convert("RGB")
    width, height = image.size
    if width <= height:
        size = (resolution, int(resolution * height / width))
    else:
        size = (int(resolution * width / height), resolution)
    resized = image.resize(size, Image.Resampling.BICUBIC)
    cropped = resized.crop((0, 0, size[0] - size[0] % patch_size,
                            size[1] - size[1] % patch_size))
    return np.asarray(cropped, dtype=np.float32) / np.float32(255)


def top_fraction_score(values, fraction=0.01):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not values.size or not np.isfinite(values).all() or not 0 < fraction <= 1:
        raise ValueError("Expected finite scores and a fraction in (0,1]")
    count = max(1, int(values.size * fraction))
    return float(np.partition(values, values.size - count)[-count:].mean())


class TemplateInspector:
    def __init__(self, resolution=448, sigma=1.0, top_fraction=0.01):
        if resolution < 14 or sigma < 0 or not 0 < top_fraction <= 1:
            raise ValueError("Invalid baseline settings")
        self.resolution, self.sigma, self.top_fraction = resolution, sigma, top_fraction
        self.templates, self.references = [], []
        self.bank_ms = None

    def fit_references(self, paths, seed=0):
        if not paths:
            raise ValueError("At least one normal reference is required")
        start = time.perf_counter()
        templates = []
        for path in paths:
            with Image.open(path) as image:
                templates.append(prepare_rgb(image, self.resolution))
        if len({t.shape for t in templates}) != 1:
            raise ValueError("References have incompatible resized shapes; no silent stretching")
        self.templates, self.references = templates, [str(p) for p in paths]
        self.bank_ms = (time.perf_counter() - start) * 1000

    def predict(self, path: Path, render_map=True):
        if not self.templates:
            raise RuntimeError("Fit references before prediction")
        start = time.perf_counter()
        with Image.open(path) as image:
            image_shape = [image.height, image.width]
            rgb = prepare_rgb(image, self.resolution)
        if rgb.shape != self.templates[0].shape:
            raise ValueError("Image/reference resized shapes differ; cannot compare aligned RGB")
        distances = [float(np.abs(rgb - template).mean()) for template in self.templates]
        selected = int(np.argmin(distances))
        difference = np.abs(rgb - self.templates[selected]).mean(axis=2)
        template_map = gaussian_filter(difference, sigma=self.sigma).astype(np.float32)
        score = top_fraction_score(template_map, self.top_fraction)
        elapsed = (time.perf_counter() - start) * 1000
        anomaly_map = cv2.resize(template_map, (image_shape[1], image_shape[0]),
                                 interpolation=cv2.INTER_LINEAR) if render_map else None
        return {"score": score, "template_map": template_map,
                "anomaly_map": anomaly_map, "image_shape": image_shape,
                "spatial_shape": list(rgb.shape[:2]), "inference_ms": elapsed,
                "nearest_reference": self.references[selected]}
