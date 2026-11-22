"""Match joint final-layer descriptors on a shared fine patch grid."""
from copy import copy

import numpy as np

from src.part3.features import fuse_features


def resize_shape(height, width, short_edge):
    """Uncropped shape of the pinned torchvision short-edge Resize."""
    if height <= width:
        return short_edge, short_edge * width // height
    return short_edge * height // width, short_edge


def align_coarse_features(features, coarse_grid, fine_grid, coarse_shape, fine_shape, patch_size):
    """Sample coarse tokens at fine centers in uncropped source-image coordinates."""
    source = np.asarray(features, dtype=np.float32).reshape(*coarse_grid, -1)

    def coordinates(axis):
        fine_center = np.arange(fine_grid[axis], dtype=np.float32) + .5
        index = fine_center * (coarse_shape[axis] / fine_shape[axis]) - .5
        index = np.clip(index, 0, coarse_grid[axis] - 1)
        left = index.astype(np.intp)
        weight = (index - left).astype(np.float32)
        return left, np.minimum(left + 1, coarse_grid[axis] - 1), weight

    if any(shape // patch_size != grid for dims, grids in [
        (coarse_shape, coarse_grid), (fine_shape, fine_grid)
    ] for shape, grid in zip(dims, grids)):
        raise ValueError("Unexpected resized grid")
    y0, y1, wy = coordinates(0)
    x0, x1, wx = coordinates(1)
    horizontal = source[:, x0, :] * (1 - wx[None, :, None]) + source[:, x1, :] * wx[None, :, None]
    aligned = horizontal[y0, :, :] * (1 - wy[:, None, None]) + horizontal[y1, :, :] * wy[:, None, None]
    return aligned.reshape(-1, source.shape[-1])


def attach_multiscale(backbone):
    """Reuse one frozen model and the exact upstream preprocessing at both scales."""
    from PIL import Image
    from torchvision import transforms

    coarse = copy(backbone)
    coarse.smaller_edge_size = 448
    original_resize, *remaining = backbone.transform.transforms
    coarse.transform = transforms.Compose([
        transforms.Resize(448, interpolation=original_resize.interpolation,
                          antialias=original_resize.antialias),
        *remaining,
    ])
    prepare_fine = backbone.prepare_image
    extract_fine = backbone.extract_features
    original_mask = backbone.compute_background_mask
    pending = None
    last_fine_features = None

    def prepare(image):
        nonlocal pending
        if isinstance(image, str):
            with Image.open(image) as handle:
                image = handle.convert("RGB")
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        height, width = image.shape[:2]
        coarse_tensor, coarse_grid = coarse.prepare_image(image)
        fine_tensor, fine_grid = prepare_fine(image)
        pending = (fine_tensor, coarse_tensor, coarse_grid, fine_grid,
                   resize_shape(height, width, 448), resize_shape(height, width, 672))
        return fine_tensor, fine_grid

    def extract(tensor):
        nonlocal pending, last_fine_features
        if pending is None or pending[0] is not tensor:
            raise RuntimeError("Prepare the paired image before extracting features")
        _, coarse_tensor, coarse_grid, fine_grid, coarse_shape, fine_shape = pending
        pending = None
        coarse_features = coarse.extract_features(coarse_tensor)
        last_fine_features = extract_fine(tensor)
        aligned = align_coarse_features(coarse_features, coarse_grid, fine_grid,
                                        coarse_shape, fine_shape, backbone.model.patch_size)
        return fuse_features(aligned, last_fine_features)

    def mask(features, grid, *args, **kwargs):
        if last_fine_features is None or len(features) != len(last_fine_features):
            raise RuntimeError("Extract paired features before computing the mask")
        return original_mask(last_fine_features, grid, *args, **kwargs)

    backbone.prepare_image = prepare
    backbone.extract_features = extract
    backbone.compute_background_mask = mask
