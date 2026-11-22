"""Verify the 672 control and spatial fusion on normal training images only."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    from src.part2.inspector import AnomalyDINOInspector
    from src.part3.multiscale import attach_multiscale

    torch.cuda.set_per_process_memory_fraction(.15, 0)
    manifest = json.loads((ROOT / "manifests/visa_1cls.json").read_text())
    cases = []
    for category, item in manifest["categories"].items():
        cases.append((category, item["references"]["8"]["0"], item["calibration"][0]))
    masks = {}
    measurements = []
    for variant in ("scale672", "fusion448_672"):
        torch.cuda.reset_peak_memory_stats(0)
        inspector = AnomalyDINOInspector(
            ROOT / "third_party/AnomalyDINO", ROOT / "third_party/dinov2",
            ROOT / "weights/dinov2_vits14_pretrain.pth", "cuda:0",
            resolution=672, knn_backend="torch-gpu",
        )
        inspector.backbone.model.requires_grad_(False)
        if variant == "fusion448_672":
            attach_multiscale(inspector.backbone)
        for category, refs, normal in cases:
            torch.cuda.reset_peak_memory_stats(0)
            inspector.fit_references([args.data_root / ref for ref in refs], seed=0)
            tensor, grid = inspector.backbone.prepare_image(str(args.data_root / normal))
            features = inspector.backbone.extract_features(tensor)
            np.random.seed(10)
            mask = inspector.backbone.compute_background_mask(features, grid, masking_type=True)
            if variant == "scale672":
                masks[category] = mask
            elif not np.array_equal(mask, masks[category]):
                raise ValueError(f"Unexpected change in the 672 mask for {category}")
            dim = 384 if variant == "scale672" else 768
            if features.shape != (grid[0] * grid[1], dim):
                raise ValueError("Unexpected feature grid or width")
            result = inspector.predict(args.data_root / normal, render_map=False)
            if min(grid) != 48 or not np.isfinite(result["patch_map"]).all() or not np.isfinite(result["score"]):
                raise ValueError("Expected a finite 672-grid prediction")
            row = {"category": category, "variant": variant, "normal_image": normal, "grid": grid,
                   "feature_dim": dim, "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 2**20}
            measurements.append(row)
            print(json.dumps(row), flush=True)
        del inspector
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"passed": True, "normal_only": True, "matching_fine_masks": True, "settings": "8-shot seed 0",
              "protocol_sha256": hashlib.sha256((ROOT / "configs/part3_multiscale_protocol.json").read_bytes()).hexdigest(),
              "measurements": measurements}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
