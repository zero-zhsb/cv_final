"""Compare the adapter against the unchanged official detector on one real image (CPU)."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/visa_1cls.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import tifffile
    from src.part2.inspector import AnomalyDINOInspector
    upstream = ROOT / "third_party/AnomalyDINO"
    manifest = json.loads(args.manifest.read_text())
    data = manifest["categories"]["cashew"]
    ref = args.data_root / data["references"]["1"]["0"][0]
    sample = args.data_root / next(x["image_path"] for x in data["tests"] if x["label"] == 0)
    inspector = AnomalyDINOInspector(upstream, ROOT / "third_party/dinov2", ROOT / "weights/dinov2_vits14_pretrain.pth")
    inspector.fit_references([ref], seed=0)
    actual = inspector.predict(sample)
    sys.path.insert(0, str(upstream))
    import src
    src.__path__.append(str(upstream / "src"))
    from src.detection import run_anomaly_detection
    with tempfile.TemporaryDirectory(prefix="cv-parity-") as directory:
        root = Path(directory)
        for folder, path in [("train/good", ref), ("test/good", sample)]:
            target = root / "cashew" / folder
            target.mkdir(parents=True)
            (target / path.name).symlink_to(path.resolve())
        np.random.seed(0)
        torch.manual_seed(0)
        # The official CPU path still invokes cuda.synchronize solely for timing.
        # No CUDA tensors are created, but avoid touching the shared GPU at all.
        synchronize = torch.cuda.synchronize
        torch.cuda.synchronize = lambda *a, **kw: None
        try:
            scores, _, _ = run_anomaly_detection(inspector.backbone, "cashew", str(root), 1,
                {"cashew": ["good"]}, str(root / "output"), save_examples=False,
                masking=True, mask_ref_images=False, rotation=True, faiss_on_cpu=True,
                seed=0, save_patch_dists=True, save_tiffs=True)
        finally:
            torch.cuda.synchronize = synchronize
        original_score = scores[f"good/{sample.name}"]
        original_map = tifffile.imread(root / "output/anomaly_maps/seed=0/cashew/test/good" / f"{sample.stem}.tiff")
        original_patch = np.load(root / "output/anomaly_maps/seed=0/cashew/test/good" / f"{sample.stem}.npy")
        np.testing.assert_allclose(actual["score"], original_score, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(actual["anomaly_map"], original_map, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(actual["patch_map"], original_patch, atol=1e-6, rtol=1e-5)
        report = {"status": "passed", "scope": "one real normal cashew test image, 1-shot seed=0, CPU only; not a benchmark",
                  "manifest_id": manifest["manifest_id"], "reference": str(ref.relative_to(args.data_root)),
                  "image": str(sample.relative_to(args.data_root)), "adapter_score": actual["score"],
                  "official_score": original_score,
                  "maximum_map_absolute_error": float(np.max(np.abs(actual["anomaly_map"] - original_map))),
                  "maximum_patch_absolute_error": float(np.max(np.abs(actual["patch_map"] - original_patch)))}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
