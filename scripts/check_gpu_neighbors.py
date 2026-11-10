"""Validate GPU exhaustive neighbors against CPU FAISS on real DINO features."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "2"
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
    from src.part2.gpu_index import ExactGPUIndex
    torch.cuda.set_per_process_memory_fraction(0.15)
    manifest = json.loads((ROOT / "manifests/visa_1cls.json").read_text())
    inspector = AnomalyDINOInspector(ROOT / "third_party/AnomalyDINO", ROOT / "third_party/dinov2", ROOT / "weights/dinov2_vits14_pretrain.pth", "cuda:0")
    records = []
    class Capture:
        def __init__(self, index): self.index = index
        def search(self, query, k=1):
            self.query = query.copy()
            self.distance, match = self.index.search(query, k)
            return self.distance, match
    for category, shot, seed in [("cashew", 1, 0), ("candle", 8, 0), ("capsules", 2, 1)]:
        item = manifest["categories"][category]
        inspector.fit_references([args.data_root / p for p in item["references"][str(shot)][str(seed)]], seed)
        cpu = inspector.index
        gpu = ExactGPUIndex(cpu.reconstruct_n(0, cpu.ntotal))
        samples = [x for label in [0, 1] for x in [x for x in item["tests"] if x["label"] == label][:2]]
        for sample in samples:
            state = np.random.get_state()
            cpu_capture = Capture(cpu)
            inspector.index = cpu_capture
            expected = inspector.predict(args.data_root / sample["image_path"], render_map=False)
            np.random.set_state(state)
            gpu_capture = Capture(gpu)
            inspector.index = gpu_capture
            actual = inspector.predict(args.data_root / sample["image_path"], render_map=False)
            score_error = abs(actual["score"] - expected["score"])
            patch_error = float(np.max(np.abs(actual["patch_map"] - expected["patch_map"])))
            direct_distance, _ = gpu.search(cpu_capture.query)
            direct_error = float(np.max(np.abs(direct_distance-cpu_capture.distance))) / 2
            records.append({"category": category, "shot": shot, "seed": seed,
                "image": sample["image_path"], "score_absolute_error": score_error,
                "maximum_patch_absolute_error": patch_error,
                "same_query_maximum_distance_error": direct_error,
                "mask_nonzero_disagreements": int(np.sum((actual["patch_map"] != 0) != (expected["patch_map"] != 0))),
                "feature_query_maximum_error": float(np.max(np.abs(cpu_capture.query-gpu_capture.query))) if cpu_capture.query.shape == gpu_capture.query.shape else None,
                "cpu_faiss_inference_ms": expected["inference_ms"], "exact_gpu_inference_ms": actual["inference_ms"]})
    passed = all(r["score_absolute_error"] <= 1e-6 and r["maximum_patch_absolute_error"] <= 2e-6
                 and r["same_query_maximum_distance_error"] <= 2e-6
                 and r["mask_nonzero_disagreements"] == 0 and r["feature_query_maximum_error"] == 0 for r in records)
    report = {"status": "passed" if passed else "investigate", "scope": "12 real images, 3 categories, 1/2/8-shot; direct same-query distances separated from end-to-end numerical variation",
              "criteria": {"image_score_atol": 1e-6, "patch_and_distance_atol": 2e-6, "identical_feature_queries_and_masks": True},
              "exhaustive_search": True, "tf32_enabled": False, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
