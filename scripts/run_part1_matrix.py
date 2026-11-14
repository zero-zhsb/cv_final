"""Resumable full Part 1 matrix with read-only data and exact streaming evaluation.

No all-test maps or decoded-image cache are saved. Deterministic maps are
recomputed in a second pass; scores must match the first pass bit for bit.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--pixel-backend", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    import cv2
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from src.data.visa import VisAReader
    from src.evaluation.float_keys import FloatKeyAUROC, CUDAFloatKeyAUROC, bounds
    from src.part1.inspector import TemplateInspector
    cv2.setNumThreads(2)
    manifest = json.loads((ROOT / "manifests/visa_1cls.json").read_text())
    reader = VisAReader(args.source, manifest)
    categories = args.categories or list(manifest["categories"])
    for selected, allowed in [(categories, manifest["categories"]), (args.shots, manifest["shots"]), (args.seeds, manifest["seeds"])]:
        if len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
            parser.error("Invalid or duplicated configuration")
    files = [Path(__file__), ROOT / "src/part1/inspector.py", ROOT / "src/data/visa.py", ROOT / "src/evaluation/float_keys.py"]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    identity = {"manifest_id": manifest["manifest_id"], "split_sha256": reader.split_sha256,
                "source_sha256": source_hashes, "categories": categories, "shots": args.shots, "seeds": args.seeds,
                "pixel_backend": args.pixel_backend}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        parser.error("Existing run has a different identity; use a new output")
    save(config_path, identity)
    meta = {**identity, "method": "template_difference", "status": "running", "device": "cpu",
            "run_id": args.output.name, "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "code_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "map_storage": "none; deterministic second-pass reconstruction; selected visualizations separately",
            "inference_time_scope": "uncached first pass: RGB read, transform, nearest template, difference, smoothing and score; excludes map resize and evaluation",
            "pixel_evaluation": "exact float32 original resolution; all normal+abnormal tests; no per-image normalization",
            "threads": 2, "started_at_utc": datetime.now(timezone.utc).isoformat()}
    save(args.output / "run.json", meta)
    all_metrics = []
    for category in categories:
        item = manifest["categories"][category]
        for shot in args.shots:
            for seed in args.seeds:
                key = f"{category}/shot_{shot}/seed_{seed}"
                folder = args.output / key.replace("/", "__")
                folder.mkdir(exist_ok=True)
                metric_file = folder / "metrics.json"
                if metric_file.exists():
                    all_metrics.append(json.loads(metric_file.read_text()))
                    continue
                if os.statvfs(args.output).f_bavail * os.statvfs(args.output).f_frsize < 96 * 1024**2:
                    raise RuntimeError("Less than 96 MiB free; safely stopping before configuration")
                print("START", key, flush=True)
                refs = item["references"][str(shot)][str(seed)]
                assert len(refs) == shot and len(set(refs)) == shot and not set(refs) & set(item["calibration"])
                inspector = TemplateInspector()
                inspector.fit_references([reader.path(p) for p in refs], seed)
                inspector.predict(reader.path(refs[0]), render_map=False)
                calibration = [{"image_path": p, "score": inspector.predict(reader.path(p), False)["score"]} for p in item["calibration"]]
                threshold = float(np.quantile([r["score"] for r in calibration], 0.95, method="linear"))
                save(folder / "calibration.json", {"references": refs, "scores": calibration, "threshold": threshold})
                rows = []; low = high = None
                for sample in item["tests"]:
                    result = inspector.predict(reader.path(sample["image_path"]), render_map=True)
                    lo, hi = bounds(result["anomaly_map"])
                    if lo is not None:
                        low = lo if low is None else min(low, lo)
                        high = hi if high is None else max(high, hi)
                    nearest = refs[inspector.references.index(result["nearest_reference"])]
                    rows.append({"method": "template_difference", "manifest_id": manifest["manifest_id"],
                                 "category": category, "shot": shot, "seed": seed, **sample,
                                 "score": result["score"], "threshold": threshold,
                                 "decision": "NG" if result["score"] > threshold else "OK",
                                 "inference_ms": result["inference_ms"], "nearest_reference": nearest})
                with (folder / "predictions.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                metric = (CUDAFloatKeyAUROC if args.pixel_backend == "cuda" else FloatKeyAUROC)(low, high)
                for sample, row in zip(item["tests"], rows):
                    prediction = inspector.predict(reader.path(sample["image_path"]), render_map=True)
                    if prediction["score"] != row["score"]:
                        raise ValueError("Recomputed score differs; refusing pixel metric")
                    image_map = prediction["anomaly_map"]
                    metric.add(image_map, reader.mask(sample, image_map.shape))
                result = {"configuration": key, "category": category, "shot": shot, "seed": seed,
                          "manifest_id": manifest["manifest_id"], "test_count": len(rows),
                          "image_auroc": float(roc_auc_score([r["label"] for r in rows], [r["score"] for r in rows])),
                          "pixel_auroc": metric.compute(), "mean_inference_ms": float(np.mean([r["inference_ms"] for r in rows])),
                          "memory_bank_ms": inspector.bank_ms, "peak_gpu_memory_mib": None,
                          "pixel_histogram_mib": metric.total.nbytes * 2 / 1024**2,
                          "pixel_backend": args.pixel_backend,
                          "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
                if args.pixel_backend == "cuda":
                    metric.close()
                del metric
                save(metric_file, result); all_metrics.append(result)
                save(args.output / "status.json", {"completed": len(all_metrics), "expected": len(categories)*len(args.shots)*len(args.seeds), "last": result})
                print("DONE", key, result["image_auroc"], result["pixel_auroc"], flush=True)
    meta["status"] = "completed"; meta["configurations"] = len(all_metrics)
    save(args.output / "run.json", meta)
    print("MATRIX_COMPLETED", len(all_metrics), flush=True)


if __name__ == "__main__":
    main()
