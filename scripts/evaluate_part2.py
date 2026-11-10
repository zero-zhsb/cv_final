"""Validate complete configurations and compute course AUROC without rescaling maps."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import sys
import hashlib
from datetime import datetime, timezone

for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/visa_1cls.json")
    parser.add_argument("--configuration", help="For example: cashew/shot_1/seed_0")
    parser.add_argument("--memory-budget-gib", type=float, default=8)
    parser.add_argument("--pixel-device", choices=["cpu", "cuda:0"], default="cpu")
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from src.part2.maps import load_anomaly_map
    from src.part2.metrics import PixelAUROC
    metadata = json.loads((args.run / "run.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    if metadata["manifest_id"] != manifest["manifest_id"] or metadata["partial_smoke_test"]:
        parser.error("Manifest mismatch or partial smoke run; refusing full-test metrics")
    completed = {entry["configuration"]: entry for entry in metadata.get("configurations", [])}
    if args.configuration:
        if args.configuration not in completed:
            parser.error("Requested configuration has not completed")
        completed = {args.configuration: completed[args.configuration]}
    if not completed:
        parser.error("No completed configurations")
    with (args.run / "predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    for key, configuration in completed.items():
        output = args.run / "metrics" / (key.replace("/", "__") + ".json")
        if output.exists():
            print(f"EXISTS {key}: preserving existing metrics", flush=True)
            continue
        rows = [r for r in predictions if f"{r['category']}/shot_{r['shot']}/seed_{r['seed']}" == key]
        category = key.split("/")[0]
        expected = {r["image_path"]: r for r in manifest["categories"][category]["tests"]}
        if len(rows) != len(expected) or {r["image_path"] for r in rows} != set(expected):
            raise ValueError(f"Incomplete or duplicated test predictions: {key}")
        labels, scores, latencies = [], [], []
        for row in rows:
            source = expected[row["image_path"]]
            if int(row["label"]) != source["label"] or (row["mask_path"] or None) != source["mask_path"]:
                raise ValueError("Prediction labels/masks differ from shared manifest")
            labels.append(source["label"]); scores.append(float(row["score"]))
            latencies.append(float(row["inference_ms"]))
        def pairs():
            for index, row in enumerate(rows):
                source = expected[row["image_path"]]
                archive_path = (args.run / row["map_path"]).resolve(strict=True)
                archive_path.relative_to(args.run.resolve())
                anomaly_map = load_anomaly_map(archive_path)
                with Image.open(args.data_root / row["image_path"]) as image:
                    image_shape = (image.height, image.width)
                if anomaly_map.shape != image_shape:
                    raise ValueError("Anomaly map is not at original image resolution")
                if source["mask_path"]:
                    with Image.open(args.data_root / source["mask_path"]) as mask_image:
                        mask = np.asarray(mask_image)
                        if mask.ndim != 2:
                            raise ValueError("Expected a two-dimensional binary ground-truth mask")
                        mask = mask > 0
                else:
                    mask = np.zeros(image_shape, dtype=bool)
                if index % 50 == 0:
                    print(f"EVAL {key}: {index+1}/{len(rows)}", flush=True)
                yield anomaly_map, mask
        if args.pixel_device == "cpu":
            metric = PixelAUROC(args.memory_budget_gib)
            for anomaly_map, mask in pairs():
                metric.add(anomaly_map, mask)
            pixel_auc, backend = metric.compute(), {"backend": "cpu_exact_hierarchical_histograms"}
        else:
            from src.part2.gpu_metrics import gpu_pixel_auc
            pixel_auc, backend = gpu_pixel_auc(pairs, args.pixel_device)
        if not np.isfinite(scores).all() or not np.isfinite(latencies).all():
            raise ValueError("Non-finite image scores or timing")
        result = {"run_id": metadata["run_id"], "manifest_id": metadata["manifest_id"],
                  "code_commit": metadata["code_commit"], "configuration": key,
                  "evaluation_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(), ROOT / "src/part2/maps.py", ROOT / "src/part2/metrics.py", ROOT / "src/part2/gpu_metrics.py"]},
                  "category": category, "shot": int(rows[0]["shot"]), "seed": int(rows[0]["seed"]),
                  "test_count": len(rows), "normal_count": labels.count(0), "anomaly_count": labels.count(1),
                  "image_auroc": float(roc_auc_score(labels, scores)), "pixel_auroc": pixel_auc,
                  "pixel_backend_details": backend,
                  "mean_inference_ms": float(np.mean(latencies)), "median_inference_ms": float(np.median(latencies)),
                  "peak_gpu_memory_mib": configuration["peak_gpu_memory_mib"],
                  "memory_bank_ms": configuration["memory_bank_ms"],
                  "pixel_evaluation": "exact float32 original-resolution, all normal+abnormal test images, no per-image normalization",
                  "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                  "evaluator_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
