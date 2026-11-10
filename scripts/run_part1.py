"""Run template difference on manifest-defined references, calibration and tests."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/visa_1cls.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["cashew"])
    parser.add_argument("--shots", type=int, nargs="+", default=[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    import cv2
    import numpy as np
    from src.part1.inspector import TemplateInspector
    cv2.setNumThreads(2)
    if args.output.exists():
        parser.error("Output exists; choose a new run directory")
    manifest = json.loads(args.manifest.read_text())
    canonical = json.dumps({k: v for k, v in manifest.items() if k != "manifest_id"},
                           ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != manifest["manifest_id"]:
        parser.error("Manifest identity mismatch")
    for values, allowed in [(args.categories, manifest["categories"]),
                            (args.shots, manifest["shots"]), (args.seeds, manifest["seeds"])]:
        if len(set(values)) != len(values) or not set(values) <= set(allowed):
            parser.error("Duplicated or out-of-protocol configuration")
    data = args.data_root.resolve(strict=True)
    def checked(relative):
        path = (data / relative).resolve(strict=True)
        path.relative_to(data)
        return path
    for category in args.categories:
        entry = manifest["categories"][category]
        for shot in args.shots:
            for seed in args.seeds:
                refs = entry["references"][str(shot)][str(seed)]
                if len(refs) != shot or set(refs) & set(entry["calibration"]):
                    raise ValueError("Reference/calibration isolation violated")
                for path in refs + entry["calibration"]:
                    checked(path)
        for sample in entry["tests"]:
            checked(sample["image_path"])
            if sample["mask_path"]:
                checked(sample["mask_path"])
    args.output.mkdir(parents=True)
    sources = [Path(__file__), ROOT / "src/part1/inspector.py"]
    meta = {"status": "running", "method": "template_difference", "run_id": args.output.name,
            "manifest_id": manifest["manifest_id"], "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "code_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()),
            "categories": args.categories, "shots": args.shots, "seeds": args.seeds,
            "resolution": 448, "geometry": "PIL bicubic short edge 448; right/bottom crop to multiple of 14",
            "difference": "mean absolute RGB in [0,1] against minimum global MAD template",
            "gaussian_sigma": 1.0, "score": "top 1% cropped-map pixels; floor, minimum one",
            "device": "cpu", "peak_gpu_memory_mib": None, "cache": "references only; no test feature cache",
            "partial_smoke_test": False, "scope": "full tests within selected configuration, not full 72-configuration study",
            "inference_time_scope": "image read, geometry, nearest template, difference, smoothing, score; excludes map rendering/export",
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "configurations": []}
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    save("run.json", meta)
    thresholds, calibration = {}, []
    fields = ["run_id", "method", "manifest_id", "category", "shot", "seed", "image_path", "mask_path", "label", "score", "map_path", "threshold", "decision", "inference_ms", "nearest_reference"]
    try:
        with (args.output / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for category in args.categories:
                entry = manifest["categories"][category]
                for shot in args.shots:
                    for seed in args.seeds:
                        key = f"{category}/shot_{shot}/seed_{seed}"
                        references = entry["references"][str(shot)][str(seed)]
                        inspector = TemplateInspector()
                        inspector.fit_references([checked(p) for p in references], seed)
                        inspector.predict(checked(references[0]), render_map=False)  # warmup; no test access
                        scores = []
                        for relative in entry["calibration"]:
                            score = inspector.predict(checked(relative), render_map=False)["score"]
                            scores.append(score)
                            calibration.append({"configuration": key, "image_path": relative, "score": score})
                        threshold = float(np.quantile(scores, 0.95, method="linear"))
                        thresholds[key] = {"threshold": threshold, "references": references, "calibration": entry["calibration"]}
                        for index, sample in enumerate(entry["tests"]):
                            prediction = inspector.predict(checked(sample["image_path"]), render_map=False)
                            map_path = Path("maps") / key / Path(sample["image_path"]).parent.name / (Path(sample["image_path"]).stem + ".npz")
                            (args.output / map_path).parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(args.output / map_path, template_map=prediction["template_map"],
                                                image_shape=np.asarray(prediction["image_shape"], dtype=np.int32))
                            writer.writerow({"run_id": args.output.name, "method": meta["method"], "manifest_id": manifest["manifest_id"],
                                "category": category, "shot": shot, "seed": seed, **sample,
                                "mask_path": sample["mask_path"] or "", "score": prediction["score"],
                                "map_path": map_path.as_posix(), "threshold": threshold,
                                "decision": "NG" if prediction["score"] > threshold else "OK",
                                "inference_ms": prediction["inference_ms"],
                                "nearest_reference": Path(prediction["nearest_reference"]).relative_to(data).as_posix()})
                            if index % 25 == 0:
                                print(f"PREDICT {key}: {index+1}/{len(entry['tests'])}", flush=True)
                        handle.flush()
                        meta["configurations"].append({"configuration": key, "memory_bank_ms": inspector.bank_ms,
                            "test_count": len(entry["tests"]), "peak_gpu_memory_mib": None})
                        save("thresholds.json", thresholds); save("calibration.json", calibration); save("run.json", meta)
        meta["status"] = "completed"
    except Exception as exc:
        meta["status"], meta["error"] = "failed", repr(exc)
        raise
    finally:
        meta["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save("run.json", meta)


if __name__ == "__main__":
    main()
