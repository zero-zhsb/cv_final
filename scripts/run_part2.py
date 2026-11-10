"""Run manifest-controlled AnomalyDINO inference; defaults to CPU on shared servers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# Set before importing numerical libraries; these apply only to this process.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def manifest_path(root: Path, relative: str, allowed_roots=()) -> Path:
    path = (root / relative).resolve(strict=True)
    if not any(path.is_relative_to(base.resolve(strict=True)) for base in [root, *allowed_roots]):
        raise ValueError(f"Image escapes approved roots: {relative}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/visa_1cls.json")
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party/AnomalyDINO")
    parser.add_argument("--dinov2-repo", type=Path, default=ROOT / "third_party/dinov2")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--shots", type=int, nargs="+", default=[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--allow-image-root", type=Path, action="append", default=[], help="Approved read-only image root when prepared data uses symlinks")
    parser.add_argument("--limit-per-class", type=int, help="Smoke test only: at most N normal + N abnormal test images; no AUROC claim")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new run directory")
    if args.threads < 1 or (args.limit_per_class is not None and args.limit_per_class < 1):
        parser.error("Threads and limits must be positive")
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[name] = str(args.threads)
    import numpy as np
    import torch
    from src.part2.inspector import AnomalyDINOInspector, UPSTREAM_COMMIT
    def checked_image(relative):
        return manifest_path(args.data_root, relative, args.allow_image_root)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_id = manifest["manifest_id"]
    payload = {k: v for k, v in manifest.items() if k != "manifest_id"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != manifest_id:
        parser.error("Manifest hash mismatch")
    if git_commit(args.upstream) != UPSTREAM_COMMIT:
        parser.error("AnomalyDINO must match the pinned upstream commit")
    if subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"], text=True).strip():
        parser.error("Upstream checkout is modified; use the clean pinned source")
    categories = args.categories or list(manifest["categories"])
    if len(set(categories)) != len(categories) or len(set(args.shots)) != len(args.shots) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate categories/shots/seeds")
    if not set(categories) <= set(manifest["categories"]) or not set(args.shots) <= set(manifest["shots"]) or not set(args.seeds) <= set(manifest["seeds"]):
        parser.error("Requested configuration is outside the manifest")
    # Validate every requested reference/calibration path before loading weights.
    for category in categories:
        item = manifest["categories"][category]
        if len(item["calibration"]) != 20:
            parser.error("Each category needs exactly 20 calibration images")
        for shot in args.shots:
            for seed in args.seeds:
                refs = item["references"][str(shot)][str(seed)]
                if (len(refs) != shot or len(set(refs)) != shot
                        or set(refs) & set(item["calibration"])
                        or not set(refs) <= set(item["eligible_normal"])):
                    parser.error("Reference/calibration protocol violated")
                for relative in refs + item["calibration"]:
                    if not relative.startswith(f"{category}/train/good/"):
                        parser.error("References/calibration must be normal training images")
                    checked_image(relative)
    args.output.mkdir(parents=True)
    metadata = {"status": "running", "method": "anomalydino", "run_id": args.output.name,
                "manifest_id": manifest_id, "code_commit": git_commit(ROOT),
                "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [ROOT / "scripts/run_part2.py", ROOT / "src/part2/inspector.py"]},
                "code_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()),
                "upstream_commit": UPSTREAM_COMMIT, "dinov2_commit": git_commit(args.dinov2_repo),
                "weights_sha256": digest(args.weights), "device": args.device,
                "torch": torch.__version__, "cuda_build": torch.version.cuda,
                "model": "dinov2_vits14", "resolution": 448, "preprocess": "agnostic",
                "precision": "float32", "faiss": "cpu", "threads": args.threads,
                "allowed_image_roots": [str(p.resolve()) for p in args.allow_image_root],
                "cache": "none", "categories": categories, "shots": args.shots, "seeds": args.seeds,
                "partial_smoke_test": args.limit_per_class is not None,
                "inference_time_scope": "synchronized RGB read, transforms, features, PCA mask, CPU FAISS; excludes map rendering/export",
                "peak_gpu_memory_scope": "model, reference bank construction and inference; CPU FAISS is not GPU memory",
                "notes": "Shared GPU performance must be measured in a coordinated slot. Metrics are a separate evaluation step."}
    run_path = args.output / "run.json"
    def save_status():
        run_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_status()
    try:
        inspector = AnomalyDINOInspector(args.upstream, args.dinov2_repo, args.weights, args.device, threads=args.threads)
        fields = ["run_id", "method", "manifest_id", "category", "shot", "seed", "image_path", "mask_path", "label", "score", "map_path", "threshold", "decision", "inference_ms"]
        with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            thresholds, calibration_rows, configurations = {}, [], []
            for category in categories:
                item = manifest["categories"][category]
                tests = item["tests"]
                if args.limit_per_class is not None:
                    tests = [x for label in [0, 1] for x in [t for t in tests if t["label"] == label][:args.limit_per_class]]
                for shot in args.shots:
                    for seed in args.seeds:
                        key = f"{category}/shot_{shot}/seed_{seed}"
                        print(f"START {key}: {len(tests)} test images", flush=True)
                        if args.device.startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats(args.device)
                        refs = item["references"][str(shot)][str(seed)]
                        # Warm up with a reference without accessing calibration/test labels.
                        for _ in range(2):
                            tensor, _ = inspector.backbone.prepare_image(str(checked_image(refs[0])))
                            inspector.backbone.extract_features(tensor)
                        inspector.fit_references([checked_image(p) for p in refs], seed)
                        state = np.random.get_state()
                        scores = []
                        for path in item["calibration"]:
                            prediction = inspector.predict(checked_image(path))
                            scores.append(prediction["score"])
                            calibration_rows.append({"configuration": key, "image_path": path, "score": prediction["score"]})
                        # Calibration must not change the randomized PCA sequence for test maps.
                        np.random.set_state(state)
                        threshold = float(np.quantile(scores, 0.95, method="linear"))
                        thresholds[key] = {"threshold": threshold, "references": refs, "calibration": item["calibration"]}
                        for index, sample in enumerate(tests):
                            prediction = inspector.predict(checked_image(sample["image_path"]))
                            relative = Path("maps") / key / Path(sample["image_path"]).parent.name / (Path(sample["image_path"]).stem + ".npz")
                            output_map = args.output / relative
                            output_map.parent.mkdir(parents=True, exist_ok=True)
                            np.savez_compressed(output_map, anomaly_map=prediction["anomaly_map"], patch_map=prediction["patch_map"])
                            writer.writerow({"run_id": args.output.name, "method": "anomalydino", "manifest_id": manifest_id,
                                "category": category, "shot": shot, "seed": seed, "image_path": sample["image_path"],
                                "mask_path": sample["mask_path"] or "", "label": sample["label"], "score": prediction["score"],
                                "map_path": relative.as_posix(), "threshold": threshold,
                                "decision": "NG" if prediction["score"] > threshold else "OK", "inference_ms": prediction["inference_ms"]})
                            handle.flush()
                            if index % 25 == 0:
                                print(f"PROGRESS {key} {index+1}/{len(tests)}", flush=True)
                        configurations.append({"configuration": key, "memory_bank_ms": inspector.bank_ms,
                            "test_count": len(tests), "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(args.device) / 1024**2 if args.device.startswith("cuda") else None})
            (args.output / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
            (args.output / "calibration.json").write_text(json.dumps(calibration_rows, indent=2) + "\n")
            metadata["configurations"] = configurations
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_status()
    print(f"Completed {args.output}; evaluate with the shared course evaluator.", flush=True)


if __name__ == "__main__":
    main()
