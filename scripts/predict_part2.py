"""Single product-image inference for the Part 2 demo, without ground truth."""
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
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--shot", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", type=Path, required=True, help="Run with the matching calibrated threshold")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    args = parser.parse_args()
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.part2.inspector import AnomalyDINOInspector
    from scripts.run_part2 import digest, git_commit
    manifest = json.loads((ROOT / "manifests/visa_1cls.json").read_text())
    meta = json.loads((args.run / "run.json").read_text())
    if meta["manifest_id"] != manifest["manifest_id"]:
        parser.error("Threshold run and manifest differ")
    if (digest(ROOT / "weights/dinov2_vits14_pretrain.pth") != meta["weights_sha256"]
            or git_commit(ROOT / "third_party/dinov2") != meta["dinov2_commit"]
            or git_commit(ROOT / "third_party/AnomalyDINO") != meta["upstream_commit"]):
        parser.error("Model source or weights differ from the calibrated run")
    key = f"{args.category}/shot_{args.shot}/seed_{args.seed}"
    thresholds = json.loads((args.run / "thresholds.json").read_text())
    if key not in thresholds:
        parser.error("Requested category/shot/seed has no completed calibration")
    calibration = thresholds[key]
    if calibration["references"] != manifest["categories"][args.category]["references"][str(args.shot)][str(args.seed)]:
        parser.error("Threshold references differ from the shared manifest")
    if args.output.exists():
        parser.error("Choose a new output directory")
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.15, args.device)
    backend = meta.get("knn_backend", "cpu-faiss")
    if backend == "torch-gpu" and args.device == "cpu":
        parser.error("The calibrated run uses GPU neighbors; select --device cuda:0")
    inspector = AnomalyDINOInspector(ROOT / "third_party/AnomalyDINO", ROOT / "third_party/dinov2", ROOT / "weights/dinov2_vits14_pretrain.pth", args.device, knn_backend=backend)
    inspector.fit_references([args.data_root / path for path in calibration["references"]], args.seed)
    prediction = inspector.predict(args.image)
    score, threshold = prediction["score"], calibration["threshold"]
    result = {"method": "AnomalyDINO", "category": args.category, "shot": args.shot, "seed": args.seed,
              "score": score, "threshold": threshold, "decision": "NG" if score > threshold else "OK",
              "manifest_id": manifest["manifest_id"], "reference_images": calibration["references"],
              "input_image": str(args.image), "device": args.device, "inference_ms": prediction["inference_ms"]}
    args.output.mkdir(parents=True)
    (args.output / "prediction.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output / "anomaly_map.npz", anomaly_map=prediction["anomaly_map"], patch_map=prediction["patch_map"])
    rgb = inspector.read_rgb(args.image)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    axes[0].imshow(rgb); axes[0].set_title("Product image")
    plot = axes[1].imshow(prediction["anomaly_map"], cmap="inferno", vmin=0)
    axes[1].set_title("Anomaly heatmap"); fig.colorbar(plot, ax=axes[1], fraction=0.046)
    axes[2].imshow(rgb); axes[2].imshow(prediction["anomaly_map"], cmap="inferno", alpha=0.45, vmin=0)
    axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"{args.category} | score={score:.4f} | threshold={threshold:.4f} | {result['decision']}")
    fig.tight_layout(); fig.savefig(args.output / "prediction.png", dpi=150); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
