"""Run live inspection on product images using only normal references/calibration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("template", "part2_448", "scale672", "fusion448_672", "fusion448_672_oas"),
                        default="fusion448_672")
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--shot", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.method == "fusion448_672_oas" and args.device != "cuda:0":
        parser.error("The measured OAS implementation requires --device cuda:0")
    manifest = json.loads((ROOT / "manifests/visa_1cls.json").read_text())
    if args.category not in manifest["categories"]:
        parser.error("Choose a category in the VisA manifest")
    if any(not image.is_file() for image in args.image):
        parser.error("Every product image must be a readable file")
    if args.output.exists():
        parser.error("Choose a new output directory")
    config = manifest["categories"][args.category]
    references = config["references"][str(args.shot)][str(args.seed)]
    calibration = config["calibration"]
    if (len(calibration) != 20 or len(references) != args.shot
            or set(references) & set(calibration)):
        parser.error("Reference and calibration lists do not follow the course split")

    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.method == "template":
        if args.device != "cpu":
            parser.error("The template baseline runs on the CPU")
        from src.part1.inspector import TemplateInspector
        inspector = TemplateInspector()
        backend = "cpu_template"
    else:
        import torch
        from scripts.run_oas import digest, git_commit
        from src.part2.inspector import AnomalyDINOInspector

        upstream = ROOT / "third_party/AnomalyDINO"
        dinov2 = ROOT / "third_party/dinov2"
        weights = ROOT / "weights/dinov2_vits14_pretrain.pth"
        sources = json.loads((ROOT / "results/tables/part3_multiscale/sources.json").read_text())
        source = sources["fusion448_672" if args.method == "fusion448_672_oas" else args.method]
        if not weights.is_file() or not upstream.is_dir() or not dinov2.is_dir():
            parser.error("Set up the pinned DINOv2 weights and source repositories first")
        if (digest(weights) != source["weights_sha256"]
                or git_commit(upstream) != source["upstream_commit"]
                or git_commit(dinov2) != source["dinov2_commit"]):
            parser.error("DINOv2 weights or source versions do not match the study")
        if args.device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(.15, args.device)
        backend = "torch-gpu" if args.device.startswith("cuda") else "cpu-faiss"
        if args.method == "fusion448_672_oas":
            from src.part3.oas import OASInspector
            inspector = OASInspector(upstream, dinov2, weights, args.device, knn_backend=backend)
        else:
            inspector = AnomalyDINOInspector(upstream, dinov2, weights, args.device,
                                             resolution=source["resolution"], knn_backend=backend)
        inspector.backbone.model.requires_grad_(False)
        if args.method == "fusion448_672":
            from src.part3.multiscale import attach_multiscale
            attach_multiscale(inspector.backbone)

    inspector.fit_references([args.data_root / x for x in references], args.seed)
    state = np.random.get_state()
    normal_scores = [inspector.predict(args.data_root / x, render_map=False)["score"] for x in calibration]
    threshold = float(np.quantile(normal_scores, .95, method="linear"))
    args.output.mkdir(parents=True)
    for index, path in enumerate(args.image, 1):
        # Keep single-image behavior independent of other inputs in the same invocation.
        np.random.set_state(state)
        pred = inspector.predict(path)
        score = float(pred["score"])
        decision = "NG" if score > threshold else "OK"
        result = {"method": args.method, "image": path.name, "category": args.category,
                  "shot": args.shot, "seed": args.seed, "device": args.device,
                  "backend": backend, "manifest_id": manifest["manifest_id"],
                  "score": score, "threshold": threshold, "decision": decision,
                  "image_shape": pred["image_shape"], "inference_ms": pred["inference_ms"],
                  "reference_images": references, "calibration_count": len(calibration)}
        target = args.output / f"{index:02d}_{path.stem}"
        target.mkdir()
        (target / "prediction.json").write_text(json.dumps(result, indent=2) + "\n")
        anomaly = pred["anomaly_map"]
        np.savez_compressed(target / "anomaly_map.npz", anomaly_map=anomaly)
        with Image.open(path) as image:
            rgb = np.array(image.convert("RGB"))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
        axes[0].imshow(rgb)
        axes[1].imshow(anomaly, cmap="inferno", vmin=0)
        axes[2].imshow(rgb)
        axes[2].imshow(anomaly, cmap="inferno", vmin=0, alpha=.45)
        for ax, title in zip(axes, ("Product image", "Anomaly heatmap", "Overlay")):
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"{args.method} | {path.name} | {decision} | score {score:.4f} / threshold {threshold:.4f}")
        fig.savefig(target / "prediction.png", dpi=160)
        plt.close(fig)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
