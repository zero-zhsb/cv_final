"""Single-image live Part 1 demo; optional explicitly labelled Part 2 result replay."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", default="cashew")
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.part1.inspector import TemplateInspector
    manifest = json.loads((ROOT / "manifests/visa_1cls.json").read_text())
    metadata = json.loads((args.run / "run.json").read_text())
    if metadata["manifest_id"] != manifest["manifest_id"] or metadata["method"] != "template_difference":
        parser.error("Run/manifest mismatch")
    if args.output.exists():
        parser.error("Output directory exists; preserving it")
    key = f"{args.category}/shot_{args.shot}/seed_{args.seed}"
    settings = json.loads((args.run / "thresholds.json").read_text())[key]
    if settings["references"] != manifest["categories"][args.category]["references"][str(args.shot)][str(args.seed)]:
        parser.error("Threshold/reference mismatch")
    inspector = TemplateInspector()
    inspector.fit_references([args.data_root / path for path in settings["references"]])
    prediction = inspector.predict(args.image)
    threshold = settings["threshold"]
    result = {"method": "template_difference", "mode": "live inference; no ground truth input",
              "category": args.category, "shot": args.shot, "seed": args.seed,
              "manifest_id": manifest["manifest_id"], "reference_images": settings["references"],
              "score": prediction["score"], "threshold": threshold,
              "decision": "NG" if prediction["score"] > threshold else "OK",
              "inference_ms": prediction["inference_ms"], "part2_replay": None}
    # This optional row is lookup-only. It is NEVER used by Part 1 inference.
    try:
        relative = args.image.resolve().relative_to(args.data_root.resolve()).as_posix()
    except ValueError:
        relative = None
    with (ROOT / "results/tables/part2_gpu_predictions.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["image_path"] == relative
                and r["category"] == args.category and int(r["shot"]) == args.shot and int(r["seed"]) == args.seed]
    if len(rows) == 1 and rows[0]["manifest_id"] == manifest["manifest_id"]:
        row = rows[0]
        result["part2_replay"] = {"mode": "saved Part 2 score replay; NOT live DINO inference",
                                  "score": float(row["score"]), "threshold": float(row["threshold"]),
                                  "decision": row["decision"], "source_run": row["run_id"]}
    args.output.mkdir(parents=True)
    (args.output / "prediction.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.output / "anomaly_map.npz", anomaly_map=prediction["anomaly_map"])
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(prediction["nearest_reference"]) as image:
        reference = np.asarray(image.convert("RGB"))
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    axes[0].imshow(rgb); axes[0].set_title("Product input")
    axes[1].imshow(reference); axes[1].set_title("Nearest normal template")
    plot = axes[2].imshow(prediction["anomaly_map"], cmap="inferno", vmin=0)
    axes[2].set_title("Part 1 live anomaly map")
    fig.colorbar(plot, ax=axes[2], fraction=0.046)
    axes[3].imshow(rgb)
    axes[3].imshow(prediction["anomaly_map"], cmap="inferno", vmin=0, alpha=0.45)
    axes[3].set_title("Part 1 live overlay")
    for ax in axes:
        ax.axis("off")
    caption = f"Part 1 LIVE | score={result['score']:.4f} | threshold={threshold:.4f} | {result['decision']}"
    if result["part2_replay"]:
        replay = result["part2_replay"]
        caption += f"\nPart 2 SAVED SCORE ONLY | score={replay['score']:.4f} | threshold={replay['threshold']:.4f} | {replay['decision']}"
    fig.suptitle(caption, fontsize=11)
    fig.tight_layout(); fig.savefig(args.output / "prediction.png", dpi=160); plt.close(fig)
    # Join the already published B example only when image/config/reference identity
    # matches; this is labelled replay and never presented as a second live model.
    example = ROOT / "demo/examples/part2"
    b_example = json.loads((example / "prediction.json").read_text())
    expected_suffix = "/" + relative if relative else None
    if (expected_suffix and b_example["input_image"].endswith(expected_suffix)
            and b_example["category"] == args.category and b_example["shot"] == args.shot
            and b_example["seed"] == args.seed and b_example["manifest_id"] == manifest["manifest_id"]
            and b_example["reference_images"] == settings["references"]):
        fig, axes = plt.subplots(2, 1, figsize=(14, 9))
        axes[0].imshow(plt.imread(args.output / "prediction.png"))
        axes[0].set_title("A: Part 1 LIVE inference", fontsize=12)
        axes[1].imshow(plt.imread(example / "prediction.png"))
        axes[1].set_title("B: published Part 2 1-shot CPU example REPLAY (not live, not the GPU timing benchmark)", fontsize=11)
        for ax in axes:
            ax.axis("off")
        fig.suptitle("Same product image, reference, shot and seed | cashew/test/bad/000.JPG", fontsize=12)
        fig.tight_layout(); fig.savefig(args.output / "comparison.png", dpi=150); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
