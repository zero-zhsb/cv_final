"""Render genuine saved predictions; never changes arrays used for evaluation."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    with (args.run / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:args.count]
    if not rows:
        parser.error("No predictions")
    arrays = []
    for row in rows:
        with np.load(args.run / row["map_path"]) as data:
            arrays.append(data["anomaly_map"])
    # One scale across this preview set; raw saved maps remain untouched.
    minimum = min(float(x.min()) for x in arrays)
    maximum = max(float(x.max()) for x in arrays)
    args.output.mkdir(parents=True, exist_ok=True)
    for i, (row, anomaly) in enumerate(zip(rows, arrays)):
        rgb = np.array(Image.open(args.data_root / row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(args.data_root / row["mask_path"])) > 0 if row["mask_path"] else np.zeros(rgb.shape[:2], dtype=bool)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
        axes[0].imshow(rgb); axes[0].set_title("RGB")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground truth (evaluation only)")
        im = axes[2].imshow(anomaly, cmap="inferno", vmin=minimum, vmax=maximum)
        axes[2].set_title("Raw anomaly map")
        fig.colorbar(im, ax=axes[2], fraction=0.046)
        axes[3].imshow(rgb)
        axes[3].imshow(anomaly, cmap="inferno", vmin=minimum, vmax=maximum, alpha=0.45)
        axes[3].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"CPU smoke example | {row['category']} | shot={row['shot']} seed={row['seed']} | "
                     f"score={float(row['score']):.4f}, threshold={float(row['threshold']):.4f}, {row['decision']}")
        fig.tight_layout()
        fig.savefig(args.output / f"part2_smoke_{i+1}.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
