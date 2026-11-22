"""Select real threshold errors for qualitative analysis without changing the model."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shot", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--compare-run", action="append", default=[], metavar="LABEL=PATH",
                        help="Show the same images and common color scale for another completed run")
    parser.add_argument("--method-label", default="Raw anomaly map")
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.part2.maps import load_anomaly_map
    meta = json.loads((args.run / "run.json").read_text())
    if meta["status"] != "completed" or meta["partial_smoke_test"]:
        parser.error("Qualitative report requires the completed full run")
    with (args.run / "predictions.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["shot"]) == args.shot and int(row["seed"]) == args.seed]
    if not rows:
        parser.error("Requested configuration is absent")
    comparisons = []
    for comparison in args.compare_run:
        label, separator, directory = comparison.partition("=")
        if not separator or not label or not directory:
            parser.error("--compare-run requires LABEL=PATH")
        root = Path(directory)
        control = json.loads((root / "run.json").read_text())
        if (control["status"] != "completed" or control["partial_smoke_test"]
                or control["manifest_id"] != meta["manifest_id"]):
            parser.error("Comparison must be a completed run on the same manifest")
        with (root / "predictions.csv").open(newline="") as handle:
            matching = {r["image_path"]: r for r in csv.DictReader(handle)
                        if int(r["shot"]) == args.shot and int(r["seed"]) == args.seed}
        comparisons.append((label, root, matching))
    counts, errors = [], []
    for category in meta["categories"]:
        group = [r for r in rows if r["category"] == category]
        count = {"category": category, "shot": args.shot, "seed": args.seed, "TP": 0, "TN": 0, "FP": 0, "FN": 0}
        for row in group:
            correct = int(row["label"]) == int(row["decision"] == "NG")
            kind = ("T" if correct else "F") + ("P" if row["decision"] == "NG" else "N")
            count[kind] += 1
            if not correct:
                row = dict(row)
                row["error_type"] = kind
                row["relative_threshold_margin"] = abs(float(row["score"]) - float(row["threshold"])) / max(abs(float(row["threshold"])), 1e-12)
                errors.append(row)
        counts.append(count)
    errors.sort(key=lambda r: r["relative_threshold_margin"], reverse=True)
    selected, seen = [], set()
    for row in errors:
        if row["category"] not in seen:
            selected.append(row); seen.add(row["category"])
            if len(selected) == args.count:
                break
    for row in errors:
        if len(selected) >= args.count:
            break
        if row not in selected:
            selected.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "threshold_confusion.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(counts[0])); writer.writeheader(); writer.writerows(counts)
    maps = [load_anomaly_map(args.run / row["map_path"]) for row in selected]
    comparisons_by_image = []
    for row in selected:
        methods = []
        for label, root, matching in comparisons:
            other = matching[row["image_path"]]
            if (row["label"], row["mask_path"]) != (other["label"], other["mask_path"]):
                raise ValueError("Comparison prediction does not match the ground-truth identity")
            methods.append((label, other, load_anomaly_map(root / other["map_path"])))
        comparisons_by_image.append(methods)
    all_maps = maps + [m for methods in comparisons_by_image for _, _, m in methods]
    vmax = max((float(value.max()) for value in all_maps), default=1)
    for index, (row, anomaly, other_maps) in enumerate(zip(selected, maps, comparisons_by_image)):
        rgb = np.asarray(Image.open(args.data_root / row["image_path"]).convert("RGB"))
        mask = np.asarray(Image.open(args.data_root / row["mask_path"])) > 0 if row["mask_path"] else np.zeros(rgb.shape[:2], dtype=bool)
        columns = 4 + len(other_maps)
        fig, axes = plt.subplots(1, columns, figsize=(3 * columns, 3.5))
        axes[0].imshow(rgb); axes[0].set_title("Product image")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground truth")
        methods = [*other_maps, (args.method_label, row, anomaly)]
        for ax, (label, details, heatmap) in zip(axes[2:-1], methods):
            if heatmap.shape != mask.shape:
                raise ValueError("Heatmap resolution does not match ground truth")
            image = ax.imshow(heatmap, cmap="inferno", vmin=0, vmax=vmax)
            title = f"{label}\n{details['decision']}, score={float(details['score']):.4f}" if comparisons else label
            ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=axes[-2], fraction=0.046)
        axes[-1].imshow(rgb); axes[-1].imshow(anomaly, cmap="inferno", vmin=0, vmax=vmax, alpha=0.45)
        axes[-1].set_title("Overlay")
        for ax in axes: ax.axis("off")
        description = "False positive: normal product flagged NG" if row["error_type"] == "FP" else "False negative: defective product flagged OK"
        fig.suptitle(f"{row['category']} | {description}\nshot={args.shot}, seed={args.seed}, score={float(row['score']):.4f}, threshold={float(row['threshold']):.4f}", fontsize=10)
        fig.tight_layout(); name = f"failure_{index+1}_{row['category']}.png"
        fig.savefig(args.output / name, dpi=150); plt.close(fig)
        row["figure"] = name
        if other_maps:
            row["comparisons"] = {label: {k: details[k] for k in ("score", "threshold", "decision")}
                                  for label, details, _ in other_maps}
    report = {"shot": args.shot, "seed": args.seed, "selection": "actual threshold errors, one per category first, then largest relative error margin; no parameter tuning",
              "available_error_count": len(errors), "selected_count": len(selected),
              "common_raw_vmax": vmax, "cases": selected}
    (args.output / "cases.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"available_errors": len(errors), "selected": len(selected)}))


if __name__ == "__main__":
    main()
