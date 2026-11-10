"""Export all-seed course summaries and scientific plots after complete evaluation."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    meta = json.loads((args.run / "run.json").read_text())
    expected = len(meta["categories"]) * len(meta["shots"]) * len(meta["seeds"])
    reports = [json.loads(p.read_text()) for p in (args.run / "metrics").glob("*.json")]
    if meta["status"] != "completed" or meta["partial_smoke_test"] or len(reports) != expected:
        parser.error("Full inference and all configuration evaluations must finish first")
    keys = [(r["category"], r["shot"], r["seed"]) for r in reports]
    expected_keys = {(c, k, s) for c in meta["categories"] for k in meta["shots"] for s in meta["seeds"]}
    if set(keys) != expected_keys or len(set(keys)) != expected:
        parser.error("Missing or duplicated configuration metrics")
    reports.sort(key=lambda r: (r["category"], r["shot"], r["seed"]))
    write_csv(args.output / "part2_gpu_per_configuration.csv", reports)
    category_rows = []
    for category in meta["categories"]:
        for shot in meta["shots"]:
            group = [r for r in reports if r["category"] == category and r["shot"] == shot]
            row = {"category": category, "shot": shot, "seed_count": len(group)}
            for key in ["image_auroc", "pixel_auroc", "mean_inference_ms", "peak_gpu_memory_mib"]:
                values = [r[key] for r in group]
                row[key + "_mean"] = float(np.mean(values))
                row[key + "_std_ddof0"] = float(np.std(values, ddof=0))
            category_rows.append(row)
    write_csv(args.output / "part2_gpu_by_category_shot.csv", category_rows)
    overall_rows = []
    for shot in meta["shots"]:
        seed_rows = []
        for seed in meta["seeds"]:
            group = [r for r in reports if r["shot"] == shot and r["seed"] == seed]
            seed_rows.append({"image_auroc": np.mean([r["image_auroc"] for r in group]),
                "pixel_auroc": np.mean([r["pixel_auroc"] for r in group]),
                "micro_mean_inference_ms": sum(r["mean_inference_ms"] * r["test_count"] for r in group) / sum(r["test_count"] for r in group),
                "peak_gpu_memory_mib": max(r["peak_gpu_memory_mib"] for r in group)})
        row = {"shot": shot, "category_count": len(meta["categories"]), "seed_count": len(seed_rows)}
        for key in seed_rows[0]:
            values = [r[key] for r in seed_rows]
            row[key + "_mean"] = float(np.mean(values))
            row[key + "_std_ddof0"] = float(np.std(values, ddof=0))
        overall_rows.append(row)
    write_csv(args.output / "part2_gpu_overall_by_shot.csv", overall_rows)
    args.figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(meta["shots"]))
    for ax, key, title in zip(axes, ["image_auroc", "pixel_auroc"], ["Image detection", "Pixel localization"]):
        mean = np.array([r[key + "_mean"] for r in overall_rows])
        std = np.array([r[key + "_std_ddof0"] for r in overall_rows])
        ax.errorbar(x, mean, yerr=std, marker="o", capsize=5, color="#1565C0")
        for a, b in zip(x, mean):
            ax.annotate(f"{b*100:.2f}%", (a, b), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
        ax.set_xticks(x, meta["shots"]); ax.set_xlabel("Normal reference images (shots)")
        ax.set_ylabel("AUROC"); ax.set_ylim(0.5, 1.04); ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title); ax.grid(alpha=0.2)
    fig.suptitle("AnomalyDINO | six VisA categories | three seeds\nCategory macro average; error bars: population standard deviation across seeds", fontsize=11)
    fig.tight_layout(); fig.savefig(args.figures / "part2_gpu_few_shot_curve.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in zip(axes, ["image_auroc", "pixel_auroc"], ["Image AUROC by category", "Pixel AUROC by category"]):
        for category in meta["categories"]:
            group = [r for r in category_rows if r["category"] == category]
            ax.plot(x, [r[key + "_mean"] for r in group], marker="o", label=category, linewidth=1.3)
        ax.set_xticks(x, meta["shots"]); ax.set_xlabel("Shots"); ax.set_ylim(0.5, 1.01)
        ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(title); ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("All required categories; each point averages seeds 0, 1, 2")
    fig.tight_layout(); fig.savefig(args.figures / "part2_gpu_category_curves.png", dpi=180); plt.close(fig)
    shutil.copyfile(args.run / "run.json", args.output / "part2_gpu_run.json")
    shutil.copyfile(args.run / "predictions.csv", args.output / "part2_gpu_predictions.csv")
    print(json.dumps(overall_rows, indent=2))


if __name__ == "__main__":
    main()
