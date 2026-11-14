"""Validate 72 complete configurations and export shared-protocol comparisons."""
import argparse
import csv
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/part1_full")
    parser.add_argument("--figures", type=Path, default=ROOT / "results/figures/part1_full")
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = json.loads((args.run / "run.json").read_text())
    expected = {(c, k, s) for c in meta["categories"] for k in meta["shots"] for s in meta["seeds"]}
    metrics = [json.loads(p.read_text()) for p in args.run.glob("*/metrics.json")]
    if meta["status"] != "completed" or {(r["category"], r["shot"], r["seed"]) for r in metrics} != expected or len(metrics) != 72:
        parser.error("All 72 configurations must finish")
    metrics.sort(key=lambda r: (r["category"], r["shot"], r["seed"]))
    predictions = [r for p in sorted(args.run.glob("*/predictions.csv")) for r in read(p)]
    b_rows = read(ROOT / "results/tables/part2_gpu_predictions.csv")
    def identities(rows):
        return {(r["category"], int(r["shot"]), int(r["seed"]), r["image_path"], r["mask_path"], int(r["label"]), r["manifest_id"]) for r in rows}
    if len(predictions) != 11520 or len(identities(predictions)) != len(predictions) or identities(predictions) != identities(b_rows):
        raise ValueError("Part 1/2 test identities differ")
    write(args.output / "per_configuration.csv", metrics)
    write(args.output / "predictions.csv", predictions)
    category_rows = []; overall = []
    for category in meta["categories"]:
        for shot in meta["shots"]:
            group = [r for r in metrics if r["category"] == category and r["shot"] == shot]
            row = {"category": category, "shot": shot, "seed_count": len(group)}
            for field in ["image_auroc", "pixel_auroc", "mean_inference_ms"]:
                row[field + "_mean"] = float(np.mean([r[field] for r in group]))
                row[field + "_std_ddof0"] = float(np.std([r[field] for r in group]))
            category_rows.append(row)
    for shot in meta["shots"]:
        row = {"shot": shot, "category_count": 6, "seed_count": 3}
        for field in ["image_auroc", "pixel_auroc"]:
            seeds = [np.mean([r[field] for r in metrics if r["shot"] == shot and r["seed"] == seed]) for seed in meta["seeds"]]
            row[field + "_mean"] = float(np.mean(seeds)); row[field + "_std_ddof0"] = float(np.std(seeds))
        row["mean_inference_ms"] = float(np.average([r["mean_inference_ms"] for r in metrics if r["shot"] == shot], weights=[r["test_count"] for r in metrics if r["shot"] == shot]))
        overall.append(row)
    write(args.output / "by_category_shot.csv", category_rows)
    write(args.output / "overall_by_shot.csv", overall)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.run / "run.json", args.output / "run.json")
    audit = {"same_reference_manifest": True, "identical_test_mask_label_ids": True,
             "configuration_count": 72, "prediction_count": len(predictions),
             "timing_limit": "CPU Part 1 vs historical GPU Part 2; observed latency only, not a controlled speed ratio",
             "pixel_auc": "exact nonnegative float32 key counts; original image resolution; normals included"}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    b = read(ROOT / "results/tables/part2_gpu_overall_by_shot.csv")
    comparison = []
    for a, other in zip(overall, b):
        assert a["shot"] == int(other["shot"])
        for name, row in [("Template difference", a), ("AnomalyDINO", other)]:
            comparison.append({"method": name, "shot": int(row["shot"]), **{k: float(row[k]) for k in ["image_auroc_mean", "image_auroc_std_ddof0", "pixel_auroc_mean", "pixel_auroc_std_ddof0"]}})
    write(args.output / "comparison.csv", comparison)
    args.figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, field, title in zip(axes, ["image_auroc", "pixel_auroc"], ["Image detection", "Pixel localization"]):
        for name, color in [("Template difference", "#d97706"), ("AnomalyDINO", "#2563eb")]:
            rows = [r for r in comparison if r["method"] == name]
            ax.errorbar([r["shot"] for r in rows], [100*r[field+"_mean"] for r in rows],
                        yerr=[100*r[field+"_std_ddof0"] for r in rows], marker="o", capsize=4, label=name, color=color)
        ax.set(xlabel="Normal reference images", ylabel="AUROC (%)", title=title, xticks=[1, 2, 4, 8], ylim=(40, 101))
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Six VisA categories | category macro average | mean +/- population std across seeds 0,1,2")
    fig.tight_layout(); fig.savefig(args.figures / "comparison.png", dpi=170); plt.close(fig)
    print(json.dumps({"audit": audit, "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
