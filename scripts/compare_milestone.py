"""Validate identical test identities and compare A's single-config result with B."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.run / "run.json").read_text())
    with (args.run / "predictions.csv").open(newline="") as handle:
        a = list(csv.DictReader(handle))
    configuration = meta["configurations"][0]["configuration"]
    if len(meta["configurations"]) != 1:
        parser.error("This milestone comparison expects one configuration")
    category, shot_text, seed_text = configuration.split("/")
    shot, seed = int(shot_text[5:]), int(seed_text[5:])
    def selected(row):
        return row["category"] == category and int(row["shot"]) == shot and int(row["seed"]) == seed
    with (ROOT / "results/tables/part2_gpu_predictions.csv").open(newline="") as handle:
        b = [r for r in csv.DictReader(handle) if selected(r)]
    def identity(rows):
        return {(r["image_path"], r["mask_path"], int(r["label"]), r["manifest_id"]) for r in rows}
    if len(a) != len(b) or len(identity(a)) != len(a) or identity(a) != identity(b):
        raise ValueError("Part 1/2 test identity, mask, label or manifest mismatch")
    a_metric = json.loads((args.run / "metrics" / (configuration.replace("/", "__") + ".json")).read_text())
    with (ROOT / "results/tables/part2_gpu_per_configuration.csv").open(newline="") as handle:
        b_metrics = [r for r in csv.DictReader(handle) if selected(r)]
    if len(b_metrics) != 1:
        raise ValueError("Expected exactly one Part 2 metric row")
    rows = []
    for method, values, source in [("template_difference", a_metric, args.run.name),
                                   ("anomalydino", b_metrics[0], "part2_gpu_exact_matrix_v1")]:
        rows.append({"method": method, "configuration": configuration, "manifest_id": meta["manifest_id"],
                     "test_count": len(a), "image_auroc": float(values["image_auroc"]),
                     "pixel_auroc": float(values["pixel_auroc"]),
                     "mean_inference_ms": float(values["mean_inference_ms"]),
                     "peak_gpu_memory_mib": values["peak_gpu_memory_mib"], "source_run": source,
                     "timing_note": "CPU template vs historical GPU DINO; shared host, different load; descriptive only"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"same_test_identity": True, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
