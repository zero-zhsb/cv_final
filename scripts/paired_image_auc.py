"""Paired, category/label-stratified image bootstrap for the frozen scale study."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

import numpy as np

RUNS = {
    "part2_448": "part2_gpu_exact_matrix_v1",
    "scale672": "part3_scale672_v1",
    "fusion448_672": "part3_fusion448_672_v1",
}


def load_predictions(path):
    groups = defaultdict(dict)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["category"], int(row["shot"]), int(row["seed"]))
            identity = (row["image_path"], row["mask_path"], row["label"], row["manifest_id"])
            if identity in groups[key]:
                raise ValueError(f"Duplicate image: {identity}")
            groups[key][identity] = float(row["score"])
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=6412)
    args = parser.parse_args()
    models = {variant: load_predictions(args.runs / run / "predictions.csv") for variant, run in RUNS.items()}
    configs = set(models["part2_448"])
    categories = sorted({category for category, _, _ in configs})
    shots = (1, 2, 4, 8)
    seeds = (0, 1, 2)
    if len(categories) != 6 or configs != {(c, k, s) for c in categories for k in shots for s in seeds}:
        raise ValueError("Expected the complete six-category study")
    if any(set(method) != configs for method in models.values()):
        raise ValueError("Methods do not have the same configurations")
    rng = np.random.default_rng(args.seed)
    boot = {name: np.zeros((len(shots), args.repeats)) for name in models}
    point = {name: np.zeros(len(shots)) for name in models}
    for category in categories:
        identities = sorted(models["part2_448"][(category, 1, 0)])
        positives = np.array([x[2] == "1" for x in identities])
        count_pos, count_neg = int(positives.sum()), int((~positives).sum())
        if not count_pos or not count_neg:
            raise ValueError(f"Both classes required: {category}")
        # Draw each test image once per replicate and reuse that draw across methods and reference seeds.
        weights_pos = rng.multinomial(count_pos, np.full(count_pos, 1 / count_pos), size=args.repeats).astype(np.float32)
        weights_neg = rng.multinomial(count_neg, np.full(count_neg, 1 / count_neg), size=args.repeats).astype(np.float32)
        scale = 1 / (len(categories) * len(seeds))
        for position, shot in enumerate(shots):
            for seed in seeds:
                for name, configs in models.items():
                    predictions = configs[(category, shot, seed)]
                    if set(predictions) != set(identities):
                        raise ValueError(f"Image identity differs: {name} {category} {shot} {seed}")
                    scores = np.array([predictions[x] for x in identities])
                    if not np.isfinite(scores).all():
                        raise ValueError("Nonfinite image scores")
                    a, b = scores[positives, None], scores[~positives][None, :]
                    wins = (a > b).astype(np.float32) + .5 * (a == b).astype(np.float32)
                    boot[name][position] += scale * ((weights_pos @ wins) * weights_neg).sum(
                        axis=1, dtype=np.float64) / (count_pos * count_neg)
                    point[name][position] += scale * wins.mean(dtype=np.float64)
    contrasts = []
    for control in ("part2_448", "scale672"):
        draws = 100 * (boot["fusion448_672"] - boot[control])
        effect = 100 * (point["fusion448_672"] - point[control])
        for index, shot in enumerate(shots):
            contrasts.append({
                "shot": shot,
                "control": control,
                "image_auroc_delta_pp": float(effect[index]),
                "percentile_ci95_pp": np.quantile(draws[index], [.025, .975]).tolist(),
                "bonferroni_interval_for_eight_contrasts_pp": np.quantile(
                    draws[index], [.025 / 8, 1 - .025 / 8]).tolist(),
            })
    print(json.dumps({
        "repeats": args.repeats,
        "rng_seed": args.seed,
        "population": "Observed VisA categories and fixed references; uncertainty from resampling test images",
        "paired_across": "all variants, shot counts and reference seeds",
        "strata": "category by test label",
        "contrasts": contrasts,
    }, indent=2))


if __name__ == "__main__":
    main()
