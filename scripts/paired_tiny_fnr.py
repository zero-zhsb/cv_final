"""Paired image bootstrap of small-defect miss rate at the fixed test FPR cap."""
import argparse
import json
from pathlib import Path

from paired_image_auc import RUNS, load_predictions, np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/part3_multiscale_protocol.json")
    parser.add_argument("--repeats", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2618)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    cap = protocol["matched_test_fpr"]
    max_area = protocol["tiny_defect_mask_fraction_max"]
    models = {name: load_predictions(args.runs / run / "predictions.csv") for name, run in RUNS.items()}
    configs = set(models["part2_448"])
    categories = sorted({category for category, _, _ in configs})
    shots = (1, 2, 4, 8)
    seeds = (0, 1, 2)
    if len(categories) != 6 or configs != {(c, k, s) for c in categories for k in shots for s in seeds}:
        raise ValueError("Expected complete paired six-category results")
    if any(set(method) != configs for method in models.values()):
        raise ValueError("Methods do not have the same configurations")
    rng = np.random.default_rng(args.seed)
    draws = {name: np.zeros((len(shots), args.repeats)) for name in models}
    point = {name: np.zeros(len(shots)) for name in models}
    counts = {}
    for cat in categories:
        identities = set(models["part2_448"][(cat, 1, 0)])
        normal = sorted(i for i in identities if i[2] == "0")
        tiny = []
        for identity in sorted(i for i in identities if i[2] == "1"):
            with Image.open(args.data_root / identity[1]) as handle:
                fraction = (np.asarray(handle.convert("L")) > 0).mean()
            if 0 < fraction <= max_area:
                tiny.append(identity)
        if not tiny or not normal:
            raise ValueError(f"Both tiny defects and normal test images required in {cat}")
        counts[cat] = len(tiny)
        sampled_normal = rng.integers(0, len(normal), size=(args.repeats, len(normal)))
        sampled_tiny = rng.integers(0, len(tiny), size=(args.repeats, len(tiny)))
        # The reduction uses the lowest empirical threshold satisfying the normal FPR cap.
        index = len(normal) - 1 - int(len(normal) * cap)
        weight = 1 / (len(categories) * len(seeds))
        for si, shot in enumerate(shots):
            for seed in seeds:
                for name, predictions in models.items():
                    group = predictions[(cat, shot, seed)]
                    if set(group) != identities:
                        raise ValueError(f"Mismatched identities: {name}/{cat}/{shot}/{seed}")
                    negatives = np.array([group[x] for x in normal])
                    defects = np.array([group[x] for x in tiny])
                    cutoff = np.partition(negatives, index)[index]
                    boot_cutoff = np.partition(negatives[sampled_normal], index, axis=1)[:, index]
                    point[name][si] += weight * (defects <= cutoff).mean()
                    draws[name][si] += weight * (defects[sampled_tiny] <= boot_cutoff[:, None]).mean(axis=1)

    contrasts = []
    for control in ("part2_448", "scale672"):
        difference = 100 * (draws["fusion448_672"] - draws[control])
        for si, shot in enumerate(shots):
            contrasts.append({
                "shot": shot, "control": control,
                "control_tiny_fnr_percent": float(100 * point[control][si]),
                "fusion_tiny_fnr_percent": float(100 * point["fusion448_672"][si]),
                "delta_pp": float(100 * (point["fusion448_672"][si] - point[control][si])),
                "percentile_ci95_pp": np.quantile(difference[si], [.025, .975]).tolist(),
                "bonferroni_interval_for_eight_contrasts_pp": np.quantile(
                    difference[si], [.025 / 8, 1 - .025 / 8]).tolist(),
            })
    print(json.dumps({
        "repeats": args.repeats, "rng_seed": args.seed, "test_fpr_cap": cap,
        "tiny_mask_fraction_max": max_area, "tiny_counts_by_category": counts,
        "estimand": "Conditional on the six categories and fixed reference selections",
        "bootstrapping": "Paired normal and tiny image draws within each category, reused across all methods, shots and seeds; threshold recomputed in each draw",
        "contrasts": contrasts,
    }, indent=2))


if __name__ == "__main__":
    main()
