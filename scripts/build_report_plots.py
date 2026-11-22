"""Build the CVPR report figures from the audited result tables."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report/figures"
TABLES = ROOT / "results/tables"
PALETTE = {"Template": "#6B7280", "448": "#1F5A99", "672": "#C26213", "448+672": "#00786E"}


def read(name):
    with (TABLES / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.65))
    ax.set(xlim=(0, 10), ylim=(0, 2.8))
    ax.axis("off")

    def box(x, y, w, h, label, border, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                   facecolor=fill, edgecolor=border, linewidth=1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.8, color="#142132")

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#495970"})

    box(.09, 1.07, 1.15, .65, "Product or\nreference", "#6B7280", "#F2F5F8")
    box(1.65, 1.84, 1.42, .54, "448 / block 12", PALETTE["448"], "#EAF3FE")
    box(1.65, .40, 1.42, .54, "672 / block 12", PALETTE["672"], "#FFF2E4")
    box(3.46, 1.84, 1.35, .54, "Align to the\n672 grid", PALETTE["448"], "#EAF3FE")
    box(5.18, 1.07, 1.26, .65, "Unit vectors\nEqual weights", PALETTE["448+672"], "#E4F5F1")
    box(6.85, 1.07, 1.23, .65, "Exact nearest\nnormal patch", PALETTE["448+672"], "#E4F5F1")
    box(8.50, 1.07, 1.38, .65, "Score / OK–NG\nand heatmap", "#6B7280", "#F2F5F8")
    box(4.08, .23, 1.12, .48, "PCA mask\n(test only)", PALETTE["672"], "#FFF2E4")
    for start, end in [((1.27, 1.52), (1.60, 2.06)), ((1.27, 1.27), (1.60, .72)),
                       ((3.11, 2.12), (3.41, 2.12)), ((4.86, 2.10), (5.27, 1.76)),
                       ((3.11, .79), (5.12, 1.24)), ((3.11, .53), (4.02, .47)),
                       ((5.25, .47), (6.96, 1.03)), ((6.49, 1.41), (6.80, 1.41)),
                       ((8.12, 1.41), (8.45, 1.41))]:
        arrow(start, end)
    ax.text(2.37, 2.66, "Same frozen DINOv2 ViT-S/14 at both scales",
            ha="center", fontsize=9, color="#142132")
    ax.text(6.93, .27, "References build the joint patch bank", ha="center", fontsize=8, color="#495970")
    fig.savefig(OUTPUT / "pipeline.pdf", bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


def metrics():
    rows = read("part3_multiscale/overall_by_shot.csv")
    pairs = {"448": "part2_448", "672": "scale672", "448+672": "fusion448_672"}
    tiny = read("part3_multiscale/tiny_defects_by_shot.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.65), layout="constrained")
    labels = ["Image AUROC (%)", "Pixel AUROC (%)", "Small-defect miss rate (%)"]
    for label, key in pairs.items():
        selected = [r for r in rows if r["variant"] == key]
        for ax, name in zip(axes[:2], ("image_auroc", "pixel_auroc")):
            ax.errorbar([int(r["shot"]) for r in selected],
                        [100 * float(r[name + "_mean"]) for r in selected],
                        yerr=[100 * float(r[name + "_std_ddof0"]) for r in selected],
                        color=PALETTE[label], marker="o", ms=3, capsize=2.5, label=label)
        selected = [r for r in tiny if r["variant"] == key]
        axes[2].plot([int(r["shot"]) for r in selected],
                     [100 * float(r["tiny_fnr_at_test_fpr_cap"]) for r in selected],
                     color=PALETTE[label], marker="o", ms=3, label=label)
    for ax, label in zip(axes, labels):
        ax.set(ylabel=label, xlabel="Normal references", xticks=[1, 2, 4, 8])
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.23)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.savefig(OUTPUT / "metrics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    metrics()
