"""Compare five genuine fusion errors against the preserved Part 2 on the same image."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.part2.maps import load_anomaly_map

    baseline = ROOT / "runs/part2_gpu_exact_matrix_v1"
    proposed = ROOT / "runs/part3_fusion6_12_v1"
    data = Path("/data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls")
    output = ROOT / "results/figures/part3/paired_failures"
    selected_report = json.loads((ROOT / "results/figures/part3_fusion_failures/cases.json").read_text())
    if selected_report["selected_count"] < 5:
        raise RuntimeError("Need five genuine errors; do not fabricate cases")
    def read(run):
        if json.loads((run / "run.json").read_text())["status"] != "completed":
            raise RuntimeError("Full completed study required")
        with (run / "predictions.csv").open() as handle:
            return {(r["category"], int(r["shot"]), int(r["seed"]), r["image_path"]): r
                    for r in csv.DictReader(handle)}
    old_rows, new_rows = read(baseline), read(proposed)
    pairs = []
    for case in selected_report["cases"][:5]:
        key = (case["category"], int(case["shot"]), int(case["seed"]), case["image_path"])
        a, b = old_rows[key], new_rows[key]
        if a["label"] != b["label"] or a["mask_path"] != b["mask_path"]:
            raise ValueError("Mismatched image identity")
        pairs.append((case, a, b, load_anomaly_map(baseline / a["map_path"]),
                      load_anomaly_map(proposed / b["map_path"])))
    # A single raw color scale for ALL cases and both methods, never per-image normalization.
    vmax = max(float(m.max()) for _, _, _, m1, m2 in pairs for m in [m1, m2])
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    lines = ["# Part 3：五个真实困难案例与 Part 2 同图对照", "",
             "固定选取融合方法在 8-shot、seed=0 的真实阈值错误：优先覆盖不同类别，再按相对阈值偏差排序。"
             "这是事后错误分析，不用于选层或调整权重。所有图片、两方法使用同一原始分数色标；没有逐图归一化。", "",
             "图像 AUROC 衡量排序，OK/NG 则由各方法正常校准集的 95% 分位数决定，两者需要分别解释。", ""]
    for index, (case, a, b, m1, m2) in enumerate(pairs, 1):
        image = np.asarray(Image.open(data / b["image_path"]).convert("RGB"))
        mask = (np.asarray(Image.open(data / b["mask_path"])) > 0
                if b["mask_path"] else np.zeros(image.shape[:2], dtype=bool))
        if m1.shape != mask.shape or m2.shape != mask.shape:
            raise ValueError("Original-resolution map and ground truth must agree")
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), layout="constrained")
        axes[0].imshow(image); axes[0].set_title("Product image")
        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground truth")
        for ax, m, label, row in [(axes[2], m1, "Part 2 / layer 12", a),
                                  (axes[3], m2, "Part 3 / fusion 6+12", b)]:
            heat = ax.imshow(m, cmap="inferno", vmin=0, vmax=vmax)
            ax.set_title(f"{label}\n{row['decision']}, score={float(row['score']):.4f}, t={float(row['threshold']):.4f}", fontsize=10)
        for ax in axes: ax.axis("off")
        fig.colorbar(heat, ax=axes[2:], shrink=.75, label="Raw anomaly distance (common scale)")
        fig.suptitle(f"{case['category']} | {case['error_type']} | 8-shot, seed 0 | {Path(b['image_path']).name}")
        filename = f"case_{index}_{case['category']}.png"
        fig.savefig(output / filename, dpi=160); plt.close(fig)
        # Quantify map concentration rather than assigning an unsupported visual cause.
        detail = {"case": index, "category": case["category"], "image_path": b["image_path"],
                  "error_type": case["error_type"], "label": int(b["label"]),
                  "defect_pixel_fraction": float(mask.mean()), "figure": filename,
                  "baseline": {k: a[k] for k in ["score", "threshold", "decision"]},
                  "fusion": {k: b[k] for k in ["score", "threshold", "decision"]}}
        for name, m in [("baseline", m1), ("fusion", m2)]:
            cutoff = float(np.quantile(m, .99))
            detail[name]["fraction_top1pct_map_pixels_inside_gt"] = (float(mask[m >= cutoff].mean()) if mask.any() else None)
            detail[name]["mean_map_inside_gt"] = float(m[mask].mean()) if mask.any() else None
            detail[name]["mean_map_outside_gt"] = float(m[~mask].mean()) if (~mask).any() else None
        cases.append(detail)
        kind = "正常图误报" if case["error_type"] == "FP" else "异常图漏检"
        lines.extend([f"## 案例 {index}：{case['category']} — {kind}", "",
                      f"图片 `{b['image_path']}`；GT 缺陷面积占比 {100*mask.mean():.3f}%。"
                      f"Part 2：{a['decision']}（分数 {float(a['score']):.4f} / 阈值 {float(a['threshold']):.4f}）；"
                      f"融合：{b['decision']}（{float(b['score']):.4f} / {float(b['threshold']):.4f}）。", "",
                      f"![同图原始热图对照](../results/figures/part3/paired_failures/{filename})", "",
                      "观察需结合原图、GT 与共享色标热图，不能仅凭阈值错误归因于背景或缺陷大小。"
                      "`cases.json` 另外记录缺陷内/外均值及原分辨率热图 top-1% 的 GT 覆盖比例；后者不是图像分数，也不是 pixel AUROC。", ""])
    (output / "cases.json").write_text(json.dumps({"selection": selected_report["selection"],
        "common_raw_vmin": 0, "common_raw_vmax": vmax, "cases": cases}, indent=2) + "\n")
    (ROOT / "docs/16_part3_failure_analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
