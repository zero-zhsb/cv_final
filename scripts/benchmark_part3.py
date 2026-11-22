"""Counterbalanced, same-session efficiency sample; never changes model settings."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["multilayer", "multiscale"], default="multilayer")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls"))
    args = parser.parse_args()
    multiscale = args.study == "multiscale"
    import numpy as np
    import torch
    from src.part2.inspector import AnomalyDINOInspector
    from src.part3.inspector import MultilayerInspector
    from src.part3.multiscale import attach_multiscale

    status = "part3_multiscale_status.json" if multiscale else "part3_study_status.json"
    study = json.loads((ROOT / "runs" / status).read_text())
    ready = {"benchmarking", "artifacts_ready"} if multiscale else {"completed"}
    if study["stage"] not in ready:
        raise RuntimeError("Finish the full inference and evaluation study first")
    report_dir = "part3_multiscale" if multiscale else "part3"
    output = ROOT / "results/tables" / report_dir / "efficiency.json"
    if output.exists():
        raise FileExistsError("Preserving existing benchmark: " + str(output))
    torch.cuda.set_per_process_memory_fraction(.15, 0)
    data = args.data_root
    manifest = json.loads((ROOT / "manifests/visa_1cls.json").read_text())
    variants = ["part2_448", "scale672", "fusion448_672"] if multiscale else ["layer12", "layer6", "fusion6_12"]
    rows = []
    common = (ROOT / "third_party/AnomalyDINO", ROOT / "third_party/dinov2",
              ROOT / "weights/dinov2_vits14_pretrain.pth", "cuda:0")
    for category_index, (category, item) in enumerate(manifest["categories"].items()):
        # Fixed balanced sample from the existing manifest; no accuracy tuning.
        images = [t["image_path"] for label in [0, 1]
                  for t in [x for x in item["tests"] if x["label"] == label][:2]]
        for shot_index, shot in enumerate([1, 8]):
            order_offset = (category_index * 2 + shot_index) % 3
            order = variants[order_offset:] + variants[:order_offset]
            for variant in order:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(0)
                if multiscale:
                    resolution = 448 if variant == "part2_448" else 672
                    inspector = AnomalyDINOInspector(*common, resolution=resolution, knn_backend="torch-gpu")
                    inspector.backbone.model.requires_grad_(False)
                    if variant == "fusion448_672":
                        attach_multiscale(inspector.backbone)
                elif variant == "layer12":
                    inspector = AnomalyDINOInspector(*common, knn_backend="torch-gpu")
                else:
                    inspector = MultilayerInspector(*common, variant=variant,
                                                    knn_backend="torch-gpu")
                references = item["references"][str(shot)]["0"]
                for _ in range(25):
                    tensor, _ = inspector.backbone.prepare_image(str(data / references[0]))
                    inspector.backbone.extract_features(tensor)
                inspector.fit_references([data / x for x in references], seed=0)
                # Warm up the complete matching/PCA pipeline on normal train data.
                for _ in range(3):
                    inspector.predict(data / item["calibration"][0], render_map=False)
                snapshot = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader"], text=True).strip()
                timings = []
                for repeat in range(3):
                    np.random.seed(700 + repeat)
                    for image in images:
                        value = inspector.predict(data / image, render_map=False)
                        timings.append({"repeat": repeat, "image_path": image,
                                        "inference_ms": value["inference_ms"]})
                row = {"category": category, "shot": shot, "seed": 0,
                       "variant": variant, "variant_order": order,
                       "mean_ms": float(np.mean([x["inference_ms"] for x in timings])),
                       "median_ms": float(np.median([x["inference_ms"] for x in timings])),
                       "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(0) / 2**20,
                       "bank_ms": inspector.bank_ms, "gpu_snapshot": snapshot,
                       "references": references, "timings": timings}
                rows.append(row)
                print(f"BENCHMARK {category} shot={shot} {variant}: {row['mean_ms']:.2f} ms", flush=True)
                del inspector
    summary = []
    for shot in [1, 8]:
        for variant in variants:
            selected = [r for r in rows if r["shot"] == shot and r["variant"] == variant]
            summary.append({"variant": variant, "shot": shot,
                            "category_macro_mean_ms": float(np.mean([r["mean_ms"] for r in selected])),
                            "max_peak_gpu_allocated_mib": max(r["peak_gpu_allocated_mib"] for r in selected)})
    report = {"measured_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_id": manifest["manifest_id"], "device": torch.cuda.get_device_name(0),
              "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol": {"shots": [1, 8], "seed": 0, "categories": 6,
                           "images_per_category": 4, "repeats": 3, "backbone_warmup": 25,
                           "full_pipeline_warmup": 3, "cache": "none",
                           "order": "counterbalanced cyclic variant order across 12 category/shot blocks",
                           "scope": "synchronized RGB read, transforms, features, PCA mask, exact kNN; excludes rendering/export",
                           "memory_scope": "max PyTorch allocated bytes, one variant resident; excludes driver and other users",
                           "limitations": "Shared GPU, no exclusive reservation; counterbalancing reduces but does not remove load confounding. This is an efficiency sample, not the full accuracy matrix."},
              "summary": summary, "configurations": rows}
    output.write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Part 3：同一会话的效率抽样", "",
             "六类 × 1/8-shot × seed=0；每类固定 2 张正常、2 张异常图，每图重复 3 次。"
             "每个类别/shot 分块循环轮换三个方法的顺序，共 36 个分块、432 次计时预测。"
             "原 Part 2 直接调用未修改的实现。每次仅驻留一个方法，先 25 次 backbone 预热，"
             "再用正常校准图做 3 次完整流程预热，不缓存特征。", "",
             "| 方法 | shot | 六类 macro 平均耗时 (ms/图) | 抽样配置最大 PyTorch 显存 (MiB) |",
             "|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['variant']} | {row['shot']} | {row['category_macro_mean_ms']:.2f} | {row['max_peak_gpu_allocated_mib']:.2f} |")
    lines.extend(["", "计时包含同步 RGB 读取、预处理、特征提取、PCA 前景 mask、精确近邻；"
                  "不含热图原分辨率渲染、磁盘导出与评价。参考库构建时间单独保存在 JSON。", "",
                  "这是共享 RTX A6000 上的同一会话抽样，没有独占预留。轮换顺序可减轻顺序偏差，"
                  "仍不能排除其他用户负载变化；不据此保证部署延迟或实时性能。"
                  "显存为该进程 PyTorch 分配峰值，不含驱动/上下文/其他用户，"
                  "完整准确率实验的 72 配置显存另见 `results/tables/part3/per_configuration.csv`。", "",
                  "逐图计时、顺序、参考图及 GPU 负载快照：`results/tables/part3/efficiency.json`。"])
    doc = "20_part3_multiscale_efficiency.md" if multiscale else "17_part3_efficiency.md"
    lines = [line.replace("results/tables/part3/", f"results/tables/{report_dir}/") for line in lines]
    (ROOT / "docs" / doc).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
