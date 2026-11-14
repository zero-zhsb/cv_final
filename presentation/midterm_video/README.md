# 期中视频素材：Part 1/2 全量完成版

本版对应 Canvas 的 ≤5 分钟期中 checkpoint，目标 **4:40（280 秒）**。两方法均已完成六类、四种 shot、三个 seed，各 72 配置、11,520 条测试预测；Part 3 仍为后续计划。本版替代旧单类别主比较版本。

## 打开和录制

1. 打开 `CV_Midterm_FullComparison_Final.pptx`（ZIP 内附），使用备注中的中英文讲稿及 A/B 分工。仓库版本在 `../midterm_ppt/`。
2. 或双击 `index.html` 离线展示，方向键切换，F 全屏，空格按时长自动翻页，R 重置。08 正常例是可选画面，占用已有时段，不额外加时间。
3. `assets/` 的 01–07 为同一份 PPT 的 1920×1080 导出图，可导入剪辑软件，按 `timeline.csv` 安排。
4. 录制 `script_en.md` 或 `script_zh.md`，中英文二选一。`subtitles_en.srt` 是新英文稿的时码草案，需配音后按实测重新对齐。`speaker_notes_full.md` 含每页完整备注和来源。
5. 原有录音如果提到“Part 1 只完成单配置”“全量比较待补齐”或把 65.34% 当作总体数据，需要重录。录屏命令见 `recording_steps.md`，可选的运行片段替换 03/04 中约 5–10 秒，不增加总时长。
6. 本次更新的是录制内容与素材，没有真人配音或最终成片。使用真实保存输出时按实验结果介绍，PPT 放映本身不会调用模型。

## 时间与分工

| 时段 | 主讲 | 画面 | 内容 |
|---|---|---|---|
| 00:00–00:25 | A | 01 | 任务与真实缺陷产品 |
| 00:25–01:05 | A | 02 | 两方法与正常校准 |
| 01:05–02:00 | A | 03 | Part 1 全量完成；单图解释方法和漏检 |
| 02:00–03:00 | B | 04 | Part 2 同图检出；分数和色标边界 |
| 03:00–03:25 | B | 05 | 8-shot 六类均值与标准差比较 |
| 03:25–03:45 | B | 06 | 两方法四种 shot 的完整检测曲线 |
| 03:45–04:10 | B | 07 | 两方法都漏检的小缺陷例子 |
| 04:10–04:40 | A | 07 | 固定 448、多层融合、受控消融 |

A 约 150 秒，B 约 130 秒。280 秒为排练计划，不保证人声实录自动等长；导出含片头片尾和转场须不超过 05:00。

## 当前结果口径

| 8-shot 六类总体 | Image AUROC (%) | Pixel AUROC (%) |
|---|---:|---:|
| Part 1 模板差分 | 60.80 ± 0.32 | 76.91 ± 0.33 |
| Part 2 AnomalyDINO | 96.62 ± 0.57 | 98.21 ± 0.02 |

每个 seed 内对六类 AUROC 等权平均，再跨 seed 0/1/2 算 mean±population std（ddof=0）。差 35.82 个百分点是均值差，不是相对改善率。模板图像均值从 1-shot 53.17% 到 8-shot 60.80%；Part 2 从 94.61% 到 96.62%。完整数据在 `data/full_comparison.csv`，以及两方法各自 `*_overall_by_shot.csv`，身份核验在 `full_comparison_audit.json`。

03/04 的图片仍是 cashew/test/bad/000.JPG，1-shot/seed=0，reference 为 train/good/024.JPG。旧 cashew 单配置 AUROC 65.34% / 92.90% 仍然有效，但不再作为本次主对比表；较低的六类模板平均值不代表同一实验分数下降，而是统计范围不同。

07 为 cashew/test/bad/075.JPG、8-shot/seed=0。Part 1 score≈0.4941、threshold≈0.5476；Part 2 score≈0.1088、threshold≈0.1708，两者均输出 OK，真实标签为异常。真值细节放大，原图和叠加图保留全画面。它与 03/04 不是同一图片或配置。

不同方法的图像分数、色标不直接比较；AUROC 也不等于 accuracy。两方法运行设备和负载不同，不做受控速度倍数结论。Part 1 推理为 CPU，CUDA 仅加速其精确像素评价。代表案例不是随机样本，不用几张图估算总体错误率。

## 来源及再生成

Part 1 正式运行 `part1_full_stream_cuda_v1`，结果提交 `616701a`；Part 2 为 `part2_gpu_exact_matrix_v1`。已核验两方法全部 11,520 行 category/shot/seed/image/mask/label/manifest 身份一致、Part 1 源码 hash 与运行记录匹配，并重算六类宏平均/标准差。

素材来源：[VisA / spot-diff](https://github.com/amazon-science/spot-diff)、[AnomalyDINO](https://github.com/dammsi/AnomalyDINO)、[DINOv2](https://github.com/facebookresearch/dinov2)。只打包少量代表数据，不含完整数据集和模型权重，原许可继续适用。

`provenance.json` 保存运行和文件来源。ZIP 的 raw 含 15 个原图、真值、compact maps、选择记录及科学图源文件，raw 不提交 GitHub；成品 PNG、PPT 可独立查看。

视频画面以本次 PPT 为唯一版式来源。有 Artifact Tool 的 Node 环境设置 `RUNTIME_NODE_MODULES` 后，在仓库根目录运行：

```bash
node scripts/export_midterm_video_from_ppt.mjs
python scripts/package_midterm_video.py
```

旧 Python 图卡生成器仅保留历史实现；全量数据存在时会提示新入口，避免生成过时画面。再生成只渲染资料，不运行模型、不占 GPU。

录制前补齐真实姓名，确认贡献；中英文只读一种，实测调整字幕和语速；最后完整播放成片检查音量、可读性及 ≤5 分钟，再上传 Canvas。Part 3 只能说计划，不能说已取得提升。
