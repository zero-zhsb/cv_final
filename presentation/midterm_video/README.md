# 五分钟以内的期中展示素材包

依据用户粘贴的 Canvas 通知：不超过 5 分钟，展示 Current Approach、Baseline Implementation、Visualization and Preliminary Results、Future Improvements；本次不是最终展示，不要求已完成创新方案。通知没有要求同屏实时运行两个方法，也没有在其中要求 Part 1 在 checkpoint 前跑完 72 配置。团队视频目标为 **4:40（280 秒）**，不含额外片头；留 20 秒余量。

## 打开与使用

1. 双击本目录 `index.html`，可离线查看全部画面；方向键切换，F 全屏，空格自动按时间翻页，R 回到开头。默认只播放 01–07；正常样例 08 是可选插入画面，需占用现有时段而非增加总时长。
2. 将 `assets/01_*.png` 至 `07_*.png` 按顺序导入剪映、Premiere 或其他剪辑软件，分辨率均为 **1920×1080，16:9**。按 `timeline.csv` 设置时长，依照 `script_en.md` 或 `script_zh.md` 录制人声。中英文讲稿是二选一，不要两种语言全读。
3. `subtitles_en.srt` 是英文讲稿按建议时段拆分的字幕草案；录完配音后对齐真实说话时间，不将规划时码当作已录音的精确对齐。英文总词数见脚本文件末尾，须实际排练确定语速。
4. 03、04、07 画面均由真实已保存的图像与数值生成，**不是新录制的屏幕操作**。可以直接作为实验结果展示；想插入实际操作证据时，按 `recording_steps.md` 分别录制两个方法的执行和输出，各截取约 5–10 秒，替换对应图卡时段的一部分。无需制作实时联合网页。
5. 本包不含真人配音、实际屏幕录制片段或最终提交视频；Canvas 上传由你们完成。

## 建议时间与分工

| 时段 | 负责人 | 画面 | 内容 |
|---|---|---|---|
| 00:00–00:25 | A | 01 | 正常/异常产品、任务与输出 |
| 00:25–01:05 | A | 02 | 两方法路线与独立正常校准 |
| 01:05–02:00 | A | 03 | Part 1 已实现；同图漏检与局限 |
| 02:00–03:00 | B | 04 | Part 2 冻结特征、同图正确检出 |
| 03:00–03:25 | B | 05 | 单配置公平比较，明确 150 张测试图 |
| 03:25–03:45 | B | 06 | Part 2 全六类进展，说明 seed 标准差 |
| 03:45–04:10 | B | 07 | Part 2 小缺陷真实漏检；解释仍待验证 |
| 04:10–04:40 | A | 07 | 补齐实验、尝试多尺度、进行受控消融 |

A 约 150 秒，B 约 130 秒。时间为团队建议，不是教师指定分段。未来改进部分有 55 秒，不能用结果数字挤占掉。

## 真实素材与数值范围

- 01/03/04：`cashew/test/bad/000.JPG`；正常图 `cashew/test/good/006.JPG`；reference `cashew/train/good/024.JPG`。均来自 VisA 官方 1cls 划分。
- 03/04/05：cashew、1-shot、seed=0，Part 1 已完成的 150 张测试图与 Part 2 主运行同配置，参考图/测试身份一致。Part 1 image/pixel AUROC=65.34%/77.80%，Part 2=92.90%/98.69%。这不是六类平均。
- 06：仅 Part 2，六类、1/2/4/8-shot、seed 0/1/2，共 72 配置，11,520 条配置内测试预测。先按 seed 对六类指标等权平均，再对三个 seed 报 mean±std，ddof=0。不能用此成绩直接与 Part 1 单类成绩画一组比较柱。
- 07：`cashew/test/bad/075.JPG`，Part 2 的 **8-shot/seed=0**；score=0.1087610，threshold=0.1707948，输出 OK，为真实漏检。不要把它说成 03/04 的同一张图片或同一配置。
- 两种方法的分数和热力图色标不同，不直接比较分数大小；不要说“Part 2 分数更小所以更好”。GPU/CPU 延迟不是受控速度比较，不建议在本次短视频做倍速结论。
- 画面是代表性例子，不是随机样本。正常图展示、Part 1 漏检及 Part 2 漏检均保留，避免只展示成功图。
- `data/preliminary_comparison.csv` 与 `data/part2_overall_by_shot.csv` 是两张定量图对应的真实结果表，可直接导入表格或幻灯片软件。

## 来源、可复核性与再生成

`provenance.json` 记录每张选中图片的实际预测行、原运行名、manifest ID、来源 commit 及选取的原始文件 hash。Part 1 来自 `part1_cashew_1shot_seed0_v1`；Part 2 来自完整 GPU 主运行 `part2_gpu_exact_matrix_v1`，没有把旧 CPU 示例图冒充 GPU 主结果。

完整结果见仓库 `results/tables/`；方法来源为 [AnomalyDINO](https://github.com/dammsi/AnomalyDINO)、[DINOv2](https://github.com/facebookresearch/dinov2)，产品图与真值来自 [VisA / spot-diff](https://github.com/amazon-science/spot-diff)。本包只含课程分析所用代表性可视化，数据与模型的原许可继续适用；它不是完整数据集副本。

本地 ZIP 的 `raw/` 含 12 个用于复核的小样例文件（原图、真值、compact maps、选择记录）；此目录不提交 GitHub。GitHub 版的成品 PNG 不依赖它即可查看。将 ZIP 的 `raw/` 放到本目录后，在项目根目录运行：

```bash
python scripts/build_midterm_video_assets.py
```

使用环境需含 numpy、Pillow、matplotlib、scipy、opencv；不加载模型、不占 GPU、不重新跑实验。异常图通过项目公共 loader 在原图尺寸重建，绘图色彩仅用于展示，不回写指标。

## 导出前核对

- 总时长含片头/片尾、转场、演示等待，实际导出必须不超过 05:00；建议维持约 04:40。
- 两人姓名、学号等身份信息由本人补充；当前画面没有虚构姓名。
- 文字/热力图在 1080p 下能看清，图表范围与方法名称明确。
- 未来工作说“计划/尝试”，不声称改进已经完成或必然有效。
- 预先计算结果按实验输出介绍；只对真正录制的执行片段说“这段是实际运行”。
- 完整播放检查配音、字幕和音量后再上传 Canvas。
