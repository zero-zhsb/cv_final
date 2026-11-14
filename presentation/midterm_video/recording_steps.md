# A/B 录屏操作与剪辑清单

不必录制同时运行两个模型的联合界面。已有 1080p 图卡可直接用于展示真实实验结果；以下步骤用于可选的 5–10 秒操作证据片段。不要重新运行 72 配置，不修改他人目录，不在录屏中输入或展示密码。

## A：Part 1（CPU）

A 在自己的服务器账号中运行，工作目录 `/data/disk1/home/zzhang657/cv_final`。先把终端字体调大，屏幕只保留命令、输出和结果图。目录中的 `take01` 必须不存在，已有时换成 `take02`，不要删除原结果。

```bash
cd /data/disk1/home/zzhang657/cv_final
.venv-a/bin/python scripts/predict_part1.py \
  --image data/VisA_pytorch/1cls/cashew/test/bad/000.JPG \
  --data-root data/VisA_pytorch/1cls \
  --run runs/part1_cashew_1shot_seed0_v1 \
  --category cashew --shot 1 --seed 0 \
  --output runs/part1_midterm_take01
```

录下命令执行与 JSON 结果，然后在远程 VS Code 打开 `runs/part1_midterm_take01/prediction.png`。预期 score≈0.4983、threshold≈0.6099、OK，为真实漏检。脚本附带的 Part 2 行明确是回放，不能称为当场运行了两个方法。

## B：Part 2（GPU；单图，无需重跑矩阵）

B 在自己的服务器账号中运行，工作目录 `/data/disk1/home/sshen560/cv_final_member_b`。先看共享 GPU 的可用情况，忙碌时协调时段。使用 **1-shot** 才与 A 的配置一致；不要直接沿用其他文档的 8-shot 示例。

```bash
cd /data/disk1/home/sshen560/cv_final_member_b
nice -n 10 .venv-b/bin/python scripts/predict_part2.py \
  --image /data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls/cashew/test/bad/000.JPG \
  --data-root /data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls \
  --run runs/part2_gpu_exact_matrix_v1 \
  --category cashew --shot 1 --seed 0 --device cuda:0 \
  --output runs/part2_midterm_take01
```

预期 score≈0.2320、threshold≈0.2195、NG。单图重跑和批量运行可有细小 FP32 差异，视频中显示实际输出，不手工改值。终端执行片段和新生成 `prediction.png` 可替换 04 图卡的一部分。

本次素材准备没有重新调用模型，也没有替你们录制实际屏幕操作或配音。上面命令为已有单图入口的录制步骤；此前已验证同图 Part 1 重跑和 Part 2 入口，不将本次准备图卡说成一次新推理。若选择不插入操作录屏，保留图卡“saved output”的说明，讲解已经完成的实现与实验即可。

## 剪辑顺序

1. 建立 1920×1080 工程，按 `timeline.csv` 放入 01–07，07 在两个说话时段保持同一画面。
2. A 录 01、02、03 和结尾计划；B 录 04、05、06 及失败分析。使用同一种配音语言。
3. 如需操作证据：03/04 的开头各替换 5–10 秒录屏，随后回到清晰图卡，不额外延长时间。
4. 08 正常样例可在 03 开头插入约 5 秒；同样占用 03 的已有时段。
5. 加入字幕并对齐人声。没有配音时，不将字幕时码视为成片已完成。
6. 导出实际视频，完整播放检查，确保总长含片头片尾和转场 ≤05:00。目标约 04:40，不需要为了凑满时长延长静态等待。

## 录屏之外的来源卡（可放视频说明或片尾短行）

Dataset: VisA. Methods: RGB template difference; AnomalyDINO with frozen DINOv2 ViT-S/14. Code: https://github.com/zero-zhsb/cv_final (branch codex/midterm-video-materials). Experimental outputs and references are documented in the repository.
