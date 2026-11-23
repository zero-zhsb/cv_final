# 最终结果复现

完整的 448+672 对照冻结于 `cd336a4`。本分支在这组已测结果上提供报告与单图推理。独立的效率实验可以从该提交另起分支，并为运行创建独立目录。比较优化前后使用同一 manifest、参考图、shot、seed、设备和计时范围。

## 环境和数据

使用 Python 3.10。GPU 正式实验使用 PyTorch 2.5.1 + CUDA 12.4、torchvision 0.20.1。根据实际驱动从 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 配置对应 wheel，再安装 `requirements/part2.txt`。CPU 模板可使用 `requirements/part1.txt`。

从 [VisA 官方仓库](https://github.com/amazon-science/spot-diff#data-download) 下载并按官方 1cls 划分准备图像。具体路径结构见 [服务器手册](03_server_runbook.md)。设置 `VISA_1CLS` 指向包含 `cashew` 等六个类别目录的根路径。`manifests/visa_1cls.json` 已固定每次实验的参考、校准和测试文件名。

## 权重和外部代码

在仓库根目录准备以下文件。Git 提交和 SHA-256 会在单图模型启动时核对。

```bash
git clone https://github.com/dammsi/AnomalyDINO.git third_party/AnomalyDINO
git -C third_party/AnomalyDINO checkout b9d1c2648e3a5247437d4d953d907a8f3d994457
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
git -C third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
mkdir -p weights
curl -L --fail -o weights/dinov2_vits14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
sha256sum weights/dinov2_vits14_pretrain.pth
```

权重 SHA-256 为 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。模型使用现成预训练权重，课程实验直接执行正常参考库构建、校准和推理。

## 运行入口

- [单图输入、实时推理、代表输出和 ZIP 生成](../demo/final_demo.md)
- [模板基线实验](13_member_a_full_results.md) · [448 对照实验](09_gpu_experiments.md) · [672 与融合实验](18_part3_multiscale_protocol.md)
- [完整 AUROC 与小缺陷分析](19_part3_multiscale_results.md) · [同一 A6000 会话的效率结果](20_part3_multiscale_efficiency.md)
- [CVPR 报告及构建方式](../report/README.md)

多尺度 672 组与融合组使用同一 fine-grid PCA mask。计划比较检测质量时应核对相同参考图、图像及掩码身份。计划比较推理速度时应优先使用同会话抽样协议，记录按方法拆分的端到端耗时与显存，同时保留小缺陷漏检率和总体指标。

## 从公开版本复跑一个完整配置

从公开的 `codex/final-delivery` 分支开始。以下命令处理 **cashew 的全部 150 张测试图**，使用相同的 8 张参考和 seed 0。它们各自创建新的运行目录。若使用符号链接版 VisA，在推理命令后加 `--allow-image-root "$VISA_ORIGINAL_ROOT"`，将参数指向官方图像与掩码所在的只读数据根目录。

```bash
common=(--data-root "$VISA_1CLS" --categories cashew --shots 8 --seeds 0)
gpu=(--device cuda:0 --knn-backend torch-gpu --map-storage patch)

python scripts/run_part1.py "${common[@]}" --output runs/check_template
python scripts/run_part2.py "${common[@]}" "${gpu[@]}" --output runs/check_448
python scripts/run_part3.py "${common[@]}" "${gpu[@]}" --variant scale672 --output runs/check_672
python scripts/run_part3.py "${common[@]}" "${gpu[@]}" --variant fusion448_672 --output runs/check_fusion
python scripts/run_oas.py "${common[@]}" "${gpu[@]}" --output runs/check_oas

for method in template 448 672 fusion oas; do
  python scripts/evaluate_part2.py --run "runs/check_$method" \
    --data-root "$VISA_1CLS" --configuration cashew/shot_8/seed_0 \
    --pixel-device cuda:0-positive-keys
done
```

`--pixel-device cpu` 也可执行精确评价。主结果比较逐配置的图像与原图像素 AUROC。`inference_ms` 会随设备与负载变化，复现检查不要求其与公开表相等。

### 已核验的复跑

2026-11-23 从公开源码提交建立干净工作目录，复用服务器已安装的 Python 环境、固定权重与课程数据。五种方法的参考图、20 张校准图和 150 张测试图身份一致。下表两项 AUROC 在保存精度下都与此前发布表相同。这是每种方法一个完整配置的核验。

| cashew / 8-shot / seed 0 | 图像 AUROC | 像素 AUROC |
|---|---:|---:|
| RGB 模板 | 64.00% | 75.40% |
| AnomalyDINO 448 | 94.98% | 99.32% |
| 单尺度 672 | 96.58% | 99.50% |
| 448+672 | 97.02% | 99.50% |
| 448+672 + OAS | 97.48% | 99.69% |

模板路径在复跑中修复为导出 manifest 中的参考名，兼容课程数据的符号链接。Windows 端从 GitHub 无凭据克隆后重新生成 demo，验证了 1920×1080、56 秒视频和 ZIP 内的 27 项文件。[完整核验记录](../results/tables/final_delivery_repro_audit.json)
