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
