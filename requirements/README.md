# 依赖说明

`base.txt` 用于 Part 1、公共评估和绘图。`scripts/prepare_manifests.py` 及其测试只使用 Python 标准库。

Part 2 应使用独立环境，先安装与服务器驱动兼容的 PyTorch / torchvision，再按 [服务器操作手册](../docs/03_server_runbook.md) 安装经适配的官方依赖。当前官方 requirements 默认列出 `faiss-gpu`；采用 `--faiss_on_cpu` 时替换为 `faiss-cpu`，不要直接假定原文件能在所有服务器上安装。

首次在真实服务器验证成功后，将 `python -m pip freeze` 保存为对应环境的锁定清单并提交，记录 GPU、驱动和上游提交号。目前没有声称任何 CUDA 组合已经验证。
