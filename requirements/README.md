# 依赖说明

`base.txt` 用于 Part 1、公共评估和绘图。`scripts/prepare_manifests.py` 及其测试只使用 Python 标准库。

A 的独立 CPU 环境使用 **Python 3.10 + `part1.txt`**，服务器实测快照见 `part1.server.freeze.txt`。该 venv 不安装 PyTorch/FAISS，环境来源及 ensurepip 缺失时的创建步骤见 [A 的说明](../docs/11_member_a_milestone.md)。

Part 2 应使用独立环境，先安装与服务器驱动兼容的 PyTorch / torchvision，再按 [服务器操作手册](../docs/03_server_runbook.md) 安装经适配的官方依赖。当前官方 requirements 默认列出 `faiss-gpu`；采用 `--faiss_on_cpu` 时替换为 `faiss-cpu`，不要直接假定原文件能在所有服务器上安装。

B 已验证的模型环境另见 `part2.server.freeze.txt` 和 `docs/member_b_environment.json`；不能将 B 的环境验证视为 A 的 venv 已具有 CUDA 模型依赖。后续新增环境仍需提交实际 `pip freeze` 与硬件记录。
