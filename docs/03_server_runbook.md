# 服务器操作手册（Linux / NVIDIA）

以下命令供你们在自己的服务器执行；当前仓库初始化没有登录服务器，也没有实际运行模型。先将路径改成服务器可写的项目目录。GitHub 管理代码和记录，服务器保存数据/权重并执行实验。

## 1. 两个成员各自 clone

```bash
git clone https://github.com/zero-zhsb/cv_final.git
cd cv_final
git switch -c codex/part1-baseline    # A；B 改为 codex/part2-anomalydino
python3 --version
nvidia-smi
df -h .
```

Python 推荐从 3.10/3.11 开始适配。A、B 使用不同 checkout；数据可共享只读路径，runs、日志、环境使用各自路径。不要同时修改共用的官方源码或 pip 环境。磁盘除数据本体外需容纳权重、浮点图、缓存与多次 runs，先用单配置测量再估算总量。

## 2. A：创建基础环境

```bash
python3 -m venv .venv-a
source .venv-a/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/base.txt
python scripts/check_environment.py
python -m unittest discover -s tests -v
```

依赖安装成功后，将 `python -m pip freeze` 保存到 `requirements/part1.server.freeze.txt`，补充 Python 版本和系统信息。冻结文件是服务器安装完成后产生的，不预先伪造。

## 3. B：独立模型环境

```bash
python3 -m venv .venv-b
source .venv-b/bin/activate
python -m pip install --upgrade pip
mkdir -p third_party
git clone https://github.com/dammsi/AnomalyDINO.git third_party/AnomalyDINO
git -C third_party/AnomalyDINO checkout b9d1c2648e3a5247437d4d953d907a8f3d994457
```

先根据 `nvidia-smi` 与 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装匹配的 torch、torchvision。不能仅根据“有 GPU”猜 CUDA wheel；驱动兼容性与实际 `torch.cuda.is_available()` 均需检查。

本仓库检查的官方 requirements 包含 `faiss-gpu`，但我们初期选择 CPU FAISS。在本地派生安装清单，保留原始文件：

```bash
python -c 'from pathlib import Path; p=Path("third_party/AnomalyDINO/requirements.txt"); lines=["faiss-cpu" if x.strip().startswith("faiss-gpu") else x for x in p.read_text().splitlines()]; Path("requirements/part2.server.local.txt").write_text("\n".join(lines)+"\n")'
python -m pip install -r requirements/part2.server.local.txt
python -m pip check
python scripts/check_environment.py --require-cuda
python -m pip freeze > requirements/part2.server.freeze.txt
```

当前上游 NumPy 要求 1.x。如 FAISS/其他依赖不能解析，先在独立环境解决兼容版本并记录；不要为了安装成功直接改用 NumPy 2.x，或在 A 的环境中反复升级。上述组合尚未在你们的服务器验证。

`third_party/` 被 Git 忽略。课程适配完成后，把明确的补丁导出到 `patches/` 并提交：

```bash
git -C third_party/AnomalyDINO diff > patches/anomalydino-course.patch
```

此命令只包含已跟踪文件的修改；若适配增加了新文件，应把自己的适配代码放在本仓库 `src/part2/` 并提交，或明确将新增文件纳入补丁。不要只在服务器改第三方目录而不交付改动。

## 4. A：下载及准备 VisA

在 [VisA 官方仓库](https://github.com/amazon-science/spot-diff#data-download) 使用 Data download 链接下载并解压。共享数据目录示例：`/srv/cv-final/data/`，先确认权限。`VisA` 下应直接包含 `candle` 等类别，避免多套一层目录。

```bash
git clone https://github.com/amazon-science/spot-diff.git third_party/spot-diff
git -C third_party/spot-diff checkout 2a692ab575001cbde74d402d897a7286086c6199
cd third_party/spot-diff
python utils/prepare_data.py --split-type 1cls --data-folder /srv/cv-final/data/VisA --save-folder /srv/cv-final/data/VisA_pytorch --split-file ./split_csv/1cls.csv
cd ../..
python scripts/prepare_manifests.py --data-root /srv/cv-final/data/VisA_pytorch/1cls --output manifests/visa_1cls.json
```

工具会检查所有六类的数据结构和异常掩码、排除校准图、按当前官方规则生成 72 套 reference。正式运行前 A/B 一起检查输出计数并提交名单。名单中保存相对路径，不含服务器凭据。

不要为了排除校准图直接移动/删除共享原始训练图片；在名单和加载器中过滤，保证原数据保持可追溯。不要把校准图复制到 test 文件夹来“顺便推理”。

## 5. B：适配后先验证一个类别

先按 [分工文档 B2](02_team_workplan.md) 接入共享名单、校准推理、导出。官方脚本默认不会读取我们的 JSON，**直接执行命令不会自动完成课程适配**。

在 `get_dataset_info` 中临时只保留 `cashew` 进行单类别排错，并使用独立 tag。记录这项限制，正式实验恢复六类；不通过选择类别提高主表分数。

```bash
cd third_party/AnomalyDINO
python run_anomalydino.py --dataset VisA --model_name dinov2_vits14 --resolution 448 --shots 1 --num_seeds 1 --preprocess agnostic --eval_segm --faiss_on_cpu --data_root /srv/cv-final/data/VisA_pytorch/1cls/ --tag smoke_v1
```

以上是上游脚本支持的命令；课程适配入口及 manifest 路径由 B 实现并记入 README。先核对实际 reference、输入几何和结果图，不将此 smoke 结果当作完整复现。

## 6. 正式运行与后台任务

先恢复六类，在同一个适配版本上运行：

```bash
python run_anomalydino.py --dataset VisA --model_name dinov2_vits14 --resolution 448 --shots 1 2 4 8 --num_seeds 3 --preprocess agnostic --eval_segm --faiss_on_cpu --data_root /srv/cv-final/data/VisA_pytorch/1cls/ --tag course_v1
```

如服务器有 tmux，可在 session 中运行并保存日志（此段仍在 AnomalyDINO 目录）：

```bash
tmux new -s cv-part2
mkdir -p ../../logs
set -o pipefail
python run_anomalydino.py --dataset VisA --model_name dinov2_vits14 --resolution 448 --shots 1 2 4 8 --num_seeds 3 --preprocess agnostic --eval_segm --faiss_on_cpu --data_root /srv/cv-final/data/VisA_pytorch/1cls/ --tag course_v1 2>&1 | tee ../../logs/part2-course-v1.log
```

用 Ctrl+B 再按 D 分离 session；用 `tmux attach -t cv-part2` 返回。关闭 SSH 后进程可以继续，但这不代表聊天助手会持续监控。服务器无 tmux 时先按平台规则安装或使用平台后台任务。

注意：官方脚本会按已有 metrics 文件跳过运行。调整数据、名单或代码后使用新 tag，先核查哪些配置真的完成。GPU OOM 时先关闭竞争进程/减少并发；不要静默降低课程规定分辨率再与原配置混报。

## 7. 保存可复现证据

每个 run 保存：本仓库 commit、上游 commit、manifest_id、设置、环境清单、硬件、日志、预测文件和时间测量定义。单图浮点图/缓存保存服务器，GitHub 提交指标 CSV、少量 PNG、运行说明和自己编写的适配代码/补丁。

Pre 前 A/B 从各自环境交叉运行，冻结标签 `pre-v1`；另外下载演示录屏、PNG 和结果表到展示电脑，提前测试无网络时能否播放。
