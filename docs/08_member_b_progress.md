# 成员 B 工作记录与可运行入口

更新日期：2026-11-10。本页记录 **Part 2 数据/环境、CPU 小样例与官方单图一致性检查**。此后开始的 GPU 阶段见 [GPU 实验记录](09_gpu_experiments.md)；不要将本页的小样例当作全部 72 个配置的正式结果。

## 已完成

- [x] 独立成员 B checkout 与 Python venv，不修改 A 的工作目录或系统 Python。
- [x] 找到服务器预置的课程 VisA，停止自行启动的重复下载。
- [x] 使用固定版本 spot-diff 官方脚本整理 `1cls`，保留公共原始数据不变。
- [x] 生成六类的共享 manifest，校准图与参考池隔离，包含 72 套 reference 配置。
- [x] 安装 PyTorch、torchvision、CPU FAISS 等；`pip check` 通过。
- [x] 下载官方 DINOv2 ViT-S/14 权重，加载本地 DINOv2 源码，避免推理时临时联网。
- [x] 实现 `scripts/run_part2.py`：固定官方特征/掩码/旋转、共享名单、独立校准、浮点异常图和 CSV 导出。
- [x] cashew，1-shot，seed=0：20 张正常校准图、2 张正常和 2 张异常测试图，CPU 两线程完成流程检查。
- [x] 同一真实正常图与未修改官方 detector 对比：图像分数完全一致、patch 图一致，渲染图最大误差约 7.45e-9。
- [x] 生成四个真实结果预览；这些是预先固定顺序的小样例，不是精选准确率证明。

## 目录和数据

| 用途 | 实际服务器路径 |
|---|---|
| 原工作目录（未修改） | `/data/disk1/home/sshen560/cv_final` |
| B 的独立工作目录 | `/data/disk1/home/sshen560/cv_final_member_b` |
| B 的独立环境 | `/data/disk1/home/sshen560/cv_final_member_b/.venv-b` |
| 课程预置原始 VisA（只读取） | `/data/disk1/shared/aiaa3201/project2_anomaly/VisA` |
| 官方整理后的数据 | `/data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls` |
| 本次最终 CPU smoke 输出 | `cv_final_member_b/runs/cashew_cpu_smoke_v3` |
| 运行日志 | `cv_final_member_b/logs/` |

公共数据完整性检查后再使用；没有把课程公共文件夹改权限、移动或覆盖。课程公共说明建议直接使用共享数据，因此整理后的图像逐文件核对字节一致后替换为指向原图的符号链接，自己的目录只保留划分结构及官方转换后的二值掩码，避免长期保留多 GB 图像副本。环境安装和 CPU 运行仍会使用共享磁盘/网络/CPU，本次将推理进程数值库线程限制为 2，没有声称共享资源“零影响”。

工具默认拒绝越出 data-root 的链接。使用这份轻量数据视图时，需要显式传入 `--allow-image-root /data/disk1/shared/aiaa3201/project2_anomaly/VisA`，仅允许指向该公共原图目录，不放开任意外部路径。manifest 仍保存相对路径，引用名单与哈希不变。

### 生成的六类计数

| 类别 | 正常训练 | 校准（从训练中保留） | 可用 reference 池 | 正常测试 | 异常测试 |
|---|---:|---:|---:|---:|---:|
| candle | 900 | 20 | 880 | 100 | 100 |
| capsules | 542 | 20 | 522 | 60 | 100 |
| cashew | 450 | 20 | 430 | 50 | 100 |
| chewinggum | 453 | 20 | 433 | 50 | 100 |
| fryum | 450 | 20 | 430 | 50 | 100 |
| pipe_fryum | 450 | 20 | 430 | 50 | 100 |

共享名单：[visa_1cls.json](../manifests/visa_1cls.json)。`manifest_id` 为 `51934290ccd342f9f7fa03f9f56c712494cb7b576b7038516efdb41d43b9b2c5`。A 的 Part 1 应复用它，不另抽 reference。

## 环境与来源

- Python 3.10.12；torch 2.5.1，CUDA build 12.4；torchvision 0.20.1；NumPy 1.26.4；FAISS CPU 1.8.0.post1。
- GPU 驱动可访问，但此阶段只验证可用性，没有 GPU 模型计算。缺少 xFormers 的提示不妨碍 CPU 功能检查。
- 环境：[检查结果](member_b_environment.json)、[完整依赖冻结](../requirements/part2.server.freeze.txt)、[安装清单](../requirements/part2.txt)。
- AnomalyDINO：`b9d1c2648e3a5247437d4d953d907a8f3d994457`。
- DINOv2：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
- spot-diff：`2a692ab575001cbde74d402d897a7286086c6199`。
- 权重来自 `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth`；run.json 保存 SHA-256。源码分别遵守其许可证，结果图片中的 VisA 数据来自课程预置、官方数据许可 CC BY 4.0，论文引用见来源文档。

## 在当前服务器运行

以下命令在 B 目录执行，输出目录必须是新的。CPU 冒烟不是完整 benchmark，不计算或宣传四张图片的 AUROC。

```bash
cd /data/disk1/home/sshen560/cv_final_member_b
source .venv-b/bin/activate
python -m pip check
python -m unittest discover -s tests -v
python scripts/run_part2.py --data-root /data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls --allow-image-root /data/disk1/shared/aiaa3201/project2_anomaly/VisA --categories cashew --shots 1 --seeds 0 --device cpu --threads 2 --limit-per-class 2 --output runs/cashew_cpu_smoke_new
python scripts/render_part2_examples.py --run runs/cashew_cpu_smoke_new --data-root /data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls --output results/figures/part2_smoke_new
```

输出：run.json、predictions.csv、thresholds.json、calibration.json、原图分辨率的浮点 `anomaly_map`、原始 `patch_map`。小样例在 run.json 标记 `partial_smoke_test=true`。20 张校准图只用于阈值，不进入 reference 或正式测试行。为防止校准消耗 PCA 随机数改变正式预测，校准前保存、结束后恢复 NumPy RNG 状态。

`predict()` 不接收真值。测试 label 和 mask 仅从 manifest 复制到评估 CSV 或用于画图，不参与特征、阈值或推理。热力图统一预览色标与原始评估数组分开。

官方一致性核对：

```bash
python scripts/check_part2_parity.py --data-root /data/disk1/home/sshen560/cv_final_data/VisA_pytorch/1cls --output results/tables/part2_parity_new.json
```

该检查仅在自动创建的临时目录中链接指定正常 reference 和单张真实正常测试图，调用未修改的官方 `run_anomaly_detection`；对比 score、patch 图与渲染图。官方函数中仅 CUDA 计时同步在这个独立 CPU 检查进程里临时替换为空操作，特征、mask、距离和分数代码不修改，不使用共享 GPU。

结果证据：[官方一致性](../results/tables/part2_parity.json)、[CPU 小样例分数](../results/tables/part2_cpu_smoke_predictions.csv)、[运行元数据](../results/tables/part2_cpu_smoke_run.json)、[真实预览](../results/figures/part2_smoke/)。CSV 的 map_path 相对于服务器原 run 目录，完整浮点图没有上传到 GitHub。验证运行的代码树含未提交文件，run.json 如实记录 `code_dirty=true`，并保存实际执行源文件的 SHA-256，便于与提交文件核对。

## 之后正式实验的边界

1. 按共享设备规则使用 GPU，不停止他人的任务；单进程和资源记录方式见 GPU 实验记录。
2. 去掉 `--limit-per-class`，使用四个 shots 和三个 seeds。默认六类来自 manifest；需要正式 GPU 运行时显式选择 `--device cuda:0`。命令支持完整矩阵，不代表它已执行。
3. 先完成 cashew 单类全测试图核对，再启动六类 72 配置。正式结果使用新 run 目录。
4. 由 A 的公共评估计算逐类别图像/像素 AUROC、六类等权平均和跨 seed 标准差。B 本次提供输入，不另定一套有利于自己的评估。
5. 单独在协调好的设备状态下测量 GPU 时间/显存；本次 CPU 小样例的时间不当作正式效率成绩。
6. 完整实验后再做稳定性曲线和五个实际困难案例分析。本次四张预览不替代课程要求的五个困难案例。

还未完成：全部 72 配置、正式 AUROC 汇总、GPU 效率、与 A 的 Part 1 公平对比、正式演示集成与 Pre 最终材料。
