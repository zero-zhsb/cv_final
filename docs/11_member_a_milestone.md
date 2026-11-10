# 成员 A：第一个里程碑

## 范围与验收

按 README 原定分工：A 完成一个类别的 Part 1，B 提供同类别 Part 2；同一份 reference、同一批 test 和公共评估，输出可接入同一演示。本次仅运行 **cashew / 1-shot / seed=0**，完整 150 张 test（50 normal、100 anomaly），20 张正常 calibration，reference 为 `cashew/train/good/024.JPG`。

不开展 72 配置全量实验、Part 3、最终报告或正式录屏。既有 Part 2 结果直接复用 B 发布的 GPU 主矩阵，不重复占用 GPU。B 的模型方法未修改；公共 map loader 仅增加 Part 1 压缩图格式分支，已有 DINO patch-map 路径保持原样。

## 环境和数据

服务器工作目录：`~/cv_final`，分支 `codex/part1-integration`。

系统 `/usr/bin/python3` 是 3.10.12；PATH 默认的 conda Python 为 3.14，不能混用。独立环境 `.venv-a` 安装 `requirements/part1.txt`，无 PyTorch/FAISS，不使用 GPU。服务器缺少 ensurepip，创建环境的方法为：

```bash
cd ~/cv_final
/usr/bin/python3 -m venv --without-pip .venv-a
# 现有 conda pip 只作为安装工具；--python 指定写入个人 venv，不改系统环境。
/opt/miniconda3/bin/python3 -m pip --python .venv-a/bin/python install pip -r requirements/part1.txt
source .venv-a/bin/activate
python --version
python -m pip check
```

如默认 PyPI 连通性不佳，可在安装命令中临时指定 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`，不修改全局 pip 配置。实际安装快照见 `requirements/part1.server.freeze.txt`。

准备数据（目的目录必须尚无对应类别，脚本不覆盖既有数据）：

```bash
python scripts/prepare_visa_subset.py \
  --source /data/disk1/shared/aiaa3201/project2_anomaly/VisA \
  --output data/VisA_pytorch/1cls --categories cashew
```

共复制 600 张图片，仅 cashew 类约 177 MiB。按官方 `1cls.csv` 分配并将非零 mask 转为 255；复制前核对 B 的 train/test/mask 名单。未改动共享目录，也没有重新抽取 reference。共享 manifest 的 SHA-256 仍为 `51934290ccd342f9f7fa03f9f56c712494cb7b576b7038516efdb41d43b9b2c5`。

来源：数据准备逻辑参考 [spot-diff prepare_data.py](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/utils/prepare_data.py)（Apache-2.0）；输入几何参考 [AnomalyDINO backbones.py](https://github.com/dammsi/AnomalyDINO/blob/b9d1c2648e3a5247437d4d953d907a8f3d994457/src/backbones.py)。

## Part 1 方法与运行

先 PIL bicubic 将短边缩放至 448，再从右/下裁为 14 的整数倍，保留 RGB [0,1]。以全局 RGB 平均绝对差选最近模板；通道平均差分图经 sigma=1 高斯平滑，以最高 1% 像素（向下取整、至少一个）的均值作为图像分数。校准阈值是 20 张保留正常图分数的线性 95% 分位数。上述评分和平滑是团队固定的起始方案，不是教师指定公式；没有用 test 标签选参数。

```bash
python scripts/run_part1.py --data-root data/VisA_pytorch/1cls \
  --output runs/part1_cashew_1shot_seed0_v1
python scripts/evaluate.py --data-root data/VisA_pytorch/1cls \
  --run runs/part1_cashew_1shot_seed0_v1 --memory-budget-gib 10
python scripts/compare_milestone.py --run runs/part1_cashew_1shot_seed0_v1 \
  --output results/tables/part1_part2_cashew_milestone.csv
```

已有 run 不会被覆盖；复现时使用新目录名。`scripts/evaluate.py` 调用与 B 相同的评价实现，normal test 的 GT 为全零，pixel AUROC 使用全部原图像素；不逐图归一化。`template_map` 存储裁剪/平滑后的 float32 图和原图尺寸，由公共 loader 线性上采样，不叠加 DINO 专用的 sigma=4。

图像级与像素级指标、同图身份核验及时间见 [两方法比较 CSV](../results/tables/part1_part2_cashew_milestone.csv)。本次是单个 seed 的一个配置，没有跨 seed 均值或标准差。A 为 CPU 方法，B 的历史结果为 GPU 方法，运行负载和时间点不同，延迟仅描述，不能作为受控速度比。CPU baseline 的 GPU 显存为 N/A。

| 方法 | Image AUROC | Pixel AUROC |
|---|---:|---:|
| Part 1 模板差分 | 65.34% | 77.80% |
| Part 2 AnomalyDINO（B 已提交结果） | 92.90% | 98.69% |

已核对两方法的全部 150 个 image/mask/label/manifest 身份完全一致。原始预测、校准分数、阈值、run 元数据和精确评价 JSON 保存在 `results/tables/part1_milestone/`。推理是在开发工作区运行的，因此 `code_dirty=true`，`code_commit` 是 B 的基底；真实运行源码以 `source_sha256` 记录，不能将基底提交号误认为已经包含 A 的实现。最终提交保存对应代码与结果。

4 GiB 上限的首次精确像素评价被内存保护主动拒绝，未产出指标；确认服务器可用内存后用 10 GiB 上限、2 计算线程和 nice=10 重跑。没有降采样、分箱近似或改变 AUROC 定义。

## 演示接口

`TemplateInspector.fit_references(paths, seed)` / `predict(path)` 与 B 的使用模式一致，返回连续 score、原图分辨率 anomaly_map 和 inference_ms，额外返回 nearest_reference。推理不接收 GT。

```bash
python scripts/predict_part1.py \
  --image data/VisA_pytorch/1cls/cashew/test/bad/000.JPG \
  --data-root data/VisA_pytorch/1cls \
  --run runs/part1_cashew_1shot_seed0_v1 \
  --output demo/examples/part1_milestone
```

输出 prediction.json、prediction.png、anomaly_map.npz。**Part 1 是实时推理，Part 2 分数是 B 已提交预测的明确标注回放**；这不表示 A 的 venv 能运行 DINO。对于身份一致的 bad/000.JPG，comparison.png 拼接 B 已发布的 1-shot CPU 演示图，标题明确 REPLAY；历史 CPU 图片不冒充正式 GPU 时延结果。任意外部产品图也能运行 Part 1；没有对应已保存记录时不会虚构 Part 2 输出。

样例 bad/000.JPG 为真实缺陷，Part 1 分数约 0.4983、阈值约 0.6099，输出 OK，是漏检；B 同配置主矩阵分数约 0.2320、阈值约 0.2195，输出 NG。模板差异会受到姿态、背景和正常外观变化影响；该样例用于展示局限，不用于调参。

## 验证与后续

```bash
python -m unittest discover -s tests -v
```

CPU 环境 17 项检查通过、4 项 GPU/模型环境相关检查跳过。覆盖几何、最近模板、局部差异、压缩图复原、同分 AUROC、normal 像素参与、参考图/校准隔离与名单稳定性。GPU/模型检查的跳过不计作通过；未在 A 环境重跑 B 的完整模型。

先确认本里程碑，再扩展 Part 1 的六类/多 shot/多 seed、实时联合界面和正式 Pre 交付。Part 3 按用户要求暂缓。
