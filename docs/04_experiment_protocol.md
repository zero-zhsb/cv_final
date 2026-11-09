# 实验与输出协议

## 1. 数据身份和 reference 规则

`configs/course_protocol.json` 是团队共享设置。`scripts/prepare_manifests.py` 读取该配置并生成 `manifests/visa_1cls.json`。其中 `manifest_id` 是对名单内容的 SHA-256，预测输出必须关联它。

每类正常训练文件只取图片，按文件名排序；前 20 张是 calibration，剩余是 eligible。为贴近所检查的官方 `src/detection.py`，对每个 shot=k、seed=s 选择：

```python
reference = eligible[s * k : (s + 1) * k]
```

**这不是随机抽样。** 课程给出三个 seeds；当前上游将其实现为排序分段。本仓库在校准集排除后保留该规则，并公开名单；如果老师要求真正 RNG 抽样，应先更新协议与名单，双方统一重跑。不要仅把种子数设置为 3 就认为两份代码一定选到同样图片。

工具在任何写入前校验全部类别；不满足类别、测试正常/异常图、掩码或 reference 容量时退出，不生成部分成功的名单。默认不覆盖已有名单，只有确认实验版本更新后才用 `--overwrite`。

## 2. 空间变换与预处理

所检查的上游 `src/backbones.py`：RGB，Bicubic 将短边缩放到 448，再从右/下裁掉不足 14 倍数的边缘；不等于强制拉伸成 448×448。Part 1 复用相同几何变换，保留 [0,1] RGB 求差分。DINOv2 的 ImageNet 数值标准化与 agnostic masking/rotation 属于其特征方法，按官方保留并记录。

共享几何实现落地后由两人比较同图尺寸与角点位置。掩码缩放使用 nearest-neighbor。官方异常图上采样步骤也必须记录：官方脚本将 patch 图 resize 回原图大小并平滑。若改用几何逆变换，属于与上游不同的后处理，需说明、统一并另标结果，不能悄悄混用。

## 3. 输出文件（待 A/B 实现）

建议每个方法输出目录：

```text
runs/<run_id>/
  run.json
  predictions.csv
  thresholds.json
  maps/<category>/shot_<k>/seed_<s>/<test_type>/<stem>.npz
  metrics_per_run.csv
```

`run.json`：method、代码 commit、upstream commit（适用时）、manifest_id、model、resolution、preprocess、设备、精度、依赖、后处理、计时范围、缓存说明。

`predictions.csv` 每行一张测试图，字段约定：

| 字段 | 含义 |
|---|---|
| run_id / method / manifest_id | 运行、方法与数据版本 |
| category / shot / seed | 配置身份 |
| image_path / mask_path | 相对数据根的路径；正常图 mask_path 为空 |
| label | 正常 0、异常 1，仅供评估，不进入推理 |
| score | 阈值前连续图像分数 |
| map_path | 相对 run 目录的异常图路径 |
| threshold / decision | 配置专用校准阈值与 OK/NG |
| inference_ms | 实测每图延迟，run.json 说明是否含 I/O、特征、可视化 |

`.npz` 的 `anomaly_map` 应为 float32、二维、有限、明确坐标系的未逐图归一化数据。评估前尺寸应与定义的真值坐标一致；展示彩色 PNG 独立保存，不能反向从 PNG 计算 AUROC。原始 patch 图可另存，不覆盖主评估图。

校准分数单独保存，不能混入 predictions 的正式测试行。不同方法/类别/shot/seed 使用自己的阈值，并关联同一套 reference。校准图仅用于演示阈值，不用于选 smoothing、feature、mask 或模型权重。

## 4. 指标和聚合

1. 每个 category/shot/seed：图像 AUROC 使用 test/good 与 test/bad 的 label 和原始 score。
2. 像素 AUROC：该类别全部测试图，正常图补全零真值，使用浮点异常图；可分块读取降低内存，但不改变统计定义。
3. 对每个 shot/seed，将六个类别 AUROC 等权平均，不将类别合并为一个跨类 AUROC。
4. 对每个 shot 报三个 seed 结果的 mean±std，std 约定 ddof=0；同时保留各类别三 seed 汇总。
5. 不跨 shot 取最大值充当总分；不要漏掉指标较低的类或 seed。
6. 不按每张测试图归一化 score/map 来计算 AUROC。绘图时的色标操作与指标数组隔离。

## 5. 阈值、效率与公平性

- 阈值：每个演示配置在 20 张保留正常图上算 score，`quantile(scores, 0.95, method="linear")`。20 张样本的经验分位数不保证真实误报率精确为 5%。
- Part 1 的 1% 图像聚合等设置是团队方案；Part 2 保留官方图像分数规则，避免为了统一接口改变算法。
- 同一设备测试，预热、CUDA 同步、固定精度；主表报告完整特征推理，缓存命中速度另表。
- 记忆库构建和单图推理分开；峰值显存注明是否含常驻模型/记忆库及统计边界。CPU-only 的 GPU 显存用 N/A。
- 每方法应覆盖 72 个配置。两方法共 144 个类别配置，不代表需要训练 144 个模型。

## 6. 汇总结果文件

`results/tables/` 只提交真实 CSV。至少有逐配置结果、各类均值±标准差、六类平均、效率表。`results/figures/` 提交 shot 曲线、参考图示例和五个困难案例。当前目录仅有说明，没有实验成绩。
