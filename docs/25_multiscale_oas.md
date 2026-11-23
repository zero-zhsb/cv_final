# 多尺度融合＋OAS 正常统计度量

在原 448+672 等权融合和共同参考 patch 最近邻匹配之外，追加参考图拟合的 OAS 度量。主干、两尺度对齐、fine PCA mask、参考旋转、top-1% 图像聚合和原分辨率热图处理保持不变。OAS 自动收缩协方差，不扫描正则或维度；满秩白化后除以 sqrt(768)，不再次做 L2 归一化。

正常统计仅来自该配置指定的 few-shot 参考图。原定 20 张独立正常校准图只用于重新计算 95% 分位阈值，不拟合协方差、不扩充 bank。校准后恢复原 NumPy RNG，避免随机 PCA 掩码序列混入比较。

## 完整实验

六类 × 1/2/4/8-shot × seed0/1/2，72 配置、11,520 测试预测；像素评价为包含正常图的原分辨率精确 AUROC，没有逐图归一化。先每个 seed 六类等权平均，再求三个 seed 的 mean/population SD。

| Shot | 原融合 Image AUROC (%) | OAS Image AUROC (%) | 原融合 Pixel AUROC (%) | OAS Pixel AUROC (%) |
|---|---:|---:|---:|---:|
| 1 | 95.72 | 94.51 | 97.89 | 98.41 |
| 2 | 96.21 | 95.55 | 98.03 | 98.52 |
| 4 | 96.82 | 96.59 | 98.19 | 98.58 |
| 8 | 97.19 | 97.08 | 98.30 | 98.63 |

8-shot OAS Image AUROC **97.079 ± 0.382%**，Pixel AUROC **98.628 ± 0.001%**。配对平均变化分别为 **-0.114 pp / +0.323 pp**。四种 shot 像素宏平均均提高约 0.323–0.520 pp，但图像宏平均均下降，不能称为全面提升；低 shot 的检测代价更明显。三个 seed 的标准差不是置信区间或显著性检验。

结果表、每配置指标和审计在 `results/tables/multiscale_oas/`。方法是在观察过同一测试集的错误后提出并选定，属于探索性证据，不是独立测试集确认。保留原多尺度作为默认对照，不根据测试类别挑选方法。

## 推理与评价

使用仓库已有的固定模型依赖和本地权重。OAS 入口要求 CUDA exact matcher，模型不训练；数据/源码/权重默认遵循现有目录结构。

```bash
# 新产品图：只使用正常参考和校准图，不读取真值。
python scripts/inspect_product.py --method fusion448_672_oas \
  --category cashew --shot 8 --seed 0 --device cuda:0 \
  --data-root "$VISA_1CLS" --image "$PRODUCT_IMAGE" --output runs/demo_oas_new

# 完整矩阵：输出必须为新目录；不覆盖已发布成绩。
python scripts/run_oas.py --data-root "$VISA_1CLS" \
  --allow-image-root "$VISA_ORIGINAL_ROOT" \
  --shots 1 2 4 8 --seeds 0 1 2 --map-storage patch \
  --threads 2 --gpu-memory-fraction 0.15 --output runs/oas_new

# 推理完成后串行精确评价，不需和推理同时占 GPU。
python scripts/evaluate_part2.py --run runs/oas_new \
  --data-root "$VISA_1CLS" --pixel-device cuda:0-positive-keys
python scripts/summarize_oas.py --run runs/oas_new \
  --baseline "$MULTISCALE_BASELINE_RUN" --pixel-complete
```

若数据不使用外部符号链接，可以省略 `--allow-image-root`。新汇总输出到 `results/tables/multiscale_oas_generated/`，与已发布历史表分开。共享 GPU 的 allocator 上限不是显存预留；性能数字不能与不同日期的运行直接构造受控速度比。

## 来源及复现限制

OAS 采用 [scikit-learn OAS](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html)。本 PR 是其作为多尺度描述子度量模块的应用，不是新的基础模型或完整新论文复现。

实际矩阵由源码提交 `19de944` 的模块化实验入口运行；OAS-only API 从该路径抽取，纯数值函数保持相同，未声称重新执行了矩阵。紧凑 provenance 保存原始源文件哈希与每配置完成信息。

随后使用公开提交 `5307ecc` 的独立 OAS 入口复跑 cashew / 8-shot / seed 0 的全部 150 张测试图。图像和像素 AUROC 均与历史表的同配置值完全一致。[核验记录](../results/tables/final_delivery_repro_audit.json)

旧运行复用 `fit_details` 字典，可能覆盖各配置的 shrinkage 元数据；该字段不参与评分或评价，拟合 index 和实际指标不受影响。紧凑记录已移除不可靠的逐配置 shrinkage 值，当前 API 每次拟合创建新字典。若分析历史每配置 shrinkage，需要用参考图重新拟合，不能从旧元数据推断。
