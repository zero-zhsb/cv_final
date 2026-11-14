# Q&A 速记卡

## 先记住这组事实

| 项目 | 准确说法 |
|---|---|
| 本次 checkpoint | ≤5 分钟视频，目标约 4:40；无需宣称最终项目完成 |
| Part 1 进度 | 全量完成，6 类×4 shots×3 seeds=72 配置，11,520 次预测 |
| Part 2 进度 | 6 类×4 shots×3 seeds=72 配置，11,520 次测试预测 |
| 测试图片数量 | 960 张不同图片，每张用于 12 个 shot/seed 配置 |
| 单配置 image AUROC | Part 1 65.34%，Part 2 92.90%，差 27.56 个百分点 |
| 单配置 pixel AUROC | Part 1 77.80%，Part 2 98.69% |
| Part 1 整体 8-shot | image 60.80±0.32%，pixel 76.91±0.33%；跨三个 seed，ddof=0 |
| 当前主比较差值 | 8-shot 六类 image AUROC 相差 35.82 个百分点 |
| Part 2 整体 8-shot | image 96.62±0.57%，pixel 98.21±0.02%；跨三个 seed，ddof=0 |
| 阈值 | 每类 20 张独立正常训练图，仅校准；95% 经验分位 |
| baseline 模型 | 冻结 DINOv2 ViT-S/14，短边 448，官方 agnostic 设置 |
| 视频里的图 | 真实保存结果；PPT 放映本身不调用模型 |
| Part 3 | 固定 448、冻结骨干、多层特征融合；尚未实现验证 |

## 高概率问答：一句话版

1. **Why twenty extra images in a one-shot system?**
   They calibrate the demo threshold only and never enter the memory bank or test set.
2. **Why high AUROC but a wrong OK decision?**
   AUROC measures ranking across thresholds, while OK/NG uses one calibrated threshold.
3. **Is 60.80% the accuracy?**
   It is Part One's six-category mean image AUROC at eight shots, averaged over three seeds.
4. **Does the 95th percentile guarantee 5% false positives?**
   No, it is a small-sample calibration rule, not a guarantee on test data.
5. **Is Part Two running live in the slides?**
   The slides show genuine saved outputs; runnable single-image commands are available separately.
6. **What is the Part Three hypothesis?**
   Different DINOv2 layers may provide complementary information for small defects.
7. **Does layer fusion increase patch resolution?**
   No. The grid stays the same at input 448; any benefit must come from the representations and must be tested.
8. **What if it does not improve?**
   We will report the result and explain it with controlled ablations and failure analysis.

## 避免混淆

- 92.90% AUROC 不等于 92.90% accuracy；27.56 个百分点不等于 27.56% 相对增长。
- 示例图片是 cashew/1-shot/seed=0；当前主比较表和曲线都使用六类总体数据。
- frozen 不等于模型没有预训练；few-shot 不等于整个系统没有额外正常校准数据。
- 95% 分位不保证测试集误报率；热力图更亮不表示分数/性能更好。
- 多层融合不改变 patch 网格；不承诺一定改善。
- GPU 精确搜索有 CPU 数值核验，不能说逐位一致或原仓库一行未改。
- 真实图片回放不冒充现场推理；不能把计划说成已完成的实验。

细节、英文短答与源码证据见 [完整 Q&A](qa_prep.md)。
