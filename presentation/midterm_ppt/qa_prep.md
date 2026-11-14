# Midterm Q&A：英文短答与中文追问准备

本文件对应 7 页视觉增强 PPT。老师提供的通知要求提交 ≤5 分钟视频，没有要求现场答辩；这里用于准备潜在追问，也用于检查讲稿。短答建议控制在 15–25 秒，先回答问题，再说明证据和边界。没有测过的效果直接说尚未验证。

## 一、任务与方法

### 1. What does “few-shot” mean in your project?【A/B】

**English:** It is the number of distinct normal reference images used to build the template set or memory bank. We evaluate one, two, four and eight references. One image can still produce many local patch features.

中文：shot 数的是原始正常参考图片，不是特征向量，也不是训练轮数。官方旋转增强来自同一图片，不增加独立 reference 数。证据：[共享名单](../../manifests/visa_1cls.json)。

### 2. If it is 1-shot, why do you use 20 more normal images?【A/B，高频】

**English:** The one reference image builds the memory bank. The twenty additional normal images only calibrate the demo threshold. They are excluded from reference selection and the test set, and they do not update the model or memory bank.

中文：必须承认系统还使用课程规定的额外正常校准数据，不能说“整个系统只看了一张正常图片”。同一类的 20 张校准图会在不同配置中重复评分，并非每次从测试集抽取。阈值是各配置正常校准分数的第 95 百分位。

### 3. Did you train DINOv2 or use labeled defects?【B】

**English:** We use official pretrained DINOv2 weights with a frozen backbone. Our required pipeline does not perform task-specific training on VisA defects. Test labels and masks are used for evaluation and qualitative error analysis, not parameter selection.

中文：预训练模型来自外部数据，不能说“模型从来没有训练过”。我们没有自行审计全部预训练数据，因此不能凭空保证预训练语料完全没有数据重叠。当前不训练 backbone，也没有实现 Part 3 adapter。

### 4. How exactly does Part 1 choose a template and score an image?【A】

**English:** We choose the reference with the smallest global mean absolute RGB difference. We average the channel differences, smooth the map with a Gaussian filter, and average the highest one percent of map values to obtain the image score.

中文：先按相同几何设置缩放/裁剪；RGB 值范围 [0,1]，sigma=1。是先选整张图的最近模板，再形成差分图，不是每个像素各自选一个模板。取最高 1% 时向下取整，至少一个值。当前没有做 ECC 配准。证据：[实现](../../src/part1/inspector.py)。

### 5. How does AnomalyDINO compute the anomaly score?【B，高频】

**English:** Frozen DINOv2 produces patch features. For each foreground test patch, we find the nearest normalized normal feature in the memory bank. Larger distances indicate stronger deviations from normality. We average the highest one percent of patch scores for image-level detection.

中文：L2 归一化特征后，距离为 squared L2 / 2，对应 1−cosine similarity。保留官方前景处理，背景 patch 分数为零。图像分数取原 patch 距离最高 1% 的均值；热力图再上采样并按官方方式平滑。不能把“平滑热力图最高值”误说成当前图像评分公式。证据：[适配器](../../src/part2/inspector.py)。

### 6. Why might pretrained features work better than pixel differences?【B】

**English:** Pixel differences are sensitive to alignment, illumination and normal appearance changes. Pretrained patch features can represent more useful visual structure. In our initial cashew configuration, feature matching gives higher AUROC, but we do not claim it always wins on every image.

中文：这是合理机制解释和本次观察，不是我们用实验彻底证明了全部因果关系。没有实现 ECC，因此不能把当前结果当作与“最佳配准模板法”的比较。

### 7. Are these heatmaps explanations of the model?【A/B】

**English:** They visualize anomaly scores and where the method responds. They help us inspect failures, but they do not establish a causal explanation for a prediction.

中文：上采样不能创造模型没有分辨出的细节。可以说“轮廓附近响应强”“小缺陷信号弱”，不要说“热力图证明原因一定是某一层”。两个方法色标不同，不比较颜色亮度。

## 二、数据与公平性

### 8. How do you ensure a fair comparison?【A/B，高频】

**English:** The full comparison uses identical reference filenames, test images and masks, and the same input geometry and AUROC definition. We verified all eleven thousand five hundred and twenty prediction identities. Pixel evaluation is exact at the original resolution for both methods. Part One and Part Two use their own calibrated thresholds.

中文：输入短边 448，右/下裁到 14 的倍数；方法各自必要的特征归一化/差分处理不同，需要披露。两方法 CPU/GPU 及运行时负载不同，所以不能宣称受控速度优势。Part 3 将固定预处理、448 输入、参考名单和评价协议。证据：[比较表](../../results/tables/part1_part2_cashew_milestone.csv)、[A 里程碑](../../docs/11_member_a_milestone.md)。

### 9. Are seeds random samples, and are larger shot sets nested?【B】

**English:** We follow the pinned official implementation’s deterministic reference selection. After excluding calibration images, it takes a sorted slice indexed by seed and shot count. We save the actual filenames, and sets across shot counts are not always nested.

中文：规则为 eligible[seed*shot:(seed+1)*shot]。不要把 seed 0/1/2 描述成我们自行设计的三次独立随机抽样。seed=0 可以嵌套，其他 seed 的区间会随 shot 改变，不能保证全部配置嵌套。

### 10. Does 11,520 predictions mean 11,520 distinct test images?【B】

**English:** No. There are nine hundred and sixty distinct test images across the six categories. Each is evaluated under twelve shot-and-seed configurations, giving eleven thousand five hundred and twenty predictions.

中文：6 类×4 shots×3 seeds=72 个类别配置；960×12=11,520 次测试预测。同一图跨配置重复出现是设计的一部分，不是新增独立样本。

### 11. Did you select examples or parameters using test labels?【A/B】

**English:** We use test labels to evaluate predictions and select illustrative failures after evaluation. We do not use those labels to set thresholds or choose model parameters. The displayed examples are representative case studies rather than a random estimate of performance.

中文：分析真实失败来提出研究问题，与用测试分数选择参数不同。后续 Part 3 的层选择、融合权重等，只能用允许的训练/校准数据确定，不根据测试 AUROC 反复挑最优方案。

### 12. Why only six VisA categories?【A/B】

**English:** The course specifies these six categories and the official one-class split. We follow that scope so we can focus on reproducible experiments, analysis and a justified improvement. All twelve categories are optional.

中文：不是跑全 12 类才有高分；也不需要为了本次 checkpoint 扩大数据集。类别为 candle、capsules、cashew、chewinggum、fryum、pipe_fryum。证据：[课程要求转述](../../docs/01_course_requirements.md)。

## 三、结果与指标

### 13. What is actually completed at this checkpoint?【A/B，高频】

**English:** Both Part One and Part Two have completed all seventy-two configurations, with eleven thousand five hundred and twenty test predictions per method. We also have single-image inference tools. Part Three remains planned work.

中文：Part 1 的全量结果已由 616701a 发布，并核验 72 配置和 11,520 条预测与 Part 2 身份一致。此前 cashew 单配置结果仍用于说明方法，但已不代表当前完成范围。Part 3 仍是计划。

### 14. What do the 60.80% and 96.62% numbers mean?【A/B，高频】

**English:** They are six-category mean image AUROC values at eight shots, averaged over three seeds. Part One reaches sixty point eight zero percent and Part Two reaches ninety-six point six two percent. The difference is thirty-five point eight two percentage points.

中文：它们是六类总体 AUROC，不是 accuracy。差 35.82 个百分点，不能说成相对提升 35.82%。对应 pixel AUROC 为 76.91±0.33% 与 98.21±0.02%。此前 65.34%/92.90% 仍是 cashew/1-shot/seed=0 的有效例子，不能混淆。

### 15. Why use AUROC rather than only accuracy?【A/B】

**English:** AUROC evaluates how well continuous scores rank anomalous samples above normal samples across thresholds. Accuracy depends on one chosen threshold. We report AUROC for evaluation and a separately calibrated threshold for the demo decision.

中文：AUROC=0.929 不代表 92.9% 图片被判断正确。图像 AUROC 对图像排序，pixel AUROC 对像素排序；正常图的零真值像素也必须参加像素评价。

### 16. How can AUROC be high while an image is classified incorrectly?【A/B，高频】

**English:** AUROC measures ranking across thresholds, while OK or NG uses one threshold. Good overall ranking does not guarantee that every defect exceeds the normal-calibrated threshold. Small defects may also receive weak image-level scores.

中文：pixel AUROC 高也不保证每张异常图都被检出，检测和定位是两个问题。大量容易区分的背景像素可能使总体像素排序掩盖局部困难，需结合图像指标和真值案例看，不能只展示漂亮热力图。

### 17. Does the 95th-percentile threshold guarantee 5% false positives?【A/B，高频】

**English:** It is an empirical percentile from twenty normal calibration images, not a guarantee on future test data. The small calibration set and distribution differences can produce a different test false-positive rate.

中文：不能通过观察测试误报后调整阈值，再把同一测试集当作公平最终评估。预先规定正常校准规则，对每个方法/类别/shot/seed 单独计算。

### 18. What does 96.62 ± 0.57% represent?【B】

**English:** It is Part Two’s eight-shot image AUROC. We first average the six category AUROCs equally for each seed, then report the mean and population standard deviation across three seeds. The plus-minus value is not a confidence interval.

中文：标准差为 0.57 个百分点，ddof=0；不是跨类别标准差，也不是图片误差比例。像素结果为 98.21±0.02%。不要把这组整体成绩直接与 Part 1 单类 65.34% 比较。证据：[整体汇总](../../results/tables/part2_gpu_overall_by_shot.csv)。

### 19. Does more reference data always improve the result?【B】

**English:** Our aggregate trend improves as the reference count increases, but individual categories, seeds or threshold decisions can behave differently. Reference sets and calibration distributions matter, so we report all seeds instead of selecting the best one.

中文：同一 bad/000.JPG 在 1-shot 可以被检出，在 8-shot 也可能低于该配置阈值。新增正常样本既可能改善正常覆盖，也可能降低某些缺陷距离。不能承诺单调改善。

## 四、实现、复现与效率

### 20. Are the slides live inference or previously computed results?【A/B，高频】

**English:** The slides show saved outputs from actual inference. Both methods also have runnable single-image commands. The earlier combined demo replays Part Two results, so we do not describe that interface as simultaneous live inference.

中文：只有在视频中实际插入新录制的执行片段，才说该片段为实际运行。PPT 放映不调用模型，截图是真实结果也不等于现场实时推理。

### 21. What is your own work beyond running an official repository?【A/B】

**English:** The feature method and pretrained model come from the cited official sources. Our implementation work includes the template baseline, shared reference and calibration manifests, output adapters, exact evaluation, reproducibility records and demonstrations. The proposed Part Three improvement has not yet been validated.

中文：不把官方 DINOv2 或 AnomalyDINO 算作原创。A/B 根据本人实际参与解释分工：A 侧重模板法/数据准备/公共比较，B 侧重模型接入/正式矩阵/效率与数值核验，双方核对协议。具体开发辅助和个人贡献按实际情况说明。

### 22. Did changing nearest-neighbor search invalidate the reproduction?【B，技术追问】

**English:** We retain the official features, preprocessing, reference selection and scoring rule. The primary run uses exhaustive GPU nearest-neighbor search, with the CPU FAISS path retained as a reference. We disclose this backend change and validate numerical and metric agreement instead of claiming bitwise identity.

中文：TF32 关闭，归一化 FP32 特征做全量 k=1 squared L2；不是 ANN。19 个完整重叠配置的 image AUROC 差为零，pixel AUROC 最大绝对差约 6.05e-8。不是 72 个配置都做了完整 CPU/GPU 双跑，更不能说所有距离逐位一致。若被要求严格的官方 CPU 后端完整复现，需补完该后端全量参照，而不是隐瞒差异。此加速不是 Part 3 已完成的算法改进。证据：[后端核验 CSV](../../results/tables/cpu_gpu_configuration_parity.csv)、[GPU 记录](../../docs/09_gpu_experiments.md)。

### 23. How did you compute pixel AUROC without storing huge full maps?【B，技术追问】

**English:** We store compact patch maps and original image dimensions, then reconstruct the official full-resolution maps for evaluation. Our evaluator counts exact float32 values without approximate score bins or per-image normalization. We cross-check it against the CPU implementation.

中文：不能说“用低分辨率图计算 pixel AUROC”。五个完整 candle 配置的 GPU/CPU 像素指标相同。compact 真图重建与保存 full-map 也做过一致性检查。热力图显示颜色缩放与评价浮点值区分开。

### 24. Is Part Two faster, and what does GPU memory mean?【A/B】

**English:** We have measured runtime, but the initial CPU template and GPU feature runs used different conditions, so we do not claim a controlled speed advantage. Inference timing includes image loading, preprocessing, feature extraction and search. Map rendering and file export are excluded.

中文：Part 2 全量无测试特征缓存，建 reference bank 另记时间。PyTorch peak allocated GPU memory 不等于整张卡总占用，评价过程也需额外显存。共享设备负载不同，不能据约 51 ms 与 74 ms 就声称一种算法的绝对速度比。

## 五、Part 3 的核心追问

### 25. Why did you choose multi-layer fusion, and what exactly will change?【A/B，高频】

**English:** Tiny defects are a concrete observed failure. We plan to combine representations from multiple DINOv2 layers while retaining the frozen backbone and input resolution of four hundred and forty-eight. We will compare this change against unchanged Part Two under the same data protocol.

中文：具体层号和融合权重尚未选定，不能编造“已证明第几层最好”。先用允许的训练/校准信息确定规则，冻结后评估。多层融合属于课程 Direction A，只做一个有理由的方向即可，不需要六个方向全做。

### 26. Does combining layers increase spatial resolution?【B，关键追问】

**English:** No. At the same input size, combining layers does not create a finer patch grid. Our hypothesis is that different layers preserve complementary visual information. It may help small defects, but that effect still needs an ablation study.

中文：ViT-S/14 的 tokenization patch 大小为 14×14；短边 448 对应 32 个 patch。多层信息不同，不会自动让一个 patch 变成 7×7，也不能保证恢复 patch 内的全部细节。不能声称“固定 448 多层融合提高空间分辨率”。

### 27. How will you prove the improvement comes from fusion?【B】

**English:** We will retain the original final-layer method as the control, then compare selected single-layer variants with the fixed fusion rule. References, seeds, preprocessing, resolution and evaluation will stay consistent. We will also report latency and memory overhead.

中文：对照要区分“换某一个层就有效”和“组合层有效”。层/权重不在测试集上反复挑选。当前没有该消融成绩，不能把计划写成结果。失败案例既看小缺陷，也检查背景误报是否增加。

### 28. What if Part Three gives no improvement?【A/B】

**English:** We will report the negative result and use controlled ablations and failure analysis to explain it. Possible outcomes include redundant features or more background false positives. We will not claim success without measured evidence.

中文：课程明确允许主指标不提升，但需要实际实现和实验分析；完全忽略 Part 3 会被判 Incomplete。小缺陷子集改善也不能替代报告总体主指标。若提出效率方向，仍需披露检测性能与成本的取舍。

### 29. Why not fine-tune a large model or use AnomalyCLIP instead?【B】

**English:** The required route uses a frozen backbone, and the course asks for one justified improvement. Multi-layer fusion directly tests a failure we observed with limited extra complexity. AnomalyCLIP is an optional extension, and we have not evaluated it.

中文：不能说微调一定更差，也不能说 AnomalyCLIP 已经被我们排除为无效。课程允许明确研究小 adapter，或在合成缺陷方向训练轻量评分头，但当前方案不需要这些训练。

## 两人练习方式

第一轮优先练 2、5、8、13–18、20、25–28。A 主答基线/阈值，B 主答特征/指标/Part 3，另一人补一句证据即可。每题按“结论—证据—边界”回答，不用背整段参数。

不确定时可以直接说：**“We have not tested that yet. We would need a controlled comparison before making that claim.”** 只有在确实没有测试时使用，不把它当作回避已实现细节的万能句。
