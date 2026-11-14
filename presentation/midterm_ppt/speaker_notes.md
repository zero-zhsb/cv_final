# Slide 1

A: 0–25 sec

ENGLISH
Our project studies few-shot industrial anomaly detection. In manufacturing, normal product images are easier to collect than examples of every possible defect. We use the VisA dataset to ask whether a few normal reference images can help detect and locate an anomaly in a new product image. The product on the right contains genuine defects.

中文
我们研究少样本工业异常检测。在实际生产中，正常产品图片容易获得，但缺陷样本往往稀少。我们希望只用少量正常参考图，判断一张新图片是否异常，并标出可疑位置。这里以 VisA 数据集的腰果产品为例，右侧轮廓标出了真实缺陷。

动作：先指正常图，再指右侧缺陷标注。

Dataset: VisA https://github.com/amazon-science/spot-diff
Methods: AnomalyDINO https://github.com/dammsi/AnomalyDINO and DINOv2 https://github.com/facebookresearch/dinov2
Evidence: repository results/tables and presentation/midterm_video/provenance.json, source commit a3e03fb891155d5ca043b9f3bbf79144d52f4569.


# Slide 2

A: 25–65 sec

ENGLISH
We have implemented two approaches. Part One is a template-difference baseline. It selects the nearest normal template and computes an RGB difference map. Part Two uses frozen DINOv2 features and a memory bank of normal image patches. A test patch is scored by its distance from the nearest normal feature. Both methods use the same reference and test manifest. We reserve twenty additional normal training images per category for threshold calibration only. These images are excluded from the reference bank and test set. We do not train the methods using defect labels.

中文
我们先实现两条技术路线。Part 1 是基础模板差分：从正常参考图中选择最相近的模板，再计算 RGB 差异，生成异常图和图像分数。Part 2 使用冻结的 DINOv2 提取局部特征，把正常参考图的特征组成记忆库，再根据最近邻距离判断测试区域是否可疑。两种方法使用相同的参考图和测试名单。另外，每类保留二十张正常训练图，只用于校准 OK、NG 阈值，不加入参考库，也不进入测试集。

动作：沿两条流程箭头依次指示；无需讲全部参数。

Dataset: VisA https://github.com/amazon-science/spot-diff
Methods: AnomalyDINO https://github.com/dammsi/AnomalyDINO and DINOv2 https://github.com/facebookresearch/dinov2
Evidence: repository results/tables and presentation/midterm_video/provenance.json, source commit a3e03fb891155d5ca043b9f3bbf79144d52f4569.


# Slide 3

A: 65–120 sec

ENGLISH
Part One is implemented and has completed all one hundred and fifty test images in our first cashew configuration, using one reference image and seed zero. It outputs the selected template, an anomaly map, a continuous score, and an OK or NG decision. Here is a genuine failure. This defective image receives a score of about zero point four nine eight, below its calibrated threshold of about zero point six one. It is therefore classified as OK. Strong responses appear around the product outline, rather than only at the annotated defects. This suggests sensitivity to normal appearance or position changes, although confirming the cause requires further experiments.

中文
我们已经实现并运行了 Part 1。当前完成的首个配置是腰果类别、一个正常参考样本、seed 为零，覆盖完整的一百五十张测试图。方法会输出参考模板、差分热力图、连续分数和最终判断。这张真实缺陷图片的分数约为零点四九八，低于正常校准得到的零点六一阈值，因此被判为 OK，是一次漏检。从热力图可以看到，较强响应集中在产品轮廓附近，而不只在真正缺陷处。这说明直接像素差分存在局限；正常形状或位置变化可能干扰判断，具体原因仍需要对照实验。

动作：可替换前 5–10 秒为真实运行录屏；在输出后指示分数与阈值。若插入正常例 08，缩短本段讲解，不增加时长。

Dataset: VisA https://github.com/amazon-science/spot-diff
Methods: AnomalyDINO https://github.com/dammsi/AnomalyDINO and DINOv2 https://github.com/facebookresearch/dinov2
Evidence: repository results/tables and presentation/midterm_video/provenance.json, source commit a3e03fb891155d5ca043b9f3bbf79144d52f4569.


# Slide 4

B: 120–180 sec

ENGLISH
Part Two has been deployed on the server GPU using AnomalyDINO. DINOv2 remains frozen; we extract local patch features rather than training the backbone. Normal reference features form a memory bank, and exact nearest-neighbor distances produce anomaly scores and maps for test images. This card shows genuine saved outputs from our primary GPU experiment, for the same image, reference, shot count, and seed as Part One. Its score is about zero point two three two, above its own threshold of about zero point two one nine, so the output is NG. This image is correctly detected. However, the methods have different score ranges and heatmap scales. We compare their decisions and evaluation metrics, rather than raw score magnitudes or color brightness.

中文
Part 2 已经接入 AnomalyDINO，并在服务器 GPU 上完成运行。我们不训练 DINOv2，而是用它提取局部 patch 特征，建立正常特征库。对于新的测试图，每个局部区域与正常特征进行精确最近邻比较，得到异常距离，再生成整体分数和热力图。这里展示同一张图片，以及相同的正常参考图、shot 和 seed。分数约为零点二三二，超过该方法自己校准的零点二一九阈值，因此输出 NG。这个样例被正确检出，但不能据此说所有图片都能正确判断。两种方法的分数范围和色标不同，不能直接比较颜色亮度或原始分数大小。

动作：指真实掩码和对应区域，再指出 score、threshold、NG；画面为 GPU 主运行保存结果。

Dataset: VisA https://github.com/amazon-science/spot-diff
Methods: AnomalyDINO https://github.com/dammsi/AnomalyDINO and DINOv2 https://github.com/facebookresearch/dinov2
Evidence: repository results/tables and presentation/midterm_video/provenance.json, source commit a3e03fb891155d5ca043b9f3bbf79144d52f4569.


# Slide 5

B: 180–205 sec

ENGLISH
The first comparison uses exactly the same one hundred and fifty cashew test images. Image AUROC is sixty-five point three four percent for Part One and ninety-two point nine zero percent for Part Two. Pixel AUROC also improves in this configuration. These are preliminary single-category results, not averages across all six categories. We are not claiming a controlled speed comparison.

中文
初步比较使用完全相同的一百五十张腰果测试图，包括五十张正常图和一百张异常图。Part 1 的图像和像素 AUROC 分别为百分之六十五点三四和七十七点八零，Part 2 为九十二点九零和九十八点六九。这只是一个类别、一个配置的结果，不代表六类总体比较。

Dataset: VisA https://github.com/amazon-science/spot-diff
Methods: AnomalyDINO https://github.com/dammsi/AnomalyDINO and DINOv2 https://github.com/facebookresearch/dinov2
Evidence: repository results/tables and presentation/midterm_video/provenance.json, source commit a3e03fb891155d5ca043b9f3bbf79144d52f4569.


# Slide 6

B: 205–225 sec

ENGLISH
Separately, Part Two has completed seventy-two configurations across six categories, four shot counts, and three seeds. The chart shows category macro means across seeds. Eight-shot image AUROC reaches ninety-six point six two percent. The adjacent values report the standard deviation across seeds. The full Part One comparison remains future work.

中文
Part 2 已完成六类别、四种 shot 和三个 seed，总共 72 个配置。图中为类别宏平均后再跨 seed 的均值，旁边列出八样本配置的标准差。Part 1 全量比较仍待补齐。

Dataset: VisA https://github.com/amazon-science/spot-diff
Methods: AnomalyDINO https://github.com/dammsi/AnomalyDINO and DINOv2 https://github.com/facebookresearch/dinov2
Evidence: repository results/tables and presentation/midterm_video/provenance.json, source commit a3e03fb891155d5ca043b9f3bbf79144d52f4569.


# Slide 7

B: 225–250 sec

ENGLISH
The method still misses defects. This different cashew image contains several tiny annotated defects, but the eight-shot configuration outputs OK because its score is below the threshold. Patch granularity may weaken small-defect signals, but the visualization alone does not establish that explanation. This example motivates our next experiments rather than a claim that the problem is solved.

中文
方法仍存在失败。这是另一张腰果图片，在八样本配置下，真值包含几个很小的缺陷，但分数低于阈值，最终输出 OK。我们怀疑局部特征的空间粒度可能影响细小缺陷的检测，不过热力图本身不能证明这个原因。

动作：指三处白色真值小点，再指错误判断。强调该图是 8-shot，和前面的 1-shot 对比不同。

A: 250–280 sec

ENGLISH
Our next steps are to complete the full Part One baseline study and test features from multiple DINOv2 layers while keeping the input resolution at four hundred and forty-eight. We will keep the backbone frozen and compare with unchanged Part Two using identical references, seeds, preprocessing and evaluation. Alongside detection and localization AUROC, we will measure latency and memory. These improvements are planned, not yet validated.

中文
下一步先补齐 Part 1 的完整实验。针对小缺陷，计划固定 448 输入分辨率，融合 DINOv2 的多个层级特征。保持骨干冻结，使用相同参考图、seed、预处理与评价协议，与不变的 Part 2 做受控对照，同时记录检测、定位、时间和显存。这是尚未验证的计划。

Dataset: VisA https://github.com/amazon-science/spot-diff
Methods: AnomalyDINO https://github.com/dammsi/AnomalyDINO and DINOv2 https://github.com/facebookresearch/dinov2
Evidence: repository results/tables and presentation/midterm_video/provenance.json, source commit a3e03fb891155d5ca043b9f3bbf79144d52f4569.

Part 3 source: course Project 2 PDF, pp.2 and 4. Direction A permits multiple DINOv2 layers or image scales. Proposed implementation keeps input 448 and backbone frozen. Hypothesis only: patch granularity may contribute to misses.
