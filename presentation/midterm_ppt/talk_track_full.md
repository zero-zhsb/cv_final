# 最新演讲稿：Part 1/2 全量完成版

7 页，目标 4:40。中英文二选一，配音后调整字幕；旧版“Part 1 待完成”的口播不再适用。

## A | 00:00-00:25 | Problem

### English

Our project studies few-shot industrial anomaly detection. In manufacturing, normal product images are easier to collect than examples of every possible defect. We use the VisA dataset to ask whether a few normal reference images can help detect and locate an anomaly in a new product image. The product on the right contains genuine defects.

### 中文

我们研究少样本工业异常检测。在实际生产中，正常产品图片容易获得，但缺陷样本往往稀少。我们希望只用少量正常参考图，判断一张新图片是否异常，并标出可疑位置。这里使用 VisA 数据集，右侧产品图片就包含真实缺陷。

动作：指向产品表面的异常区域。新版 PPT 封面只有这一张产品图，不再指“左侧正常图”或不存在的轮廓标注。

## A | 00:25-01:05 | Current approach

### English

We have implemented two approaches. Part One is a template-difference baseline. It selects the nearest normal template and computes an RGB difference map. Part Two uses frozen DINOv2 features and a memory bank of normal image patches. A test patch is scored by its distance from the nearest normal feature. Both methods use the same reference and test manifest. We reserve twenty additional normal training images per category for threshold calibration only. These images are excluded from the reference bank and test set. We do not train the methods using defect labels.

### 中文

我们实现了两条技术路线。Part 1 是模板差分：选择最接近测试图片的正常模板，计算 RGB 差异，生成异常图和图像分数。Part 2 使用冻结的 DINOv2 提取局部特征，建立正常特征库，再通过最近邻距离判断测试区域是否可疑。两种方法使用相同的参考图和测试名单。每类额外保留二十张正常训练图，只用于校准 OK、NG 阈值，不进入参考库和测试集。

## A | 01:05-02:00 | Baseline implementation

### English

Part One has now completed the full study: six categories, four shot counts and three seeds, giving seventy-two configurations. Its RGB template method remains unchanged. This slide illustrates one configuration, cashew with one reference and seed zero. The method outputs a normal template, anomaly map, continuous score and OK or NG decision. This defective image receives a score of about zero point four nine eight, below its calibrated threshold of about zero point six one, so the output is OK. Strong responses appear around the outline. Normal appearance or position differences may interfere with detection, although this explanation still requires controlled testing.

### 中文

Part 1 现在已经完成全量实验，覆盖六类别、四种参考图数量和三个 seed，共七十二个配置。模板差分方法本身保持不变。这里仍用腰果类别、一个参考样本、seed 为零的配置解释它的工作方式。方法输出正常模板、热力图、连续分数和判断结果。这张异常图片的分数约为零点四九八，低于零点六一的校准阈值，因此输出 OK，是一次漏检。热力图的较强响应集中在轮廓附近，正常外观或位置差异可能干扰检测，但具体原因还需要对照实验验证。

动作：先说明“全量已完成”，再解释此图仅是一个具体配置，不把单图结论当作总体结果。

## B | 02:00-03:00 | Feature-based implementation

### English

Part Two has been deployed on the server GPU using AnomalyDINO. DINOv2 remains frozen; we extract local patch features rather than training the backbone. Normal reference features form a memory bank, and exact nearest-neighbor distances produce anomaly scores and maps for test images. This card shows genuine saved outputs from our primary GPU experiment, for the same image, reference, shot count, and seed as Part One. Its score is about zero point two three two, above its own threshold of about zero point two one nine, so the output is NG. This image is correctly detected. However, the methods have different score ranges and heatmap scales. We compare their decisions and evaluation metrics, rather than raw score magnitudes or color brightness.

### 中文

Part 2 已经接入 AnomalyDINO，并在服务器 GPU 上完成运行。我们保持 DINOv2 参数冻结，用它提取局部 patch 特征，建立正常特征库。测试图中的每个局部区域与正常特征进行精确最近邻比较，得到异常距离，再生成整体分数和热力图。这里展示同一张图片，使用相同的正常参考图、shot 和 seed。分数约为零点二三二，超过该方法校准的零点二一九阈值，因此输出 NG，正确检出异常。两种方法的分数范围和色标不同，不能直接比较原始分数或颜色亮度，应结合判断结果和统一指标评价。

## B | 03:00-03:25 | Full six-category comparison

### English

Both methods have now completed the same full evaluation protocol. At eight shots, six-category mean image AUROC is sixty point eight zero percent for template difference and ninety-six point six two percent for AnomalyDINO, a gap of thirty-five point eight two percentage points. The table also reports pixel AUROC and standard deviations across three seeds. These are full-study results, not just the illustrative cashew example.

### 中文

两种方法现在都完成了同一协议下的全量实验。八样本配置中，六类平均图像 AUROC 分别为百分之六十点八零和九十六点六二，相差三十五点八二个百分点。表中同时给出像素 AUROC 和跨三个 seed 的标准差。这是六类总体比较，不再只是前面腰果样例所在的单配置结果。

## B | 03:25-03:45 | Both methods across shot counts

### English

Across the four shot counts, template image AUROC increases from fifty-three point one seven to sixty point eight zero percent. AnomalyDINO increases from ninety-four point six one to ninety-six point six two percent. Each method has seventy-two configurations and eleven thousand five hundred and twenty test predictions. The feature method remains stronger in the aggregate results.

### 中文

随着参考图从一张增加到八张，模板法的平均图像 AUROC 从百分之五十三点一七提高到六十点八零，AnomalyDINO 从九十四点六一提高到九十六点六二。每种方法都覆盖七十二个配置，各有一万一千五百二十条测试预测。总体上，特征方法保持更好的检测表现。

## B | 03:45-04:10 | Limitations

### English

Both methods still miss some defects. This different cashew image contains several tiny annotated defects. In the eight-shot, seed-zero configuration, both methods output OK. The enlarged ground truth makes the small regions visible. Patch granularity may weaken Part Two’s small-defect signals, but the visualization alone cannot prove that explanation. This failure motivates our Part Three experiment.

### 中文

但两种方法仍然存在失败。这是另一张腰果图片，包含几处非常小的真实缺陷。我们放大了真值掩码。在八样本、seed 为零的配置下，两种方法都输出 OK。Part 2 的 patch 特征空间粒度可能削弱小缺陷信号，但可视化本身不能证明原因，这个问题将作为 Part 3 的研究切入点。

## A | 04:10-04:40 | Future improvements

### English

With both baseline studies complete, our next step is Part Three. We plan to fuse features from multiple DINOv2 layers while retaining the frozen backbone and input resolution of four hundred and forty-eight. We will compare with unchanged Part Two using identical references, seeds, preprocessing and evaluation. Alongside image and pixel AUROC, we will measure latency, memory, and any increase in background false positives. The fusion is planned, not yet validated.

### 中文

两套基线实验完成后，下一步重点是 Part 3。我们计划固定四百四十八的输入分辨率，冻结 DINOv2，融合多个层级的特征。使用相同参考图、seed、预处理和评价协议，与不变的 Part 2 做受控对照。同时报告检测、定位、延迟和显存，并检查背景误报是否增加。这项改进仍是计划，目前尚未验证效果。
