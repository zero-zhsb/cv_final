# English narration — planned 4:40 cut

Timing is an editing plan, not a measured recording. Rehearse and retime subtitles after recording. Choose this OR the Chinese narration, not both.

## A | 00:00–00:25 | Problem

Our project studies few-shot industrial anomaly detection. In manufacturing, normal product images are easier to collect than examples of every possible defect. We use the VisA dataset to ask whether a few normal reference images can help detect and locate an anomaly in a new product image. The product on the right contains genuine defects.

## A | 00:25–01:05 | Current approach

We have implemented two approaches. Part One is a template-difference baseline. It selects the nearest normal template and computes an RGB difference map. Part Two uses frozen DINOv2 features and a memory bank of normal image patches. A test patch is scored by its distance from the nearest normal feature. Both methods use the same reference and test manifest. We reserve twenty additional normal training images per category for threshold calibration only. These images are excluded from the reference bank and test set. We do not train the methods using defect labels.

## A | 01:05–02:00 | Baseline implementation

Part One has now completed the full study: six categories, four shot counts and three seeds, giving seventy-two configurations. Its RGB template method remains unchanged. This slide illustrates one configuration, cashew with one reference and seed zero. The method outputs a normal template, anomaly map, continuous score and OK or NG decision. This defective image receives a score of about zero point four nine eight, below its calibrated threshold of about zero point six one, so the output is OK. Strong responses appear around the outline. Normal appearance or position differences may interfere with detection, although this explanation still requires controlled testing.

## B | 02:00–03:00 | Feature-based implementation

Part Two has been deployed on the server GPU using AnomalyDINO. DINOv2 remains frozen; we extract local patch features rather than training the backbone. Normal reference features form a memory bank, and exact nearest-neighbor distances produce anomaly scores and maps for test images. This card shows genuine saved outputs from our primary GPU experiment, for the same image, reference, shot count, and seed as Part One. Its score is about zero point two three two, above its own threshold of about zero point two one nine, so the output is NG. This image is correctly detected. However, the methods have different score ranges and heatmap scales. We compare their decisions and evaluation metrics, rather than raw score magnitudes or color brightness.

## B | 03:00–03:25 | Full six-category comparison

Both methods have now completed the same full evaluation protocol. At eight shots, six-category mean image AUROC is sixty point eight zero percent for template difference and ninety-six point six two percent for AnomalyDINO, a gap of thirty-five point eight two percentage points. The table also reports pixel AUROC and standard deviations across three seeds. These are full-study results, not just the illustrative cashew example.

## B | 03:25–03:45 | Both methods across shot counts

Across the four shot counts, template image AUROC increases from fifty-three point one seven to sixty point eight zero percent. AnomalyDINO increases from ninety-four point six one to ninety-six point six two percent. Each method has seventy-two configurations and eleven thousand five hundred and twenty test predictions. The feature method remains stronger in the aggregate results.

## B | 03:45–04:10 | Limitations

Both methods still miss some defects. This different cashew image contains several tiny annotated defects. In the eight-shot, seed-zero configuration, both methods output OK. The enlarged ground truth makes the small regions visible. Patch granularity may weaken Part Two鈥檚 small-defect signals, but the visualization alone cannot prove that explanation. This failure motivates our Part Three experiment.

## A | 04:10–04:40 | Future improvements

With both baseline studies complete, our next step is Part Three. We plan to fuse features from multiple DINOv2 layers while retaining the frozen backbone and input resolution of four hundred and forty-eight. We will compare with unchanged Part Two using identical references, seeds, preprocessing and evaluation. Alongside image and pixel AUROC, we will measure latency, memory, and any increase in background false positives. The fusion is planned, not yet validated.

Word count: **618**. Average over 4:40: **132.4 words/minute**, including visual pauses. Actual speech and screen capture are still to be recorded.
