# English narration — planned 4:40 cut

Timing is an editing plan, not a measured recording. Rehearse and retime subtitles after recording. Choose this OR the Chinese narration, not both.

## A | 00:00–00:25 | Problem

Our project studies few-shot industrial anomaly detection. In manufacturing, normal product images are easier to collect than examples of every possible defect. We use the VisA dataset to ask whether a few normal reference images can help detect and locate an anomaly in a new product image. The outlines on the right show the annotated defects.

## A | 00:25–01:05 | Current approach

We have implemented two approaches. Part One is a template-difference baseline. It selects the nearest normal template and computes an RGB difference map. Part Two uses frozen DINOv2 features and a memory bank of normal image patches. A test patch is scored by its distance from the nearest normal feature. Both methods use the same reference and test manifest. We reserve twenty additional normal training images per category for threshold calibration only. These images are excluded from the reference bank and test set. We do not train the methods using defect labels.

## A | 01:05–02:00 | Baseline implementation

Part One is implemented and has completed all one hundred and fifty test images in our first cashew configuration, using one reference image and seed zero. It outputs the selected template, an anomaly map, a continuous score, and an OK or NG decision. Here is a genuine failure. This defective image receives a score of about zero point four nine eight, below its calibrated threshold of about zero point six one. It is therefore classified as OK. Strong responses appear around the product outline, rather than only at the annotated defects. This suggests sensitivity to normal appearance or position changes, although confirming the cause requires further experiments.

## B | 02:00–03:00 | Feature-based implementation

Part Two has been deployed on the server GPU using AnomalyDINO. DINOv2 remains frozen; we extract local patch features rather than training the backbone. Normal reference features form a memory bank, and exact nearest-neighbor distances produce anomaly scores and maps for test images. This card shows genuine saved outputs from our primary GPU experiment, for the same image, reference, shot count, and seed as Part One. Its score is about zero point two three two, above its own threshold of about zero point two one nine, so the output is NG. This image is correctly detected. However, the methods have different score ranges and heatmap scales. We compare their decisions and evaluation metrics, rather than raw score magnitudes or color brightness.

## B | 03:00–03:25 | Preliminary comparison

The first comparison uses exactly the same one hundred and fifty cashew test images. Image AUROC is sixty-five point three four percent for Part One and ninety-two point nine zero percent for Part Two. Pixel AUROC also improves in this configuration. These are preliminary single-category results, not averages across all six categories. We are not claiming a controlled speed comparison.

## B | 03:25–03:45 | Broader Part 2 progress

Separately, Part Two has completed seventy-two configurations across six categories, four shot counts, and three seeds. Overall image AUROC rises from ninety-four point six one percent at one shot to ninety-six point six two percent at eight shots. Error bars show standard deviation across seeds. The full Part One comparison remains future work.

## B | 03:45–04:10 | Limitations

The method still misses defects. This different cashew image contains several tiny annotated defects, but the eight-shot configuration outputs OK because its score is below the threshold. Patch granularity may weaken small-defect signals, but the visualization alone does not establish that explanation. This example motivates our next experiments rather than a claim that the problem is solved.

## A | 04:10–04:40 | Future improvements

Our next steps are to complete the full Part One baseline study and test multi-scale feature fusion for small defects. We will compare the proposed change with unchanged Part Two using the same references, seeds, and evaluation protocol. Alongside detection and localization AUROC, we will measure latency and memory to assess the computational trade-off. These improvements are planned, not yet validated. Our current checkpoint demonstrates working implementations, reproducible preliminary results, and specific remaining limitations.

Word count: **620**. Average over 4:40: **132.9 words/minute**, including visual pauses. Actual speech and screen capture are still to be recorded.
