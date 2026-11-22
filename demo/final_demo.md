# Final inspection demonstration

The captioned MP4 walks through fresh CLI predictions produced on the shared RTX A6000 and the template CPU path. The script accepts new image files, builds the frozen normal reference bank, calibrates from 20 separate normal training images, and exports the score, threshold, OK/NG, and heatmap. Ground truth is used only in the side-by-side evaluation figure.

The three representative inputs are VisA cashew good/006, bad/044, and bad/069 at 8-shot, seed 0. The first is accepted by all methods. Fusion detects 044 while both single-scale controls and the RGB template accept it. All four methods miss 069. Each prediction JSON records the inputs to calibration, prediction device, reference paths, and decision.

VisA images and masks are by Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer, ECCV 2022, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The heatmaps, layout, overlays, and crops are derived from those inputs. [Dataset and preparation](https://github.com/amazon-science/spot-diff)

## Reproduce inference

Prepare the pinned dependencies, local DINOv2 weights, and VisA `1cls` directory as described in the [repository README](https://github.com/zero-zhsb/cv_final/tree/codex/final-delivery). From the repository root, run the following on Linux with a CUDA-enabled PyTorch build.

```bash
python scripts/inspect_product.py --method fusion448_672 \
  --category cashew --shot 8 --seed 0 --device cuda:0 \
  --data-root "$VISA_1CLS" \
  --image "$VISA_1CLS/cashew/test/good/006.JPG" \
          "$VISA_1CLS/cashew/test/bad/044.JPG" \
          "$VISA_1CLS/cashew/test/bad/069.JPG" \
  --output runs/demo_fusion
```

For a product image of your own, substitute its path after `--image` and select the corresponding product category. Use `--method part2_448`, `--method scale672`, or `--method template --device cpu` for the controls. CPU feature inference uses exact FAISS and recalibrates the threshold for that backend. The benchmark results were collected with the CUDA exact matcher.

## Rebuild the package

```bash
python scripts/package_final_demo.py --output runs/final_demo_export
```

This uses FFmpeg and the versioned prediction outputs to render the captioned video and package `anomaly_demo.zip`.
