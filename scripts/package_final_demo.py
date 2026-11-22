"""Render a captioned walkthrough and ZIP from verified CLI outputs."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "demo/examples/final"
FOREGROUND = "#EAF1FB"
MUTED = "#AAC0D7"
ACCENT = "#75E6CF"


def font(size, weight="normal", family="DejaVu Sans"):
    path = font_manager.findfont(font_manager.FontProperties(family=family, weight=weight))
    return ImageFont.truetype(path, size)


def prediction(method, item):
    path = EXAMPLES / method / item
    return json.loads((path / "prediction.json").read_text()), path / "prediction.png"


def make_card(title, note, label, plot=None, lines=()):
    bg = Image.new("RGB", (1920, 1080), "#0E1927")
    draw = ImageDraw.Draw(bg)
    draw.rectangle((90, 83, 147, 91), fill=ACCENT)
    draw.text((168, 72), "AIAA 3201  /  FEW-SHOT INDUSTRIAL INSPECTION",
              font=font(23, "bold"), fill=MUTED)
    draw.text((90, 128), title, font=font(52, "bold"), fill=FOREGROUND)
    if plot:
        with Image.open(plot) as image:
            picture = ImageOps.contain(image.convert("RGB"), (1720, 630))
        frame = ((1920 - picture.width) // 2, 230 + (630 - picture.height) // 2)
        bg.paste(picture, frame)
    else:
        draw.rounded_rectangle((90, 260, 1830, 819), radius=20, fill="#1B2D41", outline="#49617A")
        line_y = 318
        for line in lines:
            draw.text((124, line_y), line, font=font(32, family="DejaVu Sans Mono"), fill=FOREGROUND)
            line_y += 68
    draw.text((90, 888), note, font=font(29), fill=FOREGROUND)
    draw.line((90, 990, 1830, 990), fill="#415670", width=2)
    draw.text((90, 1008), "VisA | Zou et al., ECCV 2022 | CC BY 4.0",
              font=font(21), fill=MUTED)
    draw.text((1830, 1008), label, font=font(21, "bold"), fill=ACCENT, anchor="ra")
    return bg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output / "anomaly_demo.mp4"
    archive = args.output / "anomaly_demo.zip"
    if video.exists() or archive.exists():
        parser.error("Choose an output folder without a previous video or archive")

    cases = [prediction("fusion448_672", x) for x in ("01_006", "02_044", "03_069")]
    normal, recovered, missed = (value for value, _ in cases)
    if [x["decision"] for x in (normal, recovered, missed)] != ["OK", "NG", "OK"]:
        raise ValueError("Saved outputs do not match the walkthrough story")
    content = [
        (make_card("Inspect a product image", "This walkthrough uses actual server predictions from a fresh run.",
                   "INPUT", lines=(
                       "python scripts/inspect_product.py --method fusion448_672",
                       "  --category cashew --shot 8 --seed 0 --device cuda:0",
                       "  --data-root <prepared VisA> --image <006> <044> <069>",
                       "  --output runs/demo_fusion",
                       "8 normal references | 20 separate normal calibration images")), 7),
        (make_card("Normal product passes", "The image score is below the normal-only threshold.",
                   f"OK  {normal['score']:.4f} / {normal['threshold']:.4f}", cases[0][1]), 7),
        (make_card("Fusion flags a tiny defect", "At 8-shot seed 0, both single-scale controls accepted this image.",
                   f"NG  {recovered['score']:.4f} / {recovered['threshold']:.4f}", cases[1][1]), 8),
        (make_card("One small defect still slips through", "A local heatmap response does not always raise the image score above threshold.",
                   f"OK  {missed['score']:.4f} / {missed['threshold']:.4f}", cases[2][1]), 8),
        (make_card("Compare the same defect regions", "All three heatmaps share a raw scale. The cyan outline is the evaluation mask.",
                   "8-SHOT  /  SEED 0", ROOT / "report/figures/cashew_pair.png"), 10),
        (make_card("Measured trade-offs", "The 672 control gave the best overall pixel AUROC in the full comparison.",
                   "SIX VISA CATEGORIES", lines=(
                       "8-shot image AUROC: 96.62% -> 97.19%",
                       "Tiny-defect miss rate: 29.36% -> 25.75%",
                       "Matched empirical false-alarm rate: 4.33%",
                       "Shared A6000 inference: 73.30 -> 205.81 ms/image",
                       "Comparison: Part 2 at 448 vs. 448+672 fusion")), 8),
    ]
    with TemporaryDirectory(prefix="cv-demo-") as path:
        working = Path(path)
        entries = []
        for index, (card, duration) in enumerate(content):
            filename = f"card-{index:02}.png"
            card.save(working / filename)
            entries.extend((f"file '{filename}'", f"duration {duration}"))
        entries.append(f"file '{filename}'")
        (working / "timeline.txt").write_text("\n".join(entries) + "\n")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "1",
                        "-i", "timeline.txt", "-vf", "fps=30", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                        "-crf", "19", "-movflags", "+faststart", str(video.resolve())],
                       cwd=working, check=True)

    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as package:
        package.write(video, video.name)
        for path in sorted(EXAMPLES.rglob("*")):
            if path.is_file():
                package.write(path, Path("representative_outputs") / path.relative_to(EXAMPLES))
        package.write(ROOT / "report/figures/cashew_pair.png", "representative_outputs/cashew_pair.png")
        package.write(ROOT / "demo/final_demo.md", "README.md")
    print(archive.resolve())


if __name__ == "__main__":
    main()
