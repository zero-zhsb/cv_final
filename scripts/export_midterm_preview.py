"""Export the existing 280-second cards/subtitles as a silent rehearsal preview."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserving it")
    base = ROOT / "presentation/midterm_video"
    segments = json.loads((base / "script_segments.json").read_text())
    if segments[0]["start"] != 0 or segments[-1]["end"] != 280 or any(a["end"] != b["start"] for a,b in zip(segments,segments[1:])):
        raise ValueError("Invalid 280 second timeline")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cv_preview_") as directory:
        listing = Path(directory) / "cards.txt"
        lines = []
        for segment in segments:
            image = (base / "assets" / segment["asset"]).resolve(strict=True)
            lines.extend(["file '" + image.as_posix().replace("'", "'\\''") + "'", f"duration {segment['end']-segment['start']}"])
        lines.append(lines[-2])
        listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-i", str(base / "subtitles_en.srt"), "-map", "0:v:0", "-map", "1:s:0", "-vf", "fps=15,scale=1920:1080",
                        "-c:v", "libx264", "-preset", "veryfast", "-threads", "2", "-pix_fmt", "yuv420p", "-c:s", "mov_text",
                        "-t", "280", "-metadata", "title=Silent rehearsal preview - human narration not recorded", "-movflags", "+faststart", str(args.output)], check=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
