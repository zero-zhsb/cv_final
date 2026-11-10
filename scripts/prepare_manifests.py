"""Build a deterministic VisA course manifest without loading image pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_PROTOCOL = Path(__file__).resolve().parents[1] / "configs/course_protocol.json"


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Missing directory: {folder}")
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )


def build_manifest(data_root: Path, protocol: dict, allowed_image_roots=()) -> dict:
    root = data_root.resolve(strict=True)
    allowed = [Path(p).resolve(strict=True) for p in allowed_image_roots]

    def relative(path: Path) -> str:
        # Explicitly allow a read-only course data root for lightweight symlink views.
        resolved = path.resolve(strict=True)
        if not any(resolved.is_relative_to(base) for base in [root, *allowed]):
            raise ValueError(f"Image escapes approved data roots: {path}")
        return path.absolute().relative_to(root).as_posix()

    categories = protocol["categories"]
    shots, seeds = protocol["shots"], protocol["seeds"]
    count = protocol["calibration_count"]
    if (not categories or len(set(categories)) != len(categories)
            or any(Path(c).name != c or c in {".", ".."} for c in categories)):
        raise ValueError("Categories must be distinct directory names")
    if (not shots or not seeds or count <= 0
            or any(not isinstance(k, int) or k <= 0 for k in shots)
            or any(not isinstance(s, int) or s < 0 for s in seeds)):
        raise ValueError("Invalid shots, seeds or calibration count")
    if len(set(shots)) != len(shots) or len(set(seeds)) != len(seeds):
        raise ValueError("Shots and seeds must be distinct")
    result = {
        "schema_version": 1,
        "dataset": protocol["dataset"],
        "split": protocol["split"],
        "selection_rule": protocol["reference_selection"],
        "shots": shots,
        "seeds": seeds,
        "categories": {},
    }
    for category in categories:
        folder = root / category
        training = image_files(folder / "train/good")
        required = count + max((s + 1) * k for s in seeds for k in shots)
        if len(training) < required:
            raise ValueError(f"{category}: need at least {required} normal training images; got {len(training)}")
        calibration = [relative(p) for p in training[:count]]
        eligible = [relative(p) for p in training[count:]]
        references = {
            str(k): {str(s): eligible[s * k:(s + 1) * k] for s in seeds}
            for k in shots
        }
        normal_tests = image_files(folder / "test/good")
        abnormal_tests = image_files(folder / "test/bad")
        if not normal_tests or not abnormal_tests:
            raise ValueError(f"{category}: need both normal and abnormal test images")
        mask_files = image_files(folder / "ground_truth/bad")
        masks: dict[str, list[Path]] = {}
        for mask in mask_files:
            masks.setdefault(mask.stem, []).append(mask)
        tests = [{"image_path": relative(p), "mask_path": None, "label": 0} for p in normal_tests]
        for image in abnormal_tests:
            candidates = masks.get(image.stem, [])
            if len(candidates) != 1:
                raise ValueError(f"{category}: expected one mask matching stem {image.stem!r}, got {len(candidates)}")
            tests.append({"image_path": relative(image), "mask_path": relative(candidates[0]), "label": 1})
        # Verify sets after path resolution, not just distinct filename strings.
        train_paths = [str(p.resolve()) for p in training]
        test_paths = [str((root / item["image_path"]).resolve()) for item in tests]
        if len(set(train_paths)) != len(train_paths) or len(set(test_paths)) != len(test_paths):
            raise ValueError(f"{category}: duplicate resolved image paths")
        if set(train_paths) & set(test_paths):
            raise ValueError(f"{category}: training and test image overlap")
        if set(calibration) & set(eligible):
            raise ValueError(f"{category}: calibration and reference pool overlap")
        result["categories"][category] = {
            "counts": {"train_normal": len(training), "calibration": len(calibration),
                       "eligible_normal": len(eligible), "test_normal": len(normal_tests),
                       "test_abnormal": len(abnormal_tests)},
            "calibration": calibration,
            "eligible_normal": eligible,
            "references": references,
            "tests": tests,
        }
    content = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result["manifest_id"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Prepared VisA_pytorch/1cls directory")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=Path("manifests/visa_1cls.json"))
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing manifest")
    parser.add_argument("--allow-image-root", type=Path, action="append", default=[], help="Explicit read-only source root for symlinked images; repeatable")
    args = parser.parse_args()
    try:
        if args.output.exists() and not args.overwrite:
            raise ValueError(f"Output exists: {args.output}; use a new filename or --overwrite")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        result = build_manifest(args.data_root, protocol, args.allow_image_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Manifest error: {exc}\n")
    configurations = len(result["categories"]) * len(result["shots"]) * len(result["seeds"])
    print(f"Wrote {args.output}: {configurations} category/shot/seed configurations per method")
    print(f"manifest_id: {result['manifest_id']}")
    for category, entry in result["categories"].items():
        print(f"{category}: {entry['counts']}")


if __name__ == "__main__":
    main()
