"""Copy selected VisA 1cls categories using official CSV paths and mask semantics.

Adapted from amazon-science/spot-diff utils/prepare_data.py (Apache-2.0),
commit 2a692ab575001cbde74d402d897a7286086c6199. Only scope/validation differ:
copy selected categories, binarize nonzero masks to 255, verify B's manifest.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Prepared 1cls root")
    parser.add_argument("--categories", nargs="+", default=["cashew"])
    parser.add_argument("--manifest", type=Path, default=Path("manifests/visa_1cls.json"))
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    manifest = json.loads(args.manifest.read_text())
    if not set(args.categories) <= set(manifest["categories"]):
        parser.error("Categories outside the shared manifest")
    if any((args.output / category).exists() for category in args.categories):
        parser.error("Destination category exists; preserving it")
    source = args.source.resolve(strict=True)
    split = source / "split_csv/1cls.csv"
    with split.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["object"] in args.categories]
    work = []
    for row in rows:
        image = (source / row["image"]).resolve(strict=True)
        image.relative_to(source)
        label = "good" if row["label"] == "normal" else "bad"
        relative = Path(row["object"]) / row["split"] / label / image.name
        mask = (source / row["mask"]).resolve(strict=True) if row["mask"] else None
        if mask:
            mask.relative_to(source)
        work.append((row, image, relative, mask))
    # Validate the entire requested scope BEFORE copying any files.
    for category in args.categories:
        entry = manifest["categories"][category]
        train = {rel.as_posix() for r, _, rel, _ in work if r["object"] == category and r["split"] == "train"}
        test = {rel.as_posix() for r, _, rel, _ in work if r["object"] == category and r["split"] == "test"}
        if train != set(entry["calibration"] + entry["eligible_normal"]):
            raise ValueError(f"Official train CSV and B's manifest differ: {category}")
        if test != {r["image_path"] for r in entry["tests"]}:
            raise ValueError(f"Official test CSV and B's manifest differ: {category}")
        expected = {r["image_path"]: r for r in entry["tests"]}
        for row, _, relative, mask in work:
            if row["object"] == category and row["split"] == "test":
                actual = (Path(category) / "ground_truth/bad" / mask.name).as_posix() if mask else None
                if actual != expected[relative.as_posix()]["mask_path"]:
                    raise ValueError("Mask path differs from B's manifest")
    for row, image, relative, mask in work:
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination)
        if row["split"] == "test" and mask:
            mask_path = args.output / row["object"] / "ground_truth/bad" / mask.name
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(mask) as source_mask:
                binary = (np.asarray(source_mask) != 0).astype(np.uint8) * 255
            Image.fromarray(binary).save(mask_path)
    args.output.mkdir(parents=True, exist_ok=True)
    record = {"source": str(source), "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
              "manifest_id": manifest["manifest_id"], "categories": args.categories,
              "copied_images": len(work), "image_storage": "independent copies", "mask_values": [0, 255]}
    (args.output / ("preparation_" + "_".join(args.categories) + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
