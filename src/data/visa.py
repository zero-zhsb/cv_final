"""Read the official VisA CSV through a verified course-manifest mapping."""
import csv
import hashlib
import json
from pathlib import Path


class VisAReader:
    def __init__(self, source, manifest):
        self.source = Path(source).resolve(strict=True)
        self.manifest = manifest
        payload = {k: v for k, v in manifest.items() if k != "manifest_id"}
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode()).hexdigest() != manifest["manifest_id"]:
            raise ValueError("Manifest hash mismatch")
        split = self.source / "split_csv/1cls.csv"
        self.split_sha256 = hashlib.sha256(split.read_bytes()).hexdigest()
        self.mapping = {}
        tests = {}
        with split.open(newline="") as handle:
            for row in csv.DictReader(handle):
                category = row["object"]
                if category not in manifest["categories"]:
                    continue
                label = "good" if row["label"] == "normal" else "bad"
                image = (self.source / row["image"]).resolve(strict=True)
                image.relative_to(self.source)
                relative = f"{category}/{row['split']}/{label}/{image.name}"
                if relative in self.mapping:
                    raise ValueError("Duplicate official image")
                self.mapping[relative] = image
                mask_relative = None
                if row["mask"]:
                    mask = (self.source / row["mask"]).resolve(strict=True)
                    mask.relative_to(self.source)
                    mask_relative = f"{category}/ground_truth/bad/{mask.name}"
                    self.mapping[mask_relative] = mask
                if row["split"] == "test":
                    tests[relative] = (mask_relative, int(label == "bad"))
        for category, entry in manifest["categories"].items():
            expected = {r["image_path"]: (r["mask_path"], r["label"]) for r in entry["tests"]}
            actual = {k: v for k, v in tests.items() if k.startswith(category + "/")}
            if actual != expected:
                raise ValueError(f"Official test identities differ: {category}")
            expected_train = set(entry["calibration"] + entry["eligible_normal"])
            if {p for p in self.mapping if p.startswith(category + "/train/")} != expected_train:
                raise ValueError(f"Official normal train identities differ: {category}")

    def path(self, relative):
        return self.mapping[relative]

    def mask(self, sample, shape):
        import numpy as np
        from PIL import Image
        if sample["mask_path"] is None:
            return np.zeros(shape, dtype=bool)
        with Image.open(self.path(sample["mask_path"])) as image:
            result = np.asarray(image) > 0  # same binarization as official preparation
        if result.ndim != 2 or result.shape != tuple(shape):
            raise ValueError("Mask and original-image coordinates differ")
        return result
