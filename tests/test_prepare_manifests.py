"""Meaningful checks for calibration isolation and stable reference identity."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_manifests", ROOT / "scripts/prepare_manifests.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = json.loads((ROOT / "configs/course_protocol.json").read_text(encoding="utf-8"))
        # Empty image fixtures are sufficient: this tool validates names, not pixels.
        for category in self.protocol["categories"]:
            for subdir in ("train/good", "test/good", "test/bad", "ground_truth/bad"):
                (self.root / category / subdir).mkdir(parents=True)
            for i in reversed(range(44)):
                (self.root / category / "train/good" / f"{i:03d}.jpg").touch()
            (self.root / category / "train/good/README.txt").touch()
            (self.root / category / "test/good/normal.jpg").touch()
            (self.root / category / "test/bad/defect.jpg").touch()
            (self.root / category / "ground_truth/bad/defect.png").touch()

    def test_all_72_configurations_exclude_calibration_and_preserve_upstream_slices(self):
        result = MODULE.build_manifest(self.root, self.protocol)
        configs = 0
        for category, entry in result["categories"].items():
            self.assertEqual(len(entry["calibration"]), 20)
            self.assertTrue(entry["calibration"][0].endswith("000.jpg"))
            for shot in self.protocol["shots"]:
                for seed in self.protocol["seeds"]:
                    refs = entry["references"][str(shot)][str(seed)]
                    self.assertEqual(len(refs), shot)
                    self.assertFalse(set(refs) & set(entry["calibration"]))
                    expected = [f"{category}/train/good/{i:03d}.jpg" for i in range(20 + seed * shot, 20 + (seed + 1) * shot)]
                    self.assertEqual(refs, expected)
                    configs += 1
            self.assertEqual([x["label"] for x in entry["tests"]], [0, 1])
            self.assertIsNone(entry["tests"][0]["mask_path"])
        self.assertEqual(configs, 72)

    def test_repeated_generation_is_identical_and_paths_are_relative(self):
        first = MODULE.build_manifest(self.root, self.protocol)
        self.assertEqual(first, MODULE.build_manifest(self.root, self.protocol))
        self.assertNotIn(self.temp.name, json.dumps(first))

    def test_missing_mask_is_rejected(self):
        (self.root / "cashew/ground_truth/bad/defect.png").unlink()
        with self.assertRaisesRegex(ValueError, "expected one mask"):
            MODULE.build_manifest(self.root, self.protocol)

    def test_reference_capacity_is_checked_after_calibration_exclusion(self):
        (self.root / "candle/train/good/043.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "at least 44"):
            MODULE.build_manifest(self.root, self.protocol)

    def test_missing_normal_tests_are_rejected(self):
        (self.root / "candle/test/good/normal.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "both normal and abnormal"):
            MODULE.build_manifest(self.root, self.protocol)

    def test_invalid_category_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["categories"] = ["../outside"]
        with self.assertRaisesRegex(ValueError, "directory names"):
            MODULE.build_manifest(self.root, protocol)


if __name__ == "__main__":
    unittest.main()
