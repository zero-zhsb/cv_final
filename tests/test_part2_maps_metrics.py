import tempfile
from pathlib import Path
import unittest
try:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from src.part2.maps import reconstruct_map, load_anomaly_map
    from src.part2.metrics import PixelAUROC
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Part 2 numerical environment is required") from exc


class MapMetricTests(unittest.TestCase):
    def test_compact_archive_roundtrip_and_full_archive(self):
        patch = np.array([[0.0, 0.3], [0.8, 0.2]], dtype=np.float32)
        expected = reconstruct_map(patch, [19, 23])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npz"
            np.savez_compressed(path, patch_map=patch, image_shape=[19, 23])
            np.testing.assert_array_equal(load_anomaly_map(path), expected)
            np.savez_compressed(path, anomaly_map=expected)
            np.testing.assert_array_equal(load_anomaly_map(path), expected)

    def test_exact_auc_matches_sklearn_with_ties_across_images(self):
        rng = np.random.default_rng(42)
        metric = PixelAUROC()
        labels, scores = [], []
        for _ in range(5):
            score = rng.integers(-5, 8, size=(19, 23)).astype(np.float32) / 8
            mask = rng.random(score.shape) < 0.03
            metric.add(score, mask)
            labels.extend(mask.reshape(-1)); scores.extend(score.reshape(-1))
        self.assertAlmostEqual(metric.compute(), roc_auc_score(labels, scores), places=13)

    def test_normal_images_participate_in_pixel_auc(self):
        metric = PixelAUROC()
        metric.add(np.array([[0.9, 0.8]]), np.array([[0, 0]]))
        metric.add(np.array([[0.5, 0.1]]), np.array([[1, 0]]))
        self.assertAlmostEqual(metric.compute(), 1 / 3)

    def test_constant_scores_give_half_auc(self):
        metric = PixelAUROC()
        metric.add(np.ones((2, 2)), np.array([[1, 0], [0, 0]]))
        self.assertEqual(metric.compute(), 0.5)

    def test_missing_class_is_rejected(self):
        metric = PixelAUROC()
        metric.add(np.zeros((2, 2)), np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            metric.compute()


if __name__ == "__main__":
    unittest.main()
