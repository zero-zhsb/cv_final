import os
import unittest


@unittest.skipUnless(os.environ.get("CV_TEST_CUDA") == "1", "Explicit CV_TEST_CUDA=1 required for GPU tests")
class GPUAUROCTests(unittest.TestCase):
    def test_gpu_counts_match_sklearn_with_zero_ties(self):
        import numpy as np
        from sklearn.metrics import roc_auc_score
        from src.part2.gpu_metrics import gpu_pixel_auc
        rng = np.random.default_rng(9)
        pairs = []
        for _ in range(7):
            values = (np.float32(0.5) + rng.integers(0, 32, (13, 17)).astype(np.float32) * np.float32(2**-24))
            values[rng.random(values.shape) < 0.4] = 0
            mask = rng.random(values.shape) < 0.05
            pairs.append((values, mask))
        expected = roc_auc_score(np.concatenate([p[1].reshape(-1) for p in pairs]), np.concatenate([p[0].reshape(-1) for p in pairs]))
        actual, _ = gpu_pixel_auc(lambda: iter(pairs))
        self.assertAlmostEqual(actual, expected, places=13)

    def test_all_zero_scores_have_half_auc(self):
        import numpy as np
        from src.part2.gpu_metrics import gpu_pixel_auc
        pairs = [(np.zeros((2, 2), dtype=np.float32), np.array([[1, 0], [0, 0]], dtype=bool))]
        actual, _ = gpu_pixel_auc(lambda: iter(pairs))
        self.assertEqual(actual, 0.5)


if __name__ == "__main__":
    unittest.main()
