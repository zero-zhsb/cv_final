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

    def test_exhaustive_gpu_neighbors_match_cpu_faiss(self):
        import numpy as np
        import faiss
        from src.part2.gpu_index import ExactGPUIndex
        rng = np.random.default_rng(13)
        bank = rng.normal(size=(997, 384)).astype(np.float32)
        query = rng.normal(size=(37, 384)).astype(np.float32)
        faiss.normalize_L2(bank); faiss.normalize_L2(query)
        cpu = faiss.IndexFlatL2(384); cpu.add(bank)
        expected_distance, expected_match = cpu.search(query, 1)
        actual_distance, actual_match = ExactGPUIndex(bank, chunk_size=127).search(query)
        np.testing.assert_allclose(actual_distance, expected_distance, atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(actual_match, expected_match)
        distance, match = ExactGPUIndex(bank).search(np.empty((0, 384), dtype=np.float32))
        self.assertEqual(distance.shape, (0, 1))
        self.assertEqual(match.shape, (0, 1))


if __name__ == "__main__":
    unittest.main()
