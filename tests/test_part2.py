"""Run in the Part 2 environment; covers official scoring and nearest-neighbor math."""
import unittest
try:
    import numpy as np
    import faiss
    from src.part2.inspector import image_score
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Run Part 2 tests inside its requirements/part2.txt environment") from exc


class Part2Tests(unittest.TestCase):
    def test_fewer_than_100_patches_uses_maximum(self):
        self.assertEqual(image_score(np.array([0.0, 0.5, 2.0])), 2.0)

    def test_top_one_percent_uses_floor_and_includes_zero_background(self):
        values = np.zeros(250)
        values[-3:] = [1, 3, 5]
        self.assertEqual(image_score(values), 4.0)

    def test_nonfinite_or_empty_distances_are_rejected(self):
        for values in [np.array([]), np.array([np.nan]), np.array([np.inf])]:
            with self.assertRaises(ValueError):
                image_score(values)

    def test_normalized_faiss_squared_distance_over_two_is_cosine_distance(self):
        bank = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        query = np.array([[1.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        faiss.normalize_L2(bank)
        faiss.normalize_L2(query)
        index = faiss.IndexFlatL2(2)
        index.add(bank)
        distance, _ = index.search(query, 1)
        expected = (1.0 - query @ bank.T).min(axis=1)
        np.testing.assert_allclose(distance[:, 0] / 2, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
