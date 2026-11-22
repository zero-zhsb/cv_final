import unittest

import numpy as np

from src.part3.multiscale import align_coarse_features, resize_shape


class MultiscaleAlignmentTests(unittest.TestCase):
    def test_uses_original_coordinates_where_right_edge_crops_differ(self):
        # The coarse resize loses 11 pixels at the right, the fine resize loses 2.
        coarse_shape = resize_shape(100, 140, 28)
        fine_shape = resize_shape(100, 140, 42)
        self.assertEqual((coarse_shape, fine_shape), ((28, 39), (42, 58)))
        coarse_grid = (2, 2)
        fine_grid = (3, 4)
        source = np.array([[0], [1], [10], [11]], dtype=np.float32)
        actual = align_coarse_features(source, coarse_grid, fine_grid, coarse_shape, fine_shape, 14).reshape(3, 4)
        x = np.clip((np.arange(4) + .5) * 39 / 58 - .5, 0, 1)
        y = np.array([0, 5, 10])
        np.testing.assert_allclose(actual, y[:, None] + x, atol=1e-6)
        self.assertNotAlmostEqual(float(actual[1, 2]), 5.75)


if __name__ == "__main__":
    unittest.main()
