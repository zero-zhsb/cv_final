import tempfile
from pathlib import Path
import unittest

try:
    import cv2
    import numpy as np
    from PIL import Image
    from src.part1.inspector import TemplateInspector, prepare_rgb, top_fraction_score
    from src.part2.maps import load_anomaly_map
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Install requirements/part1.txt") from exc


class TemplateTests(unittest.TestCase):
    def test_geometry_crops_right_bottom_without_stretching(self):
        image = Image.fromarray(np.tile(np.arange(101, dtype=np.uint8)[None, :, None], (71, 1, 3)))
        expected = np.asarray(image.resize((637, 448), Image.Resampling.BICUBIC))[:, :630]
        np.testing.assert_allclose(prepare_rgb(image) * 255, expected.astype(np.float32), atol=1e-5)

    def test_score_floor_minimum_and_invalid(self):
        self.assertEqual(top_fraction_score([1, 2, 3]), 3)
        values = np.zeros(250); values[-3:] = [1, 3, 5]
        self.assertEqual(top_fraction_score(values), 4)
        with self.assertRaises(ValueError):
            top_fraction_score([float("nan")])

    def test_closest_template_identical_and_local_defect(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / (str(i) + ".png") for i in range(3)]
            black = np.zeros((28, 42, 3), dtype=np.uint8)
            white = np.full_like(black, 255)
            defect = black.copy(); defect[8:12, 15:20] = 255
            for path, array in zip(paths, [white, black, defect]):
                Image.fromarray(array).save(path)
            inspector = TemplateInspector(resolution=28, sigma=0)
            inspector.fit_references(paths[:2])
            identical = inspector.predict(paths[1])
            self.assertEqual(identical["score"], 0)
            self.assertEqual(identical["nearest_reference"], str(paths[1]))
            result = inspector.predict(paths[2])
            self.assertEqual(result["nearest_reference"], str(paths[1]))
            self.assertGreater(result["score"], identical["score"])
            self.assertEqual(np.unravel_index(result["anomaly_map"].argmax(), (28, 42)), (8, 15))

    def test_compact_map_reconstruction_matches_live_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.npz"
            values = np.array([[0, 0.2], [0.7, 0.9]], dtype=np.float32)
            np.savez_compressed(path, template_map=values, image_shape=[19, 23])
            expected = cv2.resize(values, (23, 19), interpolation=cv2.INTER_LINEAR)
            np.testing.assert_array_equal(load_anomaly_map(path), expected)


if __name__ == "__main__":
    unittest.main()
