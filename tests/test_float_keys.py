import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from src.evaluation.float_keys import FloatKeyAUROC, bounds


class FloatKeyTests(unittest.TestCase):
    def test_exact_ties_zero_and_normal_images(self):
        rng = np.random.default_rng(19)
        scores = rng.integers(0, 20, (17, 21)).astype(np.float32) / 20
        mask = rng.random(scores.shape) < 0.08
        mask[:3] = False
        metric = FloatKeyAUROC(*bounds(scores))
        metric.add(scores[:3], mask[:3]); metric.add(scores[3:], mask[3:])
        self.assertAlmostEqual(metric.compute(), roc_auc_score(mask.ravel(), scores.ravel()), places=13)

    def test_zero_only_and_counter_validation(self):
        metric = FloatKeyAUROC(None, None)
        metric.add(np.zeros((2, 2)), np.array([[1, 0], [0, 0]]))
        self.assertEqual(metric.compute(), 0.5)
        with self.assertRaises(ValueError):
            bounds(np.array([float("nan")]))
        with self.assertRaises(MemoryError):
            FloatKeyAUROC(0, 100, budget_gib=0)
