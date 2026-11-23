"""Numerical checks can run without CUDA, torch, or FAISS."""
import ast
import csv
import itertools
import json
import statistics
from pathlib import Path
import unittest
import numpy as np
from sklearn.covariance import OAS

# Load only pure math; model integration requires the documented server environment.
source = Path(__file__).resolve().parents[1] / 'src/part3/oas.py'
tree = ast.parse(source.read_text())
namespace = {'np': np, 'OAS': OAS}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), str(source), 'exec'), namespace)
fit_metric, apply_metric = namespace['fit_oas_metric'], namespace['apply_metric']


class OASTests(unittest.TestCase):
    def test_distance_equals_shrunk_mahalanobis(self):
        x = np.random.default_rng(8).normal(size=(40, 8)) * np.arange(1, 9)
        mean, transform, _ = fit_metric(x)
        z = apply_metric(x, mean, transform)
        covariance = OAS(store_precision=False).fit(x).covariance_
        delta = x[0] - x[1]
        self.assertAlmostEqual(float(((z[0] - z[1])**2).sum()), float(delta @ np.linalg.solve(covariance, delta) / 8), places=5)

    def test_query_transform_does_not_refit_or_mutate_statistics(self):
        x = np.random.default_rng(2).normal(size=(20, 8))
        mean, transform, shrinkage = fit_metric(x)
        before = (mean.copy(), transform.copy())
        apply_metric(x * 50, mean, transform)
        np.testing.assert_array_equal(mean, before[0])
        np.testing.assert_array_equal(transform, before[1])
        self.assertTrue(0 <= shrinkage <= 1)

    def test_rank_deficient_reference_bank_remains_finite(self):
        x = np.random.default_rng(3).normal(size=(4, 12))
        mean, transform, _ = fit_metric(x)
        self.assertTrue(np.isfinite(apply_metric(x, mean, transform)).all())

    def test_degenerate_bank_fails(self):
        with self.assertRaises(ValueError):
            fit_metric(np.ones((4, 8)))

    def test_complete_results_reproduce_macro_mean_and_population_sd(self):
        folder=source.parents[2]/'results/tables/multiscale_oas'
        with (folder/'full_per_configuration.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        with (folder/'full_overall_by_shot.csv').open(newline='') as f:
            overall=list(csv.DictReader(f))
        categories=['candle','capsules','cashew','chewinggum','fryum','pipe_fryum']
        expected=set(itertools.product(categories,[1,2,4,8],[0,1,2]))
        self.assertEqual(len(rows),72)
        self.assertEqual({(r['category'],int(r['shot']),int(r['seed'])) for r in rows},expected)
        self.assertTrue(json.loads((folder/'full_audit.json').read_text())['passed'])
        for row in overall:
            for metric in ['image_auroc','pixel_auroc']:
                for suffix in ['', '_baseline']:
                    key=metric+suffix
                    means=[statistics.mean(float(r[key]) for r in rows
                        if r['shot']==row['shot'] and int(r['seed'])==seed) for seed in range(3)]
                    self.assertAlmostEqual(statistics.mean(means),float(row[key+'_mean']),places=12)
                    self.assertAlmostEqual(statistics.pstdev(means),float(row[key+'_std_ddof0']),places=12)


if __name__ == '__main__':
    unittest.main()
