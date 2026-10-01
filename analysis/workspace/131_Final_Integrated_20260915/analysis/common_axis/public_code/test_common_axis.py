"""Synthetic tests: no repository data, private paths, or patient identifiers."""
import unittest
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import common_axis as ca


class CommonAxisTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        latent = rng.normal(size=31)
        self.delta = np.vstack([latent+rng.normal(size=31)*.7 for _ in range(12)])
        self.candidate = np.arange(6)
        self.regions = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
        self.controls = [np.array([6, 7]), np.array([8, 9, 10]), np.array([11])]

    def independent_residual(self, values):
        # Independent construction: raw ranks, sample-SD standardization, least squares.
        ranked = np.array([rankdata(row) for row in values])
        scores = np.array([ranked[ix].mean(axis=0) for ix in self.controls])
        scores = (scores-scores.mean(axis=1, keepdims=True))/scores.std(axis=1, ddof=1, keepdims=True)
        common = rankdata(scores.mean(axis=0))
        x = np.column_stack([np.ones(len(common)), common])
        y = ranked[self.candidate].T
        residual = y-x @ np.linalg.lstsq(x, y, rcond=None)[0]
        matrix = np.corrcoef(residual.T)
        blocks = [np.median(matrix[np.ix_(a, b)]) for i, a in enumerate(self.regions) for b in self.regions[i+1:]]
        return np.median(blocks)

    def test_residual_matches_independent_least_squares(self):
        actual = ca.residual_stat(self.delta, self.candidate, self.regions, self.controls)
        self.assertAlmostEqual(actual[1], self.independent_residual(self.delta), places=12)

    def test_bootstrap_recomputes_ranks_gene_score_and_regression(self):
        actual = ca.bootstrap_stats(self.delta, self.candidate, self.regions, self.controls, 13, 9)
        rng = np.random.default_rng(9)
        expected = []
        for _ in range(13):
            ix = rng.integers(0, self.delta.shape[1], self.delta.shape[1])
            expected.append(self.independent_residual(self.delta[:, ix]))
        np.testing.assert_allclose(actual[:, 1], expected, atol=1e-12, rtol=0)

    def test_duplicate_slot_windows_do_not_double_weight_genes_or_probes(self):
        pool = pd.DataFrame({"gene": ["ControlA", "ControlA", "ControlB"],
                             "probes": ["p1;p2", "p2;p3", "p4"]})
        lookup = {p: i for i, p in enumerate(["p1", "p2", "p3", "p4", "candidate"])}
        groups, genes, probes = ca.background_groups(pool, lookup, ["candidate"])
        self.assertEqual((genes, probes), (2, 4))
        np.testing.assert_array_equal(groups[0], [0, 1, 2])
        repeated = pd.concat([pool, pool.iloc[[0]]], ignore_index=True)
        again = ca.background_groups(repeated, lookup, ["candidate"])[0]
        for x, y in zip(groups, again):
            np.testing.assert_array_equal(x, y)

    def test_candidate_probe_and_gene_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Candidate probe"):
            ca.background_groups(pd.DataFrame({"gene": ["Control"], "probes": ["c"]}), {"c": 0}, ["c"])
        with self.assertRaisesRegex(ValueError, "Candidate gene"):
            ca.background_groups(pd.DataFrame({"gene": ["EYA4"], "probes": ["p"]}), {"p": 0}, ["c"])

    def test_pair_alignment_is_identifier_based_and_cache_mismatch_fails(self):
        matrix = pd.DataFrame({"ProbeID": ["p"], "bT": [.8], "aN": [.1], "aT": [.6], "bN": [.2]})
        metadata = pd.DataFrame({"sample_id": ["bN", "aT", "aN", "bT"], "patient_id": ["b", "a", "a", "b"],
                                 "tissue": ["N", "T", "N", "T"], "pair_verified": [True]*4})
        t, n = ca.paired_candidate_data(matrix, metadata, ["p"])
        np.testing.assert_allclose(t-n, [[.5, .6]])
        ca.validate_cache(t-n, ["p"], ["p"], t-n)
        with self.assertRaisesRegex(ValueError, "patient order"):
            ca.validate_cache((t-n)[:, ::-1], ["p"], ["p"], t-n)
        with self.assertRaisesRegex(ValueError, "Duplicate metadata"):
            ca.paired_candidate_data(matrix, pd.concat([metadata, metadata.iloc[[0]]]), ["p"])

    def test_correlated_normals_covariance_identity(self):
        rng = np.random.default_rng(81)
        normal = rng.uniform(.02, .1, (6, 31))
        tumor = normal+rng.uniform(.1, .3, (6, 31))
        _, terms = ca.tissue_decomposition("Synthetic", tumor, normal, self.regions)
        self.assertLess(max(x["identity_max_error"] for x in terms), 1e-12)

    def test_constant_vector_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Constant"):
            ca.rankunit(np.ones((2, 10)))

    def test_local_exclusion_retains_complete_pairs_and_45_tests(self):
        rng = np.random.default_rng(5)
        rows = []
        for i in range(87):
            for tissue in ["N", "T"]:
                rows.append({"study_id": "synthetic_"+str(i), "tissue": tissue,
                             **dict(zip(ca.GENES, rng.uniform(1, 80, 10)))})
        summary, tests = ca.local_qc(pd.DataFrame(rows), ["synthetic_0", "synthetic_1"])
        self.assertEqual([r["pairs"] for r in summary], [87, 85])
        self.assertEqual(len(tests), 90)
        self.assertTrue(all(0 <= r["BH_q"] <= 1 for r in tests))
        with self.assertRaisesRegex(ValueError, "two authorized"):
            ca.local_qc(pd.DataFrame(rows), ["synthetic_0", "synthetic_0"])


if __name__ == "__main__":
    unittest.main()
