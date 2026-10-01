import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
RULES = [
    "normal_q95_higher",
    "normal_q975_higher",
    "2.5x_normal_mean",
    "3x_normal_mean",
    "normal_mean_plus_4sd",
    "tumor_normal_youden",
]


class PairedAnalysisContractTest(unittest.TestCase):
    def setUp(self):
        self.wide = pd.read_csv(ROOT / "data/derived/methylation_wide.tsv", sep="\t")
        self.paired = pd.read_csv(ROOT / "results/paired.csv")
        self.cutoffs = pd.read_csv(ROOT / "results/cutoffs.csv")

    def test_paired_mean_difference_matches_canonical_input(self):
        for gene in GENES:
            tumor = self.wide[f"T_{gene}"].to_numpy(float)
            normal = self.wide[f"N_{gene}"].to_numpy(float)
            observed = self.paired.loc[self.paired["gene"].eq(gene), "mean_difference_pp"].iloc[0]
            self.assertAlmostEqual(observed, float(np.mean(tumor - normal)), places=12)

    def test_q95_cutoffs_and_calls_are_recomputed_from_normal_values(self):
        q95 = self.cutoffs[self.cutoffs["rule"].eq("normal_q95_higher")].set_index("gene")
        expected_positive = {
            "EYA4": 60,
            "ZNF568": 69,
            "ZNF793": 55,
            "SFMBT2": 67,
            "ADHFE1": 81,
            "HOXA2": 59,
            "BEND5": 54,
            "UNC5C": 75,
            "RALYL": 69,
            "GFRA1": 72,
        }
        for gene, expected_n in expected_positive.items():
            normal = self.wide[f"N_{gene}"].to_numpy(float)
            tumor = self.wide[f"T_{gene}"].to_numpy(float)
            cutoff = float(np.quantile(normal, 0.95, method="higher"))
            self.assertAlmostEqual(float(q95.loc[gene, "cutoff_pct"]), cutoff, places=12)
            self.assertEqual(int(q95.loc[gene, "tumor_positive_n"]), expected_n)
            self.assertEqual(int(np.sum(tumor > cutoff)), expected_n)

    def test_output_shapes_match_shared_contract(self):
        expected_shapes = {
            "paired.csv": 10,
            "paired_loo.csv": 870,
            "correlations.csv": 45,
            "pca_scores.csv": 87,
            "pca_loadings.csv": 10,
            "cutoffs.csv": 60,
            "cutoff_bootstrap.csv": 60,
            "cutoff_loo.csv": 5220,
            "cutoff_cv.csv": 6000,
        }
        for filename, expected_rows in expected_shapes.items():
            self.assertEqual(len(pd.read_csv(ROOT / "results" / filename)), expected_rows, filename)
        calls = pd.read_csv(ROOT / "results/cutoff_patient_calls.tsv", sep="\t")
        self.assertEqual(len(calls), 87)
        self.assertEqual(sorted(self.cutoffs["rule"].unique()), sorted(RULES))
        summary = json.loads((ROOT / "results/paired_summary.json").read_text())
        self.assertEqual(summary["bootstrap_replicates"], 5000)


if __name__ == "__main__":
    unittest.main()
