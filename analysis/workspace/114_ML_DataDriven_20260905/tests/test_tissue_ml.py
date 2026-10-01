import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ml.tissue import (
    GENES,
    TissueConfig,
    fit_predict_outer,
    patient_bootstrap_metrics,
    patient_kfolds,
    prepare_tissue_frame,
    run_cohort,
)


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "data/ml/tissue.tsv").exists(),
                     "restricted input data/ml/tissue.tsv is not distributed; see RUN_INSTRUCTIONS.md")
class TissueMLContractTest(unittest.TestCase):
    def setUp(self):
        self.data = prepare_tissue_frame(pd.read_csv(ROOT / "data/ml/tissue.tsv", sep="\t"))
        self.fast_cfg = TissueConfig(
            outer_folds=5,
            outer_repeats=2,
            inner_folds=4,
            logistic_C=(0.01, 0.1),
            svm_C=(0.1,),
            svm_gamma=(0.01,),
            forest_trees=5,
            forest_depth=(3,),
            forest_leaf=(5,),
            bootstrap_replicates=10,
            optimism_bootstrap=3,
            permutations=3,
        )

    def test_outer_and_inner_patient_groups_do_not_overlap(self):
        folds = patient_kfolds(self.data["patient_id"].unique(), 5, 2, 20260905)
        for f in folds:
            self.assertFalse(set(f["train_patients"]) & set(f["test_patients"]))
            train_rows = self.data[self.data["patient_id"].isin(f["train_patients"])]
            test_rows = self.data[self.data["patient_id"].isin(f["test_patients"])]
            self.assertEqual(train_rows.groupby("patient_id")["y"].sum().nunique(), 1)
            self.assertEqual(test_rows.groupby("patient_id")["y"].sum().nunique(), 1)

    def test_run_cohort_outputs_grouped_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = run_cohort(self.data, tmp, models=["ridge", "best_single_gene"], repeats=2, config=self.fast_cfg, bootstrap=10)
            oof = res["oof"]
            self.assertIn("study_id", oof.columns)
            self.assertNotIn("patient_id", oof.columns)
            self.assertEqual(set(oof["model"]), {"ridge", "best_single_gene"})
            self.assertEqual(len(oof), len(self.data) * 2 * 2)
            self.assertTrue((Path(tmp) / "oof_predictions.tsv.gz").exists())
            self.assertTrue((Path(tmp) / "models" / "ridge_r00_f00.pkl.gz").exists())
            pooled = res["metrics"].query("scope == 'all_repeats_pooled'")
            self.assertEqual(len(pooled), 2)
            self.assertEqual(set(pooled["aggregation"]), {"mean_of_repeat_metrics"})
            self.assertTrue(pooled["auc"].between(0, 1).all())
            self.assertIn("ridge_minus_best_single_gene", set(res["bootstrap_ci"]["model"]))

    def test_same_outdir_perturbed_input_uses_new_cache_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_cohort(self.data, tmp, models=["ridge"], repeats=1, config=self.fast_cfg)
            first_dirs = {p.name for p in (Path(tmp) / ".cache").iterdir() if p.is_dir()}
            perturbed = self.data.copy()
            perturbed.loc[perturbed["study_id"].eq("P001"), "EYA4"] += 1.0
            run_cohort(perturbed, tmp, models=["ridge"], repeats=1, config=self.fast_cfg)
            second_dirs = {p.name for p in (Path(tmp) / ".cache").iterdir() if p.is_dir()}
            self.assertGreater(len(second_dirs), len(first_dirs))
            self.assertFalse(first_dirs == second_dirs)

    def test_heldout_perturbation_cannot_change_train_scaler_state(self):
        fold = patient_kfolds(self.data["patient_id"].unique(), 5, 1, 20260905)[0]
        _, _, state1, _ = fit_predict_outer(self.data, fold, "ridge", self.fast_cfg)
        perturbed = self.data.copy()
        mask = perturbed["patient_id"].isin(fold["test_patients"])
        perturbed.loc[mask, GENES] = perturbed.loc[mask, GENES] + 10000.0
        _, _, state2, _ = fit_predict_outer(perturbed, fold, "ridge", self.fast_cfg)
        self.assertEqual(state1["scaler_mean"], state2["scaler_mean"])
        self.assertEqual(state1["scaler_scale"], state2["scaler_scale"])

    def test_constant_prediction_metrics_are_finite_for_auc_and_thresholds(self):
        oof = self.data[["patient_id", "sample_id", "tissue", "y"]].copy()
        oof["model"] = "constant"
        oof["repeat"] = 0
        oof["fold"] = 0
        oof["score"] = 0.0
        oof["probability"] = 0.5
        oof["pred_0_5"] = 1
        oof["pred_normal_q95"] = 0
        draws, ci = patient_bootstrap_metrics(oof, 5, 20260905)
        self.assertEqual(len(draws), 5)
        auc_ci = ci[(ci["model"].eq("constant")) & (ci["metric"].eq("auc"))]
        self.assertEqual(float(auc_ci["point"].iloc[0]), 0.5)

    def test_perfect_separation_and_duplicate_patient_bootstrap_do_not_crash(self):
        mini = self.data.copy()
        mini[GENES] = mini[GENES].where(mini["y"].eq(1), 0.0)
        mini[GENES] = mini[GENES].where(mini["y"].eq(0), 100.0)
        with tempfile.TemporaryDirectory() as tmp:
            res = run_cohort(mini, tmp, models=["ridge"], repeats=1, config=self.fast_cfg, bootstrap=10, optimism=2)
        pooled = res["metrics"].query("model == 'ridge' and scope == 'all_repeats_pooled'").iloc[0]
        self.assertGreaterEqual(float(pooled["auc"]), 0.99)
        self.assertIn("nonestimable_complete_separation", str(pooled["calibration_status"]))
        self.assertTrue(np.isnan(float(pooled["calibration_slope"])))
        self.assertEqual(len(res["ridge_optimism"]), 2)


if __name__ == "__main__":
    unittest.main()
