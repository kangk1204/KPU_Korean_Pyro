"""Independent ML contract checks and hand-derived metric tests."""
from __future__ import annotations

import json
import ast
import gzip
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_reference import (
    GENES,
    assert_group_disjoint,
    bootstrap_cluster_rows,
    breslow_baseline_hazard,
    censoring_km,
    ibs_from_saved_grid,
    ipcw_brier_at_time,
    ipcw_brier_from_saved_g,
    km_before,
    read_optional_tsv,
    survival_probability,
    summarize_tissue_metrics_from_oof,
    uno_c_from_saved_g_by_fold,
    uno_c_at_tau,
    weighted_auc_from_saved_g,
    weighted_auc_at_time,
)


ROOT = Path(__file__).resolve().parents[1]


def _read_table(path):
    if path.suffix == ".gz":
        return pd.read_csv(path, sep="\t", compression="gzip", keep_default_na="permutation" not in path.name)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t", keep_default_na="permutation" not in path.name)
    return pd.read_csv(path, keep_default_na="permutation" not in path.name)


def _canonical_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _skip_if_stale_smoke(path, target_repeats):
    frame = _read_table(path)
    if "repeat" in frame and frame["repeat"].nunique() < target_repeats:
        pytest.skip(f"{path.relative_to(ROOT)} is a stale smoke/quick output")
    return frame


def _assert_metric_tables_close(observed, expected, key_cols, metric_cols):
    obs = observed.copy()
    exp = expected.copy()
    for col in key_cols:
        if col in obs:
            obs[col] = obs[col].astype(str)
        if col in exp:
            exp[col] = exp[col].astype(str)
    merged = obs.merge(exp, on=key_cols, suffixes=("_observed", "_expected"), validate="one_to_one")
    assert len(merged) == len(expected)
    for col in metric_cols:
        if col in observed.columns and col in expected.columns:
            np.testing.assert_allclose(
                merged[f"{col}_observed"].to_numpy(float),
                merged[f"{col}_expected"].to_numpy(float),
                rtol=0,
                atol=1e-12,
                equal_nan=True,
            )


def _load_pickle_model(path):
    with gzip.GzipFile(fileobj=path.open("rb"), mode="rb") as gz:
        return pickle.loads(gz.read())


def _extract_ridge_rds_state(path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
state <- readRDS(args[[1]])
out <- args[[2]]
write.csv(
  data.frame(
    feature=names(state$fit$coef),
    coef=as.numeric(state$fit$coef),
    center=as.numeric(state$fit$center),
    scale=as.numeric(state$fit$scale)
  ),
  file=file.path(out, "params.csv"),
  row.names=FALSE
)
write.csv(state$fit$baseline_hazard, file=file.path(out, "baseline.csv"), row.names=FALSE)
writeLines(as.character(state$fit$eta_center), file.path(out, "eta_center.txt"))
'''
        run = subprocess.run(
            ["Rscript", "-e", r_code, str(path), str(tmp_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if run.returncode != 0:
            pytest.skip(f"could not read recurrence RDS state with Rscript: {run.stderr[-500:]}")
        return {
            "params": pd.read_csv(tmp_path / "params.csv"),
            "baseline": pd.read_csv(tmp_path / "baseline.csv"),
            "eta_center": float((tmp_path / "eta_center.txt").read_text().strip()),
        }


def _risk_from_baseline(baseline, horizon, linear_predictor, eta_center):
    eligible = baseline.loc[baseline["time"].le(horizon), "hazard"]
    hazard = 0.0 if eligible.empty else float(eligible.iloc[-1])
    return 1.0 - np.exp(-hazard * np.exp(np.asarray(linear_predictor, dtype=float) - eta_center))


def _canonical_recurrence_state_path(filename):
    matches = sorted((ROOT / "results/ml/recurrence/models").glob(f"*/{filename}"))
    return matches[0] if matches else None


def test_ml_input_manifest_matches_source_table_counts():
    manifest = json.loads((ROOT / "registry/ml_input_manifest.json").read_text())
    recurrence = pd.read_csv(ROOT / "data/ml/recurrence.tsv", sep="\t")
    tissue = pd.read_csv(ROOT / "data/ml/tissue.tsv", sep="\t")

    assert manifest["n_patients"] == recurrence["study_id"].nunique() == 87
    assert manifest["n_specimens"] == len(tissue) == 174
    assert manifest["n_primary"] == int(recurrence["recurrence_primary"].sum()) == 82
    assert manifest["primary_events"] == int(recurrence.loc[recurrence["recurrence_primary"].eq(1), "event"].sum()) == 14
    assert manifest["all_events"] == int(recurrence["event"].sum()) == 17
    assert manifest["cea_elevated_n"] == int(recurrence["cea_binary"].sum()) == 18
    assert manifest["genes"] == GENES


def test_tissue_table_keeps_each_patient_pair_in_one_group():
    tissue = pd.read_csv(ROOT / "data/ml/tissue.tsv", sep="\t")
    group_summary = tissue.groupby("study_id")["y"].agg(["size", "sum"])

    assert group_summary["size"].eq(2).all()
    assert group_summary["sum"].eq(1).all()
    assert set(tissue["tissue"]) == {"N", "T"}


def test_recurrence_ml_table_excludes_original_identifiers_and_dates():
    recurrence = pd.read_csv(ROOT / "data/ml/recurrence.tsv", sep="\t", nrows=1)
    forbidden = {"patient_id", "operation_date", "endpoint_date", "duration_months"}

    assert forbidden.isdisjoint(set(recurrence.columns))


def test_breslow_baseline_hazard_handles_tied_events_with_training_centering():
    times = np.array([2.0, 2.0, 3.0, 4.0])
    events = np.array([1, 1, 0, 1])
    lp = np.log(np.array([1.0, 2.0, 3.0, 4.0]))
    center = float(np.mean(lp))
    baseline = breslow_baseline_hazard(times, events, lp, center=center)
    risk = np.exp(lp - center)
    expected_at_2 = 2.0 / risk.sum()
    expected_at_4 = expected_at_2 + 1.0 / risk[3]

    np.testing.assert_allclose(baseline.times, [2.0, 4.0])
    np.testing.assert_allclose(baseline.hazard, [expected_at_2, expected_at_4])


def test_cox_survival_probability_uses_training_center_and_step_hazard():
    baseline = breslow_baseline_hazard([2, 4], [1, 1], [0.0, np.log(2.0)], center=0.0)
    survival = survival_probability(baseline, time=3.0, linear_predictor=[np.log(2.0)], center=0.0)

    np.testing.assert_allclose(survival, [np.exp(-(1.0 / 3.0) * 2.0)])


def test_censoring_km_keeps_left_limit_before_tied_censoring_drop():
    km = censoring_km(times=[1.0, 2.0, 2.0, 4.0], events=[1, 0, 1, 0])

    assert km["before"][2.0] == 1.0
    assert km["after"][2.0] == pytest.approx(2.0 / 3.0)


def test_censoring_km_between_time_left_limit_uses_previous_after_value():
    km = censoring_km(times=[1.0, 2.0, 4.0], events=[1, 0, 0])

    assert km_before(km, 3.0) == pytest.approx(0.5)


def test_uno_c_counts_tied_risk_as_half_concordant():
    c_index, concordant, comparable = uno_c_at_tau(
        times=[1.0, 2.0, 3.0],
        events=[1, 1, 0],
        risk=[0.5, 0.5, 0.1],
        tau=3.0,
        km={"before": {1.0: 1.0, 2.0: 1.0, 3.0: 1.0}, "after": {1.0: 1.0, 2.0: 1.0, 3.0: 1.0}},
    )

    assert comparable == 3.0
    assert concordant == 2.5
    assert c_index == pytest.approx(2.5 / 3.0)


def test_time_auc_returns_half_for_constant_risk_predictions():
    auc = weighted_auc_at_time(
        times=[1.0, 2.0, 5.0, 6.0],
        events=[1, 1, 0, 0],
        risk=[0.4, 0.4, 0.4, 0.4],
        horizon=3.0,
        km={"before": {1.0: 1.0, 2.0: 1.0, 5.0: 1.0, 6.0: 1.0}, "after": {1.0: 1.0, 2.0: 1.0, 5.0: 1.0, 6.0: 1.0}},
    )

    assert auc == pytest.approx(0.5)


def test_ipcw_brier_ignores_early_censored_observations():
    brier = ipcw_brier_at_time(
        times=[1.0, 2.0, 5.0],
        events=[0, 1, 0],
        event_probability=[0.99, 0.25, 0.25],
        horizon=3.0,
        km={"before": {1.0: 1.0, 2.0: 1.0, 5.0: 1.0}, "after": {1.0: 1.0, 2.0: 1.0, 5.0: 1.0}},
    )

    assert brier == pytest.approx(((1.0 - 0.25) ** 2 + 0.25**2) / 3.0)


def test_ipcw_brier_is_normalized_by_original_evaluation_n():
    brier = ipcw_brier_at_time(
        times=[1.0, 2.0, 5.0, 6.0],
        events=[0, 1, 0, 0],
        event_probability=[0.0, 0.2, 0.2, 0.2],
        horizon=3.0,
        km={"before": {1.0: 1.0, 2.0: 1.0, 5.0: 1.0, 6.0: 1.0}, "after": {1.0: 1.0, 2.0: 1.0, 5.0: 1.0, 6.0: 1.0}},
    )

    assert brier == pytest.approx(((1.0 - 0.2) ** 2 + 0.2**2 + 0.2**2) / 4.0)


def test_patient_cluster_bootstrap_duplicates_whole_tissue_pairs():
    tissue = pd.DataFrame(
        {
            "study_id": ["P001", "P001", "P002", "P002"],
            "specimen_id": ["P001_N", "P001_T", "P002_N", "P002_T"],
            "y": [0, 1, 0, 1],
        }
    )
    sampled = bootstrap_cluster_rows(tissue, "study_id", ["P001", "P001", "P002"])

    assert sampled.groupby("bootstrap_draw")["specimen_id"].apply(list).tolist() == [
        ["P001_N", "P001_T"],
        ["P001_N", "P001_T"],
        ["P002_N", "P002_T"],
    ]


def test_group_disjointness_detector_rejects_nested_fold_patient_overlap():
    with pytest.raises(AssertionError, match="P002"):
        assert_group_disjoint(train_groups=["P001", "P002"], test_groups=["P002", "P003"])


def test_saved_oof_predictions_have_unique_repeat_fold_rows_when_available():
    candidates = [
        ROOT / "results/ml/recurrence/primary_oof_predictions.csv",
        ROOT / "results/ml/recurrence/sensitivity_all87_stage_advanced_oof_predictions.csv",
        ROOT / "results/ml/tissue/oof_predictions.tsv.gz",
        ROOT / "results/ml/public/colonomics/oof_predictions.tsv.gz",
        ROOT / "results/ml/public/gse119526/oof_predictions.tsv.gz",
    ]
    existing = [_read_table(path) for path in candidates if path.exists()]
    if not existing:
        pytest.skip("OOF prediction files are not available yet")
    for frame in existing:
        key = [col for col in ["analysis", "task", "cohort", "model", "block", "input_block", "repeat", "fold", "study_id", "sample_id", "specimen_id", "horizon"] if col in frame.columns]
        assert key
        assert not frame.duplicated(key).any()


def test_recurrence_oof_predictions_include_fixed_horizon_risk_and_censoring_weights():
    path = _canonical_existing([
        ROOT / "results/ml/recurrence/primary_oof_predictions.csv",
        ROOT / "results/ml/recurrence/_smoke/primary_oof_predictions.csv",
    ])
    if path is None:
        pytest.skip("recurrence OOF predictions are not available yet")
    oof = _skip_if_stale_smoke(path, target_repeats=25)
    required = {"risk_1095", "risk_1825", "G_1095", "G_1825", "G_time"}

    assert required.issubset(set(oof.columns))


def test_recurrence_saved_uno_matches_fold_weighted_reference_when_oof_available():
    oof_path = _canonical_existing([
        ROOT / "results/ml/recurrence/primary_oof_predictions.csv",
        ROOT / "results/ml/recurrence/_smoke/primary_oof_predictions.csv",
    ])
    metric_path = _canonical_existing([
        ROOT / "results/ml/recurrence/primary_repeat_metrics.csv",
        ROOT / "results/ml/recurrence/_smoke/primary_repeat_metrics.csv",
    ])
    if oof_path is None or metric_path is None:
        pytest.skip("recurrence OOF and repeat metrics are not available yet")
    oof = _skip_if_stale_smoke(oof_path, target_repeats=25)
    metrics = _read_table(metric_path)
    required = {"repeat", "fold", "block", "model", "duration_days", "event", "risk_score", "G_time"}
    if not required.issubset(oof.columns):
        pytest.fail(f"recurrence OOF missing required columns: {sorted(required - set(oof.columns))}")
    row = metrics.query("block == 'combined' and model == 'ridge' and horizon_days == 1825").iloc[0]
    repeat = int(row["repeat"])
    z = oof.query("repeat == @repeat and block == 'combined' and model == 'ridge'")
    c_index, numerator, denominator = uno_c_from_saved_g_by_fold(z, 1825)

    assert row["uno_c"] == pytest.approx(c_index)
    assert row["uno_num"] == pytest.approx(numerator)
    assert row["uno_den"] == pytest.approx(denominator)


def test_recurrence_saved_auc_matches_ipcw_reference_when_oof_available():
    oof_path = _canonical_existing([
        ROOT / "results/ml/recurrence/primary_oof_predictions.csv",
        ROOT / "results/ml/recurrence/_smoke/primary_oof_predictions.csv",
    ])
    metric_path = _canonical_existing([
        ROOT / "results/ml/recurrence/primary_repeat_metrics.csv",
        ROOT / "results/ml/recurrence/_smoke/primary_repeat_metrics.csv",
    ])
    if oof_path is None or metric_path is None:
        pytest.skip("recurrence OOF and repeat metrics are not available yet")
    oof = _skip_if_stale_smoke(oof_path, target_repeats=25)
    required = {"risk_1825", "G_time", "G_1825"}
    if not required.issubset(oof.columns):
        pytest.fail(f"recurrence OOF missing required columns: {sorted(required - set(oof.columns))}")
    metrics = _read_table(metric_path)
    row = metrics.query("block == 'combined' and model == 'ridge' and horizon_days == 1825").iloc[0]
    repeat = int(row["repeat"])
    z = oof.query("repeat == @repeat and block == 'combined' and model == 'ridge'")
    auc = weighted_auc_from_saved_g(z, 1825, score_col="risk_1825")

    assert row["auc"] == pytest.approx(auc)


def test_recurrence_canonical_primary_repeat_metrics_match_independent_reference_values():
    oof_path = ROOT / "results/ml/recurrence/primary_oof_predictions.csv"
    metric_path = ROOT / "results/ml/recurrence/primary_repeat_metrics.csv"
    if not oof_path.exists() or not metric_path.exists():
        pytest.skip("canonical recurrence primary OOF and repeat metrics are not available yet")
    oof = _read_table(oof_path)
    metrics = _read_table(metric_path)
    if oof["repeat"].nunique() < 25:
        pytest.skip("canonical recurrence primary OOF is not at 25 repeats yet")
    assert len(metrics) == 25 * 3 * 3 * 2
    grouped = {key: frame for key, frame in oof.groupby(["repeat", "block", "model"], sort=False)}
    for _, row in metrics.iterrows():
        key = (int(row["repeat"]), row["block"], row["model"])
        z = grouped[key]
        risk_col = f"risk_{int(row.horizon_days)}"
        c_index, numerator, denominator = uno_c_from_saved_g_by_fold(z, row.horizon_days)
        auc = weighted_auc_from_saved_g(z, row.horizon_days, score_col=risk_col)
        brier = ipcw_brier_from_saved_g(z, row.horizon_days, risk_col=risk_col)
        ibs = ibs_from_saved_grid(z)
        assert row["uno_c"] == pytest.approx(c_index)
        assert row["uno_num"] == pytest.approx(numerator)
        assert row["uno_den"] == pytest.approx(denominator)
        assert row["auc"] == pytest.approx(auc)
        assert row["brier"] == pytest.approx(brier)
        assert row["ibs"] == pytest.approx(ibs)


def test_recurrence_canonical_ridge_rds_state_reconstructs_oof_predictions():
    state_path = _canonical_recurrence_state_path("primary_repeat001_fold01_combined_ridge.rds")
    oof_path = ROOT / "results/ml/recurrence/primary_oof_predictions.csv"
    data_path = ROOT / "data/ml/recurrence.tsv"
    if state_path is None or not all(path.exists() for path in [oof_path, data_path]):
        pytest.skip("canonical recurrence RDS state and OOF are not available yet")
    state = _extract_ridge_rds_state(state_path)
    params = state["params"]
    features = params["feature"].tolist()
    data = _read_table(data_path)
    oof = _read_table(oof_path).query("repeat == 1 and fold == 1 and block == 'combined' and model == 'ridge'")
    test = data.set_index("study_id").loc[oof["study_id"].tolist()].reset_index()
    x = test[features].to_numpy(float)
    linear_predictor = ((x - params["center"].to_numpy(float)) / params["scale"].to_numpy(float)).dot(params["coef"].to_numpy(float))

    np.testing.assert_allclose(oof["risk_score"].to_numpy(float), linear_predictor, rtol=0, atol=1e-12)
    for horizon in [1095, 1825]:
        expected = _risk_from_baseline(state["baseline"], horizon, linear_predictor, state["eta_center"])
        np.testing.assert_allclose(oof[f"risk_{horizon}"].to_numpy(float), expected, rtol=0, atol=1e-12)
    for index, horizon in enumerate(np.linspace(0.0, 1825.0, 101), start=1):
        expected = _risk_from_baseline(state["baseline"], horizon, linear_predictor, state["eta_center"])
        np.testing.assert_allclose(oof[f"risk_grid_{index}"].to_numpy(float), expected, rtol=0, atol=1e-12)


def test_recurrence_canonical_training_censoring_km_reconstructs_saved_g_for_fold():
    oof_path = ROOT / "results/ml/recurrence/primary_oof_predictions.csv"
    fold_path = ROOT / "results/ml/recurrence/primary_folds.csv"
    data_path = ROOT / "data/ml/recurrence.tsv"
    if not all(path.exists() for path in [oof_path, fold_path, data_path]):
        pytest.skip("canonical recurrence OOF and folds are not available yet")
    data = _read_table(data_path).query("recurrence_primary == 1")
    folds = _read_table(fold_path)
    selected = folds.query("repeat == 1 and fold == 1")
    train = data.loc[~data["study_id"].isin(selected["study_id"])]
    km = censoring_km(train["duration_days"], train["event"])
    oof = _read_table(oof_path).query("repeat == 1 and fold == 1 and block == 'combined' and model == 'ridge'")

    np.testing.assert_allclose(oof["G_time"].to_numpy(float), [km_before(km, t) for t in oof["duration_days"]], rtol=0, atol=1e-12)
    assert float(oof["G_1095"].iloc[0]) == pytest.approx(km_before(km, 1095.0))
    assert float(oof["G_1825"].iloc[0]) == pytest.approx(km_before(km, 1825.0))
    for index, horizon in enumerate(np.linspace(0.0, 1825.0, 101), start=1):
        assert float(oof[f"G_grid_{index}"].iloc[0]) == pytest.approx(km_before(km, horizon))


def test_recurrence_ridge_rds_state_reconstructs_smoke_oof_predictions():
    state_path = ROOT / "results/ml/recurrence/_smoke/models/primary_repeat001_fold01_combined_ridge.rds"
    oof_path = ROOT / "results/ml/recurrence/_smoke/primary_oof_predictions.csv"
    data_path = ROOT / "data/ml/recurrence.tsv"
    if not all(path.exists() for path in [state_path, oof_path, data_path]):
        pytest.skip("recurrence smoke RDS state and OOF are not available yet")
    state = _extract_ridge_rds_state(state_path)
    params = state["params"]
    features = params["feature"].tolist()
    data = _read_table(data_path)
    oof = _read_table(oof_path).query("repeat == 1 and fold == 1 and block == 'combined' and model == 'ridge'")
    test = data.set_index("study_id").loc[oof["study_id"].tolist()].reset_index()
    x = test[features].to_numpy(float)
    center = params["center"].to_numpy(float)
    scale = params["scale"].to_numpy(float)
    coef = params["coef"].to_numpy(float)
    linear_predictor = ((x - center) / scale).dot(coef)

    np.testing.assert_allclose(oof["risk_score"].to_numpy(float), linear_predictor, rtol=0, atol=1e-12)
    for horizon in [1095, 1825]:
        expected = _risk_from_baseline(state["baseline"], horizon, linear_predictor, state["eta_center"])
        np.testing.assert_allclose(oof[f"risk_{horizon}"].to_numpy(float), expected, rtol=0, atol=1e-12)
    grid = np.linspace(0.0, 1825.0, 101)
    for index, horizon in enumerate(grid, start=1):
        expected = _risk_from_baseline(state["baseline"], horizon, linear_predictor, state["eta_center"])
        np.testing.assert_allclose(oof[f"risk_grid_{index}"].to_numpy(float), expected, rtol=0, atol=1e-12)


def test_recurrence_training_censoring_km_reconstructs_saved_g_for_smoke_fold():
    oof_path = ROOT / "results/ml/recurrence/_smoke/primary_oof_predictions.csv"
    fold_path = ROOT / "results/ml/recurrence/_smoke/primary_folds.csv"
    data_path = ROOT / "data/ml/recurrence.tsv"
    if not all(path.exists() for path in [oof_path, fold_path, data_path]):
        pytest.skip("recurrence smoke OOF and folds are not available yet")
    data = _read_table(data_path).query("recurrence_primary == 1")
    folds = _read_table(fold_path)
    selected = folds.query("repeat == 1 and fold == 1")
    train = data.loc[~data["study_id"].isin(selected["study_id"])]
    km = censoring_km(train["duration_days"], train["event"])
    oof = _read_table(oof_path).query("repeat == 1 and fold == 1 and block == 'combined' and model == 'ridge'")

    np.testing.assert_allclose(oof["G_time"].to_numpy(float), [km_before(km, t) for t in oof["duration_days"]], rtol=0, atol=1e-12)
    assert oof["G_1095"].nunique() == 1
    assert oof["G_1825"].nunique() == 1
    assert float(oof["G_1095"].iloc[0]) == pytest.approx(km_before(km, 1095.0))
    assert float(oof["G_1825"].iloc[0]) == pytest.approx(km_before(km, 1825.0))
    grid = np.linspace(0.0, 1825.0, 101)
    for index, horizon in enumerate(grid, start=1):
        assert float(oof[f"G_grid_{index}"].iloc[0]) == pytest.approx(km_before(km, horizon))


def test_recurrence_canonical_sensitivity_all87_stage_advanced_metrics_match_independent_reference_values():
    oof_path = ROOT / "results/ml/recurrence/sensitivity_all87_stage_advanced_oof_predictions.csv"
    metric_path = ROOT / "results/ml/recurrence/sensitivity_all87_stage_advanced_repeat_metrics.csv"
    if not oof_path.exists() or not metric_path.exists():
        pytest.skip("canonical all87_stage_advanced sensitivity OOF and repeat metrics are not available yet")
    oof = _read_table(oof_path)
    metrics = _read_table(metric_path)
    if oof["repeat"].nunique() < 25:
        pytest.skip("canonical all87_stage_advanced sensitivity is not at 25 repeats yet")
    assert len(metrics) == 25 * 1 * 1 * 2
    assert set(metrics["block"]) == {"all87_stage_advanced"}
    assert set(metrics["model"]) == {"ridge"}
    grouped = {key: frame for key, frame in oof.groupby(["repeat", "block", "model"], sort=False)}
    for _, row in metrics.iterrows():
        key = (int(row["repeat"]), row["block"], row["model"])
        z = grouped[key]
        risk_col = f"risk_{int(row.horizon_days)}"
        c_index, numerator, denominator = uno_c_from_saved_g_by_fold(z, row.horizon_days)
        auc = weighted_auc_from_saved_g(z, row.horizon_days, score_col=risk_col)
        brier = ipcw_brier_from_saved_g(z, row.horizon_days, risk_col=risk_col)
        ibs = ibs_from_saved_grid(z)
        assert row["uno_c"] == pytest.approx(c_index)
        assert row["uno_num"] == pytest.approx(numerator)
        assert row["uno_den"] == pytest.approx(denominator)
        assert row["auc"] == pytest.approx(auc)
        assert row["brier"] == pytest.approx(brier)
        assert row["ibs"] == pytest.approx(ibs)


def test_recurrence_canonical_sensitivity_summary_has_seven_analyses_and_fourteen_horizon_rows():
    path = ROOT / "results/ml/recurrence/ridge_sensitivity_metrics.csv"
    if not path.exists():
        pytest.skip("canonical ridge sensitivity summary is not available yet")
    metrics = _read_table(path)
    assert len(metrics) == 14
    assert metrics["sensitivity"].nunique() == 7
    assert set(metrics["model"]) == {"ridge"}
    assert set(metrics["horizon_days"]) == {1095, 1825}
    assert metrics["n_repeats"].eq(25).all()
    assert not metrics.duplicated(["sensitivity", "horizon_days"]).any()


def test_recurrence_all_stage_smoke_repeat_metrics_match_independent_reference_values():
    oof_path = ROOT / "results/ml/recurrence/_smoke/all87_ridge_oof_predictions.csv"
    metric_path = ROOT / "results/ml/recurrence/_smoke/all87_ridge_repeat_metrics.csv"
    if not oof_path.exists() or not metric_path.exists():
        pytest.skip("all-stage recurrence smoke OOF and repeat metrics are not available yet")
    oof = _read_table(oof_path)
    metrics = _read_table(metric_path)
    assert len(metrics) == 12
    assert set(metrics["model"]) == {"ridge"}
    for _, row in metrics.iterrows():
        repeat = int(row["repeat"])
        block = row["block"]
        z = oof.query("repeat == @repeat and block == @block and model == 'ridge'")
        assert not z.empty
        risk_col = f"risk_{int(row.horizon_days)}"
        c_index, numerator, denominator = uno_c_from_saved_g_by_fold(z, row.horizon_days)
        auc = weighted_auc_from_saved_g(z, row.horizon_days, score_col=risk_col)
        brier = ipcw_brier_from_saved_g(z, row.horizon_days, risk_col=risk_col)
        ibs = ibs_from_saved_grid(z)
        assert row["uno_c"] == pytest.approx(c_index)
        assert row["uno_num"] == pytest.approx(numerator)
        assert row["uno_den"] == pytest.approx(denominator)
        assert row["auc"] == pytest.approx(auc)
        assert row["brier"] == pytest.approx(brier)
        assert row["ibs"] == pytest.approx(ibs)


def _assert_recurrence_bootstrap_ci_estimates(ci_path, metrics_path, draws_path):
    ci = _read_table(ci_path)
    metrics = _read_table(metrics_path)
    draws = _read_table(draws_path)
    metric_map = {
        "uno_c": "uno_c_mean",
        "harrell_c": "harrell_c_mean",
        "auc": "auc_mean",
        "brier": "brier_mean",
        "cal_intercept": "cal_intercept_mean",
        "cal_slope": "cal_slope_mean",
        "ibs": "ibs_mean",
    }
    checked = 0
    differs_from_bootstrap_mean = 0
    for _, row in ci.iterrows():
        if row["metric"] not in metric_map:
            continue
        source = metrics.query(
            "block == @row.block and model == @row.model and horizon_days == @row.horizon_days"
        )
        if source.empty:
            continue
        estimate = float(source.iloc[0][metric_map[row["metric"]]])
        assert float(row["estimate"]) == pytest.approx(estimate)
        boot = draws.query(
            "block == @row.block and model == @row.model and horizon_days == @row.horizon_days and metric == @row.metric"
        )
        if not boot.empty and np.isfinite(boot["value"]).any():
            if abs(float(row["estimate"]) - float(boot["value"].mean())) > 1e-15:
                differs_from_bootstrap_mean += 1
        checked += 1
    assert checked > 0
    assert differs_from_bootstrap_mean > 0


def test_recurrence_bootstrap_ci_estimate_matches_original_oof_summary_not_bootstrap_mean():
    ci_path = ROOT / "results/ml/recurrence/_smoke/primary_bootstrap_ci.csv"
    metrics_path = ROOT / "results/ml/recurrence/_smoke/primary_metrics.csv"
    draws_path = ROOT / "results/ml/recurrence/_smoke/primary_bootstrap_repeat_mean_draws.csv"
    if not all(path.exists() for path in [ci_path, metrics_path, draws_path]):
        pytest.skip("recurrence bootstrap CI smoke outputs are not available yet")
    _assert_recurrence_bootstrap_ci_estimates(ci_path, metrics_path, draws_path)


def test_recurrence_canonical_bootstrap_ci_estimate_matches_original_oof_summary_not_bootstrap_mean():
    ci_path = ROOT / "results/ml/recurrence/primary_bootstrap_ci.csv"
    metrics_path = ROOT / "results/ml/recurrence/primary_metrics.csv"
    draws_path = ROOT / "results/ml/recurrence/primary_bootstrap_repeat_mean_draws.csv"
    if not all(path.exists() for path in [ci_path, metrics_path, draws_path]):
        pytest.skip("canonical recurrence bootstrap CI outputs are not available yet")
    ci = _read_table(ci_path)
    if int(ci["bootstrap_n"].min()) < 2000:
        pytest.skip("canonical recurrence bootstrap CI is not at 2000 draws yet")
    _assert_recurrence_bootstrap_ci_estimates(ci_path, metrics_path, draws_path)


def test_recurrence_bootstrap_patient_weights_preserve_same_draw_ids_for_paired_delta_checks():
    path = ROOT / "results/ml/recurrence/_smoke/primary_bootstrap_patient_weights.rds"
    data_path = ROOT / "data/ml/recurrence.tsv"
    if not path.exists() or not data_path.exists():
        pytest.skip("recurrence bootstrap patient-weight RDS is not available yet")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
x <- readRDS(args[[1]])
write.csv(as.data.frame(x$weights), file=file.path(args[[2]], "weights.csv"), row.names=FALSE)
writeLines(x$study_id, file.path(args[[2]], "ids.txt"))
'''
        run = subprocess.run(["Rscript", "-e", r_code, str(path), str(tmp_path)], cwd=ROOT, text=True, capture_output=True, check=False)
        if run.returncode != 0:
            pytest.skip(f"could not read bootstrap patient weights with Rscript: {run.stderr[-500:]}")
        weights = pd.read_csv(tmp_path / "weights.csv")
        ids = (tmp_path / "ids.txt").read_text().splitlines()
    data = _read_table(data_path).query("recurrence_primary == 1")

    assert weights.shape[1] == len(ids) == data["study_id"].nunique()
    assert set(ids) == set(data["study_id"])
    assert (weights.to_numpy(int) >= 0).all()
    assert (weights.sum(axis=1).to_numpy(int) == len(ids)).all()


def test_recurrence_canonical_bootstrap_patient_weights_preserve_same_draw_ids_for_paired_delta_checks():
    path = ROOT / "results/ml/recurrence/primary_bootstrap_patient_weights.rds"
    data_path = ROOT / "data/ml/recurrence.tsv"
    if not path.exists() or not data_path.exists():
        pytest.skip("canonical recurrence bootstrap patient-weight RDS is not available yet")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        r_code = r'''
args <- commandArgs(trailingOnly=TRUE)
x <- readRDS(args[[1]])
write.csv(as.data.frame(x$weights), file=file.path(args[[2]], "weights.csv"), row.names=FALSE)
writeLines(x$study_id, file.path(args[[2]], "ids.txt"))
'''
        run = subprocess.run(["Rscript", "-e", r_code, str(path), str(tmp_path)], cwd=ROOT, text=True, capture_output=True, check=False)
        if run.returncode != 0:
            pytest.skip(f"could not read canonical bootstrap patient weights with Rscript: {run.stderr[-500:]}")
        weights = pd.read_csv(tmp_path / "weights.csv")
        ids = (tmp_path / "ids.txt").read_text().splitlines()
    data = _read_table(data_path).query("recurrence_primary == 1")

    assert weights.shape[0] == 2000
    assert weights.shape[1] == len(ids) == data["study_id"].nunique()
    assert set(ids) == set(data["study_id"])
    assert (weights.to_numpy(int) >= 0).all()
    assert (weights.sum(axis=1).to_numpy(int) == len(ids)).all()


def test_recurrence_smoke_repeat_metrics_match_independent_reference_values():
    oof_path = ROOT / "results/ml/recurrence/_smoke/primary_oof_predictions.csv"
    metric_path = ROOT / "results/ml/recurrence/_smoke/primary_repeat_metrics.csv"
    if not oof_path.exists() or not metric_path.exists():
        pytest.skip("recurrence smoke OOF and repeat metrics are not available yet")
    oof = _read_table(oof_path)
    metrics = _read_table(metric_path)
    assert len(metrics) == 36
    for _, row in metrics.iterrows():
        repeat = int(row["repeat"])
        block = row["block"]
        model = row["model"]
        z = oof.query("repeat == @repeat and block == @block and model == @model")
        assert not z.empty
        risk_col = f"risk_{int(row.horizon_days)}"
        c_index, numerator, denominator = uno_c_from_saved_g_by_fold(z, row.horizon_days)
        auc = weighted_auc_from_saved_g(z, row.horizon_days, score_col=risk_col)
        brier = ipcw_brier_from_saved_g(z, row.horizon_days, risk_col=risk_col)
        ibs = ibs_from_saved_grid(z)
        assert row["uno_c"] == pytest.approx(c_index)
        assert row["uno_num"] == pytest.approx(numerator)
        assert row["uno_den"] == pytest.approx(denominator)
        assert row["auc"] == pytest.approx(auc)
        assert row["brier"] == pytest.approx(brier)
        assert row["ibs"] == pytest.approx(ibs)


def test_tissue_smoke_folds_have_no_train_test_patient_overlap():
    path = _canonical_existing([
        ROOT / "results/ml/tissue/folds.tsv",
        ROOT / "results/ml/tissue/final/folds.tsv",
        ROOT / "results/ml/tissue/smoke/folds.tsv",
    ])
    if path is None:
        pytest.skip("tissue smoke folds are not available yet")
    import ast

    folds = _read_table(path)
    train_col = "train_patients" if "train_patients" in folds.columns else "train_study_ids"
    test_col = "test_patients" if "test_patients" in folds.columns else "test_study_ids"
    for _, row in folds.iterrows():
        assert_group_disjoint(ast.literal_eval(row[train_col]), ast.literal_eval(row[test_col]))


def test_tissue_oof_predictions_have_one_row_per_model_repeat_specimen():
    path = _canonical_existing([
        ROOT / "results/ml/tissue/oof_predictions.tsv.gz",
        ROOT / "results/ml/tissue/final/oof_predictions.tsv.gz",
        ROOT / "results/ml/tissue/smoke/oof_predictions.tsv.gz",
    ])
    if path is None:
        pytest.skip("tissue smoke OOF predictions are not available yet")
    oof = _read_table(path)
    id_col = "patient_id" if "patient_id" in oof.columns else "study_id"
    sample_col = "sample_id" if "sample_id" in oof.columns else "specimen_id"

    assert not oof.duplicated(["model", "repeat", id_col, sample_col]).any()


def test_tissue_all_repeat_summary_equals_mean_of_repeat_metrics_when_available():
    path = _canonical_existing([
        ROOT / "results/ml/tissue/metrics.tsv",
        ROOT / "results/ml/tissue/final/metrics.tsv",
        ROOT / "results/ml/tissue/smoke/metrics.tsv",
    ])
    if path is None:
        pytest.skip("tissue smoke metrics are not available yet")
    metrics = _skip_if_stale_smoke(path, target_repeats=20)
    repeat_rows = metrics.loc[metrics["scope"].eq("repeat_pooled")]
    summary_rows = metrics.loc[metrics["scope"].isin(["all_repeats_mean", "all_repeats_pooled"])]
    metric_cols = [
        "auc",
        "brier",
        "threshold_0_5_balanced_accuracy",
        "normal_q95_balanced_accuracy",
    ]
    for model, summary in summary_rows.groupby("model"):
        repeats = repeat_rows.loc[repeat_rows["model"].eq(model)]
        assert not repeats.empty
        assert len(summary) == 1
        for col in metric_cols:
            if col in metrics.columns and np.isfinite(repeats[col]).any():
                assert float(summary.iloc[0][col]) == pytest.approx(float(repeats[col].mean()))


def test_tissue_final_outputs_have_expected_counts_and_models():
    oof_path = ROOT / "results/ml/tissue/oof_predictions.tsv.gz"
    folds_path = ROOT / "results/ml/tissue/folds.tsv"
    states_path = ROOT / "results/ml/tissue/model_states.tsv.gz"
    candidates_path = ROOT / "results/ml/tissue/candidate_results.tsv.gz"
    if not all(path.exists() for path in [oof_path, folds_path, states_path, candidates_path]):
        pytest.skip("canonical tissue final outputs are not available yet")
    oof = _read_table(oof_path)
    folds = _read_table(folds_path)
    states = _read_table(states_path)
    candidates = _read_table(candidates_path)

    assert len(oof) == 174 * 20 * 5
    assert oof["repeat"].nunique() == 20
    assert set(oof["model"]) == {"ridge", "elastic_net", "random_forest", "rbf_svm", "best_single_gene"}
    assert len(folds) == 100
    assert len(states) == 500
    assert not candidates.empty
    assert "stage" not in oof.columns
    assert "event" not in oof.columns
    assert "duration_days" not in oof.columns


def test_tissue_final_metrics_match_independent_oof_recalculation():
    oof_path = ROOT / "results/ml/tissue/oof_predictions.tsv.gz"
    metric_path = ROOT / "results/ml/tissue/metrics.tsv"
    if not oof_path.exists() or not metric_path.exists():
        pytest.skip("canonical tissue final metrics are not available yet")
    observed = _read_table(metric_path)
    expected = summarize_tissue_metrics_from_oof(_read_table(oof_path))
    metric_cols = [
        "n",
        "n_patients",
        "auc",
        "brier",
        "threshold_0_5_sensitivity",
        "threshold_0_5_specificity",
        "threshold_0_5_balanced_accuracy",
        "normal_q95_sensitivity",
        "normal_q95_specificity",
        "normal_q95_balanced_accuracy",
    ]

    assert len(observed.query("scope == 'repeat_pooled'")) == 100
    assert len(observed.query("scope == 'all_repeats_pooled'")) == 5
    _assert_metric_tables_close(observed, expected, ["model", "scope", "repeat"], metric_cols)


@pytest.mark.parametrize("cohort,n_patients", [("colonomics", 92), ("gse119526", 48)])
def test_public_tissue_metrics_match_independent_oof_recalculation(cohort, n_patients):
    oof_path = ROOT / "results/ml/public" / cohort / "oof_predictions.tsv.gz"
    metric_path = ROOT / "results/ml/public" / cohort / "metrics.tsv"
    if not oof_path.exists() or not metric_path.exists():
        pytest.skip(f"{cohort} public tissue outputs are not available yet")
    oof = _read_table(oof_path)
    observed = _read_table(metric_path)
    expected = summarize_tissue_metrics_from_oof(oof)
    metric_cols = [
        "n",
        "n_patients",
        "auc",
        "brier",
        "threshold_0_5_sensitivity",
        "threshold_0_5_specificity",
        "threshold_0_5_balanced_accuracy",
        "normal_q95_sensitivity",
        "normal_q95_specificity",
        "normal_q95_balanced_accuracy",
    ]

    assert oof["study_id"].nunique() == n_patients
    assert oof["repeat"].nunique() == 20
    assert set(oof["model"]) == {"ridge"}
    assert len(observed.query("scope == 'repeat_pooled'")) == 20
    assert len(observed.query("scope == 'all_repeats_pooled'")) == 1
    _assert_metric_tables_close(observed, expected, ["model", "scope", "repeat"], metric_cols)


def test_tissue_final_folds_have_no_group_overlap_and_cover_all_pairs():
    path = ROOT / "results/ml/tissue/folds.tsv"
    if not path.exists():
        pytest.skip("canonical tissue folds are not available yet")
    folds = _read_table(path)
    expected = {f"P{i:03d}" for i in range(1, 88)}
    for repeat, repeat_folds in folds.groupby("repeat"):
        covered = set()
        for _, row in repeat_folds.iterrows():
            train = set(ast.literal_eval(row["train_study_ids"]))
            test = set(ast.literal_eval(row["test_study_ids"]))
            assert_group_disjoint(train, test)
            covered.update(test)
        assert covered == expected


def test_tissue_saved_state_predictions_match_oof_for_representative_models():
    data_path = ROOT / "data/ml/tissue.tsv"
    oof_path = ROOT / "results/ml/tissue/oof_predictions.tsv.gz"
    states_path = ROOT / "results/ml/tissue/model_states.tsv.gz"
    if not all(path.exists() for path in [data_path, oof_path, states_path]):
        pytest.skip("canonical tissue model states are not available yet")
    data = _read_table(data_path).rename(columns={"specimen_id": "sample_id"})
    oof = _read_table(oof_path)
    states = _read_table(states_path)
    for model_name in ["ridge", "random_forest", "rbf_svm"]:
        state = states.query("model == @model_name and repeat == 0 and fold == 0").iloc[0]
        model_path = ROOT / "results/ml/tissue/models" / f"{model_name}_r00_f00.pkl.gz"
        payload = _load_pickle_model(model_path)
        genes = ast.literal_eval(state["genes"])
        test_ids = ast.literal_eval(state["test_study_ids"])
        test = data.loc[data["study_id"].isin(test_ids)].copy()
        X = test[genes].to_numpy(float)
        if payload["scaler"] is not None:
            X = payload["scaler"].transform(X)
        estimator = payload["model"]
        if model_name == "rbf_svm":
            score = estimator.decision_function(X)
            probability = np.full(len(score), np.nan)
        else:
            probability = estimator.predict_proba(X)[:, 1]
            score = probability
        predicted = test[["sample_id"]].copy()
        predicted["score"] = score
        predicted["probability"] = probability
        predicted = predicted.sort_values("sample_id")
        saved = oof.query("model == @model_name and repeat == 0 and fold == 0").sort_values("sample_id")
        assert saved["sample_id"].tolist() == predicted["sample_id"].tolist()
        np.testing.assert_allclose(saved["score"].to_numpy(float), predicted["score"].to_numpy(float), rtol=0, atol=1e-12, equal_nan=True)
        if model_name != "rbf_svm":
            np.testing.assert_allclose(saved["probability"].to_numpy(float), predicted["probability"].to_numpy(float), rtol=0, atol=1e-12)


@pytest.mark.parametrize("cohort", ["colonomics", "gse119526"])
def test_public_ridge_saved_state_predictions_match_oof_for_representative_fold(cohort):
    oof_path = ROOT / "results/ml/public" / cohort / "oof_predictions.tsv.gz"
    states_path = ROOT / "results/ml/public" / cohort / "model_states.tsv.gz"
    source_path = ROOT / "results/external/public_gene_scores.tsv"
    if not all(path.exists() for path in [oof_path, states_path, source_path]):
        pytest.skip(f"public {cohort} model states are not available yet")
    source = _read_table(source_path)
    gene_column_map = {f"{gene}_beta": gene for gene in GENES}
    source = source.loc[source["cohort"].eq(cohort)].rename(columns={"patient": "study_id", "sample": "sample_id", **gene_column_map})
    source["study_id"] = cohort + "_" + source["study_id"].astype(str)
    if cohort == "gse119526":
        source["sample_id"] = source["study_id"] + "_" + source["tissue"].astype(str)
    else:
        source["sample_id"] = cohort + "_" + source["sample_id"].astype(str)
    oof = _read_table(oof_path)
    state = _read_table(states_path).query("model == 'ridge' and repeat == 0 and fold == 0").iloc[0]
    payload = _load_pickle_model(ROOT / "results/ml/public" / cohort / "models/ridge_r00_f00.pkl.gz")
    genes = ast.literal_eval(state["genes"])
    test_ids = ast.literal_eval(state["test_study_ids"])
    test = source.loc[source["study_id"].isin(test_ids)].copy()
    X = test[genes].to_numpy(float)
    X = payload["scaler"].transform(X)
    probability = payload["model"].predict_proba(X)[:, 1]
    predicted = test[["sample_id"]].copy()
    predicted["probability"] = probability
    predicted = predicted.sort_values("sample_id")
    saved = oof.query("repeat == 0 and fold == 0").sort_values("sample_id")

    assert saved["sample_id"].tolist() == predicted["sample_id"].tolist()
    np.testing.assert_allclose(saved["probability"].to_numpy(float), predicted["probability"].to_numpy(float), rtol=0, atol=1e-12)


def test_tissue_bootstrap_ci_uses_saved_draw_quantiles_and_original_points():
    draws_path = ROOT / "results/ml/tissue/bootstrap_draws.tsv.gz"
    ci_path = ROOT / "results/ml/tissue/bootstrap_ci.tsv"
    metrics_path = ROOT / "results/ml/tissue/metrics.tsv"
    if not all(path.exists() for path in [draws_path, ci_path, metrics_path]):
        pytest.skip("local tissue bootstrap outputs are not available yet")
    draws = _read_table(draws_path)
    ci = _read_table(ci_path)
    metrics = _read_table(metrics_path).query("scope == 'all_repeats_pooled'")
    points = metrics.set_index("model")

    assert draws["bootstrap"].nunique() == 2000
    for _, row in ci.iterrows():
        metric = row["metric"]
        model = row["model"]
        if "_minus_best_single_gene" in model:
            model_a = model.replace("_minus_best_single_gene", "")
            a = draws.loc[draws["model"].eq(model_a), ["bootstrap", metric]].rename(columns={metric: "a"})
            b = draws.loc[draws["model"].eq("best_single_gene"), ["bootstrap", metric]].rename(columns={metric: "b"})
            vals = a.merge(b, on="bootstrap", validate="one_to_one")
            values = (vals["a"] - vals["b"]).dropna().to_numpy(float)
            point = float(points.loc[model_a, metric] - points.loc["best_single_gene", metric])
        else:
            values = draws.loc[draws["model"].eq(model), metric].dropna().to_numpy(float)
            point = float(points.loc[model, metric])
        if len(values) == 0:
            assert pd.isna(row["point"])
            assert pd.isna(row["ci_low"])
            assert pd.isna(row["ci_high"])
            continue
        low, high = np.quantile(values, [0.025, 0.975])
        assert float(row["point"]) == pytest.approx(point)
        assert float(row["ci_low"]) == pytest.approx(low)
        assert float(row["ci_high"]) == pytest.approx(high)
        assert int(row["bootstrap_replicates"]) == 2000


@pytest.mark.parametrize("cohort", ["colonomics", "gse119526"])
def test_public_bootstrap_ci_uses_saved_draw_quantiles(cohort):
    draws_path = ROOT / "results/ml/public" / cohort / "bootstrap_draws.tsv.gz"
    ci_path = ROOT / "results/ml/public" / cohort / "bootstrap_ci.tsv"
    if not draws_path.exists() or not ci_path.exists():
        pytest.skip(f"{cohort} bootstrap outputs are not available yet")
    draws = _read_table(draws_path)
    ci = _read_table(ci_path)
    assert draws["bootstrap"].nunique() == 2000
    for _, row in ci.iterrows():
        vals = draws.loc[draws["model"].eq(row["model"]), row["metric"]].dropna().to_numpy(float)
        low, high = np.quantile(vals, [0.025, 0.975])
        assert row["ci_low"] == pytest.approx(low)
        assert row["ci_high"] == pytest.approx(high)
