import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "revise_recurrence_metrics.py"
OUTDIR = ROOT / "results" / "recurrence_revision"

spec = importlib.util.spec_from_file_location("revise_recurrence_metrics", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def toy_survival_frame(scores):
    rows = []
    patients = [
        (1, "case_a", 1, 100, 0.80),
        (1, "control_a", 0, 3000, 0.20),
        (2, "case_b", 1, 100, 0.80),
        (2, "control_b", 0, 3000, 0.20),
    ]
    for fold, pid, event, duration, default_score in patients:
        score = scores.get(pid, default_score)
        rows.append(
            {
                "repeat": 1,
                "fold": fold,
                "study_id": pid,
                "event": event,
                "duration_days": duration,
                "risk_score": score,
                "risk_1825": score,
                "G_time": 1.0,
                "G_1825": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_within_fold_auc_is_invariant_to_fold_baseline_offsets():
    base = toy_survival_frame({})
    shifted = toy_survival_frame(
        {
            "case_a": 100.8,
            "control_a": 100.2,
            "case_b": -99.2,
            "control_b": -99.8,
        }
    )

    assert mod.within_fold_auc(base, "risk_1825") == 1.0
    assert mod.within_fold_auc(shifted, "risk_1825") == 1.0
    assert mod.ipcw_auc(base, "risk_1825") == 1.0
    assert mod.ipcw_auc(shifted, "risk_1825") < 1.0


def test_constant_null_scores_rank_at_chance():
    null_frame = toy_survival_frame(
        {"case_a": 0.5, "control_a": 0.5, "case_b": 0.5, "control_b": 0.5}
    )
    assert mod.within_fold_auc(null_frame, "risk_1825") == 0.5
    assert mod.ipcw_auc(null_frame, "risk_1825") == 0.5


def test_corrected_table_s3_contract():
    table = pd.read_csv(OUTDIR / "table_s3_recurrence_primary_corrected.tsv", sep="\t")
    delta = pd.read_csv(OUTDIR / "table_s3_delta_ci.tsv", sep="\t")
    decisions = pd.read_csv(OUTDIR / "metric_retention_decisions.tsv", sep="\t")
    support = pd.read_csv(OUTDIR / "horizon_support_counts.tsv", sep="\t")

    assert list(table["block"]) == ["clinical", "tumor10", "combined"]
    assert set(table["horizon_days"]) == {1825}
    assert table["uno_c_5y"].between(0, 1).all()
    assert table["decision_note"].str.contains("pooled AUC/calibration retired").all()

    delta_uno = delta[delta["metric"].eq("delta_uno_c")].iloc[0]
    assert abs(delta_uno["estimate"] - 0.0087154111761351) < 1e-12
    assert delta_uno["ci_low"] < 0 < delta_uno["ci_high"]
    assert delta_uno["bootstrap_n"] == 2000

    support_5y = support[support["horizon_days"].eq(1825)].iloc[0]
    assert support_5y["n_patients"] == 82
    assert support_5y["events_by_horizon"] == 11
    assert support_5y["controls_at_horizon"] == 35
    assert support_5y["censored_before_horizon_unknown_status"] == 36
    assert support_5y["median_observed_time_days"] == 1807

    status_by_metric = dict(zip(decisions["metric"], decisions["status"]))
    assert status_by_metric["5-year Uno C"] == "retained_primary"
    assert status_by_metric["pooled risk_1825 AUC"] == "retired_from_claims"
    assert status_by_metric["calibration slope/intercept"] == "retired_from_claims"
    assert status_by_metric["3-year endpoint"] == "not_substituted"


def test_elastic_net_wording_contract():
    reg = pd.read_csv(OUTDIR / "regularization_boundary_diagnostic.tsv", sep="\t")
    oof = pd.read_csv(OUTDIR / "elastic_net_oof_diagnostic.tsv", sep="\t")

    elastic = reg[reg["model"].eq("elastic_net")]
    assert elastic["n_at_grid_max_100"].sum() == 295
    assert elastic["n_fits"].sum() == 300

    clinical = oof[oof["block"].eq("clinical")].iloc[0]
    assert clinical["max_unique_risk_score_per_repeat"] == 1
    methylation = oof[oof["block"].isin(["tumor10", "combined"])]
    assert (methylation["max_unique_risk_score_per_repeat"] > 1).all()
