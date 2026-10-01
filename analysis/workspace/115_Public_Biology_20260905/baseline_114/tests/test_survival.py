import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg


ROOT = Path(__file__).resolve().parents[1]
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(order) + 1))[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out


def load_model_frame(primary):
    clinical = pd.read_csv(ROOT / "data/derived/clinical.tsv", sep="\t")
    methylation = pd.read_csv(ROOT / "data/derived/methylation_wide.tsv", sep="\t")
    data = clinical.merge(methylation, on=["patient_id", "study_id"], how="inner")
    if primary:
        data = data.loc[data["recurrence_primary"] == 1].copy()
    return data


def statsmodels_cox(data, gene):
    complete = data[["duration_days", "event", f"T_{gene}"]].dropna().copy()
    exog = (complete[[f"T_{gene}"]] / 10.0).to_numpy()
    fit = PHReg(complete["duration_days"], exog, status=complete["event"], ties="efron").fit(disp=0)
    return fit.params[0], fit.bse[0], fit.pvalues[0], complete.shape[0], int(complete["event"].sum())


def test_survival_rerun_and_outputs_match_independent_phreg():
    subprocess.run(["Rscript", "scripts/analyze_survival.R"], cwd=ROOT, check=True)

    primary = pd.read_csv(ROOT / "results/recurrence.csv")
    all_stage = pd.read_csv(ROOT / "results/recurrence_all_stage.csv")
    summary = json.loads((ROOT / "results/survival_summary.json").read_text())

    assert summary["primary_population"]["n"] == 82
    assert summary["primary_population"]["events"] == 14
    assert summary["sensitivity_population"]["n"] == 87
    assert summary["sensitivity_population"]["events"] == 17
    assert primary["estimable"].all()
    assert all_stage["estimable"].all()
    assert not primary["ph_violation_p_lt_0_05"].fillna(False).any()
    assert not all_stage["ph_violation_p_lt_0_05"].fillna(False).any()
    np.testing.assert_allclose(primary["p_bh"].to_numpy(), bh_adjust(primary["p_value"].to_numpy()), rtol=0, atol=1e-12)
    np.testing.assert_allclose(all_stage["p_bh"].to_numpy(), bh_adjust(all_stage["p_value"].to_numpy()), rtol=0, atol=1e-12)

    for table, data in [(primary, load_model_frame(primary=True)), (all_stage, load_model_frame(primary=False))]:
        assert table["gene"].tolist() == GENES
        for _, row in table.iterrows():
            coef, se, p_value, n_complete, events = statsmodels_cox(data, row["gene"])
            assert row["complete_cases"] == n_complete
            assert row["events"] == events
            np.testing.assert_allclose(row["coef_per10"], coef, rtol=0, atol=1e-7)
            np.testing.assert_allclose(row["se_per10"], se, rtol=0, atol=1e-7)
            np.testing.assert_allclose(row["p_value"], p_value, rtol=0, atol=1e-7)


def test_survival_diagnostic_tables_have_expected_shape():
    ph = pd.read_csv(ROOT / "results/survival_ph_diagnostics.csv")
    residuals = pd.read_csv(ROOT / "results/survival_schoenfeld_residuals.csv")
    followup = pd.read_csv(ROOT / "results/survival_followup_summary.csv")
    event_time = pd.read_csv(ROOT / "results/survival_event_time_distribution.csv")

    assert ph.shape[0] == 20
    assert set(ph["gene"]) == set(GENES)
    assert residuals.shape[0] == (14 + 17) * len(GENES)
    assert followup["n"].tolist() == [82, 87]
    assert followup["events"].tolist() == [14, 17]
    assert event_time.groupby("analysis_population")["milestone_days"].min().eq(0).all()
