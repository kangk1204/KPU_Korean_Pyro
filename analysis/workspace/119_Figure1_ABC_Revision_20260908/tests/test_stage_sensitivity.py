import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer"


def test_stage_runner_self_test():
    got = subprocess.run(
        ["Rscript", str(ROOT / "scripts" / "run_stage_sensitivity.R"), "--self-test"],
        cwd=ROOT.parent,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "stage_sensitivity self-tests passed" in got.stdout


def test_stage_patient_audit_contract():
    audit = pd.read_csv(ROOT / "private" / "stage_patient_audit.tsv", sep="\t")
    assert len(audit) == 82
    assert int(audit["event"].sum()) == 14
    included = audit[audit["analysis_included"]]
    excluded = audit[~audit["analysis_included"]]
    assert len(included) == 79
    assert int(included["event"].sum()) == 14
    assert len(excluded) == 3  # identifiers and raw TNM strings stay in the restricted audit file
    assert int(audit["provider_tnm_mismatch"].sum()) == 10
    assert not (ROOT / "submission" / "stage_patient_audit.tsv").exists()


def test_stage_common_folds_and_no_train_test_leakage():
    folds = pd.read_csv(OUT / "stage_folds.csv")
    oof = pd.read_csv(OUT / "stage_oof_predictions.csv", usecols=["repeat", "fold", "study_id", "block"])
    selected = pd.read_csv(
        OUT / "stage_selected_hyperparameters.csv",
        usecols=["repeat", "fold", "block", "train_study_ids", "test_study_ids", "status"],
    )
    blocks = {
        "provider_clinical",
        "provider_combined",
        "tnm_clinical",
        "tnm_combined",
        "nostage_clinical",
        "nostage_combined",
    }
    assert len(folds) == 79 * 25
    assert set(oof["block"]) == blocks
    assert len(oof) == 79 * 25 * 6
    assert len(selected) == 25 * 4 * 6
    assert set(selected["status"]) == {"complete"}

    fold_map = folds.set_index(["repeat", "study_id"])["fold"].to_dict()
    for (repeat, block), z in oof.groupby(["repeat", "block"]):
        assert len(z) == 79
        assert not z["study_id"].duplicated().any()
        assert z.apply(lambda row: fold_map[(row["repeat"], row["study_id"])] == row["fold"], axis=1).all()

    for row in selected.itertuples(index=False):
        train = set(row.train_study_ids.split(";"))
        test = set(row.test_study_ids.split(";"))
        assert train
        assert test
        assert train.isdisjoint(test)
        expected_test = set(folds[(folds["repeat"] == row.repeat) & (folds["fold"] == row.fold)]["study_id"])
        assert test == expected_test


def test_stage_inner_assignment_and_metric_schema():
    inner = pd.read_csv(OUT / "stage_inner_assignment_proof.csv")
    selected = pd.read_csv(OUT / "stage_selected_hyperparameters.csv")
    assert set(inner["inner_fold"]) == {1, 2, 3}
    assert len(inner) == sum(len(x.split(";")) for x in selected["train_study_ids"])

    metrics = pd.read_csv(OUT / "stage_metrics.tsv", sep="\t")
    diffs = pd.read_csv(OUT / "stage_differences.tsv", sep="\t")
    assert list(metrics.columns) == [
        "stage_definition",
        "block",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "bootstrap_n",
        "n_estimable",
        "status",
    ]
    assert list(diffs.columns) == [
        "stage_definition",
        "contrast",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "bootstrap_n",
        "n_estimable",
        "status",
    ]
    assert set(metrics["stage_definition"]) == {"provider", "tnm", "none"}
    assert set(metrics["block"]) == {"clinical", "combined"}
    assert set(metrics["metric"]) == {"uno_c", "auc", "brier"}
    assert len(metrics) == 18
    assert len(diffs) == 9
    assert set(diffs["contrast"]) == {"combined-minus-clinical"}
    assert "p_value" not in metrics.columns
    assert "p_value" not in diffs.columns
    assert set(metrics["bootstrap_n"]) == {2000}
    assert set(diffs["bootstrap_n"]) == {2000}
    assert set(metrics["status"]) == {"estimable"}
    assert set(diffs["status"]) == {"estimable"}
