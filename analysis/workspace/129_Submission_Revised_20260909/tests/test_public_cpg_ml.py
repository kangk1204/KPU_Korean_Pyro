from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pathlib
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_public_cpg_ml as ml  # noqa: E402



def _require_retained(path, what):
    """Skip when a retained input/evidence file is absent (clean archive checkouts; see reproducibility/INPUT_REQUIREMENTS.tsv)."""
    if not pathlib.Path(path).exists():
        pytest.skip(f"{what} not present in this checkout: {path}")

def test_fixed_registry_expands_to_77_individual_features_without_gene_means():
    fixed = ml.read_fixed_probes(ROOT / "registry" / "fixed_probes.json")

    assert len(fixed) == 77
    assert fixed["cpg"].is_unique
    assert fixed["feature"].str.startswith("cpg__").all()
    assert not set(ml.GENES).intersection(fixed["feature"])


def test_gse119526_description_recovers_48_patient_pairs():
    _require_retained(ROOT.parent / "114_ML_DataDriven_20260905" / "data" / "public" / "processed" / "gse119526_table1_signal_intensities.tsv.gz", "GSE119526 processed signal table")
    fixed = ml.read_fixed_probes(ROOT / "registry" / "fixed_probes.json")
    frame, qc, _ = ml.load_gse119526(fixed)

    assert frame["study_id"].nunique() == 48
    assert len(frame) == 96
    assert frame.groupby("study_id")["y"].agg(["size", "sum"]).eq({"size": 2, "sum": 1}).all().all()
    assert qc["all_missing"].sum() == 6
    assert set(frame["tissue"]) == {"N", "T"}


def test_colonomics_keeps_curated_complete_pairs_only():
    _require_retained(ROOT.parent / "119_Figure1_ABC_Revision_20260908" / "data" / "derived" / "Colonomics_beta.tsv.gz", "prepared Colonomics beta matrix")
    fixed = ml.read_fixed_probes(ROOT / "registry" / "fixed_probes.json")
    frame, qc, _ = ml.load_colonomics(fixed)

    assert frame["study_id"].nunique() == 92
    assert len(frame) == 184
    assert qc["all_missing"].sum() == 0
    assert not {"EYA4", "panel_mean", "score_beta"}.intersection(frame.columns)


def test_preprocessor_drops_all_missing_columns_inside_training_only_and_imputes_by_training_median():
    train = pd.DataFrame(
        {
            "cpg__a": [0.0, 1.0, np.nan, 1.0],
            "cpg__b": [np.nan, np.nan, np.nan, np.nan],
            "cpg__c": [0.2, 0.2, 0.2, 0.2],
        }
    )

    usable, median, scaler = ml.fit_preprocessor(train, ["cpg__a", "cpg__b", "cpg__c"])
    transformed = ml.transform_with_preprocessor(
        pd.DataFrame({"cpg__a": [np.nan], "cpg__b": [99.0], "cpg__c": [0.2]}),
        usable,
        median,
        scaler,
    )

    assert usable == ["cpg__a", "cpg__c"]
    assert median[0] == pytest.approx(1.0)
    assert transformed.shape == (1, 2)


def test_patient_kfolds_have_no_train_test_overlap():
    folds = ml.patient_kfolds([f"P{i}" for i in range(17)], n_splits=5, repeats=3, seed=123)

    assert len(folds) == 15
    assert all(set(f["train_patients"]).isdisjoint(set(f["test_patients"])) for f in folds)


def test_small_run_writes_oof_hashes_and_overlap_audit(tmp_path):
    rows = []
    for patient in range(12):
        for tissue, y, shift in [("N", 0, 0.0), ("T", 1, 0.5)]:
            rows.append(
                {
                    "study_id": f"P{patient}",
                    "sample_id": f"P{patient}_{tissue}",
                    "tissue": tissue,
                    "y": y,
                    "cpg__a": patient / 100 + shift,
                    "cpg__b": np.nan if patient % 3 == 0 else shift,
                    "cpg__all_missing": np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    fixed = pd.DataFrame(
        [
            {"gene": "EYA4", "cpg": "a", "feature": "cpg__a", "gene_cpg_order": 1},
            {"gene": "EYA4", "cpg": "b", "feature": "cpg__b", "gene_cpg_order": 2},
            {"gene": "EYA4", "cpg": "all_missing", "feature": "cpg__all_missing", "gene_cpg_order": 3},
        ]
    )
    qc = ml.feature_qc(frame, fixed, "colonomics", "toy")
    config = ml.MLConfig(outer_folds=3, outer_repeats=2, inner_folds=2, logistic_C=(0.1, 1.0), bootstrap_replicates=20, max_workers=1)

    manifest = ml.run_cohort("colonomics", frame, qc, tmp_path, config, fixed, expected_patients=12)

    assert manifest["patient_fold_overlap_max"] == 0
    assert manifest["features_requested"] == 3
    assert manifest["features_nonmissing_any"] == 2
    assert (tmp_path / "colonomics" / "oof_predictions.tsv.gz").exists()
    assert len(pd.read_csv(tmp_path / "colonomics" / "fold_audit.tsv", sep="\t")) == 6
    assert json.loads((tmp_path / "colonomics" / "run_manifest.json").read_text())["outputs"]["oof_predictions.tsv.gz"]
