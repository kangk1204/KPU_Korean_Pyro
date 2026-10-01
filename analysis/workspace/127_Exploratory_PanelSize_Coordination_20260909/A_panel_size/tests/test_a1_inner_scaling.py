from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "a1_local_panel_size.py"


def load_a1_module():
    spec = importlib.util.spec_from_file_location("a1_local_panel_size_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inner_cv_scaling_is_fit_within_each_inner_training_split(monkeypatch):
    a1 = load_a1_module()

    genes = ["G1", "G2", "G3"]
    subsets = a1.all_subsets(genes)
    monkeypatch.setattr(a1, "GENES", genes)
    monkeypatch.setattr(a1, "SUBSETS", subsets)
    monkeypatch.setattr(a1, "KS", np.array([len(s) for s in subsets]))
    monkeypatch.setattr(a1, "C_GRID", (0.1, 1.0))

    patients = np.array([f"P{i}" for i in range(8)])
    groups = np.repeat(patients, 2)
    y = np.tile([0, 1], len(patients))
    patient_signal = np.repeat(np.arange(len(patients), dtype=float), 2)[:, None]
    tissue_signal = y[:, None] * np.array([[0.2, 0.5, 1.1]])
    feature_offset = np.array([[1.0, 10.0, 100.0]])
    x = feature_offset + patient_signal * np.array([[1.3, -2.1, 3.7]]) + tissue_signal

    a1._init(x, y, groups)
    outer_train = patients[:6]
    spec = {
        "repeat": 0,
        "fold": 0,
        "train_patients": outer_train,
        "test_patients": patients[6:],
    }
    full_outer_train_n = len(outer_train) * 2
    inner_training_moments = []

    def fake_fit_predict(xtr, ytr, xte, c):
        if len(ytr) < full_outer_train_n:
            inner_training_moments.append((xtr.mean(axis=0), xtr.std(axis=0, ddof=0)))
        return 1.0 / (1.0 + np.exp(-xte.sum(axis=1)))

    monkeypatch.setattr(a1, "fit_predict", fake_fit_predict)
    a1.run_fold(spec)

    assert inner_training_moments
    for mean, sd in inner_training_moments:
        np.testing.assert_allclose(mean, np.zeros_like(mean), atol=1e-12)
        np.testing.assert_allclose(sd, np.ones_like(sd), atol=1e-12)


def test_checkpoint_round_trips_one_outer_fold(tmp_path):
    a1 = load_a1_module()
    metadata = a1._checkpoint_metadata("abc123")
    res = {
        "repeat": 2,
        "fold": 3,
        "test_idx": np.array([1, 5]),
        "sweep": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "nested": np.array([[0.5, 0.6]]),
        "chosen": [{"repeat": 2, "fold": 3, "k": 1, "subset": "G1", "inner_auc": 0.75, "C": 1.0}],
        "c_full": 0.1,
    }

    a1._save_checkpoint(tmp_path, res, metadata)
    got = a1._load_checkpoint(tmp_path / "repeat02_fold03.npz")

    assert got["repeat"] == res["repeat"]
    assert got["fold"] == res["fold"]
    assert got["chosen"] == res["chosen"]
    assert got["c_full"] == res["c_full"]
    assert got["metadata"] == metadata
    np.testing.assert_array_equal(got["test_idx"], res["test_idx"])
    np.testing.assert_allclose(got["sweep"], res["sweep"])
    np.testing.assert_allclose(got["nested"], res["nested"])


def test_checkpoint_rejects_mismatched_metadata(tmp_path):
    a1 = load_a1_module()
    metadata = a1._checkpoint_metadata("expected")
    stale_metadata = a1._checkpoint_metadata("stale")
    groups = np.array(["P0", "P0", "P1", "P1"])
    spec = {"repeat": 0, "fold": 1, "test_patients": np.array(["P1"])}
    res = {
        "repeat": 0,
        "fold": 1,
        "test_idx": np.array([2, 3]),
        "sweep": np.full((len(a1.SUBSETS), 2), 0.5),
        "nested": np.full((len(a1.GENES), 2), 0.5),
        "chosen": [
            {"repeat": 0, "fold": 1, "k": k, "subset": "G1", "inner_auc": 0.75, "C": a1.C_GRID[0]}
            for k in range(1, len(a1.GENES) + 1)
        ],
        "c_full": a1.C_GRID[0],
    }

    a1._save_checkpoint(tmp_path, res, stale_metadata)
    got = a1._load_checkpoint(tmp_path / "repeat00_fold01.npz")

    try:
        a1._validate_checkpoint(got, spec, groups, metadata)
    except ValueError as exc:
        assert "metadata mismatch" in str(exc)
    else:
        raise AssertionError("stale checkpoint was accepted")


def test_checkpoint_rejects_wrong_test_indices(tmp_path):
    a1 = load_a1_module()
    metadata = a1._checkpoint_metadata("abc123")
    groups = np.array(["P0", "P0", "P1", "P1"])
    spec = {"repeat": 0, "fold": 1, "test_patients": np.array(["P1"])}
    res = {
        "repeat": 0,
        "fold": 1,
        "test_idx": np.array([0, 1]),
        "sweep": np.full((len(a1.SUBSETS), 2), 0.5),
        "nested": np.full((len(a1.GENES), 2), 0.5),
        "chosen": [
            {"repeat": 0, "fold": 1, "k": k, "subset": "G1", "inner_auc": 0.75, "C": a1.C_GRID[0]}
            for k in range(1, len(a1.GENES) + 1)
        ],
        "c_full": a1.C_GRID[0],
    }

    a1._save_checkpoint(tmp_path, res, metadata)
    got = a1._load_checkpoint(tmp_path / "repeat00_fold01.npz")

    try:
        a1._validate_checkpoint(got, spec, groups, metadata)
    except ValueError as exc:
        assert "test index mismatch" in str(exc)
    else:
        raise AssertionError("checkpoint with wrong test indices was accepted")


def _valid_tissue_frame(a1):
    rows = []
    for patient in ["P0", "P1"]:
        for tissue, y in [("N", 0), ("T", 1)]:
            row = {"study_id": patient, "tissue": tissue, "y": y}
            row.update({gene: 10.0 + y for gene in a1.GENES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_load_local_rejects_fractional_labels(tmp_path, monkeypatch):
    a1 = load_a1_module()
    path = tmp_path / "fractional_labels.tsv"
    monkeypatch.setattr(a1, "TISSUE_TSV", path)
    for row, value in [(0, 0.5), (1, 1.5), (0, -0.5)]:
        df = _valid_tissue_frame(a1)
        df["y"] = df["y"].astype(float)
        df.loc[row, "y"] = value
        df.to_csv(path, sep="\t", index=False)
        try:
            a1.load_local()
        except ValueError as exc:
            assert "y labels must be exactly 0 and 1" in str(exc)
        else:
            raise AssertionError(f"fractional label {value} was accepted")


def test_load_local_rejects_tissue_y_mismatch(tmp_path, monkeypatch):
    a1 = load_a1_module()
    df = _valid_tissue_frame(a1)
    df.loc[0, "y"] = 1
    path = tmp_path / "bad_tissue.tsv"
    df.to_csv(path, sep="\t", index=False)
    monkeypatch.setattr(a1, "TISSUE_TSV", path)

    try:
        a1.load_local()
    except ValueError as exc:
        assert "N tissues must have y=0" in str(exc)
    else:
        raise AssertionError("mismatched tissue/y labels were accepted")


def test_load_local_rejects_nonfinite_or_out_of_range_values(tmp_path, monkeypatch):
    a1 = load_a1_module()
    df = _valid_tissue_frame(a1)
    df.loc[0, a1.GENES[0]] = float("inf")
    path = tmp_path / "bad_tissue.tsv"
    df.to_csv(path, sep="\t", index=False)
    monkeypatch.setattr(a1, "TISSUE_TSV", path)

    try:
        a1.load_local()
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite methylation values were accepted")

    df = _valid_tissue_frame(a1)
    df.loc[0, a1.GENES[0]] = 101.0
    df.to_csv(path, sep="\t", index=False)

    try:
        a1.load_local()
    except ValueError as exc:
        assert "within 0-100" in str(exc)
    else:
        raise AssertionError("out-of-range methylation values were accepted")


def test_load_local_rejects_extra_tissue_label_or_missing_patient_id(tmp_path, monkeypatch):
    a1 = load_a1_module()
    df = _valid_tissue_frame(a1)
    df.loc[0, "tissue"] = "A"
    path = tmp_path / "bad_tissue.tsv"
    df.to_csv(path, sep="\t", index=False)
    monkeypatch.setattr(a1, "TISSUE_TSV", path)

    try:
        a1.load_local()
    except ValueError as exc:
        assert "tissue labels must be exactly N and T" in str(exc)
    else:
        raise AssertionError("unexpected tissue label was accepted")

    df = _valid_tissue_frame(a1)
    df.loc[0, "study_id"] = None
    df.to_csv(path, sep="\t", index=False)

    try:
        a1.load_local()
    except ValueError as exc:
        assert "missing study_id" in str(exc)
    else:
        raise AssertionError("missing study_id was accepted")
