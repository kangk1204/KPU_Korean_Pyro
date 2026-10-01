from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_public_cpg_context as apc  # noqa: E402


def test_bh77_keeps_planned_denominator_and_na_rows():
    q = apc.bh77([0.001, math.nan, 0.02])

    assert q[0] == pytest.approx(0.077)
    assert math.isnan(q[1])
    assert q[2] == pytest.approx(0.77)


def test_fixed_cpg_table_is_unique_77_and_promoter_is_stratification_only():
    cpgs = apc.fixed_cpg_table()

    assert len(cpgs) == 77
    assert cpgs["cpg"].is_unique
    assert set(cpgs["gene"]) == set(apc.GENES)
    assert cpgs["promoter_match"].sum() == 44
    assert not {"score_beta", "panel_mean"}.intersection(cpgs.columns)


def test_same_cpg_technical_replicates_are_collapsed_without_cross_cpg_averaging():
    beta = pd.DataFrame(
        {
            "S1": [0.20, 0.80],
            "S2": [0.40, 0.60],
            "S3": [0.90, 0.10],
        },
        index=["cg_a", "cg_b"],
    )
    meta = pd.DataFrame(
        [
            {"sample": "S1", "patient": "P1", "tissue": "T", "age": 60, "sex": "M"},
            {"sample": "S2", "patient": "P1", "tissue": "T", "age": 60, "sex": "M"},
            {"sample": "S3", "patient": "P2", "tissue": "T", "age": 70, "sex": "F"},
        ]
    )

    collapsed, audit = apc.collapse_same_cpg_technical_replicates(beta, meta)
    p1 = collapsed.loc[collapsed["patient"].eq("P1")].iloc[0]

    assert p1["cg_a"] == pytest.approx(0.30)
    assert p1["cg_b"] == pytest.approx(0.70)
    assert "panel_mean" not in collapsed.columns
    assert audit.loc[audit["patient"].eq("P1"), "status"].iloc[0] == "collapsed_same_cpg"


def test_rank_residual_spearman_refits_covariates_and_recovers_signal():
    df = pd.DataFrame(
        {
            "meth": np.arange(12, dtype=float),
            "expr": -np.arange(12, dtype=float),
            "age": [50, 61, 53, 67, 58, 70, 55, 63, 72, 59, 66, 57],
            "sex": ["M", "F"] * 6,
            "site": ["Left"] * 6 + ["Right"] * 6,
            "stromal_score": [0.3, 0.1, 0.8, 0.2, 0.6, 0.4, 0.9, 0.5, 0.7, 0.15, 0.45, 0.25],
        }
    )

    out = apc.rank_residual_spearman(df, "meth", "expr", ["age", "sex", "site", "stromal_score"])

    assert out["status"] == "estimated"
    assert out["n"] == 12
    assert out["effect"] < -0.9
    assert out["df_model"] >= 3


def test_missing_probe_rows_are_retained_in_expression_family(monkeypatch):
    cpgs = pd.DataFrame(
        [
            {"gene": "EYA4", "cpg": "cg_present", "fixed_probe_order_within_gene": 1, "promoter_match": True, "promoter_groups": "TSS1500"},
            {"gene": "EYA4", "cpg": "cg_absent", "fixed_probe_order_within_gene": 2, "promoter_match": False, "promoter_groups": ""},
        ]
    )
    frame = pd.DataFrame(
        {
            "patient": [f"P{i}" for i in range(8)],
            "cg_present": np.linspace(0.1, 0.8, 8),
            "EYA4_expr": np.linspace(8, 1, 8),
            "age": np.arange(50, 58),
            "sex": ["M", "F"] * 4,
            "site": ["Left"] * 4 + ["Right"] * 4,
            "stromal_score": np.linspace(0.2, 0.9, 8),
        }
    )
    monkeypatch.setattr(apc, "expression_colonomics_frame", lambda _cpgs: frame)
    monkeypatch.setattr(apc, "expression_colocare_frame", lambda _cpgs: (frame, pd.DataFrame()))

    rows, _ = apc.expression_rows(cpgs, n_boot=20)
    absent = rows.loc[rows["cpg"].eq("cg_absent")]

    assert len(rows) == 6
    assert absent["status"].eq("probe_absent").all()
    assert absent["q_BH77"].isna().all()
