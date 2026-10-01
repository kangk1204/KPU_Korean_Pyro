import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_expression as ae  # noqa: E402


def test_bh_preserves_nan_and_monotone_adjusts():
    out = ae.bh([0.01, math.nan, 0.04, 0.03])
    assert out[1] != out[1]
    assert np.allclose([out[0], out[2], out[3]], [0.04, 0.05333333333333334, 0.05333333333333334])


def test_normalize_symbol_excludes_ambiguous_mappings():
    assert ae.normalize_symbol("EYA4") == "EYA4"
    assert ae.normalize_symbol("EYA4 /// EYA4") == "EYA4"
    assert ae.normalize_symbol("EYA4 /// OTHER") == ""
    assert ae.normalize_symbol("---") == ""


def test_collapse_to_genes_uses_median_without_outcome_selection():
    expr = pd.DataFrame(
        {
            "S1": [1.0, 3.0, 100.0],
            "S2": [2.0, 4.0, 200.0],
        },
        index=["p1", "p2", "p3"],
    )
    annot = pd.DataFrame({"ID": ["p1", "p2", "p3"], "normalized_symbol": ["EYA4", "EYA4", "ZNF568"]})
    genes, qc = ae.collapse_to_genes(expr, annot)
    assert genes.loc["S1", "EYA4"] == 2.0
    assert genes.loc["S2", "EYA4"] == 3.0
    assert qc.loc[qc["gene"] == "EYA4", "n_present_probesets"].iat[0] == 2


def test_partial_rank_spearman_handles_covariates():
    df = pd.DataFrame(
        {
            "meth": [1, 2, 3, 4, 5, 6, 7, 8],
            "expr": [8, 7, 6, 5, 4, 3, 2, 1],
            "age": [60, 61, 62, 63, 64, 65, 66, 67],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "site": ["Left", "Left", "Right", "Right", "Left", "Left", "Right", "Right"],
        }
    )
    r, p, n, k = ae.partial_rank_spearman(df, "meth", "expr", ["age", "sex", "site"])
    assert n == 8
    assert k >= 2
    assert r < 0
    assert 0 <= p <= 1


def test_partial_rank_spearman_uses_matrix_rank_for_degrees_of_freedom():
    df = pd.DataFrame(
        {
            "meth": [1, 2, 3, 4, 5, 6, 7, 8],
            "expr": [1, 3, 2, 5, 4, 7, 6, 8],
            "age": [60, 61, 62, 63, 64, 65, 66, 67],
            "age_duplicate": [60, 61, 62, 63, 64, 65, 66, 67],
        }
    )
    r, p, n, rank = ae.partial_rank_spearman(df, "meth", "expr", ["age", "age_duplicate"])
    dfree = n - rank - 1
    expected_t = r * math.sqrt(dfree / (1 - r * r))
    expected_p = float(2 * ae.stats.t.sf(abs(expected_t), dfree))
    assert rank == 2
    assert np.isclose(p, expected_p)


def test_collapse_verified_technical_replicates_keeps_biological_groups_separate():
    scores = pd.DataFrame(
        {
            "EYA4": [0.2, 0.4, 0.9],
            "ZNF568": [0.1, 0.3, 0.8],
            "ZNF793": [0.1, 0.3, 0.8],
            "SFMBT2": [0.1, 0.3, 0.8],
            "ADHFE1": [0.1, 0.3, 0.8],
            "HOXA2": [0.1, 0.3, 0.8],
            "BEND5": [0.1, 0.3, 0.8],
            "UNC5C": [0.1, 0.3, 0.8],
            "RALYL": [0.1, 0.3, 0.8],
            "GFRA1": [0.1, 0.3, 0.8],
            "panel_mean": [0.11, 0.31, 0.81],
        },
        index=["GSM1", "GSM2", "GSM3"],
    )
    meta = pd.DataFrame(
        {
            "sample": ["GSM1", "GSM2", "GSM3"],
            "patient": ["P1", "P1", "P1"],
            "tissue": ["T", "T", "M"],
            "age": [60, 60, 60],
            "sex": ["M", "M", "M"],
            "title": ["P1_A", "P1_A1", "P1_C"],
        }
    )
    collapsed, audit = ae.collapse_verified_technical_replicates(scores, meta, "toy")
    tumor = collapsed[(collapsed["patient"] == "P1") & (collapsed["tissue"] == "T")].iloc[0]
    mucosa = collapsed[(collapsed["patient"] == "P1") & (collapsed["tissue"] == "M")].iloc[0]
    assert tumor["n_technical_replicates"] == 2
    assert np.isclose(tumor["EYA4"], 0.3)
    assert mucosa["n_technical_replicates"] == 1
    assert set(audit["tissue"]) == {"T", "M"}
