import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_promoter as ap  # noqa: E402


def test_promoter_probe_context_is_gene_aligned_and_fixed_44():
    context = ap.ensure_probe_context()
    assert len(context) == 77
    assert context["probe"].is_unique
    promoters = context.loc[context["promoter_match"].astype(bool)]
    assert len(promoters) == 44
    assert set(promoters["fixed_gene"]) == set(ap.common.GENES)
    for _, row in promoters.iterrows():
        names = str(row["UCSC_RefGene_Name"]).split(";")
        groups = str(row["UCSC_RefGene_Group"]).split(";")
        n = max(len(names), len(groups))
        names += [""] * (n - len(names))
        groups += [""] * (n - len(groups))
        assert any(name == row["fixed_gene"] and group in ap.PROMOTER_GROUPS for name, group in zip(names, groups))


def test_promoter_score_uses_promoter_denominator_for_80_percent():
    probes = {"EYA4": [f"eya4_{i}" for i in range(5)]}
    for gene in ap.common.GENES:
        probes.setdefault(gene, [f"{gene.lower()}_0"])
    beta = pd.DataFrame(
        {
            "sample_pass": [0.1, 0.2, 0.3, 0.4, math.nan, *([0.5] * 9)],
            "sample_fail": [0.1, 0.2, 0.3, math.nan, math.nan, *([0.5] * 9)],
        },
        index=[*probes["EYA4"], *[f"{gene.lower()}_0" for gene in ap.common.GENES if gene != "EYA4"]],
    )
    scores, coverage = ap.score_beta(beta, probes, "toy")
    assert scores.loc["sample_pass", "EYA4"] == pytest.approx(0.25)
    assert np.isnan(scores.loc["sample_fail", "EYA4"])
    eya4 = coverage.loc[coverage["gene"].eq("EYA4")].iloc[0]
    assert eya4["n_promoter_fixed"] == 5
    assert eya4["minimum_valid"] == 4


def test_estimate_pair_uses_identical_complete_cases_for_full_and_promoter():
    frame = pd.DataFrame(
        {
            "promoter": [1, 2, 3, 4, 5, 6, 7, np.nan],
            "full": [1, 2, 3, 4, 5, 6, np.nan, 8],
            "expr": [8, 7, 6, 5, 4, 3, 2, 1],
            "patient": [f"P{i}" for i in range(8)],
        }
    )
    out = ap.estimate_pair(frame, "promoter", "full", "expr", [], False, ("toy",), n_boot=100)
    assert out["n_complete_case"] == 6
    assert out["promoter_status"] == "estimated"
    assert out["full_matched_status"] == "estimated"


def test_adjusted_rank_spearman_blocks_nonpositive_residual_df():
    frame = pd.DataFrame(
        {
            "meth": [1, 2, 3, 4, 5, 6],
            "expr": [6, 5, 4, 3, 2, 1],
            "c1": [0, 1, 2, 3, 4, 5],
            "c2": ["a", "b", "c", "d", "e", "f"],
        }
    )
    out = ap.residualized_rank_spearman(frame, "meth", "expr", ["c1", "c2"])
    assert out["status"] == "not_estimable"
    assert out["df_resid"] <= 0
