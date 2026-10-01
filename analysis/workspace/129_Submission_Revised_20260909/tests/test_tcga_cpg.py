from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_tcga_cpg as tcga  # noqa: E402


def test_bh_fixed_uses_77_denominator_and_retains_na():
    q = tcga.bh_fixed([0.001, math.nan, 0.02])

    assert q[0] == pytest.approx(0.077)
    assert math.isnan(q[1])
    assert q[2] == pytest.approx(0.77)


def test_stream_fixed_probe_values_reads_only_targets_from_zip(tmp_path):
    archive = tmp_path / "beta.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("f1.txt", "cg_a\t0.1\ncg_x\t0.9\ncg_b\t0.2\n")
        zf.writestr("f2.txt", "cg_a\t0.3\ncg_b\t0.4\n")
    sample_map = pd.DataFrame(
        [
            {"file_id": "f1", "archive_member": "f1.txt", "case": "C1", "sample": "S1", "project": "TCGA-X", "sample_type": "Primary Tumor", "tissue": "T"},
            {"file_id": "f2", "archive_member": "f2.txt", "case": "C1", "sample": "S2", "project": "TCGA-X", "sample_type": "Solid Tissue Normal", "tissue": "N"},
        ]
    )
    cpgs = pd.DataFrame(
        [
            {"gene": "G1", "cpg": "cg_a", "fixed_probe_order_within_gene": 1},
            {"gene": "G1", "cpg": "cg_b", "fixed_probe_order_within_gene": 2},
        ]
    )

    out = tcga.stream_fixed_probe_values(archive, sample_map, cpgs)

    assert len(out) == 4
    assert set(out["cpg"]) == {"cg_a", "cg_b"}
    assert "cg_x" not in set(out["cpg"])
    assert out.loc[out["file_id"].eq("f1") & out["cpg"].eq("cg_b"), "beta"].iloc[0] == pytest.approx(0.2)


def test_paired_cpg_stats_keeps_one_row_per_cpg_and_bh77():
    cpgs = pd.DataFrame(
        [
            {"gene": "G1", "cpg": "cg_a", "fixed_probe_order_within_gene": 1},
            {"gene": "G1", "cpg": "cg_b", "fixed_probe_order_within_gene": 2},
        ]
    )
    collapsed = pd.DataFrame(
        [
            {"case": "C1", "project": "TCGA-X", "tissue": "T", "gene": "G1", "cpg": "cg_a", "fixed_probe_order_within_gene": 1, "beta": 0.8},
            {"case": "C1", "project": "TCGA-X", "tissue": "N", "gene": "G1", "cpg": "cg_a", "fixed_probe_order_within_gene": 1, "beta": 0.2},
            {"case": "C2", "project": "TCGA-X", "tissue": "T", "gene": "G1", "cpg": "cg_a", "fixed_probe_order_within_gene": 1, "beta": 0.6},
            {"case": "C2", "project": "TCGA-X", "tissue": "N", "gene": "G1", "cpg": "cg_a", "fixed_probe_order_within_gene": 1, "beta": 0.4},
            {"case": "C1", "project": "TCGA-X", "tissue": "T", "gene": "G1", "cpg": "cg_b", "fixed_probe_order_within_gene": 2, "beta": 0.5},
        ]
    )

    out = tcga.paired_cpg_stats(collapsed, cpgs)

    assert len(out) == 2
    assert out.loc[out["cpg"].eq("cg_a"), "n_pairs"].iloc[0] == 2
    assert out.loc[out["cpg"].eq("cg_a"), "mean_delta_beta"].iloc[0] == pytest.approx(0.4)
    assert out.loc[out["cpg"].eq("cg_b"), "status"].iloc[0] == "not_estimable"


def test_expression_gap_rows_retains_77_family_shape():
    cpgs = pd.DataFrame([{"gene": "G1", "cpg": f"cg_{i}", "fixed_probe_order_within_gene": i} for i in range(3)])

    out = tcga.expression_gap_rows(cpgs, [])

    assert len(out) == 6
    assert set(out["scope"]) == {"tumor_only", "pooled_tumor_normal"}
    assert out["status"].eq("source_unavailable").all()
    assert out["q_BH77"].isna().all()
