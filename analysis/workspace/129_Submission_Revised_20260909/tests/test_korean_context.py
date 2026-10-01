from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_korean_context as akc  # noqa: E402


GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "GFRA1", "UNC5C", "RALYL", "BEND5"]
COUNTS = {"EYA4": 17, "ZNF568": 8, "ZNF793": 8, "SFMBT2": 8, "ADHFE1": 8, "HOXA2": 8, "GFRA1": 6, "UNC5C": 5, "RALYL": 5, "BEND5": 4}


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    probes = {gene: [f"cg{gene}_{i:02d}" for i in range(count)] for gene, count in COUNTS.items()}
    probes["EYA4"][0] = "cg_missing"
    fixed = tmp_path / "fixed_probes.json"
    fixed.write_text(json.dumps(probes))
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"seed": 123, "n_boot": 20, "secondary_context_family": 77, "paired_family": 77}))

    samples = []
    meth_rows = []
    rna_rows = []
    cms_rows = []
    cms_labels = ["CMS2", "CMS2", "CMS3", "CMS3", "CMS1", "CMS4"]
    for i, cms in enumerate(cms_labels):
        sample = f"P{i}-T-A1"
        samples.append(sample)
        meth_rows.append({"cohort": "toy", "patient_id": f"P{i}", "sample_id": sample, "tissue": "T", "pair_verified": "true"})
        rna_rows.append(
            {
                "cohort": "toy",
                "patient_id": f"P{i}",
                "sample_id": sample,
                "rna_sample_id": sample,
                "tissue": "tumor",
                "sex": "M" if i % 2 else "F",
                "age": str(50 + i),
                "site": ["A", "B", "C"][i % 3],
                "MSI": "MSS",
                "sample_in_log2cpm_342": "True",
                "methylation_sample_id_candidate": sample,
            }
        )
        cms_rows.append(
            {
                "cohort": "toy",
                "patient_id": f"P{i}",
                "sample_id": sample,
                "rna_sample_id": sample,
                "CMS_source": cms,
                "CMS_label_status": "assigned",
            }
        )

    beta_rows = {}
    for gene, gene_probes in probes.items():
        for probe in gene_probes:
            if probe == "cg_missing":
                continue
            beta_rows[probe] = np.linspace(0.2, 0.7, len(samples)) + GENES.index(gene) * 0.001
    beta = pd.DataFrame(beta_rows, index=samples).T
    beta_path = tmp_path / "beta.tsv"
    beta.to_csv(beta_path, sep="\t")
    meth_path = tmp_path / "meth.tsv"
    pd.DataFrame(meth_rows).to_csv(meth_path, sep="\t", index=False)
    rna_path = tmp_path / "rna.tsv"
    pd.DataFrame(rna_rows).to_csv(rna_path, sep="\t", index=False)
    cms_path = tmp_path / "cms.tsv"
    pd.DataFrame(cms_rows).to_csv(cms_path, sep="\t", index=False)
    expression = pd.DataFrame(
        {sample: np.linspace(10, 5, len(GENES)) + i * 0.1 for i, sample in enumerate(samples)},
        index=GENES,
    )
    expr_path = tmp_path / "expr.tsv"
    expression.to_csv(expr_path, sep="\t")
    return {
        "beta": beta_path,
        "meth": meth_path,
        "rna": rna_path,
        "cms": cms_path,
        "expr": expr_path,
        "contract": contract,
        "fixed": fixed,
    }


def test_per_cpg_outputs_preserve_planned_rows_without_gene_panel_means(tmp_path):
    paths = _write_fixture(tmp_path)
    outdir = tmp_path / "out"
    outputs = akc.run_context(
        beta_path=paths["beta"],
        methylation_meta_path=paths["meth"],
        rna_meta_path=paths["rna"],
        cms_path=paths["cms"],
        expression_path=paths["expr"],
        contract_path=paths["contract"],
        fixed_probes_path=paths["fixed"],
        outdir=outdir,
        review_path=tmp_path / "gate.md",
        n_boot=12,
    )

    cms = pd.read_csv(outputs["cms_cpg_contrasts"], sep="\t")
    expr = pd.read_csv(outputs["expression_cpg_associations"], sep="\t")
    settings = json.loads(outputs["settings"].read_text())
    assert len(cms) == 77
    assert len(expr) == 77
    assert settings["analysis_unit"] == "individual_CpG"
    assert settings["no_gene_or_panel_methylation_averages"] is True
    assert not any(col in cms.columns for col in ["gene_score", "panel_mean", "methylation_mean"])

    missing = cms.loc[cms["cpg"].eq("cg_missing")].iloc[0]
    assert missing["status"] == "missing_cpg"
    assert pd.isna(missing["welch_p"])
    assert pd.isna(missing["welch_bh_q"])
    assert cms["welch_bh_q"].notna().sum() == 76


def test_whitespace_stripping_can_create_duplicate_key_and_is_rejected(tmp_path):
    paths = _write_fixture(tmp_path)
    meth = pd.read_csv(paths["meth"], sep="\t")
    duplicate = meth.iloc[[0]].copy()
    duplicate["sample_id"] = duplicate["sample_id"].iloc[0] + " "
    meth = pd.concat([meth, duplicate], ignore_index=True)
    meth.to_csv(paths["meth"], sep="\t", index=False)

    targets = akc.read_fixed_targets(paths["fixed"])
    beta, _ = akc.load_beta_cpgs(paths["beta"], targets)
    with pytest.raises(ValueError, match="duplicate keys after stripping"):
        akc.load_context_inputs(paths["meth"], paths["rna"], paths["cms"], paths["expr"], beta.index)


def test_expression_uses_each_cpg_mapped_gene_rna_row(tmp_path):
    paths = _write_fixture(tmp_path)
    targets = akc.read_fixed_targets(paths["fixed"])
    beta, _ = akc.load_beta_cpgs(paths["beta"], targets)
    meth, rna, cms, expression = akc.load_context_inputs(paths["meth"], paths["rna"], paths["cms"], paths["expr"], beta.index)
    tumor, _ = akc.methylation_context_frame(beta, targets, meth, rna, cms)
    result = akc.expression_cpg_tests(tumor, targets, expression, akc.read_contract(paths["contract"]))

    by_cpg = result.set_index("cpg")
    assert by_cpg.loc["cgEYA4_01", "rna_gene"] == "EYA4"
    assert by_cpg.loc["cgZNF568_00", "rna_gene"] == "ZNF568"
    assert by_cpg.loc["cg_missing", "status"] == "missing_cpg"


def test_cli_rejects_nonpositive_bootstrap(tmp_path):
    paths = _write_fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_korean_context.py"),
            "--beta",
            str(paths["beta"]),
            "--methylation-metadata",
            str(paths["meth"]),
            "--rna-metadata",
            str(paths["rna"]),
            "--cms",
            str(paths["cms"]),
            "--expression",
            str(paths["expr"]),
            "--contract",
            str(paths["contract"]),
            "--fixed-probes",
            str(paths["fixed"]),
            "--outdir",
            str(tmp_path / "out"),
            "--n-boot",
            "0",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "n_boot must be positive" in result.stderr


def test_partial_spearman_uses_pearson_rank_residuals_and_correct_df():
    beta = np.array([1, 2, 4, 3, 7, 8, 5, 6], dtype=float)
    expression = np.array([8, 1, 7, 2, 4, 3, 6, 5], dtype=float)
    raw_age = np.array([40, 41, 42, 43, 44, 47, 48, 52], dtype=float)
    covariates = np.column_stack([np.ones(len(beta)), raw_age])

    observed_rho, observed_p = akc._partial_spearman_from_arrays(beta, expression, covariates)

    beta_rank = stats.rankdata(beta, method="average")
    expression_rank = stats.rankdata(expression, method="average")
    beta_resid = beta_rank - covariates @ np.linalg.lstsq(covariates, beta_rank, rcond=None)[0]
    expression_resid = expression_rank - covariates @ np.linalg.lstsq(covariates, expression_rank, rcond=None)[0]
    expected_rho = np.corrcoef(beta_resid, expression_resid)[0, 1]
    df = len(beta) - np.linalg.matrix_rank(covariates) - 1
    expected_t = expected_rho * np.sqrt(df / (1 - expected_rho**2))
    expected_p = 2 * stats.t.sf(abs(expected_t), df)
    wrong_double_rank_rho = stats.spearmanr(beta_resid, expression_resid).statistic

    assert observed_rho == pytest.approx(expected_rho)
    assert observed_p == pytest.approx(expected_p)
    assert observed_rho != pytest.approx(wrong_double_rank_rho)
