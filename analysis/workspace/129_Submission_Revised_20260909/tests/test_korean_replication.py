from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_korean_replication as akr  # noqa: E402


GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "GFRA1", "UNC5C", "RALYL", "BEND5"]


def _fixture_registry(tmp_path: Path, n_pairs: int = 5) -> tuple[Path, Path, akr.Contract, dict[str, list[str]]]:
    probes = {
        "EYA4": [f"EYA4_{i}" for i in range(8)],
        **{gene: [f"{gene}_0"] for gene in GENES if gene != "EYA4"},
    }
    contract = akr.Contract(
        genes=GENES,
        min_valid={"EYA4": 7, **{gene: 1 for gene in GENES if gene != "EYA4"}},
        seed=123,
        n_boot=30,
        n_permutation=40,
        fixed_probe_sha256=None,
    )
    contract_path = tmp_path / "contract.json"
    fixed_path = tmp_path / "fixed_probes.json"
    contract_path.write_text(
        json.dumps(
            {
                "genes": contract.genes,
                "min_valid": contract.min_valid,
                "seed": contract.seed,
                "n_boot": contract.n_boot,
                "n_permutation": contract.n_permutation,
            }
        )
    )
    fixed_path.write_text(json.dumps(probes))
    return contract_path, fixed_path, contract, probes


def _make_beta_metadata(
    tmp_path: Path,
    probes: dict[str, list[str]],
    n_pairs: int = 5,
    eya4_missing_normal_probes: int = 0,
    drop_gene: str | None = None,
) -> tuple[Path, Path]:
    samples = []
    meta_rows = []
    for i in range(n_pairs):
        for tissue in ("T", "N"):
            sample = f"P{i}_{tissue}"
            samples.append(sample)
            meta_rows.append(
                {
                    "cohort": "toy",
                    "patient_id": f"P{i}",
                    "sample_id": sample,
                    "tissue": tissue,
                    "pair_verified": "true",
                }
            )
    rows = {}
    for gene, gene_probes in probes.items():
        if gene == drop_gene:
            for probe in gene_probes:
                rows[probe] = [math.nan] * len(samples)
            continue
        for probe in gene_probes:
            values = []
            gene_shift = (GENES.index(gene) + 1) * 0.001
            for i in range(n_pairs):
                n_value = 0.20 + 0.01 * i + gene_shift
                t_value = n_value + 0.10 + 0.005 * i
                values.extend([t_value, n_value])
            rows[probe] = values
    for probe in probes["EYA4"][:eya4_missing_normal_probes]:
        for col_index, sample in enumerate(samples):
            if sample.endswith("_N"):
                rows[probe][col_index] = math.nan
    beta = pd.DataFrame(rows, index=samples).T
    beta_path = tmp_path / "beta.tsv"
    metadata_path = tmp_path / "metadata.tsv"
    beta.to_csv(beta_path, sep="\t")
    pd.DataFrame(meta_rows).to_csv(metadata_path, sep="\t", index=False)
    return beta_path, metadata_path


def test_coverage_uses_frozen_80pct_denominator_6_of_8_fails_7_of_8_passes(tmp_path):
    _, _, contract, probes = _fixture_registry(tmp_path)
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes, eya4_missing_normal_probes=2)
    beta = akr.load_beta(beta_path)
    akr.load_metadata(metadata_path, beta.columns)
    scores, coverage = akr.score_beta(beta, probes, contract)

    normal_samples = [sample for sample in scores.index if sample.endswith("_N")]
    tumor_samples = [sample for sample in scores.index if sample.endswith("_T")]
    assert scores.loc[normal_samples, "EYA4"].isna().all()
    assert scores.loc[tumor_samples, "EYA4"].notna().all()
    eya4 = coverage.loc[coverage["gene"].eq("EYA4")].iloc[0]
    assert eya4["n_fixed"] == 8
    assert eya4["minimum_valid"] == 7

    beta_7_of_8_path, _ = _make_beta_metadata(tmp_path, probes, eya4_missing_normal_probes=1)
    scores_7_of_8, _ = akr.score_beta(akr.load_beta(beta_7_of_8_path), probes, contract)
    assert scores_7_of_8.loc[normal_samples, "EYA4"].notna().all()


def test_real_registry_thresholds_match_frozen_contract():
    contract = akr.read_contract(ROOT / "registry" / "analysis_contract.json")
    probes = akr.read_fixed_probes(ROOT / "registry" / "fixed_probes.json")

    assert sum(len(values) for values in probes.values()) == 77
    assert {gene: math.ceil(0.8 * len(probes[gene])) for gene in contract.genes} == contract.min_valid


def test_panel_is_na_when_only_nine_gene_scores_are_valid(tmp_path):
    _, _, contract, probes = _fixture_registry(tmp_path)
    beta_path, _ = _make_beta_metadata(tmp_path, probes, drop_gene="BEND5")
    scores, _ = akr.score_beta(akr.load_beta(beta_path), probes, contract)

    assert scores[GENES[:-1]].notna().all().all()
    assert scores["BEND5"].isna().all()
    assert scores["panel_mean"].isna().all()


def test_known_paired_mean_delta_is_recovered_in_percentage_points(tmp_path):
    _, _, contract, probes = _fixture_registry(tmp_path, n_pairs=4)
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes, n_pairs=4)
    beta = akr.load_beta(beta_path)
    meta = akr.load_metadata(metadata_path, beta.columns)
    scores, _ = akr.score_beta(beta, probes, contract)
    frame = meta.merge(scores.reset_index(), on="sample_id", validate="one_to_one")

    paired = akr.paired_gene_effects(frame, contract, n_boot=20, seed=77)
    eya4 = paired.loc[paired["gene"].eq("EYA4")].iloc[0]

    expected = np.mean([10.0, 10.5, 11.0, 11.5])
    assert eya4["status"] == "estimated"
    assert eya4["n_pairs"] == 4
    assert eya4["mean_delta_pp"] == pytest.approx(expected)


def test_duplicate_verified_patient_tissue_fails_pair_validation(tmp_path):
    _, _, _, probes = _fixture_registry(tmp_path)
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes)
    metadata = pd.read_csv(metadata_path, sep="\t")
    duplicate = metadata.iloc[[0]].copy()
    duplicate["sample_id"] = "extra_sample"
    metadata = pd.concat([metadata, duplicate], ignore_index=True)
    beta = pd.read_csv(beta_path, sep="\t", index_col=0)
    beta["extra_sample"] = beta.iloc[:, 0]
    metadata.to_csv(metadata_path, sep="\t", index=False)
    beta.to_csv(beta_path, sep="\t")

    with pytest.raises(ValueError, match="duplicate verified patient/tissue"):
        akr.load_metadata(metadata_path, akr.load_beta(beta_path).columns)


def test_duplicate_beta_header_is_rejected_before_pandas_mangles_it(tmp_path):
    path = tmp_path / "duplicate_header.tsv"
    path.write_text("probe\tS1\tS1\ncg1\t0.1\t0.2\n")

    with pytest.raises(ValueError, match="duplicate sample identifiers"):
        akr.load_beta(path)


def test_beta_rejects_nonnumeric_and_infinite_nonmissing_tokens(tmp_path):
    bad_text = tmp_path / "bad_text.tsv"
    bad_text.write_text("probe\tS1\ncg1\tfailed\n")
    with pytest.raises(ValueError, match="nonnumeric beta value|non-finite"):
        akr.load_beta(bad_text)

    bad_inf = tmp_path / "bad_inf.tsv"
    bad_inf.write_text("probe\tS1\ncg1\tinf\n")
    with pytest.raises(ValueError, match="non-finite"):
        akr.load_beta(bad_inf)


def test_metadata_missing_or_blank_ids_are_rejected_before_string_conversion(tmp_path):
    path = tmp_path / "metadata.tsv"
    pd.DataFrame(
        [
            {"cohort": "toy", "patient_id": "", "sample_id": "S1", "tissue": "T", "pair_verified": "false"},
            {"cohort": "toy", "patient_id": np.nan, "sample_id": "S2", "tissue": "N", "pair_verified": "false"},
        ]
    ).to_csv(path, sep="\t", index=False)

    with pytest.raises(ValueError, match="patient_id contains missing or blank"):
        akr.load_metadata(path, ["S1", "S2"])


def test_no_verified_pairs_keep_explicit_empty_delta_schema_and_na_outputs(tmp_path):
    _, _, contract, probes = _fixture_registry(tmp_path, n_pairs=1)
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes, n_pairs=1)
    meta = pd.read_csv(metadata_path, sep="\t")
    meta["pair_verified"] = "false"
    meta.to_csv(metadata_path, sep="\t", index=False)
    beta = akr.load_beta(beta_path)
    metadata = akr.load_metadata(metadata_path, beta.columns)
    scores, _ = akr.score_beta(beta, probes, contract)
    frame = metadata.merge(scores.reset_index(), on="sample_id", validate="one_to_one")

    delta = akr._paired_delta_matrix(frame, contract.genes)
    assert list(delta.columns) == ["cohort", "patient_id", *contract.genes]
    paired = akr.paired_gene_effects(frame, contract)
    correlations, summary = akr.delta_correlations(frame, contract)
    pca = akr.pca_loadings(frame, contract)
    assert len(paired) == 10
    assert paired["status"].eq("insufficient_data").all()
    assert len(correlations) == 45
    assert summary.iloc[0]["n_complete_10gene_pairs"] == 0
    assert len(pca) == 10


def test_reproducibility_with_fixed_seed(tmp_path):
    contract_path, fixed_path, _, probes = _fixture_registry(tmp_path, n_pairs=6)
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes, n_pairs=6)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    akr.run_analysis(beta_path, metadata_path, out_a, contract_path, fixed_path, n_boot=25, n_permutation=30, seed=99)
    akr.run_analysis(beta_path, metadata_path, out_b, contract_path, fixed_path, n_boot=25, n_permutation=30, seed=99)

    for name in [
        "paired_gene_effects.tsv",
        "paired_panel_effect.tsv",
        "delta_correlations.tsv",
        "coordination_summary.tsv",
        "eligible_coordination_summary.tsv",
        "pairwise_complete_delta_correlations.tsv",
        "pairwise_complete_n_matrix.tsv",
        "pca_loadings.tsv",
        "settings.json",
    ]:
        assert (out_a / name).read_text() == (out_b / name).read_text()


def test_cli_writes_expected_result_shapes(tmp_path):
    contract_path, fixed_path, _, probes = _fixture_registry(tmp_path, n_pairs=5)
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes, n_pairs=5)
    outdir = tmp_path / "cli_out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_korean_replication.py"),
            "--beta",
            str(beta_path),
            "--metadata",
            str(metadata_path),
            "--outdir",
            str(outdir),
            "--contract",
            str(contract_path),
            "--fixed-probes",
            str(fixed_path),
            "--n-boot",
            "20",
            "--n-permutation",
            "25",
            "--seed",
            "456",
        ],
        check=True,
    )

    paired = pd.read_csv(outdir / "paired_gene_effects.tsv", sep="\t")
    panel = pd.read_csv(outdir / "paired_panel_effect.tsv", sep="\t")
    corr = pd.read_csv(outdir / "delta_correlations.tsv", sep="\t")
    eligible = pd.read_csv(outdir / "eligible_coordination_summary.tsv", sep="\t")
    pairwise = pd.read_csv(outdir / "pairwise_complete_delta_correlations.tsv", sep="\t")
    n_matrix = pd.read_csv(outdir / "pairwise_complete_n_matrix.tsv", sep="\t")
    settings = json.loads((outdir / "settings.json").read_text())
    assert len(paired) == 10
    assert len(panel) == 1
    assert len(corr) == 45
    assert len(eligible) == 1
    assert len(pairwise) == 45
    assert len(n_matrix) == 10
    assert settings["n_boot"] == 20
    assert settings["n_permutation"] == 25
    assert settings["pairwise_complete_delta_correlation_rows_per_cohort"] == 45
    assert set(["beta_sha256", "metadata_sha256", "contract_sha256", "fixed_probe_sha256"]).issubset(settings)


def test_contract_fixed_probe_sha256_is_enforced(tmp_path):
    contract_path, fixed_path, _, probes = _fixture_registry(tmp_path)
    contract_data = json.loads(contract_path.read_text())
    contract_data["fixed_probe_sha256"] = "bad"
    contract_path.write_text(json.dumps(contract_data))
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        akr.run_analysis(beta_path, metadata_path, tmp_path / "out", contract_path, fixed_path)


def test_cli_rejects_nonpositive_bootstrap_and_permutation_counts(tmp_path):
    contract_path, fixed_path, _, probes = _fixture_registry(tmp_path)
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_korean_replication.py"),
            "--beta",
            str(beta_path),
            "--metadata",
            str(metadata_path),
            "--outdir",
            str(tmp_path / "bad"),
            "--contract",
            str(contract_path),
            "--fixed-probes",
            str(fixed_path),
            "--n-boot",
            "0",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "n_boot must be positive" in result.stderr


def test_bca_bias_correction_uses_midrank_ties():
    delta = np.array([1.0, 1.0, 2.0, 5.0])
    n_boot = 200
    seed = 22
    rng = np.random.default_rng(seed)
    lo, hi, method, valid = akr._bca_mean_ci(delta, rng, n_boot)

    rng = np.random.default_rng(seed)
    theta = float(np.mean(delta))
    boot = delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))].mean(axis=1)
    less = np.mean(boot < theta) + 0.5 * np.mean(boot == theta)
    less = min(max(less, 1 / (2 * len(boot))), 1 - 1 / (2 * len(boot)))
    z0 = stats.norm.ppf(less)
    jack = np.array([(np.sum(delta) - x) / (len(delta) - 1) for x in delta])
    centered = jack.mean() - jack
    accel = float(np.sum(centered**3) / (6.0 * np.sum(centered**2) ** 1.5))
    z_alpha = stats.norm.ppf([0.025, 0.975])
    adjusted = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - accel * (z0 + z_alpha)))
    expected = np.quantile(boot, adjusted)

    assert method == "bca"
    assert valid == n_boot
    assert (lo, hi) == pytest.approx(tuple(expected))


def test_fast_rank_spearman_matches_scipy_with_ties():
    values = np.array(
        [
            [1, 1, 3],
            [2, 1, 2],
            [2, 3, 2],
            [4, 4, 1],
            [5, 4, 1],
        ],
        dtype=float,
    )

    observed = akr._spearman_matrix(values)
    expected = np.array(
        [
            [stats.spearmanr(values[:, i], values[:, j]).statistic for j in range(values.shape[1])]
            for i in range(values.shape[1])
        ]
    )

    assert observed == pytest.approx(expected)


def test_coordination_bootstrap_reranks_raw_resampled_vectors_with_ties():
    genes = ["G1", "G2", "G3", "G4"]
    contract = akr.Contract(
        genes=genes,
        min_valid={gene: 1 for gene in genes},
        seed=918,
        n_boot=80,
        n_permutation=12,
        fixed_probe_sha256=None,
    )
    complete = pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(7)],
            "G1": [0.0, 1.0, 1.0, 4.0, 6.0, 7.0, 11.0],
            "G2": [3.0, 3.0, 2.0, 8.0, 8.0, 13.0, 21.0],
            "G3": [5.0, 1.0, 4.0, 4.0, 10.0, 9.0, 9.0],
            "G4": [2.0, 2.0, 3.0, 5.0, 5.0, 8.0, 14.0],
        }
    )

    summary = akr._coordination_summary("toy", complete, contract)
    matrix = complete[genes].to_numpy(dtype=float)
    expected_observed = []
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            expected_observed.append(stats.spearmanr(matrix[:, i], matrix[:, j]).statistic)
    rng = np.random.default_rng(akr.stable_seed(contract.seed, "toy", "coordination"))
    expected_boot = []
    wrong_precomputed_rank_boot = []
    precomputed_ranks = akr._rank_matrix(matrix)
    for _ in range(contract.n_boot):
        indices = rng.integers(0, len(matrix), size=len(matrix))
        raw_sample = matrix[indices, :]
        rhos = []
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                rho = stats.spearmanr(raw_sample[:, i], raw_sample[:, j]).statistic
                if np.isfinite(rho):
                    rhos.append(rho)
        expected_boot.append(float(np.median(rhos)))
        wrong_values = akr._corr_values(akr._rank_corrcoef_from_ranks(precomputed_ranks[indices, :]))
        wrong_precomputed_rank_boot.append(float(np.nanmedian(wrong_values)))
    expected_ci = np.quantile(expected_boot, [0.025, 0.975])

    assert summary["median_rho"] == pytest.approx(float(np.median(expected_observed)))
    assert (summary["rho_ci_low"], summary["rho_ci_high"]) == pytest.approx(tuple(expected_ci))
    assert tuple(expected_ci) != pytest.approx(tuple(np.quantile(wrong_precomputed_rank_boot, [0.025, 0.975])))


def test_entirely_missing_gene_preserves_primary_rows_and_reports_eligible_subset(tmp_path):
    _, _, contract, probes = _fixture_registry(tmp_path, n_pairs=6)
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes, n_pairs=6, drop_gene="BEND5")
    beta = akr.load_beta(beta_path)
    meta = akr.load_metadata(metadata_path, beta.columns)
    scores, _ = akr.score_beta(beta, probes, contract)
    frame = meta.merge(scores.reset_index(), on="sample_id", validate="one_to_one")

    primary_corr, primary_summary = akr.delta_correlations(frame, contract)
    eligible = akr.eligible_gene_coordination(frame, contract)
    pairwise, n_matrix = akr.pairwise_complete_delta_correlations(frame, contract)

    assert len(primary_corr) == 45
    assert primary_corr["status"].eq("insufficient_data").all()
    assert primary_summary.iloc[0]["status"] == "insufficient_data"
    assert eligible.iloc[0]["status"] in {"estimated", "partial_estimated"}
    assert eligible.iloc[0]["n_eligible_genes"] == 9
    assert eligible.iloc[0]["n_complete_10gene_pairs"] == 0
    assert "BEND5" not in eligible.iloc[0]["eligible_genes"].split(";")
    assert len(pairwise) == 45
    assert pairwise.loc[pairwise["gene_y"].eq("BEND5"), "status"].eq("insufficient_data").all()
    bend5_row = n_matrix.loc[n_matrix["gene"].eq("BEND5")].iloc[0]
    assert bend5_row[GENES].sum() == 0


def test_uneven_missingness_keeps_primary_complete10_separate_from_pairwise_sensitivity(tmp_path):
    _, _, contract, probes = _fixture_registry(tmp_path, n_pairs=5)
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes, n_pairs=5)
    beta = pd.read_csv(beta_path, sep="\t", index_col=0)
    # Remove one single-probe gene from different patients so there are no
    # complete-ten pairs, while many pairwise-complete gene pairs remain.
    beta.loc["GFRA1_0", ["P0_T", "P0_N"]] = math.nan
    beta.loc["RALYL_0", ["P1_T", "P1_N"]] = math.nan
    beta.to_csv(beta_path, sep="\t")
    loaded = akr.load_beta(beta_path)
    meta = akr.load_metadata(metadata_path, loaded.columns)
    scores, _ = akr.score_beta(loaded, probes, contract)
    frame = meta.merge(scores.reset_index(), on="sample_id", validate="one_to_one")

    primary_corr, primary_summary = akr.delta_correlations(frame, contract)
    eligible = akr.eligible_gene_coordination(frame, contract)
    pairwise, n_matrix = akr.pairwise_complete_delta_correlations(frame, contract)

    assert len(primary_corr) == 45
    assert primary_corr["n_complete_10gene_pairs"].eq(3).all()
    assert primary_summary.iloc[0]["status"] == "estimated"
    assert eligible.iloc[0]["status"] == "not_applicable"
    assert eligible.iloc[0]["reason"] == "complete_10gene_coordination_available"
    gf_ral = pairwise.loc[
        pairwise["gene_x"].eq("GFRA1") & pairwise["gene_y"].eq("RALYL")
    ].iloc[0]
    assert gf_ral["n_pairwise_pairs"] == 3
    eya_bend = pairwise.loc[
        pairwise["gene_x"].eq("EYA4") & pairwise["gene_y"].eq("BEND5")
    ].iloc[0]
    assert eya_bend["n_pairwise_pairs"] == 5
    row = n_matrix.loc[n_matrix["gene"].eq("EYA4")].iloc[0]
    assert row["BEND5"] == 5
    assert row["GFRA1"] == 4


def test_constant_correlation_reports_partial_estimated_when_less_than_all_45_are_finite(tmp_path):
    _, _, contract, probes = _fixture_registry(tmp_path, n_pairs=5)
    beta_path, metadata_path = _make_beta_metadata(tmp_path, probes, n_pairs=5)
    beta = pd.read_csv(beta_path, sep="\t", index_col=0)
    # Make one gene's paired delta constant; the other genes retain varying deltas.
    for i in range(5):
        beta.loc["GFRA1_0", f"P{i}_T"] = beta.loc["GFRA1_0", f"P{i}_N"] + 0.1
    beta.to_csv(beta_path, sep="\t")
    loaded = akr.load_beta(beta_path)
    meta = akr.load_metadata(metadata_path, loaded.columns)
    scores, _ = akr.score_beta(loaded, probes, contract)
    frame = meta.merge(scores.reset_index(), on="sample_id", validate="one_to_one")

    primary_corr, primary_summary = akr.delta_correlations(frame, contract)

    gf_rows = primary_corr.loc[(primary_corr["gene_x"].eq("GFRA1")) | (primary_corr["gene_y"].eq("GFRA1"))]
    assert gf_rows["status"].eq("insufficient_data").all()
    assert primary_summary.iloc[0]["estimated_correlations"] < 45
    assert primary_summary.iloc[0]["status"] == "partial_estimated"
    assert primary_summary.iloc[0]["reason"] == "some_gene_pair_correlations_constant"
