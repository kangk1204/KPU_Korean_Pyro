import math
import sys
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_biology as ab  # noqa: E402
import common  # noqa: E402

SEED = 20260905


def _stable_seed(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return (SEED + int(digest[:8], 16)) % (2**32)


def _bh_independent(p_values):
    p_values = np.asarray(p_values, float)
    adjusted = np.full(len(p_values), np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if len(finite) == 0:
        return adjusted
    ordered = finite[np.argsort(p_values[finite])]
    raw = p_values[ordered] * len(p_values) / np.arange(1, len(finite) + 1)
    adjusted[ordered] = np.minimum(1, np.minimum.accumulate(raw[::-1])[::-1])
    return adjusted


@lru_cache(maxsize=None)
def _fixed_probes_cached():
    return common.fixed_probes()


@lru_cache(maxsize=None)
def _metadata(cohort):
    return pd.read_csv(ROOT / f"data/derived/{cohort}_samples.tsv", sep="\t")


@lru_cache(maxsize=None)
def _independent_scores(cohort):
    beta = pd.read_csv(ROOT / f"data/derived/{cohort}_beta.tsv.gz", sep="\t", index_col=0)
    scores = pd.DataFrame(index=beta.columns)
    fixed = _fixed_probes_cached()
    for gene in common.GENES:
        targets = fixed[gene]
        values = beta.reindex(targets).T
        valid = values.notna().sum(axis=1)
        scores[gene] = values.mean(axis=1).where(valid >= math.ceil(0.8 * len(targets)))
    scores = scores[common.GENES]
    scores["panel_mean"] = scores[common.GENES].mean(axis=1).where(scores[common.GENES].notna().all(axis=1))
    return scores


@lru_cache(maxsize=None)
def _contrasts():
    return pd.read_csv(ROOT / "results/tissue_contrasts.tsv", sep="\t")


def _result_row(cohort, contrast, gene):
    rows = _contrasts()
    match = rows[(rows["cohort"].eq(cohort)) & (rows["contrast"].eq(contrast)) & (rows["gene"].eq(gene))]
    assert len(match) == 1
    return match.iloc[0]


def _paired_independent(cohort, gene, high="T", low="N"):
    meta = _metadata(cohort)
    scores = _independent_scores(cohort)
    frame = meta.set_index("sample").join(scores[[gene]]).reset_index()
    frame = frame.loc[frame["pair_verified"].astype(bool) & frame["tissue"].isin([high, low])]
    paired = frame.pivot(index="patient", columns="tissue", values=gene).reindex(columns=[high, low]).dropna()
    delta = paired[high].to_numpy() - paired[low].to_numpy()
    rng = np.random.default_rng(_stable_seed(cohort, gene, high, low))
    boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
    return {
        "n": len(delta),
        "mean_high": paired[high].mean(),
        "mean_low": paired[low].mean(),
        "effect": delta.mean(),
        "se": stats.sem(delta),
        "p": stats.ttest_1samp(delta, 0).pvalue,
        "rank_p": stats.wilcoxon(delta).pvalue if np.any(delta != 0) else 1.0,
        "ci_low": np.quantile(boot, 0.025),
        "ci_high": np.quantile(boot, 0.975),
    }


def _welch_independent(cohort, gene, high, low):
    meta = _metadata(cohort)
    scores = _independent_scores(cohort)
    frame = meta.set_index("sample").join(scores[[gene]]).reset_index()
    frame = frame.loc[frame["tissue"].isin([high, low])].dropna(subset=[gene])
    x = frame.loc[frame["tissue"].eq(high), gene].to_numpy()
    y = frame.loc[frame["tissue"].eq(low), gene].to_numpy()
    rng = np.random.default_rng(_stable_seed(cohort, gene, high, low))
    boot = (
        x[rng.integers(0, len(x), size=(5000, len(x)))].mean(axis=1)
        - y[rng.integers(0, len(y), size=(5000, len(y)))].mean(axis=1)
    )
    return {
        "n_high": len(x),
        "n_low": len(y),
        "n_patients": frame["patient"].nunique(),
        "mean_high": x.mean(),
        "mean_low": y.mean(),
        "effect": x.mean() - y.mean(),
        "se": math.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y)),
        "p": stats.ttest_ind(x, y, equal_var=False).pvalue,
        "rank_p": stats.mannwhitneyu(x, y, alternative="two-sided").pvalue,
        "ci_low": np.quantile(boot, 0.025),
        "ci_high": np.quantile(boot, 0.975),
    }


def _meta_reml_independent(effects, variances):
    y = np.asarray(effects, float)
    v = np.asarray(variances, float)

    def nll(tau2):
        weights = 1 / (v + tau2)
        mu = np.sum(weights * y) / np.sum(weights)
        return 0.5 * (np.log(v + tau2).sum() + np.log(np.sum(weights)) + np.sum(weights * (y - mu) ** 2))

    opt = optimize.minimize_scalar(nll, bounds=(0, 2), method="bounded", options={"xatol": 1e-12})
    tau2 = 0.0 if nll(0) <= opt.fun else float(opt.x)
    weights = 1 / (v + tau2)
    mu = float(np.sum(weights * y) / np.sum(weights))
    k = len(y)
    hk = max(1.0, float(np.sum(weights * (y - mu) ** 2) / (k - 1)))
    se = float(math.sqrt(hk / np.sum(weights)))
    crit = stats.t.ppf(0.975, k - 1)
    fixed_weights = 1 / v
    fixed = np.sum(fixed_weights * y) / np.sum(fixed_weights)
    q = float(np.sum(fixed_weights * (y - fixed) ** 2))
    return {
        "k": k,
        "effect": mu,
        "se": se,
        "ci_low": mu - crit * se,
        "ci_high": mu + crit * se,
        "p": 2 * stats.t.sf(abs(mu / se), k - 1),
        "tau2": tau2,
        "I2_percent": max(0.0, 100 * (q - (k - 1)) / q) if q > 0 else 0.0,
        "heterogeneity_Q": q,
        "hk_scale": hk,
    }


def _series_matrix_values(path, probes, samples):
    import csv
    import gzip

    probes = set(probes)
    samples = list(samples)
    out = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                header = next(csv.reader([next(handle)], delimiter="\t"))
                sample_to_index = {sample: header.index(sample) for sample in samples}
                for line in handle:
                    if line.startswith("!series_matrix_table_end"):
                        break
                    row = next(csv.reader([line], delimiter="\t"))
                    probe = row[0].strip('"')
                    if probe not in probes:
                        continue
                    for sample, index in sample_to_index.items():
                        out[(probe, sample)] = float(row[index])
                break
    return out


def _fixed_panel_negative_cells(accession):
    import csv
    import gzip

    wanted = set(sum(_fixed_probes_cached().values(), []))
    path = ROOT / f"data/raw/{accession}_series_matrix.txt.gz"
    negative = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                header = next(csv.reader([next(handle)], delimiter="\t"))
                samples = [sample.strip('"') for sample in header[1:]]
                for line in handle:
                    if line.startswith("!series_matrix_table_end"):
                        break
                    row = next(csv.reader([line], delimiter="\t"))
                    probe = row[0].strip('"')
                    if probe not in wanted:
                        continue
                    for sample, cell in zip(samples, row[1:]):
                        if cell and cell != "NA" and float(cell) < 0:
                            negative.append((probe, sample, float(cell)))
                break
    return negative


def _fixed_probe_map():
    probes = {"EYA4": [f"eya4_{i}" for i in range(5)]}
    for gene in common.GENES:
        probes.setdefault(gene, [f"{gene.lower()}_0"])
    return probes


def _panel_beta(values_by_probe):
    return pd.DataFrame(values_by_probe, index=["sample_pass", "sample_fail"]).T


def test_score_beta_uses_frozen_gene_probe_denominator_for_80pct(monkeypatch):
    monkeypatch.setattr(common, "fixed_probes", _fixed_probe_map)
    beta = _panel_beta(
        {
            "eya4_0": [0.10, 0.10],
            "eya4_1": [0.20, 0.20],
            "eya4_2": [0.30, 0.30],
            "eya4_3": [0.40, math.nan],
            "eya4_4": [math.nan, math.nan],
            **{f"{gene.lower()}_0": [0.50, 0.50] for gene in common.GENES if gene != "EYA4"},
        }
    )

    scores, coverage = common.score_beta(beta, "synthetic")

    assert scores.loc["sample_pass", "EYA4"] == pytest.approx(0.25)
    assert math.isnan(scores.loc["sample_fail", "EYA4"])
    eya4 = coverage.loc[coverage["gene"].eq("EYA4")].iloc[0]
    assert eya4["n_fixed"] == 5
    assert eya4["minimum_valid"] == 4


def test_panel_mean_is_available_only_when_all_10_gene_scores_exist(monkeypatch):
    monkeypatch.setattr(common, "fixed_probes", _fixed_probe_map)
    beta = _panel_beta(
        {
            "eya4_0": [0.10, 0.10],
            "eya4_1": [0.20, 0.20],
            "eya4_2": [0.30, 0.30],
            "eya4_3": [0.40, 0.40],
            "eya4_4": [0.50, 0.50],
            **{
                f"{gene.lower()}_0": [0.50, math.nan if gene == "GFRA1" else 0.50]
                for gene in common.GENES
                if gene != "EYA4"
            },
        }
    )

    scores, _ = common.score_beta(beta, "synthetic")

    assert scores.loc["sample_pass", "panel_mean"] == pytest.approx(np.mean([0.30] + [0.50] * 9))
    assert math.isnan(scores.loc["sample_fail", "panel_mean"])


def test_bh_keeps_unavailable_tests_in_declared_family_denominator():
    adjusted = common.bh([0.01, math.nan, 0.04])

    assert adjusted[0] == pytest.approx(0.03)
    assert math.isnan(adjusted[1])
    assert adjusted[2] == pytest.approx(0.06)


def test_paired_mean_contrast_is_invariant_to_sample_order():
    frame = pd.DataFrame(
        {
            "patient": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "tissue": ["T", "N", "T", "N", "T", "N"],
            "EYA4": [2.0, 1.0, 4.0, 1.5, 5.0, 3.0],
        }
    )

    forward = common.mean_contrast(frame, "EYA4", "T", "N", paired=True, n_boot=200, seed=17)
    reversed_rows = common.mean_contrast(frame.iloc[::-1], "EYA4", "T", "N", paired=True, n_boot=200, seed=17)

    assert reversed_rows["effect"] == pytest.approx(forward["effect"])
    assert reversed_rows["mean_high"] == pytest.approx(forward["mean_high"])
    assert reversed_rows["mean_low"] == pytest.approx(forward["mean_low"])


def test_paired_mean_contrast_uses_complete_patient_pairs_only():
    frame = pd.DataFrame(
        {
            "patient": ["p1", "p1", "p2", "p2", "p3", "p3", "unmatched"],
            "tissue": ["T", "N", "T", "N", "T", "N", "T"],
            "EYA4": [2.0, 1.0, 3.5, 2.0, 5.0, 3.0, 100.0],
        }
    )

    result = common.mean_contrast(frame, "EYA4", "T", "N", paired=True, n_boot=200, seed=19)

    assert result["n_pairs"] == 3
    assert result["effect"] == pytest.approx(1.5)


def test_partial_pairing_uses_patient_cluster_ols():
    frame = pd.DataFrame(
        {
            "patient": ["p1", "p1", "p2", "p2", "p3", "p4", "p5", "p6"],
            "tissue": ["T", "N", "T", "N", "T", "T", "N", "N"],
            "EYA4": [5.0, 1.0, 4.0, 2.0, 6.0, 7.0, 1.5, 2.5],
        }
    )

    result = common.mean_contrast(frame, "EYA4", "T", "N", paired=False, n_boot=200, seed=23)

    assert result["method"] == "patient_cluster_OLS"
    assert result["n_patients"] == 6


def test_partial_pairing_bootstrap_resamples_patient_rows():
    frame = pd.DataFrame(
        {
            "patient": ["p1", "p1", "p2", "p2", "p3", "p4", "p5", "p6"],
            "tissue": ["T", "N", "T", "N", "T", "N", "T", "N"],
            "EYA4": [4.0, 1.0, 5.0, 2.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    table = frame.pivot(index="patient", columns="tissue", values="EYA4").reindex(columns=["T", "N"]).to_numpy()
    rng = np.random.default_rng(29)
    sampled = table[rng.integers(0, len(table), size=(1, len(table)))]
    expected_bootstrap = float(np.nanmean(sampled[:, :, 0], axis=1)[0] - np.nanmean(sampled[:, :, 1], axis=1)[0])

    result = common.mean_contrast(frame, "EYA4", "T", "N", paired=False, n_boot=1, seed=29)

    assert result["ci_low"] == pytest.approx(expected_bootstrap)
    assert result["ci_high"] == pytest.approx(expected_bootstrap)


def test_meta_reml_identical_effects_reduce_to_fixed_effect_with_modified_hk_floor():
    result = ab.meta_reml([0.2, 0.2, 0.2], [0.01, 0.01, 0.01])

    assert result["effect"] == pytest.approx(0.2)
    assert result["tau2"] == pytest.approx(0.0)
    assert result["I2_percent"] == pytest.approx(0.0)
    assert result["hk_scale"] == pytest.approx(1.0)
    assert result["se"] == pytest.approx(math.sqrt(1 / 300))


def test_meta_reml_rejects_less_than_three_finite_effects():
    with pytest.raises(ValueError, match=">=3 finite effects"):
        ab.meta_reml([0.1, 0.2], [0.01, 0.01])


def test_colonomics_metadata_recovers_48_healthy_samples_and_92_tn_pairs():
    meta = pd.read_csv(ROOT / "data/derived/Colonomics_samples.tsv", sep="\t")
    counts = meta.groupby(["patient", "tissue"]).size().unstack(fill_value=0)

    assert int((meta["tissue"] == "H").sum()) == 48
    assert int(((counts.get("T", 0) == 1) & (counts.get("N", 0) == 1)).sum()) == 92


def test_colonomics_eya4_effect_from_raw_beta_matches_derived_gene_scores():
    fixed = common.fixed_probes()
    beta = pd.read_csv(ROOT / "data/derived/Colonomics_beta.tsv.gz", sep="\t", index_col=0)
    meta = pd.read_csv(ROOT / "data/derived/Colonomics_samples.tsv", sep="\t")
    scores = pd.read_csv(ROOT / "data/derived/Colonomics_scores.tsv", sep="\t", index_col=0)

    independent_eya4 = beta.reindex(fixed["EYA4"]).T.mean(axis=1).rename("EYA4")
    independent = meta.set_index("sample").join(independent_eya4).reset_index()
    derived = meta.set_index("sample").join(scores[["EYA4"]]).reset_index()

    independent_pairs = independent[independent.tissue.isin(["T", "N"])].pivot(
        index="patient", columns="tissue", values="EYA4"
    ).dropna()
    derived_pairs = derived[derived.tissue.isin(["T", "N"])].pivot(
        index="patient", columns="tissue", values="EYA4"
    ).dropna()

    assert float((independent_pairs["T"] - independent_pairs["N"]).mean()) == pytest.approx(
        float((derived_pairs["T"] - derived_pairs["N"]).mean())
    )


@pytest.mark.parametrize(
    ("cohort", "gene"),
    [
        ("GSE193535", "EYA4"),
        ("GSE42752", "EYA4"),
        ("GSE77718", "ADHFE1"),
    ],
)
def test_actual_paired_tn_effects_match_raw_fixed_beta_and_sample_maps(cohort, gene):
    expected = _paired_independent(cohort, gene)
    observed = _result_row(cohort, "T-N", gene)

    assert observed["method"] == "paired_t"
    assert bool(observed["paired"])
    assert observed["n_high"] == expected["n"]
    assert observed["n_low"] == expected["n"]
    assert observed["n_pairs"] == expected["n"]
    assert observed["mean_high"] == pytest.approx(expected["mean_high"])
    assert observed["mean_low"] == pytest.approx(expected["mean_low"])
    assert observed["effect"] == pytest.approx(expected["effect"])
    assert observed["se"] == pytest.approx(expected["se"])
    assert observed["p"] == pytest.approx(expected["p"])
    assert observed["rank_p"] == pytest.approx(expected["rank_p"])
    assert observed["ci_low"] == pytest.approx(expected["ci_low"])
    assert observed["ci_high"] == pytest.approx(expected["ci_high"])


@pytest.mark.parametrize("contrast", ["N-H", "A-H"])
def test_gse48684_welch_effects_match_raw_fixed_beta_and_sample_maps(contrast):
    high, low = contrast.split("-")
    expected = _welch_independent("GSE48684", "EYA4", high, low)
    observed = _result_row("GSE48684", contrast, "EYA4")

    assert observed["method"] == "Welch_t"
    assert not bool(observed["paired"])
    assert observed["n_high"] == expected["n_high"]
    assert observed["n_low"] == expected["n_low"]
    assert observed["n_patients"] == expected["n_patients"]
    assert observed["mean_high"] == pytest.approx(expected["mean_high"])
    assert observed["mean_low"] == pytest.approx(expected["mean_low"])
    assert observed["effect"] == pytest.approx(expected["effect"])
    assert observed["se"] == pytest.approx(expected["se"])
    assert observed["p"] == pytest.approx(expected["p"])
    assert observed["rank_p"] == pytest.approx(expected["rank_p"])
    assert observed["ci_low"] == pytest.approx(expected["ci_low"])
    assert observed["ci_high"] == pytest.approx(expected["ci_high"])


def test_gse77718_original_denominator_coverage_leaves_five_valid_genes():
    coverage = pd.read_csv(ROOT / "results/probe_coverage.tsv", sep="\t")
    gse = coverage.loc[coverage["cohort"].eq("GSE77718")]

    valid = set(gse.loc[gse["n_valid_scores"].gt(0), "gene"])
    invalid = set(gse.loc[gse["n_valid_scores"].eq(0), "gene"])
    assert valid == {"ZNF568", "ZNF793", "ADHFE1", "HOXA2", "UNC5C"}
    assert invalid == {"EYA4", "SFMBT2", "GFRA1", "RALYL", "BEND5"}
    assert gse.set_index("gene").loc["EYA4", "minimum_valid"] == 14
    assert gse.set_index("gene").loc["EYA4", "n_present"] == 11


def test_gse77718_raw_negative_fixed_panel_cells_are_set_missing_before_scoring():
    fixed = _fixed_probes_cached()
    negative = _fixed_panel_negative_cells("GSE77718")
    beta = pd.read_csv(ROOT / "data/derived/GSE77718_beta.tsv.gz", sep="\t", index_col=0)
    adhfe1_values = _series_matrix_values(
        ROOT / "data/raw/GSE77718_series_matrix.txt.gz",
        fixed["ADHFE1"],
        ["GSM2057632"],
    )

    assert len(negative) == 13
    assert ("cg18065361", "GSM2057632", -0.001) in negative
    assert ("cg20912169", "GSM2057632", -0.001) in negative
    assert adhfe1_values[("cg18065361", "GSM2057632")] == pytest.approx(-0.001)
    assert adhfe1_values[("cg20912169", "GSM2057632")] == pytest.approx(-0.001)
    assert int(beta.reindex(fixed["ADHFE1"])["GSM2057632"].notna().sum()) == 6


def test_tissue_contrast_bh_uses_declared_family_sizes():
    contrasts = _contrasts()

    assert contrasts.groupby("family").size().to_dict() == {
        "healthy_reference": 60,
        "lesion": 30,
        "tissue_replication": 60,
    }
    for _, family in contrasts.groupby("family"):
        observed = family["q_BH"].to_numpy(float)
        expected = _bh_independent(family["p"])
        assert np.array_equal(np.isnan(observed), np.isnan(expected))
        assert observed[np.isfinite(expected)] == pytest.approx(expected[np.isfinite(expected)])


def test_tn_meta_analysis_uses_three_or_more_paired_cohorts_in_beta_units():
    contrasts = _contrasts()
    meta = pd.read_csv(ROOT / "results/meta_analysis.tsv", sep="\t")
    observed = meta[(meta["contrast"].eq("T-N")) & (meta["gene"].eq("EYA4"))].iloc[0]
    source = contrasts[
        contrasts["contrast"].eq("T-N")
        & contrasts["gene"].eq("EYA4")
        & contrasts["status"].eq("estimated")
        & contrasts["paired"].astype(bool)
    ]
    expected = _meta_reml_independent(source["effect"], source["se"] ** 2)

    assert observed["cohorts"] == ";".join(source["cohort"])
    assert observed["k"] == 3
    assert abs(observed["effect"]) < 1
    assert observed["effect"] == pytest.approx(expected["effect"])
    assert observed["se"] == pytest.approx(expected["se"])
    assert observed["ci_low"] == pytest.approx(expected["ci_low"])
    assert observed["ci_high"] == pytest.approx(expected["ci_high"])
    assert observed["p"] == pytest.approx(expected["p"])
    assert observed["tau2"] == pytest.approx(expected["tau2"])
    assert observed["I2_percent"] == pytest.approx(expected["I2_percent"])
    assert observed["heterogeneity_Q"] == pytest.approx(expected["heterogeneity_Q"])


def test_sample_pair_mapping_counts_are_semantically_consistent():
    gse193535 = _metadata("GSE193535")
    gse42752 = _metadata("GSE42752")
    gse77718 = _metadata("GSE77718")

    assert int(gse193535["pair_verified"].astype(bool).sum()) == 108
    assert int(gse193535.groupby(["patient", "tissue"]).size().unstack(fill_value=0).eval("T == 1 and N == 1").sum()) == 54
    assert int(gse42752["pair_verified"].astype(bool).sum()) == 44
    assert int((gse42752["tissue"].eq("H") & ~gse42752["pair_verified"].astype(bool)).sum()) == 19
    assert int(gse77718["pair_verified"].astype(bool).sum()) == 190
    false_rows = gse77718.loc[~gse77718["pair_verified"].astype(bool), ["patient", "tissue"]]
    assert set(map(tuple, false_rows.to_numpy())) == {(338, "N"), (388, "T")}


def test_match_sfmbt2_unavailability_removes_panel_mean_from_context_analysis():
    coverage = pd.read_csv(ROOT / "results/probe_coverage.tsv", sep="\t")
    analysis = pd.read_csv(ROOT / "data/derived/GSE164811_analysis.tsv", sep="\t")
    context = pd.read_csv(ROOT / "results/molecular_context.tsv", sep="\t")
    match_coverage = coverage.loc[coverage["cohort"].eq("GSE164811")].set_index("gene")

    assert match_coverage.loc["SFMBT2", "n_valid_scores"] == 0
    assert int(analysis["SFMBT2"].notna().sum()) == 0
    assert int(analysis["panel_mean"].notna().sum()) == 0

    sfmbt2 = context.loc[(context["cohort"].eq("GSE164811")) & (context["gene"].eq("SFMBT2"))].iloc[0]
    panel = context.loc[(context["cohort"].eq("GSE164811")) & (context["gene"].eq("panel_mean"))].iloc[0]
    assert sfmbt2["status"] == "insufficient_data"
    assert panel["status"] == "insufficient_data"
    assert sfmbt2["n"] == 0
    assert panel["n"] == 0


def test_colonomics_context_counts_are_tumor_only_and_semantic():
    meta = _metadata("Colonomics")
    tumor = meta.loc[meta["tissue"].eq("T")]
    context = pd.read_csv(ROOT / "results/molecular_context.tsv", sep="\t")

    assert len(tumor) == 96
    assert int(tumor["stromal_score"].notna().sum()) == 94
    assert int(tumor["cms"].notna().sum()) == 83
    assert tumor["cms"].value_counts().to_dict() == {"CMS2": 29, "CMS4": 28, "CMS3": 21, "CMS1": 5}
    assert int(tumor["cms"].isna().sum()) == 13
    assert int(meta.loc[meta["tissue"].isin(["N", "H"]), "stromal_score"].notna().sum()) == 0

    stroma_rows = context.loc[context["analysis"].eq("stromal_spearman")]
    cms_rows = context.loc[context["analysis"].eq("CMS_global_Kruskal")]
    assert set(stroma_rows["n"]) == {94}
    assert set(cms_rows["n"]) == {83}


def test_gse77954_raw_m_value_is_converted_to_beta():
    fixed = _fixed_probes_cached()
    values = _series_matrix_values(
        ROOT / "data/raw/GSE77954_series_matrix.txt.gz",
        [fixed["EYA4"][0]],
        ["GSM2062364"],
    )
    m_value = values[(fixed["EYA4"][0], "GSM2062364")]
    expected_beta = 2**m_value / (1 + 2**m_value)
    beta = pd.read_csv(ROOT / "data/derived/GSE77954_beta.tsv.gz", sep="\t", index_col=0)

    assert m_value == pytest.approx(1.16915014060458)
    assert beta.loc[fixed["EYA4"][0], "GSM2062364"] == pytest.approx(expected_beta)


def test_gse77954_ta_welch_effect_matches_m_value_converted_beta():
    expected = _welch_independent("GSE77954", "EYA4", "T", "A")
    observed = _result_row("GSE77954", "T-A", "EYA4")

    assert observed["method"] == "Welch_t"
    assert observed["n_high"] == expected["n_high"]
    assert observed["n_low"] == expected["n_low"]
    assert observed["effect"] == pytest.approx(expected["effect"])
    assert observed["se"] == pytest.approx(expected["se"])
    assert observed["ci_low"] == pytest.approx(expected["ci_low"])
    assert observed["ci_high"] == pytest.approx(expected["ci_high"])
