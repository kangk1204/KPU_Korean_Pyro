"""Paired methylation, outcome-blind patterns, and fixed cutoff analyses."""
from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from common import GENES, N_BOOT, ROOT, SEED, bh, ensure_dirs, write_json


RULES = [
    "normal_q95_higher",
    "normal_q975_higher",
    "2.5x_normal_mean",
    "3x_normal_mean",
    "normal_mean_plus_4sd",
    "tumor_normal_youden",
]
N_CV_REPEATS = 20


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(ROOT / "data/derived/clinical.tsv", sep="\t")
    wide = pd.read_csv(ROOT / "data/derived/methylation_wide.tsv", sep="\t")
    legacy = pd.read_csv(ROOT / "qc/legacy_calls.tsv", sep="\t")
    assert len(clinical) == 87
    assert clinical["patient_id"].is_unique
    assert list(wide["patient_id"]) == list(clinical["patient_id"])
    assert list(legacy["patient_id"]) == list(clinical["patient_id"])
    tumor = wide[[f"T_{g}" for g in GENES]].copy()
    tumor.columns = GENES
    normal = wide[[f"N_{g}" for g in GENES]].copy()
    normal.columns = GENES
    for frame in (tumor, normal):
        assert frame.notna().all().all()
        assert frame.ge(0).all().all() and frame.le(100).all().all()
    assert legacy[GENES].isin([0, 1]).all().all()
    return clinical, tumor, normal, legacy[GENES].astype(int)


def _bca_mean_ci(values: np.ndarray, boot_idx: np.ndarray) -> dict[str, float | int]:
    n = len(values)
    theta_hat = float(np.mean(values))
    assert boot_idx.shape == (N_BOOT, n)
    boot = values[boot_idx].mean(axis=1)
    jack = np.array([np.delete(values, i).mean() for i in range(n)], dtype=float)
    jack_bar = jack.mean()
    num = np.sum((jack_bar - jack) ** 3)
    den = 6.0 * (np.sum((jack_bar - jack) ** 2) ** 1.5)
    accel = float(num / den) if den > 0 else 0.0
    prop = (np.sum(boot < theta_hat) + 0.5 * np.sum(boot == theta_hat)) / N_BOOT
    prop = min(max(float(prop), 1.0 / (2 * N_BOOT)), 1 - 1.0 / (2 * N_BOOT))
    z0 = float(stats.norm.ppf(prop))
    adjusted = []
    for alpha in (0.025, 0.975):
        z_alpha = stats.norm.ppf(alpha)
        denom = 1 - accel * (z0 + z_alpha)
        adjusted_alpha = stats.norm.cdf(z0 + (z0 + z_alpha) / denom) if denom != 0 else alpha
        adjusted.append(float(min(max(adjusted_alpha, 0.0), 1.0)))
    low, high = np.quantile(boot, adjusted, method="linear")
    return {
        "mean_diff_bca95_low": float(low),
        "mean_diff_bca95_high": float(high),
        "bca_z0": z0,
        "bca_acceleration": accel,
        "bootstrap_replicates": int(N_BOOT),
    }


def _fmt_iqr(values: np.ndarray) -> tuple[float, float]:
    q1, q3 = np.quantile(values, [0.25, 0.75], method="linear")
    return float(q1), float(q3)


def _paired_summary(clinical: pd.DataFrame, tumor: pd.DataFrame, normal: pd.DataFrame) -> pd.DataFrame:
    boot_idx = np.random.default_rng(SEED).integers(0, len(clinical), size=(N_BOOT, len(clinical)))
    rows = []
    p_t, p_w = [], []
    for gene in GENES:
        t = tumor[gene].to_numpy(float)
        n = normal[gene].to_numpy(float)
        d = t - n
        t_iqr = _fmt_iqr(t)
        n_iqr = _fmt_iqr(n)
        d_iqr = _fmt_iqr(d)
        ttest = stats.ttest_rel(t, n)
        try:
            wil = stats.wilcoxon(t, n, zero_method="wilcox", alternative="two-sided")
            wil_p = float(wil.pvalue)
        except ValueError:
            wil_p = math.nan
        p_t.append(float(ttest.pvalue))
        p_w.append(wil_p)
        row = {
            "gene": gene,
            "n_pairs": int(len(d)),
            "tumor_mean_pct": float(np.mean(t)),
            "tumor_sd_pct": float(np.std(t, ddof=1)),
            "tumor_median_pct": float(np.median(t)),
            "tumor_q1_pct": t_iqr[0],
            "tumor_q3_pct": t_iqr[1],
            "normal_mean_pct": float(np.mean(n)),
            "normal_sd_pct": float(np.std(n, ddof=1)),
            "normal_median_pct": float(np.median(n)),
            "normal_q1_pct": n_iqr[0],
            "normal_q3_pct": n_iqr[1],
            "mean_difference_pp": float(np.mean(d)),
            "sd_difference_pp": float(np.std(d, ddof=1)),
            "median_difference_pp": float(np.median(d)),
            "difference_q1_pp": d_iqr[0],
            "difference_q3_pp": d_iqr[1],
            "tumor_greater_own_normal_n": int(np.sum(t > n)),
            "tumor_greater_own_normal_fraction": float(np.mean(t > n)),
            "paired_t_statistic": float(ttest.statistic),
            "paired_t_p": float(ttest.pvalue),
            "wilcoxon_p": wil_p,
        }
        row.update(_bca_mean_ci(d, boot_idx))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["paired_t_BH_q"] = bh(p_t)
    out["wilcoxon_BH_q"] = bh(p_w)
    return out


def _paired_loo(clinical: pd.DataFrame, tumor: pd.DataFrame, normal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ids = clinical["patient_id"].to_numpy(int)
    study_ids = clinical["study_id"].to_numpy(str)
    for gene in GENES:
        d = tumor[gene].to_numpy(float) - normal[gene].to_numpy(float)
        full = float(np.mean(d))
        for i, patient_id in enumerate(ids):
            without = float(np.delete(d, i).mean())
            rows.append(
                {
                    "gene": gene,
                    "patient_id": int(patient_id),
                    "study_id": study_ids[i],
                    "left_out_difference_pp": float(d[i]),
                    "full_mean_difference_pp": full,
                    "loo_mean_difference_pp": without,
                    "loo_minus_full_pp": without - full,
                    "absolute_loo_minus_full_pp": abs(without - full),
                }
            )
    return pd.DataFrame(rows)


def _patterns(tumor: pd.DataFrame, normal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    delta = tumor[GENES].to_numpy(float) - normal[GENES].to_numpy(float)
    corr_rows = []
    pvals = []
    for i, gene_a in enumerate(GENES):
        for gene_b in GENES[i + 1 :]:
            rho, p = stats.spearmanr(delta[:, i], delta[:, GENES.index(gene_b)])
            pvals.append(float(p))
            corr_rows.append(
                {
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "n_pairs": int(delta.shape[0]),
                }
            )
    corr = pd.DataFrame(corr_rows)
    corr["spearman_BH_q"] = bh(pvals)

    scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(delta)
    pca = PCA(n_components=2, random_state=SEED)
    scores = pca.fit_transform(scaled)
    loadings = pca.components_.T.copy()
    for comp in range(2):
        anchor = int(np.argmax(np.abs(loadings[:, comp])))
        if loadings[anchor, comp] < 0:
            loadings[:, comp] *= -1
            scores[:, comp] *= -1
    score_df = pd.DataFrame(
        {
            "patient_id": tumor.index.to_numpy(int),
            "study_id": [f"P{i:03d}" for i in range(1, delta.shape[0] + 1)],
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
        }
    )
    loading_df = pd.DataFrame(
        {
            "gene": GENES,
            "PC1_loading": loadings[:, 0],
            "PC2_loading": loadings[:, 1],
        }
    )
    summary = {
        "n_pairs": int(delta.shape[0]),
        "n_genes": int(delta.shape[1]),
        "spearman_pairs": int(len(corr)),
        "pca_explained_variance_ratio": {
            "PC1": float(pca.explained_variance_ratio_[0]),
            "PC2": float(pca.explained_variance_ratio_[1]),
        },
        "pca_standardization": "tumor-minus-normal differences standardized by gene with sklearn StandardScaler",
        "pca_sign_rule": "each component flipped so the largest absolute loading is positive",
        "correlation_multiplicity_family": "45 pairwise Spearman correlations among fixed genes",
    }
    return corr, score_df, loading_df, summary


def _youden_threshold(tumor: np.ndarray, normal: np.ndarray) -> float:
    unique = np.unique(np.concatenate([tumor, normal]))
    candidates = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1] + unique[1:]) / 2, unique[-1]]
    sens = (tumor[:, None] > candidates).mean(axis=0)
    spec = (normal[:, None] <= candidates).mean(axis=0)
    j = sens + spec - 1
    return float(candidates[np.flatnonzero(np.isclose(j, np.max(j), atol=1e-12, rtol=0))[-1]])


def _thresholds(tumor: np.ndarray, normal: np.ndarray) -> dict[str, float]:
    return {
        "normal_q95_higher": float(np.quantile(normal, 0.95, method="higher")),
        "normal_q975_higher": float(np.quantile(normal, 0.975, method="higher")),
        "2.5x_normal_mean": float(2.5 * np.mean(normal)),
        "3x_normal_mean": float(3.0 * np.mean(normal)),
        "normal_mean_plus_4sd": float(np.mean(normal) + 4.0 * np.std(normal, ddof=1)),
        "tumor_normal_youden": _youden_threshold(tumor, normal),
    }


def _cutoff_bootstrap(gene: str, tumor: np.ndarray, normal: np.ndarray, boot_idx: np.ndarray) -> list[dict]:
    n = len(normal)
    assert boot_idx.shape == (N_BOOT, n)
    rows = []
    values_by_rule = {rule: np.empty(N_BOOT, dtype=float) for rule in RULES}
    for b, draw in enumerate(boot_idx):
        th = _thresholds(tumor[draw], normal[draw])
        for rule in RULES:
            values_by_rule[rule][b] = th[rule]
    for rule, values in values_by_rule.items():
        low, high = np.quantile(values, [0.025, 0.975], method="linear")
        rows.append(
            {
                "gene": gene,
                "rule": rule,
                "bootstrap_replicates": int(N_BOOT),
                "cutoff_bootstrap_mean_pct": float(np.mean(values)),
                "cutoff_bootstrap_sd_pct": float(np.std(values, ddof=1)),
                "cutoff_bootstrap_95ci_low_pct": float(low),
                "cutoff_bootstrap_95ci_high_pct": float(high),
                "n_unique_bootstrap_cutoffs": int(len(np.unique(values))),
            }
        )
    return rows


def _cutoffs(
    clinical: pd.DataFrame, tumor: pd.DataFrame, normal: pd.DataFrame, legacy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = clinical["patient_id"].to_numpy(int)
    study_ids = clinical["study_id"].to_numpy(str)
    boot_idx = np.random.default_rng(SEED).integers(0, len(ids), size=(N_BOOT, len(ids)))
    full_rows = []
    boot_rows = []
    loo_rows = []
    cv_rows = []
    call_df = clinical[["patient_id", "study_id"]].copy()
    rkf = RepeatedKFold(n_splits=5, n_repeats=N_CV_REPEATS, random_state=SEED)
    splits = list(rkf.split(np.arange(len(ids))))

    for gene in GENES:
        t = tumor[gene].to_numpy(float)
        n = normal[gene].to_numpy(float)
        old = legacy[gene].to_numpy(int)
        full = _thresholds(t, n)
        call_df[f"{gene}_tumor_pct"] = t
        call_df[f"{gene}_normal_pct"] = n
        call_df[f"{gene}_stored_call"] = old
        for rule, cutoff in full.items():
            tumor_pos = t > cutoff
            normal_pos = n > cutoff
            call_df[f"{gene}_{rule}_call"] = tumor_pos.astype(int)
            full_rows.append(
                {
                    "gene": gene,
                    "rule": rule,
                    "cutoff_pct": float(cutoff),
                    "n_pairs": int(len(t)),
                    "tumor_positive_n": int(np.sum(tumor_pos)),
                    "tumor_positive_fraction": float(np.mean(tumor_pos)),
                    "normal_positive_n": int(np.sum(normal_pos)),
                    "normal_positive_fraction": float(np.mean(normal_pos)),
                    "normal_negative_fraction": float(1.0 - np.mean(normal_pos)),
                    "stored_tumor_call_disagreement_n": int(np.sum(tumor_pos != old)),
                }
            )
        boot_rows.extend(_cutoff_bootstrap(gene, t, n, boot_idx))

        for i, patient_id in enumerate(ids):
            train = np.arange(len(ids)) != i
            th = _thresholds(t[train], n[train])
            for rule, cutoff in th.items():
                full_call = t > full[rule]
                loo_call = t > cutoff
                loo_rows.append(
                    {
                        "gene": gene,
                        "rule": rule,
                        "left_out_patient_id": int(patient_id),
                        "left_out_study_id": study_ids[i],
                        "left_out_normal_pct": float(n[i]),
                        "left_out_tumor_pct": float(t[i]),
                        "full_cutoff_pct": float(full[rule]),
                        "loo_cutoff_pct": float(cutoff),
                        "loo_minus_full_cutoff_pct": float(cutoff - full[rule]),
                        "changed_tumor_calls_on_same_87": int(np.sum(full_call != loo_call)),
                    }
                )

        coverage = np.zeros((N_CV_REPEATS, len(ids)), dtype=int)
        for split_number, (train_idx, test_idx) in enumerate(splits):
            repeat = split_number // 5
            fold = split_number % 5
            coverage[repeat, test_idx] += 1
            th = _thresholds(t[train_idx], n[train_idx])
            for rule, cutoff in th.items():
                test_t_pos = t[test_idx] > cutoff
                test_n_pos = n[test_idx] > cutoff
                cv_rows.append(
                    {
                        "gene": gene,
                        "rule": rule,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "cutoff_pct": float(cutoff),
                        "n_train_pairs": int(len(train_idx)),
                        "n_test_pairs": int(len(test_idx)),
                        "test_tumor_positive_n": int(np.sum(test_t_pos)),
                        "test_normal_positive_n": int(np.sum(test_n_pos)),
                        "test_tumor_positive_fraction": float(np.mean(test_t_pos)),
                        "test_normal_negative_fraction": float(1.0 - np.mean(test_n_pos)),
                    }
                )
        assert np.all(coverage == 1)

    return (
        pd.DataFrame(full_rows),
        pd.DataFrame(boot_rows),
        pd.DataFrame(loo_rows),
        pd.DataFrame(cv_rows),
        call_df,
    )


def _write_outputs() -> dict:
    ensure_dirs()
    clinical, tumor, normal, legacy = _read_inputs()
    tumor.index = clinical["patient_id"].to_numpy(int)
    normal.index = clinical["patient_id"].to_numpy(int)
    legacy.index = clinical["patient_id"].to_numpy(int)

    paired = _paired_summary(clinical, tumor, normal)
    paired_loo = _paired_loo(clinical, tumor, normal)
    corr, pca_scores, pca_loadings, pattern_summary = _patterns(tumor, normal)
    cutoffs, cutoff_bootstrap, cutoff_loo, cutoff_cv, cutoff_calls = _cutoffs(clinical, tumor, normal, legacy)

    result_paths = {
        "paired": ROOT / "results/paired.csv",
        "paired_loo": ROOT / "results/paired_loo.csv",
        "correlations": ROOT / "results/correlations.csv",
        "pca_scores": ROOT / "results/pca_scores.csv",
        "pca_loadings": ROOT / "results/pca_loadings.csv",
        "cutoffs": ROOT / "results/cutoffs.csv",
        "cutoff_bootstrap": ROOT / "results/cutoff_bootstrap.csv",
        "cutoff_loo": ROOT / "results/cutoff_loo.csv",
        "cutoff_cv": ROOT / "results/cutoff_cv.csv",
        "cutoff_patient_calls": ROOT / "results/cutoff_patient_calls.tsv",
    }
    paired.to_csv(result_paths["paired"], index=False)
    paired_loo.to_csv(result_paths["paired_loo"], index=False)
    corr.to_csv(result_paths["correlations"], index=False)
    pca_scores.to_csv(result_paths["pca_scores"], index=False)
    pca_loadings.to_csv(result_paths["pca_loadings"], index=False)
    cutoffs.to_csv(result_paths["cutoffs"], index=False)
    cutoff_bootstrap.to_csv(result_paths["cutoff_bootstrap"], index=False)
    cutoff_loo.to_csv(result_paths["cutoff_loo"], index=False)
    cutoff_cv.to_csv(result_paths["cutoff_cv"], index=False)
    cutoff_calls.to_csv(result_paths["cutoff_patient_calls"], sep="\t", index=False)

    paired_summary = {
        "analysis": "paired methylation, outcome-blind pattern, and fixed cutoff analysis",
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "n_patients": int(len(clinical)),
        "n_genes": int(len(GENES)),
        "gene_order": GENES,
        "paired_primary_estimand": "mean tumor-minus-normal methylation percentage-point difference",
        "paired_ci": "patient-paired BCa bootstrap 95% CI",
        "paired_p_family": "10 paired t tests; BH adjusted across genes",
        "wilcoxon_family": "10 paired Wilcoxon signed-rank tests; BH adjusted across genes",
        "cutoff_rules": RULES,
        "cutoff_positive_rule": "tumor methylation strictly greater than cutoff",
        "cutoff_bootstrap_unit": "patient pair; same draw indexes tumor and normal for Youden; normal values define normal-only rules",
        "bootstrap_indexing": "one shared patient-index bootstrap matrix generated with seed 20260905 is reused across all ten genes",
        "cutoff_bootstrap_ci": "percentile 2.5% and 97.5% tails of fixed-rule cutoff bootstrap distributions, not BCa",
        "cutoff_cv": "patient-level RepeatedKFold, 5 folds x 20 repeats, thresholds fitted on training patients only",
        "outputs": {name: str(path.relative_to(ROOT)) for name, path in result_paths.items()},
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    write_json(ROOT / "results/pattern_summary.json", pattern_summary)
    write_json(ROOT / "results/paired_summary.json", paired_summary)
    return {
        "paired_rows": int(len(paired)),
        "correlation_rows": int(len(corr)),
        "paired_loo_rows": int(len(paired_loo)),
        "cutoff_rows": int(len(cutoffs)),
        "cutoff_bootstrap_rows": int(len(cutoff_bootstrap)),
        "cutoff_loo_rows": int(len(cutoff_loo)),
        "cutoff_cv_rows": int(len(cutoff_cv)),
        "cutoff_patient_call_rows": int(len(cutoff_calls)),
        "q95_tumor_positive_n": {
            row["gene"]: int(row["tumor_positive_n"])
            for row in cutoffs.loc[cutoffs["rule"].eq("normal_q95_higher")].to_dict("records")
        },
    }


def main() -> None:
    summary = _write_outputs()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
