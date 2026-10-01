"""Public cohort CpG-level contrast engine.

This script reruns the public array tissue contrasts at the frozen 77-CpG
level. It intentionally does not compute gene means or panel averages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "data" / "public_inputs"
DEFAULT_OUTDIR = ROOT / "results" / "public_contrasts"
DEFAULT_KOREAN_CPG_ROOT = ROOT / "results" / "individual_cpg"
DEFAULT_FIXED_PROBES = ROOT / "registry" / "fixed_probes.json"
DEFAULT_CONTRACT = ROOT / "registry" / "public_contrast_contract.json"
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
COHORTS = ["Colonomics", "GSE48684", "GSE42752", "GSE193535", "GSE77718", "GSE77954", "GSE164811"]
CONTRASTS = {
    "Colonomics": [("N", "H", "healthy_reference"), ("T", "H", "healthy_reference"), ("T", "N", "tissue_replication")],
    "GSE48684": [("N", "H", "healthy_reference"), ("T", "H", "healthy_reference"), ("T", "N", "tissue_replication"), ("A", "H", "lesion"), ("T", "A", "lesion")],
    "GSE42752": [("N", "H", "healthy_reference"), ("T", "H", "healthy_reference"), ("T", "N", "tissue_replication")],
    "GSE193535": [("T", "N", "tissue_replication")],
    "GSE77718": [("T", "N", "tissue_replication")],
    "GSE77954": [("T", "N", "tissue_replication"), ("T", "A", "lesion")],
    "GSE164811": [],
}
SEED = 20260908
N_BOOT = 5000


@dataclass(frozen=True)
class ProbeTarget:
    gene: str
    probe: str
    gene_probe_index: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return (SEED + int(digest[:8], 16)) % (2**32)


def read_fixed_probes(path: Path = DEFAULT_FIXED_PROBES) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    return {gene: list(data[gene]) for gene in GENES}


def probe_targets(fixed: dict[str, list[str]]) -> list[ProbeTarget]:
    out: list[ProbeTarget] = []
    for gene in GENES:
        for i, probe in enumerate(fixed[gene], start=1):
            out.append(ProbeTarget(gene=gene, probe=probe, gene_probe_index=i))
    if len(out) != 77:
        raise ValueError(f"fixed CpG registry must contain 77 probes, found {len(out)}")
    if len({target.probe for target in out}) != 77:
        raise ValueError("fixed CpG registry contains duplicate probes")
    return out


def bh_adjust_with_family_denominator(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return q
    ordered = finite[np.argsort(p[finite])]
    raw = p[ordered] * len(p) / np.arange(1, len(finite) + 1)
    q[ordered] = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    return q


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_cohort(cohort: str, public_root: Path = PUBLIC_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta_path = public_root / "data" / "derived" / f"{cohort}_beta.tsv.gz"
    metadata_path = public_root / "data" / "derived" / f"{cohort}_samples.tsv"
    beta = pd.read_csv(beta_path, sep="\t", index_col=0)
    meta = pd.read_csv(metadata_path, sep="\t", keep_default_na=False, na_values=["", "NaN", "nan"])
    if beta.index.duplicated().any() or beta.columns.duplicated().any():
        raise ValueError(f"{cohort}: duplicated probe or beta sample identifiers")
    if meta["sample"].duplicated().any():
        raise ValueError(f"{cohort}: duplicated sample metadata")
    if set(beta.columns) != set(meta["sample"]):
        raise ValueError(f"{cohort}: beta-metadata sample set mismatch")
    values = beta.to_numpy(dtype=float)
    if np.any(np.isfinite(values) & ((values < 0) | (values > 1))):
        raise ValueError(f"{cohort}: beta values outside [0,1]")
    meta = meta.set_index("sample").loc[beta.columns]
    meta.index.name = "sample"
    meta = meta.reset_index()
    if "pair_verified" not in meta:
        meta["pair_verified"] = False
    meta["pair_verified"] = parse_bool(meta["pair_verified"])
    for col in ["age", "stromal_score"]:
        if col in meta:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    return beta, meta


def is_paired(meta: pd.DataFrame, high: str, low: str) -> bool:
    sub = meta.loc[meta["tissue"].isin([high, low])]
    verified = sub.loc[sub["pair_verified"]]
    shared = set(verified.loc[verified["tissue"].eq(high), "patient"]) & set(
        verified.loc[verified["tissue"].eq(low), "patient"]
    )
    return high == "T" and low == "N" and len(shared) >= 3


def empty_contrast_result(method: str, status: str, reason: str) -> dict[str, object]:
    return {
        "n_high": 0,
        "n_low": 0,
        "n_patients": 0,
        "n_pairs": 0,
        "mean_high_pp": np.nan,
        "mean_low_pp": np.nan,
        "effect_pp": np.nan,
        "se_pp": np.nan,
        "ci_low_pp": np.nan,
        "ci_high_pp": np.nan,
        "p": np.nan,
        "p_for_bh": 1.0,
        "rank_p": np.nan,
        "rank_p_for_bh": 1.0,
        "method": method,
        "status": status,
        "reason": reason,
        "bootstrap_replicates": 0,
    }


def probe_contrast(
    meta: pd.DataFrame,
    values: pd.Series,
    high: str,
    low: str,
    paired: bool,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    frame = meta.loc[:, ["sample", "patient", "tissue", "pair_verified"]].copy()
    frame["beta"] = values.reindex(frame["sample"]).to_numpy(dtype=float)
    use = frame.loc[frame["pair_verified"]].copy() if paired else frame
    d = use.loc[use["tissue"].isin([high, low]), ["patient", "tissue", "beta"]].dropna()
    if d.duplicated(["patient", "tissue"]).any():
        raise ValueError("multiple measurements per patient/tissue require curated replicate handling")
    method = "paired_t" if paired else "Welch_t"
    if paired:
        wide = d.pivot(index="patient", columns="tissue", values="beta").reindex(columns=[high, low]).dropna()
        if len(wide) < 3:
            return empty_contrast_result(method, "insufficient_data", "fewer_than_3_verified_pairs")
        delta = (wide[high] - wide[low]).to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        boots = delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))].mean(axis=1)
        rank_p = float(stats.wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        effect = float(delta.mean())
        p_value = float(stats.ttest_1samp(delta, 0.0).pvalue)
        se = float(stats.sem(delta))
        mean_high = float(wide[high].mean())
        mean_low = float(wide[low].mean())
        n_high = n_low = n_pairs = n_patients = int(len(wide))
    else:
        x = d.loc[d["tissue"].eq(high), "beta"].to_numpy(dtype=float)
        y = d.loc[d["tissue"].eq(low), "beta"].to_numpy(dtype=float)
        if min(len(x), len(y)) < 3:
            return empty_contrast_result(method, "insufficient_data", "fewer_than_3_per_group")
        rng = np.random.default_rng(seed)
        overlap = set(d.loc[d["tissue"].eq(high), "patient"]) & set(d.loc[d["tissue"].eq(low), "patient"])
        if overlap:
            design = d.assign(high=(d["tissue"] == high).astype(int))
            fit = sm.OLS(design["beta"], sm.add_constant(design[["high"]])).fit(
                cov_type="cluster", cov_kwds={"groups": design["patient"]}
            )
            table = d.pivot(index="patient", columns="tissue", values="beta").reindex(columns=[high, low]).to_numpy()
            bt = table[rng.integers(0, len(table), size=(n_boot, len(table)))]
            high_bt = bt[:, :, 0]
            low_bt = bt[:, :, 1]
            high_count = np.isfinite(high_bt).sum(axis=1)
            low_count = np.isfinite(low_bt).sum(axis=1)
            high_mean = np.divide(
                np.nansum(high_bt, axis=1),
                high_count,
                out=np.full(n_boot, np.nan),
                where=high_count > 0,
            )
            low_mean = np.divide(
                np.nansum(low_bt, axis=1),
                low_count,
                out=np.full(n_boot, np.nan),
                where=low_count > 0,
            )
            boots = high_mean - low_mean
            method = "patient_cluster_OLS"
            se = float(fit.bse["high"])
            p_value = float(fit.pvalues["high"])
            rank_p = np.nan
        else:
            boots = (
                x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
                - y[rng.integers(0, len(y), size=(n_boot, len(y)))].mean(axis=1)
            )
            se = float(np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y)))
            p_value = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
            rank_p = float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
        effect = float(x.mean() - y.mean())
        mean_high = float(x.mean())
        mean_low = float(y.mean())
        n_high = int(len(x))
        n_low = int(len(y))
        n_pairs = 0
        n_patients = int(d["patient"].nunique())
    finite = boots[np.isfinite(boots)]
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return {
        "n_high": n_high,
        "n_low": n_low,
        "n_patients": n_patients,
        "n_pairs": n_pairs,
        "mean_high_pp": mean_high * 100.0,
        "mean_low_pp": mean_low * 100.0,
        "effect_pp": effect * 100.0,
        "se_pp": se * 100.0,
        "ci_low_pp": float(lo) * 100.0,
        "ci_high_pp": float(hi) * 100.0,
        "p": p_value,
        "p_for_bh": p_value if np.isfinite(p_value) else 1.0,
        "rank_p": rank_p,
        "rank_p_for_bh": rank_p if np.isfinite(rank_p) else 1.0,
        "method": method,
        "status": "estimated",
        "reason": "",
        "bootstrap_replicates": int(len(finite)),
    }


def adjusted_probe_contrast(meta: pd.DataFrame, values: pd.Series, high: str, low: str) -> dict[str, object]:
    frame = meta.copy()
    frame["beta"] = values.reindex(frame["sample"]).to_numpy(dtype=float)
    sub = frame.loc[frame["tissue"].isin([high, low])].copy()
    candidates = [
        col
        for col in ["age", "sex", "site"]
        if col in sub and sub[col].notna().sum() > 0 and sub[col].nunique(dropna=True) > 1
    ]
    ans = {
        "adjusted_effect_pp": np.nan,
        "adjusted_low_pp": np.nan,
        "adjusted_high_pp": np.nan,
        "adjusted_p": np.nan,
        "adjusted_p_for_bh": 1.0,
        "adjusted_n": 0,
        "adjusted_covariates": ";".join(candidates),
        "adjusted_status": "no_covariates",
        "adjusted_interval": "",
        "adjusted_covariance": "",
    }
    if not candidates:
        return ans
    sub = sub.dropna(subset=["beta", "patient", *candidates])
    if min((sub["tissue"] == high).sum(), (sub["tissue"] == low).sum()) < 5:
        ans["adjusted_status"] = "insufficient_complete_cases"
        return ans
    design = pd.DataFrame({"high": (sub["tissue"] == high).astype(float)}, index=sub.index)
    for col in candidates:
        if col == "age":
            design[col] = (sub[col] - sub[col].mean()) / 10.0
        else:
            design = pd.concat([design, pd.get_dummies(sub[col], prefix=col, drop_first=True, dtype=float)], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    ans["adjusted_n"] = int(len(sub))
    if np.linalg.matrix_rank(design) < design.shape[1] or len(sub) - design.shape[1] < 10:
        ans["adjusted_status"] = "rank_or_residual_df"
        return ans
    overlapping = sub["patient"].duplicated().any()
    fit = sm.OLS(sub["beta"], design).fit(
        cov_type="cluster", cov_kwds={"groups": sub["patient"]}
    ) if overlapping else sm.OLS(sub["beta"], design).fit(cov_type="HC3")
    lo, hi = fit.conf_int().loc["high"]
    p_value = float(fit.pvalues["high"])
    ans.update(
        adjusted_effect_pp=float(fit.params["high"]) * 100.0,
        adjusted_low_pp=float(lo) * 100.0,
        adjusted_high_pp=float(hi) * 100.0,
        adjusted_p=p_value,
        adjusted_p_for_bh=p_value if np.isfinite(p_value) else 1.0,
        adjusted_status="estimated",
        adjusted_interval="robust_Wald",
        adjusted_covariance="patient_cluster" if overlapping else "HC3",
    )
    return ans


def meta_reml(effect_pp: Iterable[float], variance_pp: Iterable[float]) -> dict[str, object]:
    y = np.asarray(list(effect_pp), dtype=float)
    v = np.asarray(list(variance_pp), dtype=float)
    if len(y) < 3 or not np.all(np.isfinite(y)) or not np.all(v > 0):
        raise ValueError("Meta-analysis requires >=3 finite effects with positive variances")

    def nll(tau2: float) -> float:
        weights = 1.0 / (v + tau2)
        mu = np.sum(weights * y) / np.sum(weights)
        return 0.5 * (np.log(v + tau2).sum() + np.log(np.sum(weights)) + np.sum(weights * (y - mu) ** 2))

    opt = optimize.minimize_scalar(nll, bounds=(0, 2000), method="bounded", options={"xatol": 1e-10})
    if not opt.success:
        raise RuntimeError("REML optimization failed")
    tau2 = 0.0 if nll(0.0) <= opt.fun else float(opt.x)
    weights = 1.0 / (v + tau2)
    mu = float(np.sum(weights * y) / np.sum(weights))
    k = len(y)
    hk = max(1.0, float(np.sum(weights * (y - mu) ** 2) / (k - 1)))
    se = float(np.sqrt(hk / np.sum(weights)))
    crit = stats.t.ppf(0.975, k - 1)
    fixed_weights = 1.0 / v
    fixed = np.sum(fixed_weights * y) / np.sum(fixed_weights)
    q = float(np.sum(fixed_weights * (y - fixed) ** 2))
    p_value = float(2 * stats.t.sf(abs(mu / se), k - 1))
    return {
        "k": k,
        "effect_pp": mu,
        "se_pp": se,
        "ci_low_pp": float(mu - crit * se),
        "ci_high_pp": float(mu + crit * se),
        "p": p_value,
        "p_for_bh": p_value if np.isfinite(p_value) else 1.0,
        "tau2_pp2": tau2,
        "I2_percent": max(0.0, 100.0 * (q - (k - 1)) / q) if q > 0 else 0.0,
        "heterogeneity_Q": q,
        "hk_scale": hk,
        "method": "REML_modified_Hartung_Knapp",
    }


def make_probe_coverage(
    cohort: str,
    beta: pd.DataFrame,
    meta: pd.DataFrame,
    targets: list[ProbeTarget],
) -> list[dict[str, object]]:
    rows = []
    for target in targets:
        present = target.probe in beta.index
        row = {
            "cohort": cohort,
            "gene": target.gene,
            "probe": target.probe,
            "gene_probe_index": target.gene_probe_index,
            "probe_present": bool(present),
            "n_samples": int(beta.shape[1]),
            "n_valid_values": int(beta.loc[target.probe].notna().sum()) if present else 0,
        }
        for tissue, group in meta.groupby("tissue", dropna=False):
            row[f"n_{tissue}_samples"] = int(len(group))
            row[f"n_{tissue}_valid_values"] = int(beta.loc[target.probe, group["sample"]].notna().sum()) if present else 0
        rows.append(row)
    return rows


def build_contrasts(
    public_root: Path,
    fixed_path: Path,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fixed = read_fixed_probes(fixed_path)
    targets = probe_targets(fixed)
    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    source_files: dict[str, dict[str, str]] = {}
    cohort_counts = []
    for cohort in COHORTS:
        beta, meta = load_cohort(cohort, public_root)
        source_files[f"{cohort}_beta"] = {
            "path": str(public_root / "data" / "derived" / f"{cohort}_beta.tsv.gz"),
            "sha256": file_sha256(public_root / "data" / "derived" / f"{cohort}_beta.tsv.gz"),
        }
        source_files[f"{cohort}_samples"] = {
            "path": str(public_root / "data" / "derived" / f"{cohort}_samples.tsv"),
            "sha256": file_sha256(public_root / "data" / "derived" / f"{cohort}_samples.tsv"),
        }
        coverage_rows.extend(make_probe_coverage(cohort, beta, meta, targets))
        for tissue, group in meta.groupby("tissue", dropna=False):
            cohort_counts.append(
                {
                    "cohort": cohort,
                    "tissue": tissue,
                    "n_samples": int(len(group)),
                    "n_patients": int(group["patient"].nunique()),
                    "n_pair_verified": int(group["pair_verified"].sum()),
                }
            )
        for high, low, family in CONTRASTS[cohort]:
            paired = is_paired(meta, high, low)
            for target in targets:
                base = {
                    "cohort": cohort,
                    "contrast": f"{high}-{low}",
                    "high_tissue": high,
                    "low_tissue": low,
                    "family": family,
                    "gene": target.gene,
                    "probe": target.probe,
                    "gene_probe_index": target.gene_probe_index,
                    "paired": paired,
                    "pairing_scope": "verified_patient_pairs" if paired else "unpaired_or_partially_paired_observations",
                    "unit": "beta_percentage_points",
                    "n_boot": n_boot,
                }
                if target.probe not in beta.index:
                    result = empty_contrast_result(
                        "paired_t" if paired else "Welch_t", "probe_absent", "probe_not_present_in_beta_matrix"
                    )
                    adjusted = {}
                    if family in {"healthy_reference", "lesion"}:
                        adjusted = {
                            "adjusted_effect_pp": np.nan,
                            "adjusted_low_pp": np.nan,
                            "adjusted_high_pp": np.nan,
                            "adjusted_p": np.nan,
                            "adjusted_p_for_bh": 1.0,
                            "adjusted_n": 0,
                            "adjusted_covariates": "",
                            "adjusted_status": "probe_absent",
                            "adjusted_interval": "",
                            "adjusted_covariance": "",
                        }
                else:
                    result = probe_contrast(
                        meta,
                        beta.loc[target.probe],
                        high,
                        low,
                        paired,
                        n_boot,
                        stable_seed(cohort, target.gene, target.probe, high, low),
                    )
                    adjusted = adjusted_probe_contrast(meta, beta.loc[target.probe], high, low) if family in {
                        "healthy_reference",
                        "lesion",
                    } else {}
                rows.append({**base, **result, **adjusted})
    contrasts = pd.DataFrame(rows)
    for _, idx in contrasts.groupby("family").groups.items():
        contrasts.loc[idx, "q_BH_family"] = bh_adjust_with_family_denominator(contrasts.loc[idx, "p_for_bh"])
        contrasts.loc[idx, "rank_q_BH_family"] = bh_adjust_with_family_denominator(contrasts.loc[idx, "rank_p_for_bh"])
        if "adjusted_p_for_bh" in contrasts:
            contrasts.loc[idx, "adjusted_q_BH_family"] = bh_adjust_with_family_denominator(
                contrasts.loc[idx, "adjusted_p_for_bh"].fillna(1.0)
            )
    for _, idx in contrasts.groupby(["cohort", "contrast"]).groups.items():
        contrasts.loc[idx, "q_BH_contrast_77"] = bh_adjust_with_family_denominator(contrasts.loc[idx, "p_for_bh"])
        contrasts.loc[idx, "rank_q_BH_contrast_77"] = bh_adjust_with_family_denominator(
            contrasts.loc[idx, "rank_p_for_bh"]
        )
    return contrasts, pd.DataFrame(coverage_rows), pd.DataFrame(cohort_counts), source_files


MIN_PAIRS_PRIMARY = 20
MIN_PAIRS_SENSITIVITY = 3
META_SENSITIVITY_NAME = "public_probe_meta_analysis_all_paired_cohorts_sensitivity.tsv"
META_PRIMARY_AMENDMENT = (
    "2026-09-09: after the GSE77954 pairing correction left that cohort with four source-defined pairs, "
    "the primary tumor-adjacent meta-analysis pool was restricted to paired cohorts with at least "
    f"{MIN_PAIRS_PRIMARY} pairs (Colonomics, GSE42752, GSE193535, GSE77718). This restriction was adopted after "
    "inspecting the five-cohort result; the five-cohort pool (all paired cohorts with >=3 pairs) is written as "
    "a sensitivity analysis. Both files are produced from the same contrast table."
)


def build_meta(contrasts: pd.DataFrame, min_pairs: int = MIN_PAIRS_PRIMARY, include_gse77954_tn: bool | None = None) -> pd.DataFrame:
    """Pool the same contrast and CpG across independent public cohorts.

    T-N pooling uses paired cohorts only. ``min_pairs`` is the minimum number of source-defined
    pairs a cohort needs to enter the pool: MIN_PAIRS_PRIMARY (20) for the primary analysis and
    MIN_PAIRS_SENSITIVITY (3) for the all-paired-cohorts sensitivity analysis. The legacy flag
    ``include_gse77954_tn`` is kept for backward compatibility: True maps to min_pairs=3 and
    False to min_pairs=20.
    """
    if include_gse77954_tn is not None:
        min_pairs = MIN_PAIRS_SENSITIVITY if include_gse77954_tn else MIN_PAIRS_PRIMARY
    rows = []
    estimated = contrasts.loc[contrasts["status"].eq("estimated")].copy()
    for (contrast, gene, probe), group in estimated.groupby(["contrast", "gene", "probe"], sort=True):
        family = group["family"].iloc[0]
        eligible = group.copy()
        if contrast == "T-N":
            eligible = eligible.loc[eligible["paired"]]
            eligible = eligible.loc[pd.to_numeric(eligible["n_pairs"], errors="coerce").fillna(0) >= min_pairs]
        eligible = eligible.loc[np.isfinite(eligible["effect_pp"]) & np.isfinite(eligible["se_pp"]) & (eligible["se_pp"] > 0)]
        if len(eligible) < 3:
            rows.append(
                {
                    "contrast": contrast,
                    "family": family,
                    "gene": gene,
                    "probe": probe,
                    "k": int(len(eligible)),
                    "cohorts": ";".join(eligible["cohort"]),
                    "status": "insufficient_independent_cohorts",
                    "reason": "requires_at_least_3_eligible_public_cohorts",
                    "p_for_bh": 1.0,
                }
            )
            continue
        result = meta_reml(eligible["effect_pp"], eligible["se_pp"] ** 2)
        rows.append(
            {
                **result,
                "contrast": contrast,
                "family": family,
                "gene": gene,
                "probe": probe,
                "cohorts": ";".join(eligible["cohort"]),
                "status": "estimated",
                "reason": "",
            }
        )
    meta = pd.DataFrame(rows)
    if len(meta):
        for _, idx in meta.groupby("contrast").groups.items():
            meta.loc[idx, "q_BH_contrast_77"] = bh_adjust_with_family_denominator(meta.loc[idx, "p_for_bh"])
    return meta


def korean_effects_with_se(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"n_pairs", "sd_delta_pp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns for SE calculation: {sorted(missing)}")
    n_pairs = pd.to_numeric(df["n_pairs"], errors="coerce")
    sd = pd.to_numeric(df["sd_delta_pp"], errors="coerce")
    df["se_delta_pp"] = sd / np.sqrt(n_pairs.where(n_pairs > 0))
    return df


def build_korean_only_meta(korean_root: Path = DEFAULT_KOREAN_CPG_ROOT) -> pd.DataFrame:
    frames = []
    for cohort in ["CMCBSN", "SNUH", "ASAN"]:
        path = korean_root / cohort / "paired_cpg_effects.tsv"
        if not path.exists():
            continue
        df = korean_effects_with_se(path)
        df["source_file"] = str(path)
        df["source_sha256"] = file_sha256(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    all_effects = pd.concat(frames, ignore_index=True)
    rows = []
    for (gene, probe), group in all_effects.groupby(["gene", "probe"], sort=True):
        eligible = group.loc[
            group["status"].eq("estimated")
            & np.isfinite(group["mean_delta_pp"])
            & np.isfinite(group["se_delta_pp"])
            & (group["se_delta_pp"] > 0)
        ].copy()
        if len(eligible) < 3:
            rows.append(
                {
                    "gene": gene,
                    "probe": probe,
                    "k": int(len(eligible)),
                    "cohorts": ";".join(eligible["cohort"]),
                    "status": "insufficient_korean_cohorts",
                    "reason": "requires_three_estimated_Korean_institution_cohorts",
                    "p_for_bh": 1.0,
                }
            )
            continue
        result = meta_reml(eligible["mean_delta_pp"], eligible["se_delta_pp"] ** 2)
        rows.append(
            {
                **result,
                "gene": gene,
                "probe": probe,
                "cohorts": ";".join(eligible["cohort"]),
                "status": "estimated",
                "reason": "",
            }
        )
    meta = pd.DataFrame(rows)
    if len(meta):
        meta["q_BH_77"] = bh_adjust_with_family_denominator(meta["p_for_bh"])
        meta["source_scope"] = "Korean institutions only; not pooled with public discovery cohorts"
        meta["source_identity_note"] = "Independent institution/source IDs matched; genotype identity not verified"
    return meta


def write_korean_reports(outdir: Path, contrasts: pd.DataFrame, meta: pd.DataFrame, coverage: pd.DataFrame) -> None:
    family_sizes = contrasts.groupby("family").size().to_dict()
    estimated = contrasts["status"].eq("estimated").sum()
    absent = contrasts["status"].eq("probe_absent").sum()
    (outdir / "공개코호트_CpG_분석요약.md").write_text(
        "\n".join(
            [
                "# 공개 코호트 CpG 단위 재분석 요약",
                "",
                "기존 공개 분석은 77개 CpG를 유전자별로 평균낸 gene score에 대해 contrast를 계산했다.",
                "이번 재분석은 평균을 내지 않고 고정 77개 CpG 각각을 같은 cohort-contrast 설계로 평가했다.",
                "",
                f"- 전체 계획 행: {len(contrasts)}개",
                f"- 추정 가능 행: {int(estimated)}개",
                f"- probe 부재 행: {int(absent)}개",
                f"- family 크기: {family_sizes}",
                "- 효과 단위: beta 차이 × 100, percentage point",
                "- 결측 probe: 행은 유지하고 표시 p는 NA, BH 입력 p는 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    meta_estimated = int(meta["status"].eq("estimated").sum()) if len(meta) else 0
    (outdir / "공개코호트_CpG_meta_요약.md").write_text(
        "\n".join(
            [
                "# 공개 코호트 CpG 단위 meta-analysis 요약",
                "",
                "동일 contrast와 동일 CpG에 대해 독립 공개 코호트가 3개 이상인 경우에만 REML modified Hartung-Knapp meta-analysis를 수행했다.",
                "T-N meta-analysis는 기존 계약과 동일하게 verified paired cohort만 eligible로 두었다.",
                "",
                f"- meta-analysis 계획/검토 행: {len(meta)}개",
                f"- 추정 가능 meta-analysis 행: {meta_estimated}개",
                "- BH 보정: contrast별 77개 CpG 기준",
                "- Korean cohort 결과는 이 파일에 pooled하지 않았다.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    coverage_summary = coverage.groupby("cohort").agg(
        probes_present=("probe_present", "sum"),
        probes_total=("probe", "size"),
    )
    (outdir / "공개코호트_CpG_coverage_요약.md").write_text(
        "# 공개 코호트 77 CpG coverage 요약\n\n"
        + coverage_summary.to_markdown()
        + "\n\n모든 downstream contrast는 이 고정 77 CpG registry를 denominator로 사용했다.\n",
        encoding="utf-8",
    )


def ensure_contract(path: Path, fixed_path: Path, public_root: Path, n_boot: int, source_files: dict[str, dict[str, str]]) -> None:
    fixed = read_fixed_probes(fixed_path)
    contract = {
        "status": "public_array_cpg_level_contract",
        "contract_creation_timestamp": "timestamp_unrecorded",
        "amended_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "amendment_reason": "Use portable 124 public inputs; GSE77954 T-N analysed as four source-label pairs; primary meta-analysis pool restricted to cohorts with >=20 pairs (see meta_primary_amendment)",
        "public_root": str(public_root),
        "fixed_probe_registry": str(fixed_path),
        "fixed_probe_sha256": file_sha256(fixed_path),
        "n_fixed_cpg": sum(len(v) for v in fixed.values()),
        "genes": GENES,
        "cohorts": COHORTS,
        "contrasts": {cohort: [{"high": h, "low": l, "family": f} for h, l, f in values] for cohort, values in CONTRASTS.items()},
        "contrast_family_sizes": {"healthy_reference": 462, "tissue_replication": 462, "lesion": 231},
        "n_boot": n_boot,
        "effect_unit": "beta percentage points",
        "estimand": "individual CpG beta difference; no gene means, no panel means",
        "missing_probe_handling": "retain planned row; report p as NA; use p_for_bh=1 in family and contrast BH denominators",
        "tests": "paired T-N uses verified patient pairs with paired t and Wilcoxon; unpaired uses Welch t or patient-cluster OLS when patient overlap exists",
        "meta_analysis": "same public contrast and CpG only; REML modified Hartung-Knapp; >=3 independent eligible public cohorts per CpG; primary T-N pool = source-defined paired cohorts with >=20 pairs (Colonomics, GSE42752, GSE193535, GSE77718); sensitivity T-N pool = all paired cohorts with >=3 pairs (adds the corrected four-pair GSE77954); no Korean pooling",
        "meta_primary_amendment": META_PRIMARY_AMENDMENT,
        "source_files": source_files,
    }
    write_json(path, contract)


def write_gse77954_unpaired_tn_sensitivity(public_root: Path, fixed_path: Path, outdir: Path, n_boot: int) -> None:
    """Write a clearly separated 13T:4N GSE77954 T-N sensitivity table.

    Primary T-N inference for GSE77954 uses the four verified same-patient pairs.
    This sensitivity retains all carcinoma observations against the four adjacent
    normal carcinoma observations and is not consumed by the primary BH family or
    public meta-analysis.
    """
    fixed = read_fixed_probes(fixed_path)
    targets = probe_targets(fixed)
    beta, meta = load_cohort("GSE77954", public_root)
    rows = []
    for target in targets:
        base = {
            "cohort": "GSE77954",
            "contrast": "T-N",
            "high_tissue": "T",
            "low_tissue": "N",
            "family": "tissue_replication_sensitivity_not_primary",
            "gene": target.gene,
            "probe": target.probe,
            "gene_probe_index": target.gene_probe_index,
            "paired": False,
            "pairing_scope": "all_carcinoma_vs_four_adjacent_normals_sensitivity_only",
            "unit": "beta_percentage_points",
            "n_boot": n_boot,
        }
        if target.probe not in beta.index:
            result = empty_contrast_result("Welch_t", "probe_absent", "probe_not_present_in_beta_matrix")
        else:
            result = probe_contrast(
                meta,
                beta.loc[target.probe],
                "T",
                "N",
                paired=False,
                n_boot=n_boot,
                seed=stable_seed("GSE77954", target.gene, target.probe, "T", "N", "unpaired_sensitivity"),
            )
        rows.append({**base, **result})
    sens = pd.DataFrame(rows)
    sens["q_BH_77"] = bh_adjust_with_family_denominator(sens["p_for_bh"])
    sens.to_csv(outdir / "GSE77954_T-N_13T4N_unpaired_sensitivity_not_primary.tsv", sep="\t", index=False)



def run_analysis(
    public_root: Path = PUBLIC_ROOT,
    outdir: Path = DEFAULT_OUTDIR,
    fixed_path: Path = DEFAULT_FIXED_PROBES,
    contract_path: Path = DEFAULT_CONTRACT,
    n_boot: int = N_BOOT,
    korean_cpg_root: Path = DEFAULT_KOREAN_CPG_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    contrasts, coverage, cohort_counts, source_files = build_contrasts(public_root, fixed_path, n_boot)
    meta = build_meta(contrasts, min_pairs=MIN_PAIRS_PRIMARY)
    meta_all_paired = build_meta(contrasts, min_pairs=MIN_PAIRS_SENSITIVITY)
    ensure_contract(contract_path, fixed_path, public_root, n_boot, source_files)
    contrasts.to_csv(outdir / "public_probe_contrasts.tsv", sep="\t", index=False)
    meta.to_csv(outdir / "public_probe_meta_analysis.tsv", sep="\t", index=False)
    meta_all_paired.to_csv(outdir / META_SENSITIVITY_NAME, sep="\t", index=False)
    korean_meta = build_korean_only_meta(korean_cpg_root)
    if len(korean_meta):
        korean_meta.to_csv(outdir / "korean_only_meta.tsv", sep="\t", index=False)
        korean_inputs = []
        for cohort in ["CMCBSN", "SNUH", "ASAN"]:
            path = korean_cpg_root / cohort / "paired_cpg_effects.tsv"
            if path.exists():
                df = korean_effects_with_se(path)
                df["input_file"] = str(path)
                korean_inputs.append(df)
        if korean_inputs:
            pd.concat(korean_inputs, ignore_index=True).to_csv(
                outdir / "korean_only_meta_input_with_se.tsv", sep="\t", index=False
            )
    coverage.to_csv(outdir / "public_probe_coverage.tsv", sep="\t", index=False)
    write_gse77954_unpaired_tn_sensitivity(public_root, fixed_path, outdir, n_boot)
    cohort_counts.to_csv(outdir / "public_cohort_counts.tsv", sep="\t", index=False)
    manifest = {
        "status": "complete",
        "contrast_rows": int(len(contrasts)),
        "expected_contrast_rows": 1155,
        "estimated_contrast_rows": int(contrasts["status"].eq("estimated").sum()),
        "probe_absent_rows": int(contrasts["status"].eq("probe_absent").sum()),
        "meta_rows": int(len(meta)),
        "estimated_meta_rows": int(meta["status"].eq("estimated").sum()) if len(meta) else 0,
        "estimated_meta_all_paired_cohorts_sensitivity_rows": int(meta_all_paired["status"].eq("estimated").sum()) if len(meta_all_paired) else 0,
        "meta_primary_rule": f"paired cohorts with >= {MIN_PAIRS_PRIMARY} pairs",
        "meta_sensitivity_rule": f"paired cohorts with >= {MIN_PAIRS_SENSITIVITY} pairs",
        "korean_only_meta_rows": int(len(korean_meta)),
        "estimated_korean_only_meta_rows": int(korean_meta["status"].eq("estimated").sum()) if len(korean_meta) else 0,
        "family_sizes": {str(k): int(v) for k, v in contrasts.groupby("family").size().to_dict().items()},
        "outputs": {
            name: file_sha256(outdir / name)
            for name in [
                "public_probe_contrasts.tsv",
                "public_probe_meta_analysis.tsv",
                META_SENSITIVITY_NAME,
                "public_probe_coverage.tsv",
                "public_cohort_counts.tsv",
                "GSE77954_T-N_13T4N_unpaired_sensitivity_not_primary.tsv",
                *([] if korean_meta.empty else ["korean_only_meta.tsv"]),
                *([] if korean_meta.empty else ["korean_only_meta_input_with_se.tsv"]),
            ]
        },
        "contract_sha256": file_sha256(contract_path),
    }
    write_json(outdir / "public_contrast_manifest.json", manifest)
    write_korean_reports(outdir, contrasts, meta, coverage)
    manifest["outputs"].update(
        {
            name: file_sha256(outdir / name)
            for name in [
                "공개코호트_CpG_분석요약.md",
                "공개코호트_CpG_meta_요약.md",
                "공개코호트_CpG_coverage_요약.md",
            ]
        }
    )
    write_json(outdir / "public_contrast_manifest.json", manifest)
    return contrasts, meta


def rebuild_meta_only(outdir: Path, fixed_path: Path, contract_path: Path, public_root: Path, n_boot: int) -> None:
    """Recompute only the deterministic REML pooling from the saved contrast table.

    Used when the pooling rule changes (2026-09-09 primary-pool amendment); the contrast
    estimates, bootstrap intervals and BH families are left untouched.
    """
    contrasts = pd.read_csv(outdir / "public_probe_contrasts.tsv", sep="\t")
    coverage = pd.read_csv(outdir / "public_probe_coverage.tsv", sep="\t")
    meta = build_meta(contrasts, min_pairs=MIN_PAIRS_PRIMARY)
    meta_all_paired = build_meta(contrasts, min_pairs=MIN_PAIRS_SENSITIVITY)
    meta.to_csv(outdir / "public_probe_meta_analysis.tsv", sep="\t", index=False)
    meta_all_paired.to_csv(outdir / META_SENSITIVITY_NAME, sep="\t", index=False)
    manifest_path = outdir / "public_contrast_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    source_files = manifest.get("source_files", {})
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else {}
    ensure_contract(contract_path, fixed_path, public_root, n_boot, contract.get("source_files", source_files))
    manifest.update({
        "meta_rows": int(len(meta)),
        "estimated_meta_rows": int(meta["status"].eq("estimated").sum()),
        "estimated_meta_all_paired_cohorts_sensitivity_rows": int(meta_all_paired["status"].eq("estimated").sum()),
        "meta_primary_rule": f"paired cohorts with >= {MIN_PAIRS_PRIMARY} pairs",
        "meta_sensitivity_rule": f"paired cohorts with >= {MIN_PAIRS_SENSITIVITY} pairs",
        "meta_primary_amendment": META_PRIMARY_AMENDMENT,
        "meta_rebuilt_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    manifest.setdefault("outputs", {})
    manifest["outputs"].pop("public_probe_meta_analysis_leave_GSE77954_out_sensitivity.tsv", None)
    manifest["outputs"]["public_probe_meta_analysis.tsv"] = file_sha256(outdir / "public_probe_meta_analysis.tsv")
    manifest["outputs"][META_SENSITIVITY_NAME] = file_sha256(outdir / META_SENSITIVITY_NAME)
    manifest["contract_sha256"] = file_sha256(contract_path)
    write_json(manifest_path, manifest)
    write_korean_reports(outdir, contrasts, meta, coverage)
    print(f"primary estimated={int(meta['status'].eq('estimated').sum())} sensitivity estimated={int(meta_all_paired['status'].eq('estimated').sum())}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", type=Path, default=PUBLIC_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--fixed-probes", type=Path, default=DEFAULT_FIXED_PROBES)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--korean-cpg-root", type=Path, default=DEFAULT_KOREAN_CPG_ROOT)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--meta-only", action="store_true", help="Rebuild the two meta-analysis tables, the contract and the manifest from the saved public_probe_contrasts.tsv without recomputing contrasts or bootstraps")
    args = parser.parse_args()
    if args.meta_only:
        rebuild_meta_only(args.outdir, args.fixed_probes, args.contract, args.public_root, args.n_boot)
        return
    contrasts, meta = run_analysis(
        args.public_root, args.outdir, args.fixed_probes, args.contract, args.n_boot, args.korean_cpg_root
    )
    summary = contrasts.groupby(["family", "cohort"]).agg(
        n_rows=("probe", "size"),
        estimated=("status", lambda x: int((x == "estimated").sum())),
        absent=("status", lambda x: int((x == "probe_absent").sum())),
    )
    print(summary)
    print(f"meta_rows={len(meta)} estimated_meta={int(meta['status'].eq('estimated').sum()) if len(meta) else 0}")


if __name__ == "__main__":
    main()
