"""CMCBSN per-CpG molecular context analysis.

This runner treats each frozen CpG as the analysis unit. It preserves all
77 planned CpGs, writes NA rows for missing beta probes, and never computes
gene-level or panel-level methylation averages.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT.parent / "120_Korean_External_Validation_20260908"
DEFAULT_CONTRACT = ROOT / "registry" / "cpg_analysis_contract.json"
DEFAULT_FIXED_PROBES = ROOT / "registry" / "fixed_probes.json"
DEFAULT_BETA = ROOT / "data" / "derived" / "CMCBSN_beta.tsv"
DEFAULT_METHYLATION_META = ROOT / "data" / "derived" / "CMCBSN_samples.tsv"
DEFAULT_RNA_META = PRIOR / "data" / "derived" / "CMCBSN_rna_samples.tsv"
DEFAULT_CMS = PRIOR / "review" / "cmc_metadata" / "CMCBSN_cms_source_labels.tsv"
DEFAULT_EXPRESSION = PRIOR / "data" / "raw" / "cmc_rna" / "CMCBSN_log2cpm_342.txt"
DEFAULT_OUTDIR = ROOT / "results" / "context"
DEFAULT_REVIEW = ROOT / "review" / "context_gate_report.md"


@dataclass(frozen=True)
class CpgContract:
    seed: int
    n_boot: int
    paired_family: int
    secondary_context_family: int


def read_contract(path: Path = DEFAULT_CONTRACT) -> CpgContract:
    data = json.loads(path.read_text())
    return CpgContract(
        seed=int(data["seed"]),
        n_boot=int(data["n_boot"]),
        paired_family=int(data.get("paired_family", 77)),
        secondary_context_family=int(data.get("secondary_context_family", 77)),
    )


def read_fixed_targets(path: Path = DEFAULT_FIXED_PROBES) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows = []
    order = 0
    for gene, probes in data.items():
        for probe in probes:
            rows.append({"target_order": order, "gene": str(gene), "cpg": str(probe)})
            order += 1
    targets = pd.DataFrame(rows)
    if len(targets) != 77 or targets["cpg"].duplicated().any():
        raise ValueError("fixed CpG registry must contain 77 unique CpGs")
    return targets


def stable_seed(seed: int, *parts: object) -> int:
    import hashlib

    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return (int(seed) + int(digest[:8], 16)) % (2**32)


def bh_adjust_planned(p_values: Iterable[float], family_size: int) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    if len(p) != family_size:
        raise ValueError(f"BH family size mismatch: expected {family_size}, observed {len(p)}")
    q = np.full(len(p), np.nan)
    order = np.argsort(p)
    raw = p[order] * family_size / np.arange(1, family_size + 1)
    q[order] = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    return q


def _strip_ids(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = frame[col].astype(str).str.strip()
    return frame


def _reject_duplicate_ids(frame: pd.DataFrame, cols: list[str], name: str) -> None:
    duplicated = frame.duplicated(cols, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, cols].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{name} has duplicate keys after stripping whitespace: {examples}")


def load_beta_cpgs(beta_path: Path, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta = pd.read_csv(beta_path, sep="\t", index_col=0)
    beta.index = beta.index.astype(str).str.strip()
    beta.columns = beta.columns.astype(str).str.strip()
    if beta.index.duplicated().any():
        raise ValueError("beta matrix has duplicate CpG identifiers after stripping whitespace")
    if beta.columns.duplicated().any():
        raise ValueError("beta matrix has duplicate sample identifiers after stripping whitespace")
    values = beta.to_numpy(dtype=float)
    if np.nanmin(values) < 0 or np.nanmax(values) > 1:
        raise ValueError("beta values must be in [0,1]")
    present = targets["cpg"].isin(beta.index)
    observed = beta.reindex(targets["cpg"]).T
    observed.index.name = "sample_id"
    coverage = targets.copy()
    coverage["present_in_beta"] = present.to_numpy()
    return observed, coverage


def load_context_inputs(
    methylation_meta_path: Path,
    rna_meta_path: Path,
    cms_path: Path,
    expression_path: Path,
    beta_samples: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    beta_sample_set = set(map(str, beta_samples))
    meth = pd.read_csv(methylation_meta_path, sep="\t", dtype=str)
    meth = _strip_ids(meth, ["cohort", "patient_id", "sample_id", "tissue", "pair_verified"])
    _reject_duplicate_ids(meth, ["sample_id"], "methylation metadata")
    if set(meth["sample_id"]) != beta_sample_set:
        raise ValueError("methylation metadata sample_id set must match beta columns exactly")
    if not set(meth["tissue"]).issubset({"T", "N"}):
        raise ValueError("methylation tissue must be T or N")

    rna = pd.read_csv(rna_meta_path, sep="\t", dtype=str)
    rna = _strip_ids(
        rna,
        [
            "cohort",
            "patient_id",
            "sample_id",
            "rna_sample_id",
            "tissue",
            "sex",
            "age",
            "site",
            "MSI",
            "methylation_sample_id_candidate",
        ],
    )
    _reject_duplicate_ids(rna, ["sample_id"], "RNA metadata")
    if not set(rna["tissue"]).issubset({"tumor", "adjacent_normal"}):
        raise ValueError("RNA tissue must be tumor or adjacent_normal")

    cms = pd.read_csv(cms_path, sep="\t", dtype=str)
    cms = _strip_ids(cms, ["cohort", "patient_id", "sample_id", "rna_sample_id", "CMS_source"])
    _reject_duplicate_ids(cms, ["sample_id"], "CMS source labels")

    expression = pd.read_csv(expression_path, sep="\t", index_col=0)
    expression.index = expression.index.astype(str).str.strip()
    expression.columns = expression.columns.astype(str).str.strip()
    if expression.index.duplicated().any() or expression.columns.duplicated().any():
        raise ValueError("expression matrix has duplicate row or sample identifiers after stripping whitespace")
    return meth, rna, cms, expression


def methylation_context_frame(
    cpg_beta: pd.DataFrame,
    targets: pd.DataFrame,
    methylation_meta: pd.DataFrame,
    rna_meta: pd.DataFrame,
    cms: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = cpg_beta.reset_index().melt(id_vars="sample_id", var_name="cpg", value_name="beta")
    long = long.merge(targets, on="cpg", how="right")
    long = long.merge(methylation_meta[["sample_id", "patient_id", "tissue", "cohort"]], on="sample_id", how="left")
    tumor = long.loc[long["tissue"].eq("T")].copy()

    rna_tumor = rna_meta.loc[rna_meta["tissue"].eq("tumor")].copy()
    rna_cols = [
        "sample_id",
        "rna_sample_id",
        "patient_id",
        "sex",
        "age",
        "site",
        "MSI",
        "sample_in_log2cpm_342",
    ]
    tumor = tumor.merge(
        rna_tumor[rna_cols].rename(columns={"sample_id": "methylation_sample_id", "patient_id": "rna_patient_id"}),
        left_on="sample_id",
        right_on="methylation_sample_id",
        how="left",
        validate="many_to_one",
    )
    cms_cols = ["sample_id", "CMS_source", "CMS_label_status"]
    tumor = tumor.merge(
        cms[cms_cols].rename(columns={"sample_id": "cms_sample_id"}),
        left_on="sample_id",
        right_on="cms_sample_id",
        how="left",
        validate="many_to_one",
    )
    tumor["exact_rna_match"] = tumor["rna_sample_id"].notna()
    tumor["exact_cms_match"] = tumor["cms_sample_id"].notna()
    tumor["rna_qc342"] = tumor["sample_in_log2cpm_342"].astype(str).str.lower().eq("true")
    tumor["cms_assigned"] = tumor["CMS_source"].isin(["CMS1", "CMS2", "CMS3", "CMS4"])

    audit = (
        tumor[["sample_id", "patient_id", "exact_rna_match", "rna_qc342", "exact_cms_match", "cms_assigned"]]
        .drop_duplicates()
        .sort_values("sample_id")
    )
    audit["context_link_status"] = np.select(
        [
            audit["exact_rna_match"] & audit["exact_cms_match"],
            audit["exact_rna_match"],
            audit["exact_cms_match"],
        ],
        ["matched_rna_and_cms", "matched_rna_only", "matched_cms_only"],
        default="no_context_match",
    )
    return tumor, audit


def _mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, int]:
    boot = []
    values = np.asarray(values, dtype=float)
    for _ in range(n_boot):
        sample = values[rng.integers(0, len(values), size=len(values))]
        boot.append(float(np.mean(sample)))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi), len(boot)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.isclose(np.nanstd(x), 0.0) or np.isclose(np.nanstd(y), 0.0):
        return np.nan, np.nan
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _pearson_r_no_p(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.dot(x, y) / denom)


def _spearman_r_no_p(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return np.nan
    return _pearson_r_no_p(
        _rank_average(np.asarray(x, dtype=float)),
        _rank_average(np.asarray(y, dtype=float)),
    )


def _rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def cms_cpg_tests(tumor: pd.DataFrame, targets: pd.DataFrame, contract: CpgContract) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for target in targets.itertuples(index=False):
        sub = tumor.loc[tumor["cpg"].eq(target.cpg) & tumor["beta"].notna() & tumor["cms_assigned"]].copy()
        cms2 = sub.loc[sub["CMS_source"].eq("CMS2"), "beta"].to_numpy(dtype=float)
        cms3 = sub.loc[sub["CMS_source"].eq("CMS3"), "beta"].to_numpy(dtype=float)
        groups = [g["beta"].to_numpy(dtype=float) for _, g in sub.groupby("CMS_source") if len(g)]
        row = {
            "target_order": target.target_order,
            "gene": target.gene,
            "cpg": target.cpg,
            "present_in_beta": bool(len(sub) > 0 or tumor.loc[tumor["cpg"].eq(target.cpg), "beta"].notna().any()),
            "n_cms2": int(len(cms2)),
            "n_cms3": int(len(cms3)),
            "mean_cms3_minus_cms2_beta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "welch_p": np.nan,
            "welch_bh_q": np.nan,
            "kw_n": int(sum(len(group) for group in groups)),
            "kw_groups": int(len(groups)),
            "kw_p": np.nan,
            "kw_bh_q": np.nan,
            "bootstrap_valid": 0,
            "n_boot": contract.n_boot,
            "status": "missing_cpg",
            "reason": "CpG_absent_from_beta_matrix",
        }
        if len(sub):
            row.update(status="insufficient_data", reason="fewer_than_2_CMS2_or_CMS3_samples")
        if len(cms2) >= 2 and len(cms3) >= 2:
            diff = float(np.mean(cms3) - np.mean(cms2))
            rng = np.random.default_rng(stable_seed(contract.seed, target.cpg, "cms"))
            boot = []
            for _ in range(contract.n_boot):
                b2 = cms2[rng.integers(0, len(cms2), size=len(cms2))]
                b3 = cms3[rng.integers(0, len(cms3), size=len(cms3))]
                boot.append(float(np.mean(b3) - np.mean(b2)))
            lo, hi = np.quantile(boot, [0.025, 0.975])
            row.update(
                mean_cms3_minus_cms2_beta=diff,
                ci_low=float(lo),
                ci_high=float(hi),
                welch_p=float(stats.ttest_ind(cms3, cms2, equal_var=False).pvalue),
                bootstrap_valid=len(boot),
                status="estimated",
                reason="",
            )
        if len(groups) >= 2 and all(len(group) >= 2 for group in groups):
            row["kw_p"] = float(stats.kruskal(*groups).pvalue)
        rows.append(row)
    out = pd.DataFrame(rows)
    welch_q = bh_adjust_planned([p if np.isfinite(p) else 1.0 for p in out["welch_p"]], contract.secondary_context_family)
    kw_q = bh_adjust_planned([p if np.isfinite(p) else 1.0 for p in out["kw_p"]], contract.secondary_context_family)
    out.loc[out["welch_p"].notna(), "welch_bh_q"] = welch_q[out["welch_p"].notna().to_numpy()]
    out.loc[out["kw_p"].notna(), "kw_bh_q"] = kw_q[out["kw_p"].notna().to_numpy()]
    counts = tumor.loc[tumor["cms_assigned"], ["sample_id", "CMS_source"]].drop_duplicates()["CMS_source"].value_counts()
    summary = pd.DataFrame(
        [
            {
                "analysis": "per_cpg_CMS_context",
                "planned_cpgs": len(targets),
                "observed_cpgs": int(out["present_in_beta"].sum()),
                "missing_cpgs": int((~out["present_in_beta"]).sum()),
                "matched_assigned_cms_tumors": int(counts.sum()),
                "CMS1": int(counts.get("CMS1", 0)),
                "CMS2": int(counts.get("CMS2", 0)),
                "CMS3": int(counts.get("CMS3", 0)),
                "CMS4": int(counts.get("CMS4", 0)),
                "status": "estimated",
            }
        ]
    )
    return out, summary


def expression_subset(expression: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    observed = [gene for gene in genes if gene in expression.index]
    return expression.reindex(observed).T


def _covariate_matrix(frame: pd.DataFrame) -> np.ndarray | None:
    cov = frame[["age", "sex", "site"]].copy()
    cov["age"] = pd.to_numeric(cov["age"], errors="coerce")
    if cov.isna().any().any():
        return None
    design = pd.concat(
        [
            cov[["age"]],
            pd.get_dummies(cov["sex"], prefix="sex", drop_first=True, dtype=float),
            pd.get_dummies(cov["site"], prefix="site", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    x = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    if np.linalg.matrix_rank(x) < x.shape[1] or len(design) <= x.shape[1] + 2:
        return None
    return x


def _partial_spearman(frame: pd.DataFrame) -> tuple[float, float]:
    x = _covariate_matrix(frame)
    if x is None:
        return np.nan, np.nan
    return _partial_spearman_from_arrays(
        frame["beta"].to_numpy(dtype=float),
        frame["expression"].to_numpy(dtype=float),
        x,
    )


def _partial_spearman_from_arrays(beta: np.ndarray, expression: np.ndarray, covariates: np.ndarray) -> tuple[float, float]:
    rank = int(np.linalg.matrix_rank(covariates))
    if rank < covariates.shape[1] or len(beta) <= rank + 1:
        return np.nan, np.nan
    meth_rank = stats.rankdata(np.asarray(beta, dtype=float), method="average")
    expr_rank = stats.rankdata(np.asarray(expression, dtype=float), method="average")
    meth_resid = meth_rank - covariates @ np.linalg.lstsq(covariates, meth_rank, rcond=None)[0]
    expr_resid = expr_rank - covariates @ np.linalg.lstsq(covariates, expr_rank, rcond=None)[0]
    rho = _pearson_r_no_p(meth_resid, expr_resid)
    if not np.isfinite(rho):
        return np.nan, np.nan
    df = len(beta) - rank - 1
    if df <= 0:
        return rho, np.nan
    if np.isclose(abs(rho), 1.0):
        return rho, 0.0
    t_value = rho * math.sqrt(df / (1.0 - rho**2))
    p_value = 2.0 * stats.t.sf(abs(t_value), df)
    return rho, float(p_value)


def _partial_spearman_r_no_p(beta: np.ndarray, expression: np.ndarray, covariates: np.ndarray) -> float:
    if len(beta) <= covariates.shape[1] + 2:
        return np.nan
    meth_rank = _rank_average(np.asarray(beta, dtype=float))
    expr_rank = _rank_average(np.asarray(expression, dtype=float))
    meth_resid = meth_rank - covariates @ np.linalg.lstsq(covariates, meth_rank, rcond=None)[0]
    expr_resid = expr_rank - covariates @ np.linalg.lstsq(covariates, expr_rank, rcond=None)[0]
    return _pearson_r_no_p(meth_resid, expr_resid)


def expression_cpg_tests(
    tumor: pd.DataFrame,
    targets: pd.DataFrame,
    expression: pd.DataFrame,
    contract: CpgContract,
) -> pd.DataFrame:
    genes = targets["gene"].drop_duplicates().tolist()
    expr_by_sample = expression_subset(expression, genes)
    rows = []
    for target in targets.itertuples(index=False):
        sub = tumor.loc[
            tumor["cpg"].eq(target.cpg)
            & tumor["beta"].notna()
            & tumor["rna_qc342"]
            & tumor["rna_sample_id"].notna()
        ].copy()
        row = {
            "target_order": target.target_order,
            "gene": target.gene,
            "cpg": target.cpg,
            "present_in_beta": bool(tumor.loc[tumor["cpg"].eq(target.cpg), "beta"].notna().any()),
            "rna_gene": target.gene,
            "n": 0,
            "rho": np.nan,
            "p": np.nan,
            "bh_q": np.nan,
            "rho_ci_low": np.nan,
            "rho_ci_high": np.nan,
            "partial_rho_age_sex_site": np.nan,
            "partial_p_age_sex_site": np.nan,
            "partial_bh_q_age_sex_site": np.nan,
            "partial_rho_ci_low": np.nan,
            "partial_rho_ci_high": np.nan,
            "bootstrap_valid": 0,
            "partial_bootstrap_valid": 0,
            "n_boot": contract.n_boot,
            "status": "missing_cpg",
            "reason": "CpG_absent_from_beta_matrix",
        }
        if not row["present_in_beta"]:
            pass
        elif target.gene not in expr_by_sample.columns:
            row.update(status="missing_rna_gene", reason="RNA_gene_absent_from_expression_matrix")
        elif len(sub):
            expr = expr_by_sample.reindex(sub["rna_sample_id"])[target.gene].to_numpy(dtype=float)
            sub = sub.assign(expression=expr).dropna(subset=["beta", "expression"])
            row["n"] = int(len(sub))
            row.update(status="insufficient_data", reason="fewer_than_3_matched_tumors_or_constant")
            rho, p = _spearman(sub["beta"].to_numpy(dtype=float), sub["expression"].to_numpy(dtype=float))
            if np.isfinite(rho):
                rng = np.random.default_rng(stable_seed(contract.seed, target.cpg, "expression"))
                boot = []
                for _ in range(contract.n_boot):
                    sample = sub.iloc[rng.integers(0, len(sub), size=len(sub))]
                    brho = _spearman_r_no_p(
                        sample["beta"].to_numpy(dtype=float),
                        sample["expression"].to_numpy(dtype=float),
                    )
                    if np.isfinite(brho):
                        boot.append(brho)
                lo, hi = np.quantile(boot, [0.025, 0.975]) if boot else (np.nan, np.nan)
                row.update(
                    rho=rho,
                    p=p,
                    rho_ci_low=float(lo) if np.isfinite(lo) else np.nan,
                    rho_ci_high=float(hi) if np.isfinite(hi) else np.nan,
                    bootstrap_valid=len(boot),
                    status="estimated",
                    reason="",
                )
            covariates = _covariate_matrix(sub.reset_index(drop=True)) if len(sub) >= 5 else None
            beta_values = sub["beta"].to_numpy(dtype=float)
            expression_values = sub["expression"].to_numpy(dtype=float)
            prho, pp = (
                _partial_spearman_from_arrays(beta_values, expression_values, covariates)
                if covariates is not None
                else (np.nan, np.nan)
            )
            if np.isfinite(prho):
                rng = np.random.default_rng(stable_seed(contract.seed, target.cpg, "partial_expression"))
                boot = []
                for _ in range(contract.n_boot):
                    indices = rng.integers(0, len(sub), size=len(sub))
                    brho = _partial_spearman_r_no_p(
                        beta_values[indices],
                        expression_values[indices],
                        covariates[indices, :],
                    )
                    if np.isfinite(brho):
                        boot.append(brho)
                lo, hi = np.quantile(boot, [0.025, 0.975]) if boot else (np.nan, np.nan)
                row.update(
                    partial_rho_age_sex_site=prho,
                    partial_p_age_sex_site=pp,
                    partial_rho_ci_low=float(lo) if np.isfinite(lo) else np.nan,
                    partial_rho_ci_high=float(hi) if np.isfinite(hi) else np.nan,
                    partial_bootstrap_valid=len(boot),
                )
        rows.append(row)
    out = pd.DataFrame(rows)
    q = bh_adjust_planned([p if np.isfinite(p) else 1.0 for p in out["p"]], contract.secondary_context_family)
    pq = bh_adjust_planned(
        [p if np.isfinite(p) else 1.0 for p in out["partial_p_age_sex_site"]],
        contract.secondary_context_family,
    )
    out.loc[out["p"].notna(), "bh_q"] = q[out["p"].notna().to_numpy()]
    out.loc[out["partial_p_age_sex_site"].notna(), "partial_bh_q_age_sex_site"] = pq[
        out["partial_p_age_sex_site"].notna().to_numpy()
    ]
    return out


def stromal_not_run(targets: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_order": row.target_order,
                "gene": row.gene,
                "cpg": row.cpg,
                "analysis": "stromal_score",
                "status": "not_run",
                "reason": "source_stromal_score_absent_and_R_ESTIMATE_package_unavailable; no_new_dependencies_downloaded",
            }
            for row in targets.itertuples(index=False)
        ]
    )


def write_gate_report(path: Path, coverage: pd.DataFrame, audit: pd.DataFrame, cms_summary: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CMCBSN Per-CpG Context Gate Report",
        "",
        f"- Planned CpGs: {len(coverage)}",
        f"- Observed CpGs in beta matrix: {int(coverage['present_in_beta'].sum())}",
        f"- Missing planned CpGs retained as NA rows: {int((~coverage['present_in_beta']).sum())}",
        f"- Tumor methylation samples with exact RNA match: {int(audit['exact_rna_match'].sum())}",
        f"- Tumor methylation samples with exact RNA QC342 match: {int((audit['exact_rna_match'] & audit['rna_qc342']).sum())}",
        f"- Tumor methylation samples with source CMS match: {int(audit['exact_cms_match'].sum())}",
        f"- Tumor methylation samples with assigned source CMS: {int(audit['cms_assigned'].sum())}",
        f"- Assigned CMS counts: CMS1={int(cms_summary.iloc[0]['CMS1'])}, CMS2={int(cms_summary.iloc[0]['CMS2'])}, CMS3={int(cms_summary.iloc[0]['CMS3'])}, CMS4={int(cms_summary.iloc[0]['CMS4'])}",
        "",
        "No gene-level or panel-level methylation averages are computed in this context lane.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_context(
    beta_path: Path = DEFAULT_BETA,
    methylation_meta_path: Path = DEFAULT_METHYLATION_META,
    rna_meta_path: Path = DEFAULT_RNA_META,
    cms_path: Path = DEFAULT_CMS,
    expression_path: Path = DEFAULT_EXPRESSION,
    contract_path: Path = DEFAULT_CONTRACT,
    fixed_probes_path: Path = DEFAULT_FIXED_PROBES,
    outdir: Path = DEFAULT_OUTDIR,
    review_path: Path = DEFAULT_REVIEW,
    n_boot: int | None = None,
) -> dict[str, Path]:
    contract = read_contract(contract_path)
    if n_boot is not None:
        if int(n_boot) <= 0:
            raise ValueError("n_boot must be positive")
        contract = CpgContract(contract.seed, int(n_boot), contract.paired_family, contract.secondary_context_family)
    targets = read_fixed_targets(fixed_probes_path)
    cpg_beta, coverage = load_beta_cpgs(beta_path, targets)
    meth, rna, cms, expression = load_context_inputs(
        methylation_meta_path,
        rna_meta_path,
        cms_path,
        expression_path,
        cpg_beta.index,
    )
    tumor, audit = methylation_context_frame(cpg_beta, targets, meth, rna, cms)
    cms_out, cms_summary = cms_cpg_tests(tumor, targets, contract)
    expr_out = expression_cpg_tests(tumor, targets, expression, contract)
    stroma = stromal_not_run(targets)

    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cpg_coverage": outdir / "cpg_coverage.tsv",
        "sample_linkage_audit": outdir / "sample_linkage_audit.tsv",
        "cms_cpg_contrasts": outdir / "cms_cpg_contrasts.tsv",
        "cms_context_summary": outdir / "cms_context_summary.tsv",
        "expression_cpg_associations": outdir / "expression_cpg_associations.tsv",
        "stromal_cpg_associations": outdir / "stromal_cpg_associations.tsv",
        "settings": outdir / "settings.json",
        "gate_report": review_path,
    }
    coverage.to_csv(outputs["cpg_coverage"], sep="\t", index=False)
    audit.to_csv(outputs["sample_linkage_audit"], sep="\t", index=False)
    cms_out.to_csv(outputs["cms_cpg_contrasts"], sep="\t", index=False)
    cms_summary.to_csv(outputs["cms_context_summary"], sep="\t", index=False)
    expr_out.to_csv(outputs["expression_cpg_associations"], sep="\t", index=False)
    stroma.to_csv(outputs["stromal_cpg_associations"], sep="\t", index=False)
    settings = {
        "analysis_unit": "individual_CpG",
        "planned_cpgs": int(len(targets)),
        "observed_cpgs": int(coverage["present_in_beta"].sum()),
        "missing_cpgs": int((~coverage["present_in_beta"]).sum()),
        "no_gene_or_panel_methylation_averages": True,
        "missing_test_internal_p": 1.0,
        "cms_family_size": contract.secondary_context_family,
        "expression_family_size": contract.secondary_context_family,
        "partial_spearman_model": "Pearson correlation of rank residuals after OLS adjustment for raw age plus sex/site dummy variables; p uses t distribution with df = n - rank(X) - 1",
        "seed": contract.seed,
        "n_boot": contract.n_boot,
        "beta_path": str(beta_path),
        "methylation_metadata_path": str(methylation_meta_path),
        "rna_metadata_path": str(rna_meta_path),
        "cms_path": str(cms_path),
        "expression_path": str(expression_path),
    }
    outputs["settings"].write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
    write_gate_report(review_path, coverage, audit, cms_summary)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=Path, default=DEFAULT_BETA)
    parser.add_argument("--methylation-metadata", type=Path, default=DEFAULT_METHYLATION_META)
    parser.add_argument("--rna-metadata", type=Path, default=DEFAULT_RNA_META)
    parser.add_argument("--cms", type=Path, default=DEFAULT_CMS)
    parser.add_argument("--expression", type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixed-probes", type=Path, default=DEFAULT_FIXED_PROBES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--n-boot", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_context(
            beta_path=args.beta,
            methylation_meta_path=args.methylation_metadata,
            rna_meta_path=args.rna_metadata,
            cms_path=args.cms,
            expression_path=args.expression,
            contract_path=args.contract,
            fixed_probes_path=args.fixed_probes,
            outdir=args.outdir,
            review_path=args.review,
            n_boot=args.n_boot,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
