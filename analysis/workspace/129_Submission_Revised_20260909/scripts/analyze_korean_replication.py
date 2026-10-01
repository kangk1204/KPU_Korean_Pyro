"""Fixed ten-gene Korean external replication engine.

The script scores the frozen 77-probe panel in beta matrices, validates
patient-paired tumor and adjacent-normal metadata, and reports paired effects
and coordination statistics without selecting genes from the target cohort.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "registry" / "analysis_contract.json"
DEFAULT_FIXED_PROBES = ROOT / "registry" / "fixed_probes.json"


@dataclass(frozen=True)
class Contract:
    genes: list[str]
    min_valid: dict[str, int]
    seed: int
    n_boot: int
    n_permutation: int
    fixed_probe_sha256: str | None = None


def read_contract(path: Path = DEFAULT_CONTRACT) -> Contract:
    data = json.loads(path.read_text())
    return Contract(
        genes=list(data["genes"]),
        min_valid={str(k): int(v) for k, v in data["min_valid"].items()},
        seed=int(data["seed"]),
        n_boot=int(data["n_boot"]),
        n_permutation=int(data["n_permutation"]),
        fixed_probe_sha256=data.get("fixed_probe_sha256"),
    )


def read_fixed_probes(path: Path = DEFAULT_FIXED_PROBES) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    return {gene: list(probes) for gene, probes in data.items()}


def stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return (int(seed) + int(digest[:8], 16)) % (2**32)


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return q
    ordered = finite[np.argsort(p[finite])]
    raw = p[ordered] * len(p) / np.arange(1, len(finite) + 1)
    q[ordered] = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    return q


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_positive_count(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def verify_fixed_probe_hash(contract: Contract, fixed_probes_path: Path) -> None:
    if contract.fixed_probe_sha256:
        observed = file_sha256(fixed_probes_path)
        if observed != contract.fixed_probe_sha256:
            raise ValueError(
                "fixed probe registry SHA256 mismatch; "
                f"contract={contract.fixed_probe_sha256}, observed={observed}"
            )


def _is_missing_token(value: object) -> bool:
    return pd.isna(value) or str(value).strip() in {"", "NA", "NaN", "nan", "NAN", "null", "NULL"}


def load_beta(path: Path) -> pd.DataFrame:
    with Path(path).open(newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    samples = [str(value).strip() for value in header[1:]]
    if len(samples) != len(set(samples)):
        seen = set()
        dup = []
        for sample in samples:
            if sample in seen and sample not in dup:
                dup.append(sample)
            seen.add(sample)
        raise ValueError(f"duplicate sample identifiers in beta matrix header: {dup[:5]}")
    beta_raw = pd.read_csv(path, sep="\t", index_col=0, dtype=str, keep_default_na=False)
    bad_tokens = []
    for probe, row in beta_raw.iterrows():
        for sample, value in row.items():
            if _is_missing_token(value):
                continue
            try:
                numeric = float(str(value).strip())
            except ValueError:
                bad_tokens.append((probe, sample, value))
                continue
            if not np.isfinite(numeric):
                bad_tokens.append((probe, sample, value))
    if bad_tokens:
        probe, sample, value = bad_tokens[0]
        raise ValueError(f"non-finite or nonnumeric beta value at probe={probe} sample={sample}: {value!r}")
    beta = pd.read_csv(path, sep="\t", index_col=0, na_values=["", "NA", "NaN", "nan", "NAN", "null", "NULL"])
    beta.index = beta.index.astype(str)
    beta.columns = beta.columns.astype(str)
    if beta.index.duplicated().any():
        dup = beta.index[beta.index.duplicated()].unique().tolist()[:5]
        raise ValueError(f"duplicate probe identifiers: {dup}")
    beta = beta.apply(pd.to_numeric, errors="raise")
    values = beta.to_numpy(dtype=float)
    bad = (~np.isfinite(values) & ~pd.isna(values)) | (np.isfinite(values) & ((values < 0) | (values > 1)))
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise ValueError(
            f"beta value outside [0,1] at probe={beta.index[row]} sample={beta.columns[col]}"
        )
    return beta


def _parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"invalid pair_verified value: {value!r}")


def load_metadata(path: Path, beta_samples: Iterable[str]) -> pd.DataFrame:
    meta = pd.read_csv(path, sep="\t", dtype=str)
    required = {"cohort", "patient_id", "sample_id", "tissue", "pair_verified"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing required columns: {sorted(missing)}")
    meta = meta.copy()
    for col in required:
        if meta[col].isna().any() or meta[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"metadata {col} contains missing or blank values")
        meta[col] = meta[col].str.strip()
    if meta["sample_id"].duplicated().any():
        dup = meta.loc[meta["sample_id"].duplicated(), "sample_id"].unique().tolist()[:5]
        raise ValueError(f"duplicate metadata sample_id values: {dup}")
    if not set(meta["tissue"]).issubset({"T", "N"}):
        bad = sorted(set(meta["tissue"]) - {"T", "N"})
        raise ValueError(f"metadata tissue must be T or N, found: {bad}")
    meta["pair_verified"] = meta["pair_verified"].map(_parse_bool)
    beta_samples = set(map(str, beta_samples))
    meta_samples = set(meta["sample_id"])
    if beta_samples != meta_samples:
        missing_beta = sorted(meta_samples - beta_samples)[:5]
        missing_meta = sorted(beta_samples - meta_samples)[:5]
        raise ValueError(
            "metadata and beta samples must match exactly; "
            f"missing_beta={missing_beta}, missing_metadata={missing_meta}"
        )
    verified = meta.loc[meta["pair_verified"]].copy()
    duplicate_tissue = verified.duplicated(["cohort", "patient_id", "tissue"], keep=False)
    if duplicate_tissue.any():
        bad = verified.loc[duplicate_tissue, ["cohort", "patient_id", "tissue"]].drop_duplicates()
        raise ValueError(f"duplicate verified patient/tissue rows: {bad.to_dict('records')[:5]}")
    counts = verified.groupby(["cohort", "patient_id"])["tissue"].agg(lambda x: set(x))
    incomplete = counts[counts.map(lambda s: s != {"T", "N"})]
    if len(incomplete):
        examples = [
            {"cohort": idx[0], "patient_id": idx[1], "tissues": ",".join(sorted(tissues))}
            for idx, tissues in incomplete.head(5).items()
        ]
        raise ValueError(f"pair_verified rows must contain exactly one T and one N: {examples}")
    return meta


def score_beta(
    beta: pd.DataFrame,
    fixed_probes: dict[str, list[str]],
    contract: Contract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(index=beta.columns)
    coverage_rows = []
    for gene in contract.genes:
        targets = list(fixed_probes[gene])
        minimum = int(contract.min_valid[gene])
        if minimum != math.ceil(0.8 * len(targets)):
            raise ValueError(f"contract min_valid disagrees with 80% rule for {gene}")
        values = beta.reindex(targets).T
        n_valid = values.notna().sum(axis=1)
        scores[gene] = values.mean(axis=1).where(n_valid >= minimum)
        coverage_rows.append(
            {
                "gene": gene,
                "n_fixed": len(targets),
                "minimum_valid": minimum,
                "n_present_probes": int(sum(probe in beta.index for probe in targets)),
                "n_samples": int(beta.shape[1]),
                "n_valid_scores": int(scores[gene].notna().sum()),
                "missing_probes": ";".join(probe for probe in targets if probe not in beta.index),
            }
        )
    scores["panel_mean"] = scores[contract.genes].mean(axis=1).where(
        scores[contract.genes].notna().all(axis=1)
    )
    scores.index.name = "sample_id"
    return scores, pd.DataFrame(coverage_rows)


def _bca_mean_ci(delta: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, str, int]:
    delta = np.asarray(delta, dtype=float)
    theta = float(np.mean(delta))
    boot = delta[rng.integers(0, len(delta), size=(n_boot, len(delta)))].mean(axis=1)
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return np.nan, np.nan, "no_valid_bootstrap", 0
    percentile = np.quantile(boot, [0.025, 0.975])
    if len(delta) < 3 or np.allclose(delta, delta[0]):
        return float(percentile[0]), float(percentile[1]), "percentile_fallback", int(len(boot))
    less = np.mean(boot < theta) + 0.5 * np.mean(boot == theta)
    less = min(max(less, 1 / (2 * len(boot))), 1 - 1 / (2 * len(boot)))
    z0 = stats.norm.ppf(less)
    jack = np.array([(np.sum(delta) - x) / (len(delta) - 1) for x in delta])
    jack_mean = jack.mean()
    centered = jack_mean - jack
    denom = 6.0 * np.sum(centered**2) ** 1.5
    if denom == 0 or not np.isfinite(denom):
        return float(percentile[0]), float(percentile[1]), "percentile_fallback", int(len(boot))
    accel = float(np.sum(centered**3) / denom)
    z_alpha = stats.norm.ppf([0.025, 0.975])
    adjusted = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - accel * (z0 + z_alpha)))
    if not np.all(np.isfinite(adjusted)) or adjusted[0] >= adjusted[1]:
        return float(percentile[0]), float(percentile[1]), "percentile_fallback", int(len(boot))
    ci = np.quantile(boot, adjusted)
    return float(ci[0]), float(ci[1]), "bca", int(len(boot))


def _analysis_frame(meta: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    frame = meta.merge(scores.reset_index(), on="sample_id", how="left", validate="one_to_one")
    return frame


def _paired_delta_matrix(frame: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    verified = frame.loc[frame["pair_verified"]].copy()
    rows = []
    for (cohort, patient), group in verified.groupby(["cohort", "patient_id"], sort=True):
        wide = group.set_index("tissue")[genes]
        if not {"T", "N"}.issubset(wide.index):
            continue
        delta = (wide.loc["T"] - wide.loc["N"]) * 100.0
        rows.append({"cohort": cohort, "patient_id": patient, **delta.to_dict()})
    return pd.DataFrame(rows, columns=["cohort", "patient_id", *genes])


def paired_panel_effect(
    frame: pd.DataFrame,
    contract: Contract,
    n_boot: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    n_boot = contract.n_boot if n_boot is None else int(n_boot)
    seed = contract.seed if seed is None else int(seed)
    delta = _paired_delta_matrix(frame, ["panel_mean"])
    rows = []
    for cohort in sorted(frame["cohort"].unique()):
        cohort_delta = delta.loc[delta["cohort"].eq(cohort)] if "cohort" in delta else delta.iloc[0:0]
        values = cohort_delta["panel_mean"].dropna().to_numpy(dtype=float) if "panel_mean" in cohort_delta else np.array([])
        row = {
            "cohort": cohort,
            "measure": "panel_mean",
            "n_pairs": int(len(values)),
            "mean_delta_pp": np.nan,
            "sd_delta_pp": np.nan,
            "ci_low_pp": np.nan,
            "ci_high_pp": np.nan,
            "ci_method": "not_estimated",
            "paired_t_p": np.nan,
            "wilcoxon_p": np.nan,
            "n_tumor_greater": np.nan,
            "bootstrap_valid": 0,
            "n_boot": n_boot,
            "status": "insufficient_data",
            "reason": "fewer_than_2_complete_panel_pairs",
        }
        if len(values) >= 2:
            rng = np.random.default_rng(stable_seed(seed, cohort, "panel_mean", "paired_panel"))
            lo, hi, ci_method, valid = _bca_mean_ci(values, rng, n_boot)
            t_p = stats.ttest_1samp(values, 0.0).pvalue
            row.update(
                {
                    "mean_delta_pp": float(np.mean(values)),
                    "sd_delta_pp": float(np.std(values, ddof=1)),
                    "ci_low_pp": lo,
                    "ci_high_pp": hi,
                    "ci_method": ci_method,
                    "paired_t_p": float(t_p),
                    "wilcoxon_p": float(stats.wilcoxon(values).pvalue) if np.any(values != 0) else 1.0,
                    "n_tumor_greater": int(np.sum(values > 0)),
                    "bootstrap_valid": valid,
                    "status": "estimated",
                    "reason": "",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gene_effects(
    frame: pd.DataFrame,
    contract: Contract,
    n_boot: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    n_boot = contract.n_boot if n_boot is None else int(n_boot)
    seed = contract.seed if seed is None else int(seed)
    delta = _paired_delta_matrix(frame, contract.genes)
    cohorts = sorted(frame["cohort"].unique())
    all_rows = []
    for cohort in cohorts:
        cohort_delta = delta.loc[delta["cohort"].eq(cohort)].copy()
        p_internal = []
        start = len(all_rows)
        for gene in contract.genes:
            values = cohort_delta[gene].dropna().to_numpy(dtype=float) if gene in cohort_delta else np.array([])
            row = {
                "cohort": cohort,
                "gene": gene,
                "n_pairs": int(len(values)),
                "mean_delta_pp": np.nan,
                "sd_delta_pp": np.nan,
                "ci_low_pp": np.nan,
                "ci_high_pp": np.nan,
                "ci_method": "not_estimated",
                "paired_t_p": np.nan,
                "wilcoxon_p": np.nan,
                "bh_q": np.nan,
                "n_tumor_greater": np.nan,
                "bootstrap_valid": 0,
                "n_boot": n_boot,
                "status": "insufficient_data",
                "reason": "fewer_than_2_complete_pairs",
            }
            if len(values) >= 2:
                rng = np.random.default_rng(stable_seed(seed, cohort, gene, "paired_gene"))
                lo, hi, ci_method, valid = _bca_mean_ci(values, rng, n_boot)
                t_p = stats.ttest_1samp(values, 0.0).pvalue
                try:
                    w_p = stats.wilcoxon(values).pvalue if np.any(values != 0) else 1.0
                except ValueError:
                    w_p = np.nan
                row.update(
                    {
                        "mean_delta_pp": float(np.mean(values)),
                        "sd_delta_pp": float(np.std(values, ddof=1)),
                        "ci_low_pp": lo,
                        "ci_high_pp": hi,
                        "ci_method": ci_method,
                        "paired_t_p": float(t_p),
                        "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
                        "n_tumor_greater": int(np.sum(values > 0)),
                        "bootstrap_valid": valid,
                        "status": "estimated",
                        "reason": "",
                    }
                )
                p_internal.append(float(t_p) if np.isfinite(t_p) else 1.0)
            else:
                p_internal.append(1.0)
            all_rows.append(row)
        q = bh_adjust(p_internal)
        for offset, value in enumerate(q):
            if all_rows[start + offset]["status"] == "estimated":
                all_rows[start + offset]["bh_q"] = float(value)
    return pd.DataFrame(all_rows)


def _spearman_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.isclose(np.nanstd(x), 0.0) or np.isclose(np.nanstd(y), 0.0):
        return np.nan, np.nan
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _rank_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    ranks = np.apply_along_axis(stats.rankdata, 0, values, method="average")
    constant = np.isclose(np.nanstd(values, axis=0), 0.0)
    ranks[:, constant] = np.nan
    return ranks


def _rank_corrcoef_from_ranks(ranks: np.ndarray) -> np.ndarray:
    centered = ranks - ranks.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.sum(centered**2, axis=0))
    out = np.full((ranks.shape[1], ranks.shape[1]), np.nan)
    valid = scale > 0
    if valid.any():
        z = centered[:, valid] / scale[valid]
        out[np.ix_(valid, valid)] = z.T @ z
    return out


def _spearman_matrix(values: np.ndarray) -> np.ndarray:
    return _rank_corrcoef_from_ranks(_rank_matrix(values))


def _corr_values(corr: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(corr.shape[0], k=1)
    return corr[idx]


def delta_correlations(frame: pd.DataFrame, contract: Contract) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta = _paired_delta_matrix(frame, contract.genes)
    corr_rows = []
    summary_rows = []
    for cohort in sorted(frame["cohort"].unique()):
        complete = delta.loc[delta["cohort"].eq(cohort), ["patient_id", *contract.genes]].dropna()
        p_internal = []
        start = len(corr_rows)
        for i, gene_x in enumerate(contract.genes):
            for gene_y in contract.genes[i + 1 :]:
                rho, p = _spearman_pair(complete[gene_x].to_numpy(), complete[gene_y].to_numpy())
                status = "estimated" if np.isfinite(rho) else "insufficient_data"
                corr_rows.append(
                    {
                        "cohort": cohort,
                        "gene_x": gene_x,
                        "gene_y": gene_y,
                        "n_complete_10gene_pairs": int(len(complete)),
                        "rho": rho,
                        "p": p,
                        "bh_q": np.nan,
                        "status": status,
                        "reason": "" if status == "estimated" else "fewer_than_3_complete_10gene_pairs_or_constant",
                    }
                )
                p_internal.append(float(p) if np.isfinite(p) else 1.0)
        q = bh_adjust(p_internal)
        for offset, value in enumerate(q):
            if corr_rows[start + offset]["status"] == "estimated":
                corr_rows[start + offset]["bh_q"] = float(value)
        summary_rows.append(_coordination_summary(cohort, complete, contract))
    return pd.DataFrame(corr_rows), pd.DataFrame(summary_rows)


def pairwise_complete_delta_correlations(frame: pd.DataFrame, contract: Contract) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sensitivity correlations using each gene pair's complete patients.

    This is not the primary complete-ten-gene coordination analysis. It keeps
    all 45 planned gene pairs and records the pairwise sample-size matrix.
    """
    delta = _paired_delta_matrix(frame, contract.genes)
    corr_rows = []
    n_rows = []
    for cohort in sorted(frame["cohort"].unique()):
        cohort_delta = delta.loc[delta["cohort"].eq(cohort)].copy()
        n_matrix = pd.DataFrame(0, index=contract.genes, columns=contract.genes, dtype=int)
        p_internal = []
        start = len(corr_rows)
        for i, gene_x in enumerate(contract.genes):
            n_matrix.loc[gene_x, gene_x] = int(cohort_delta[gene_x].notna().sum()) if gene_x in cohort_delta else 0
            for gene_y in contract.genes[i + 1 :]:
                pair = cohort_delta[[gene_x, gene_y]].dropna() if {gene_x, gene_y}.issubset(cohort_delta) else pd.DataFrame()
                n = int(len(pair))
                n_matrix.loc[gene_x, gene_y] = n
                n_matrix.loc[gene_y, gene_x] = n
                rho, p = (
                    _spearman_pair(pair[gene_x].to_numpy(), pair[gene_y].to_numpy())
                    if n >= 3
                    else (np.nan, np.nan)
                )
                status = "estimated" if np.isfinite(rho) else "insufficient_data"
                reason = "" if status == "estimated" else "fewer_than_3_pairwise_pairs_or_constant"
                corr_rows.append(
                    {
                        "cohort": cohort,
                        "analysis": "pairwise_complete_sensitivity",
                        "gene_x": gene_x,
                        "gene_y": gene_y,
                        "n_pairwise_pairs": n,
                        "rho": rho,
                        "p": p,
                        "bh_q": np.nan,
                        "status": status,
                        "reason": reason,
                    }
                )
                p_internal.append(float(p) if np.isfinite(p) else 1.0)
        for gene_y in contract.genes:
            n_matrix.loc[gene_y, gene_y] = int(cohort_delta[gene_y].notna().sum()) if gene_y in cohort_delta else 0
        q = bh_adjust(p_internal)
        for offset, value in enumerate(q):
            if corr_rows[start + offset]["status"] == "estimated":
                corr_rows[start + offset]["bh_q"] = float(value)
        for gene_x in contract.genes:
            n_rows.append(
                {
                    "cohort": cohort,
                    "gene": gene_x,
                    **{gene_y: int(n_matrix.loc[gene_x, gene_y]) for gene_y in contract.genes},
                }
            )
    return pd.DataFrame(corr_rows), pd.DataFrame(n_rows)


def eligible_gene_coordination(frame: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    """Global coordination for the largest predefined eligible gene subset.

    Used only as a labeled fallback when complete ten-gene pairs are not
    available. Genes are eligible when they have at least three paired deltas,
    and the reported subset requires at least three complete patients across
    the eligible genes. It is not named or interpreted as a ten-gene panel.
    """
    delta = _paired_delta_matrix(frame, contract.genes)
    rows = []
    for cohort in sorted(frame["cohort"].unique()):
        cohort_delta = delta.loc[delta["cohort"].eq(cohort), ["patient_id", *contract.genes]].copy()
        complete10 = cohort_delta.dropna()
        eligible = [gene for gene in contract.genes if gene in cohort_delta and cohort_delta[gene].notna().sum() >= 3]
        base = {
            "cohort": cohort,
            "analysis": "eligible_gene_subset_coordination",
            "n_complete_10gene_pairs": int(len(complete10)),
            "n_eligible_genes": int(len(eligible)),
            "eligible_genes": ";".join(eligible),
            "n_complete_eligible_pairs": 0,
            "median_rho": np.nan,
            "rho_ci_low": np.nan,
            "rho_ci_high": np.nan,
            "positive_correlations": 0,
            "estimated_correlations": 0,
            "permutation_p_positive_direction": np.nan,
            "permutation_method": "independent_column_permutation",
            "n_boot": contract.n_boot,
            "n_permutation": contract.n_permutation,
            "status": "not_applicable",
            "reason": "complete_10gene_coordination_available",
        }
        if len(complete10) >= 3:
            rows.append(base)
            continue
        if len(eligible) < 2:
            base.update(status="insufficient_data", reason="fewer_than_2_eligible_genes")
            rows.append(base)
            continue
        complete_eligible = cohort_delta[["patient_id", *eligible]].dropna()
        base["n_complete_eligible_pairs"] = int(len(complete_eligible))
        if len(complete_eligible) < 3:
            base.update(status="insufficient_data", reason="fewer_than_3_complete_eligible_pairs")
            rows.append(base)
            continue
        subset_contract = Contract(
            genes=eligible,
            min_valid={gene: contract.min_valid[gene] for gene in eligible},
            seed=contract.seed,
            n_boot=contract.n_boot,
            n_permutation=contract.n_permutation,
            fixed_probe_sha256=contract.fixed_probe_sha256,
        )
        summary = _coordination_summary(cohort, complete_eligible, subset_contract)
        base.update(
            {
                "median_rho": summary["median_rho"],
                "rho_ci_low": summary["rho_ci_low"],
                "rho_ci_high": summary["rho_ci_high"],
                "positive_correlations": summary["positive_correlations"],
                "estimated_correlations": summary["estimated_correlations"],
                "permutation_p_positive_direction": summary["permutation_p_positive_direction"],
                "status": summary["status"],
                "reason": summary["reason"],
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def _median_rho(matrix: pd.DataFrame, genes: list[str]) -> float:
    corr = _spearman_matrix(matrix[genes].to_numpy(dtype=float))
    values = _corr_values(corr)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def _coordination_summary(cohort: str, complete: pd.DataFrame, contract: Contract) -> dict[str, object]:
    n = len(complete)
    base = {
        "cohort": cohort,
        "n_complete_10gene_pairs": int(n),
        "median_rho": np.nan,
        "rho_ci_low": np.nan,
        "rho_ci_high": np.nan,
        "positive_correlations": 0,
        "estimated_correlations": 0,
        "permutation_p_positive_direction": np.nan,
        "permutation_method": "independent_column_permutation",
        "n_boot": contract.n_boot,
        "n_permutation": contract.n_permutation,
        "status": "insufficient_data",
        "reason": "fewer_than_3_complete_10gene_pairs",
    }
    if n < 3:
        return base
    matrix = complete[contract.genes].to_numpy(dtype=float)
    ranks = _rank_matrix(matrix)
    observed_rhos = _corr_values(_rank_corrcoef_from_ranks(ranks))
    observed_rhos = observed_rhos[np.isfinite(observed_rhos)]
    if len(observed_rhos) == 0:
        base["reason"] = "all_gene_deltas_constant"
        return base
    observed = float(np.median(observed_rhos))
    rng = np.random.default_rng(stable_seed(contract.seed, cohort, "coordination"))
    boot = []
    for _ in range(contract.n_boot):
        sample = matrix[rng.integers(0, n, size=n), :]
        values = _corr_values(_spearman_matrix(sample))
        values = values[np.isfinite(values)]
        value = float(np.median(values)) if len(values) else np.nan
        if np.isfinite(value):
            boot.append(value)
    perm = []
    for _ in range(contract.n_permutation):
        shuffled = np.column_stack([rng.permutation(ranks[:, col]) for col in range(ranks.shape[1])])
        values = _corr_values(_rank_corrcoef_from_ranks(shuffled))
        values = values[np.isfinite(values)]
        value = float(np.median(values)) if len(values) else np.nan
        if np.isfinite(value):
            perm.append(value)
    if boot:
        lo, hi = np.quantile(boot, [0.025, 0.975])
    else:
        lo, hi = np.nan, np.nan
    p_perm = (1 + np.sum(np.asarray(perm) >= observed)) / (len(perm) + 1) if perm else np.nan
    base.update(
        {
            "median_rho": observed,
            "rho_ci_low": float(lo) if np.isfinite(lo) else np.nan,
            "rho_ci_high": float(hi) if np.isfinite(hi) else np.nan,
            "positive_correlations": int(np.sum(np.asarray(observed_rhos) > 0)),
            "estimated_correlations": int(len(observed_rhos)),
            "permutation_p_positive_direction": float(p_perm) if np.isfinite(p_perm) else np.nan,
            "status": "estimated" if len(observed_rhos) == (len(contract.genes) * (len(contract.genes) - 1) // 2) else "partial_estimated",
            "reason": "" if len(observed_rhos) == (len(contract.genes) * (len(contract.genes) - 1) // 2) else "some_gene_pair_correlations_constant",
        }
    )
    return base


def pca_loadings(frame: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    delta = _paired_delta_matrix(frame, contract.genes)
    rows = []
    for cohort in sorted(frame["cohort"].unique()):
        complete = delta.loc[delta["cohort"].eq(cohort), contract.genes].dropna()
        if len(complete) < 3:
            for gene in contract.genes:
                rows.append(
                    {
                        "cohort": cohort,
                        "component": "PC1",
                        "gene": gene,
                        "loading": np.nan,
                        "variance_explained": np.nan,
                        "n_complete_10gene_pairs": int(len(complete)),
                        "status": "insufficient_data",
                    }
                )
            continue
        z = (complete - complete.mean(axis=0)) / complete.std(axis=0, ddof=1)
        if z.isna().any().any() or np.isclose(z.std(axis=0, ddof=1), 0).any():
            loadings = np.full(len(contract.genes), np.nan)
            variance = np.nan
            status = "constant_gene_delta"
        else:
            _, s, vt = np.linalg.svd(z.to_numpy(dtype=float), full_matrices=False)
            loadings = vt[0].copy()
            if np.nansum(loadings) < 0:
                loadings *= -1
            variance = float((s[0] ** 2) / np.sum(s**2))
            status = "estimated"
        for gene, loading in zip(contract.genes, loadings):
            rows.append(
                {
                    "cohort": cohort,
                    "component": "PC1",
                    "gene": gene,
                    "loading": float(loading) if np.isfinite(loading) else np.nan,
                    "variance_explained": variance,
                    "n_complete_10gene_pairs": int(len(complete)),
                    "status": status,
                }
            )
    return pd.DataFrame(rows)


def run_analysis(
    beta_path: Path,
    metadata_path: Path,
    outdir: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    fixed_probes_path: Path = DEFAULT_FIXED_PROBES,
    n_boot: int | None = None,
    n_permutation: int | None = None,
    seed: int | None = None,
) -> dict[str, Path]:
    contract = read_contract(contract_path)
    if n_boot is not None or n_permutation is not None or seed is not None:
        contract = Contract(
            genes=contract.genes,
            min_valid=contract.min_valid,
            seed=contract.seed if seed is None else int(seed),
            n_boot=contract.n_boot if n_boot is None else validate_positive_count("n_boot", n_boot),
            n_permutation=(
                contract.n_permutation
                if n_permutation is None
                else validate_positive_count("n_permutation", n_permutation)
            ),
            fixed_probe_sha256=contract.fixed_probe_sha256,
        )
    fixed_probes = read_fixed_probes(fixed_probes_path)
    verify_fixed_probe_hash(contract, fixed_probes_path)
    missing = set(contract.genes) - set(fixed_probes)
    if missing:
        raise ValueError(f"fixed probe registry missing genes: {sorted(missing)}")
    beta = load_beta(beta_path)
    meta = load_metadata(metadata_path, beta.columns)
    scores, coverage = score_beta(beta, fixed_probes, contract)
    frame = _analysis_frame(meta, scores)
    paired = paired_gene_effects(frame, contract)
    panel = paired_panel_effect(frame, contract)
    correlations, coordination = delta_correlations(frame, contract)
    eligible = eligible_gene_coordination(frame, contract)
    pairwise, pairwise_n = pairwise_complete_delta_correlations(frame, contract)
    pca = pca_loadings(frame, contract)

    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sample_scores": outdir / "sample_scores.tsv",
        "probe_coverage": outdir / "probe_coverage.tsv",
        "paired_gene_effects": outdir / "paired_gene_effects.tsv",
        "paired_panel_effect": outdir / "paired_panel_effect.tsv",
        "delta_correlations": outdir / "delta_correlations.tsv",
        "coordination_summary": outdir / "coordination_summary.tsv",
        "eligible_coordination_summary": outdir / "eligible_coordination_summary.tsv",
        "pairwise_complete_delta_correlations": outdir / "pairwise_complete_delta_correlations.tsv",
        "pairwise_complete_n_matrix": outdir / "pairwise_complete_n_matrix.tsv",
        "pca_loadings": outdir / "pca_loadings.tsv",
        "settings": outdir / "settings.json",
    }
    scores.reset_index().to_csv(outputs["sample_scores"], sep="\t", index=False)
    coverage.to_csv(outputs["probe_coverage"], sep="\t", index=False)
    paired.to_csv(outputs["paired_gene_effects"], sep="\t", index=False)
    panel.to_csv(outputs["paired_panel_effect"], sep="\t", index=False)
    correlations.to_csv(outputs["delta_correlations"], sep="\t", index=False)
    coordination.to_csv(outputs["coordination_summary"], sep="\t", index=False)
    eligible.to_csv(outputs["eligible_coordination_summary"], sep="\t", index=False)
    pairwise.to_csv(outputs["pairwise_complete_delta_correlations"], sep="\t", index=False)
    pairwise_n.to_csv(outputs["pairwise_complete_n_matrix"], sep="\t", index=False)
    pca.to_csv(outputs["pca_loadings"], sep="\t", index=False)
    settings = {
        "beta_path": str(beta_path),
        "metadata_path": str(metadata_path),
        "contract_path": str(contract_path),
        "fixed_probes_path": str(fixed_probes_path),
        "beta_sha256": file_sha256(beta_path),
        "metadata_sha256": file_sha256(metadata_path),
        "contract_sha256": file_sha256(contract_path),
        "fixed_probe_sha256": file_sha256(fixed_probes_path),
        "genes": contract.genes,
        "seed": contract.seed,
        "n_boot": contract.n_boot,
        "n_permutation": contract.n_permutation,
        "paired_gene_rows_per_cohort": 10,
        "delta_correlation_rows_per_cohort": 45,
        "pairwise_complete_delta_correlation_rows_per_cohort": 45,
        "coverage_rule": "gene mean beta when valid probes >= ceil(0.8 * frozen target probes); panel_mean requires all ten genes",
    }
    outputs["settings"].write_text(json.dumps(settings, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", required=True, type=Path, help="Probe by sample beta TSV; first column is probe ID.")
    parser.add_argument("--metadata", required=True, type=Path, help="Sample metadata TSV.")
    parser.add_argument("--outdir", required=True, type=Path, help="Directory for TSV/JSON outputs.")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT, type=Path)
    parser.add_argument("--fixed-probes", default=DEFAULT_FIXED_PROBES, type=Path)
    parser.add_argument("--n-boot", type=int, default=None, help="Override bootstrap count for tests/fixtures.")
    parser.add_argument("--n-permutation", type=int, default=None, help="Override permutation count for tests/fixtures.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for tests/fixtures.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_analysis(
        beta_path=args.beta,
        metadata_path=args.metadata,
        outdir=args.outdir,
        contract_path=args.contract,
        fixed_probes_path=args.fixed_probes,
        n_boot=args.n_boot,
        n_permutation=args.n_permutation,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
