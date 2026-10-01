#!/usr/bin/env python3
"""Individual-CpG public context analyses for the frozen 77-probe panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SOURCE_119 = ROOT.parent / "119_Figure1_ABC_Revision_20260908"
SOURCE_115_RAW = ROOT.parent / "115_Public_Biology_20260905" / "data" / "raw"
RESULTS = ROOT / "results" / "public_context"
REGISTRY = ROOT / "registry"
CONTRACT = REGISTRY / "public_context_contract.json"
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "GFRA1", "UNC5C", "RALYL", "BEND5"]
N_BOOT = 5000
SEED = 20260908


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stable_seed(*parts: object) -> int:
    return (SEED + int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)) % (2**32)


def bh77(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return q
    ordered = finite[np.argsort(p[finite])]
    raw = p[ordered] * 77 / np.arange(1, len(finite) + 1)
    q[ordered] = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    return q


def read_fixed_probes(path: Path = REGISTRY / "fixed_probes.json") -> dict[str, list[str]]:
    return {gene: list(probes) for gene, probes in json.loads(path.read_text()).items()}


def fixed_cpg_table() -> pd.DataFrame:
    rows = []
    probes_by_gene = read_fixed_probes()
    context_path = SOURCE_119 / "registry" / "reviewer" / "probe_context.tsv"
    context = pd.read_csv(context_path, sep="\t")
    promoter = context.set_index("probe")["promoter_match"].astype(bool).to_dict()
    promoter_group = context.set_index("probe")["promoter_groups"].fillna("").astype(str).to_dict()
    for gene in GENES:
        for i, cpg in enumerate(probes_by_gene[gene], start=1):
            rows.append(
                {
                    "gene": gene,
                    "cpg": cpg,
                    "fixed_probe_order_within_gene": i,
                    "promoter_match": bool(promoter.get(cpg, False)),
                    "promoter_groups": promoter_group.get(cpg, ""),
                }
            )
    table = pd.DataFrame(rows)
    if len(table) != 77 or table["cpg"].duplicated().any():
        raise ValueError("fixed CpG table must contain 77 unique probes")
    return table


def load_beta(path: Path) -> pd.DataFrame:
    beta = pd.read_csv(path, sep="\t", index_col=0)
    beta.index = beta.index.astype(str)
    beta.columns = beta.columns.astype(str)
    if beta.index.duplicated().any() or beta.columns.duplicated().any():
        raise ValueError(f"duplicate probe or sample identifiers in {path}")
    beta = beta.apply(pd.to_numeric, errors="coerce")
    values = beta.to_numpy(float)
    if np.any(np.isfinite(values) & ((values < 0) | (values > 1))):
        raise ValueError(f"beta values outside [0,1] in {path}")
    return beta


def residualize(y: np.ndarray, covars: pd.DataFrame) -> tuple[np.ndarray, int]:
    parts = []
    for col in covars.columns:
        series = covars[col]
        if pd.api.types.is_numeric_dtype(series):
            x = pd.to_numeric(series, errors="coerce")
            if x.nunique(dropna=True) > 1:
                parts.append(((x - x.mean()) / x.std(ddof=0)).to_numpy(float).reshape(-1, 1))
        else:
            dummies = pd.get_dummies(series.astype(str), prefix=col, drop_first=True, dtype=float)
            dummies = dummies.loc[:, dummies.nunique(dropna=True) > 1]
            if len(dummies.columns):
                parts.append(dummies.to_numpy(float))
    design = np.column_stack([np.ones(len(y)), *parts]) if parts else np.ones((len(y), 1))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef, int(np.linalg.matrix_rank(design))


def design_matrix(covars: pd.DataFrame) -> np.ndarray:
    parts = []
    for col in covars.columns:
        series = covars[col]
        if pd.api.types.is_numeric_dtype(series):
            x = pd.to_numeric(series, errors="coerce")
            if x.nunique(dropna=True) > 1:
                scale = x.std(ddof=0)
                parts.append(((x - x.mean()) / scale if scale else x - x.mean()).to_numpy(float).reshape(-1, 1))
        else:
            dummies = pd.get_dummies(series.astype(str), prefix=col, drop_first=True, dtype=float)
            dummies = dummies.loc[:, dummies.nunique(dropna=True) > 1]
            if len(dummies.columns):
                parts.append(dummies.to_numpy(float))
    return np.column_stack([np.ones(len(covars)), *parts]) if parts else np.ones((len(covars), 1))


def residual_corr_with_design(x: np.ndarray, y: np.ndarray, design: np.ndarray) -> float:
    if len(x) != len(y) or len(x) != design.shape[0]:
        raise ValueError("residual correlation inputs have inconsistent row counts")
    xcoef, *_ = np.linalg.lstsq(design, x, rcond=None)
    ycoef, *_ = np.linalg.lstsq(design, y, rcond=None)
    rx = x - design @ xcoef
    ry = y - design @ ycoef
    if np.std(rx) == 0 or np.std(ry) == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_frame(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, object]:
    sub = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 6 or sub[x_col].nunique() <= 1 or sub[y_col].nunique() <= 1:
        return {"n": int(len(sub)), "effect": math.nan, "p": math.nan, "df_model": 0, "df_resid": math.nan, "status": "not_estimable"}
    rho, p = stats.spearmanr(sub[x_col], sub[y_col])
    return {"n": int(len(sub)), "effect": float(rho), "p": float(p), "df_model": 0, "df_resid": int(len(sub) - 2), "status": "estimated"}


def rank_residual_spearman(df: pd.DataFrame, x_col: str, y_col: str, covars: list[str]) -> dict[str, object]:
    sub = df[[x_col, y_col, *covars]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 8 or sub[x_col].nunique() <= 1 or sub[y_col].nunique() <= 1:
        return {"n": int(len(sub)), "effect": math.nan, "p": math.nan, "df_model": 0, "df_resid": math.nan, "status": "not_estimable"}
    xr = stats.rankdata(sub[x_col].to_numpy(float))
    yr = stats.rankdata(sub[y_col].to_numpy(float))
    rx, rank = residualize(xr, sub[covars])
    ry, _ = residualize(yr, sub[covars])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return {"n": int(len(sub)), "effect": math.nan, "p": math.nan, "df_model": rank, "df_resid": math.nan, "status": "not_estimable"}
    rho = float(np.corrcoef(rx, ry)[0, 1])
    df_resid = int(len(sub) - rank - 1)
    if df_resid <= 0 or abs(rho) >= 1:
        p = math.nan
    else:
        t = rho * math.sqrt(df_resid / max(1e-12, 1 - rho * rho))
        p = float(2 * stats.t.sf(abs(t), df_resid))
    return {"n": int(len(sub)), "effect": rho, "p": p, "df_model": rank, "df_resid": df_resid, "status": "estimated" if math.isfinite(p) else "not_estimable"}


def bootstrap_corr_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covars: list[str],
    adjusted: bool,
    seed: int,
    n_boot: int,
    unit_col: str = "patient",
) -> tuple[float, float, int]:
    keep = [x_col, y_col, *covars, unit_col]
    base = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(base) < 6 or base[x_col].nunique() <= 1 or base[y_col].nunique() <= 1:
        return math.nan, math.nan, 0
    if base[unit_col].duplicated().any():
        raise ValueError(f"correlation bootstrap unit is not unique: {unit_col}")
    rng = np.random.default_rng(seed)
    vals = []
    x_all = base[x_col].to_numpy(float)
    y_all = base[y_col].to_numpy(float)
    design_all = design_matrix(base[covars]) if adjusted else None
    if not adjusted:
        idx = rng.integers(0, len(base), size=(n_boot, len(base)))
        xr = stats.rankdata(x_all[idx], axis=1)
        yr = stats.rankdata(y_all[idx], axis=1)
        xr = xr - xr.mean(axis=1, keepdims=True)
        yr = yr - yr.mean(axis=1, keepdims=True)
        denom = np.sqrt((xr * xr).sum(axis=1) * (yr * yr).sum(axis=1))
        good = denom > 0
        vals_arr = np.full(n_boot, np.nan)
        vals_arr[good] = (xr[good] * yr[good]).sum(axis=1) / denom[good]
        vals_arr = vals_arr[np.isfinite(vals_arr)]
        if len(vals_arr) == 0:
            return math.nan, math.nan, 0
        lo, hi = np.quantile(vals_arr, [0.025, 0.975])
        return float(lo), float(hi), int(len(vals_arr))
    for _ in range(n_boot):
        idx = rng.integers(0, len(base), len(base))
        val = residual_corr_with_design(stats.rankdata(x_all[idx]), stats.rankdata(y_all[idx]), design_all[idx, :])
        if math.isfinite(float(val)):
            vals.append(float(val))
    if not vals:
        return math.nan, math.nan, 0
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi), len(vals)


def expression_colonomics_frame(cpgs: pd.DataFrame) -> pd.DataFrame:
    beta = load_beta(SOURCE_119 / "data" / "derived" / "Colonomics_beta.tsv.gz").T
    beta = beta.reindex(columns=cpgs["cpg"])
    beta.index.name = "sample"
    beta = beta.reset_index()
    samples = pd.read_csv(SOURCE_119 / "data" / "derived" / "Colonomics_samples.tsv", sep="\t")
    meth = samples.merge(beta, on="sample", validate="one_to_one")
    meth_t = meth.loc[meth["tissue"].eq("T") & (~meth["excluded"].astype(bool))].copy()
    expr_meta = pd.read_csv(SOURCE_119 / "data" / "derived" / "expression_Colonomics_samples.tsv", sep="\t")
    expr = pd.read_csv(SOURCE_119 / "data" / "derived" / "expression_Colonomics_gene.tsv", sep="\t", index_col=0)
    expr_t = expr_meta.loc[expr_meta["tissue"].eq("T")].copy()
    merged = expr_t.merge(meth_t, on="author_id", how="inner", suffixes=("_expr_meta", "_meth_meta"), validate="one_to_one")
    if not merged["patient_meth_meta"].is_unique:
        raise ValueError("Colonomics matched expression-methylation rows are not unique by patient")
    merged["patient"] = merged["patient_meth_meta"].astype(str)
    values = expr.reindex(merged["sample_expr_meta"])
    values.index = merged.index
    for gene in GENES:
        merged[f"{gene}_expr"] = values[gene]
    return merged.rename(columns={"age_meth_meta": "age", "sex_meth_meta": "sex", "site_meth_meta": "site"})


def collapse_same_cpg_technical_replicates(beta: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = meta.set_index("sample").join(beta.T, how="inner")
    rows = []
    audit = []
    for (patient, tissue), group in joined.groupby(["patient", "tissue"], dropna=False):
        sample_cols = beta.index.tolist()
        out = {"patient": str(patient), "tissue": tissue, "age": group["age"].iloc[0], "sex": group["sex"].iloc[0], "sample": ";".join(group.index.astype(str))}
        out.update(group[sample_cols].mean(axis=0, skipna=True).to_dict())
        rows.append(out)
        audit.append({"patient": str(patient), "tissue": tissue, "n_technical_replicates": int(len(group)), "samples": out["sample"], "status": "collapsed_same_cpg" if len(group) > 1 else "single_sample"})
    collapsed = pd.DataFrame(rows)
    return collapsed, pd.DataFrame(audit)


def expression_colocare_frame(cpgs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta = load_beta(SOURCE_119 / "data" / "raw" / "GSE101764_fixed77_beta.tsv.gz").reindex(cpgs["cpg"])
    meta = pd.read_csv(SOURCE_119 / "data" / "derived" / "expression_GSE101764_samples_raw.tsv", sep="\t")
    meta["patient"] = meta["patient"].astype(str)
    collapsed, audit = collapse_same_cpg_technical_replicates(beta, meta)
    meth_t = collapsed.loc[collapsed["tissue"].eq("T")].copy()
    expr_meta = pd.read_csv(SOURCE_119 / "data" / "derived" / "expression_GSE106582_samples.tsv", sep="\t")
    expr_meta["patient"] = expr_meta["patient"].astype(str)
    expr = pd.read_csv(SOURCE_119 / "data" / "derived" / "expression_GSE106582_gene.tsv", sep="\t", index_col=0)
    expr_t = expr_meta.loc[expr_meta["tissue"].eq("T")].copy()
    merged = expr_t.merge(meth_t, on=["patient", "tissue"], how="inner", suffixes=("_expr_meta", "_meth_meta"), validate="one_to_one")
    if not merged["patient"].is_unique:
        raise ValueError("ColoCare matched expression-methylation rows are not unique by patient")
    values = expr.reindex(merged["sample_expr_meta"])
    values.index = merged.index
    for gene in GENES:
        merged[f"{gene}_expr"] = values[gene]
    return merged, audit


def expression_rows(cpgs: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    colonomics = expression_colonomics_frame(cpgs)
    specs = [
        ("Colonomics", "unadjusted", colonomics, [], False),
        ("Colonomics", "rank_residual_age_sex_site_stroma", colonomics, ["age", "sex", "site", "stromal_score"], True),
    ]
    colocare, audit = expression_colocare_frame(cpgs)
    specs.append(("ColoCare_GSE101764_GSE106582", "unadjusted", colocare, [], False))
    rows = []
    for cohort, model, frame, covars, adjusted in specs:
        for rec in cpgs.to_dict("records"):
            cpg = rec["cpg"]
            gene = rec["gene"]
            if cpg not in frame.columns:
                est = {"n": 0, "effect": math.nan, "p": math.nan, "df_model": 0, "df_resid": math.nan, "status": "probe_absent"}
                ci_low, ci_high, valid = math.nan, math.nan, 0
            else:
                expr_col = f"{gene}_expr"
                est = rank_residual_spearman(frame, cpg, expr_col, covars) if adjusted else spearman_frame(frame, cpg, expr_col)
                ci_low, ci_high, valid = bootstrap_corr_ci(frame, cpg, expr_col, covars, adjusted, stable_seed(cohort, model, gene, cpg), n_boot)
            rows.append({**rec, "cohort": cohort, "model": model, "analysis": "methylation_expression_spearman", **est, "ci_low": ci_low, "ci_high": ci_high, "bootstrap_valid_draws": valid, "unit": "rho"})
    out = pd.DataFrame(rows)
    for (_, _), idx in out.groupby(["cohort", "model"]).groups.items():
        out.loc[idx, "q_BH77"] = bh77(out.loc[idx, "p"])
    return out, audit


def colonomics_context_rows(cpgs: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    beta = load_beta(SOURCE_119 / "data" / "derived" / "Colonomics_beta.tsv.gz").T.reindex(columns=cpgs["cpg"])
    samples = pd.read_csv(SOURCE_119 / "data" / "derived" / "Colonomics_samples.tsv", sep="\t")
    frame = samples.merge(beta.reset_index(names="sample"), on="sample", validate="one_to_one")
    frame = frame.loc[frame["tissue"].eq("T") & (~frame["excluded"].astype(bool))].copy()
    rows = []
    for rec in cpgs.to_dict("records"):
        cpg = rec["cpg"]
        if cpg not in frame.columns:
            st = {"n": 0, "effect": math.nan, "p": math.nan, "df_model": 0, "df_resid": math.nan, "status": "probe_absent"}
            ci_low, ci_high, valid = math.nan, math.nan, 0
            kw = {"n": 0, "effect": math.nan, "p": math.nan, "status": "probe_absent"}
        else:
            st = spearman_frame(frame, cpg, "stromal_score")
            ci_low, ci_high, valid = bootstrap_corr_ci(frame, cpg, "stromal_score", [], False, stable_seed("Colonomics", "stroma", cpg), n_boot)
            cms = frame[[cpg, "cms"]].replace([np.inf, -np.inf], np.nan).dropna()
            groups = [g[cpg].to_numpy(float) for _, g in cms.groupby("cms", dropna=True)]
            if len(groups) >= 2 and min(map(len, groups)) >= 2:
                h, p = stats.kruskal(*groups)
                kw = {"n": int(len(cms)), "effect": float(h), "p": float(p), "status": "estimated"}
            else:
                kw = {"n": int(len(cms)), "effect": math.nan, "p": math.nan, "status": "not_estimable"}
        rows.append({**rec, "cohort": "Colonomics", "model": "unadjusted", "analysis": "stromal_spearman", **st, "ci_low": ci_low, "ci_high": ci_high, "bootstrap_valid_draws": valid, "unit": "rho"})
        rows.append({**rec, "cohort": "Colonomics", "model": "source_cms_labels", "analysis": "CMS_global_Kruskal", "n": kw["n"], "effect": kw["effect"], "p": kw["p"], "df_model": math.nan, "df_resid": math.nan, "status": kw["status"], "ci_low": math.nan, "ci_high": math.nan, "bootstrap_valid_draws": 0, "unit": "H"})
    out = pd.DataFrame(rows)
    for analysis, idx in out.groupby("analysis").groups.items():
        out.loc[idx, "q_BH77"] = bh77(out.loc[idx, "p"])
    return out


def welch_bootstrap_ci(x: np.ndarray, y: np.ndarray, seed: int, n_boot: int) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if min(len(x), len(y)) < 3:
        return math.nan, math.nan, 0
    rng = np.random.default_rng(seed)
    bx = x[rng.integers(0, len(x), size=(n_boot, len(x)))]
    by = y[rng.integers(0, len(y), size=(n_boot, len(y)))]
    vals = np.nanmean(bx, axis=1) - np.nanmean(by, axis=1)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return math.nan, math.nan, 0
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi), int(len(vals))


def adjusted_mean_model(df: pd.DataFrame, y_col: str, high_col: str, covars: list[str]) -> dict[str, object]:
    sub = df[[y_col, high_col, *covars]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 10 or sub[y_col].nunique() <= 1 or sub[high_col].nunique() <= 1:
        return {"n": int(len(sub)), "effect": math.nan, "p": math.nan, "df_model": 0, "df_resid": math.nan, "status": "not_estimable"}
    design = pd.DataFrame({"high": sub[high_col].astype(float)}, index=sub.index)
    for col in covars:
        dummies = pd.get_dummies(sub[col].astype(str), prefix=col, drop_first=True, dtype=float)
        if len(dummies.columns):
            design = pd.concat([design, dummies], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    if np.linalg.matrix_rank(design) < design.shape[1] or len(sub) - design.shape[1] < 5:
        return {"n": int(len(sub)), "effect": math.nan, "p": math.nan, "df_model": int(np.linalg.matrix_rank(design)), "df_resid": int(len(sub) - np.linalg.matrix_rank(design)), "status": "rank_or_residual_df"}
    fit = sm.OLS(sub[y_col].astype(float), design).fit(cov_type="HC3")
    lo, hi = fit.conf_int().loc["high"]
    return {"n": int(len(sub)), "effect": float(fit.params["high"]), "p": float(fit.pvalues["high"]), "df_model": int(np.linalg.matrix_rank(design)), "df_resid": int(fit.df_resid), "status": "estimated", "ci_low": float(lo), "ci_high": float(hi)}


def match_cms_rows(cpgs: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    beta = load_beta(SOURCE_119 / "data" / "derived" / "GSE164811_beta.tsv.gz").T.reindex(columns=cpgs["cpg"])
    samples = pd.read_csv(SOURCE_119 / "data" / "derived" / "GSE164811_samples.tsv", sep="\t")
    frame = samples.merge(beta.reset_index(names="sample"), on="sample", validate="one_to_one")
    frame = frame.loc[frame["tissue"].eq("T")].copy()
    frame["cms_original"] = frame["cms"].astype(str)
    frame["cms_pair"] = frame["cms_original"].str.replace("CMS", "", regex=False).str.replace(".0", "", regex=False)
    frame = frame.loc[frame["cms_pair"].isin(["2", "3"])].copy()
    frame["is_cms3"] = frame["cms_pair"].eq("3").astype(int)
    rows = []
    for rec in cpgs.to_dict("records"):
        cpg = rec["cpg"]
        if cpg not in frame.columns:
            base = {"n": 0, "effect": math.nan, "p": math.nan, "status": "probe_absent", "ci_low": math.nan, "ci_high": math.nan, "bootstrap_valid_draws": 0}
            adj = {"n": 0, "effect": math.nan, "p": math.nan, "status": "probe_absent", "ci_low": math.nan, "ci_high": math.nan, "bootstrap_valid_draws": 0}
        else:
            sub = frame[[cpg, "cms_pair", "is_cms3", "sex", "site"]].dropna(subset=[cpg, "cms_pair"])
            x = sub.loc[sub["cms_pair"].eq("3"), cpg].to_numpy(float)
            y = sub.loc[sub["cms_pair"].eq("2"), cpg].to_numpy(float)
            if min(len(x), len(y)) >= 3:
                effect = float(np.mean(x) - np.mean(y))
                p = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
                lo, hi, valid = welch_bootstrap_ci(x, y, stable_seed("GSE164811", "CMS3-CMS2", cpg), n_boot)
                base = {"n": int(len(sub)), "n_high": int(len(x)), "n_low": int(len(y)), "effect": effect, "p": p, "status": "estimated", "ci_low": lo, "ci_high": hi, "bootstrap_valid_draws": valid}
            else:
                base = {"n": int(len(sub)), "n_high": int(len(x)), "n_low": int(len(y)), "effect": math.nan, "p": math.nan, "status": "not_estimable", "ci_low": math.nan, "ci_high": math.nan, "bootstrap_valid_draws": 0}
            adj_est = adjusted_mean_model(sub, cpg, "is_cms3", ["sex", "site"])
            adj = {**adj_est, "bootstrap_valid_draws": 0}
        rows.append({**rec, "cohort": "GSE164811_MATCH", "model": "Welch_CMS3_minus_CMS2", "analysis": "CMS3-CMS2", **base, "df_model": math.nan, "df_resid": math.nan, "unit": "beta_difference"})
        rows.append({**rec, "cohort": "GSE164811_MATCH", "model": "sex_site_adjusted_OLS_CMS3_minus_CMS2", "analysis": "CMS3-CMS2_adjusted", **adj, "unit": "beta_difference"})
    out = pd.DataFrame(rows)
    for model, idx in out.groupby("model").groups.items():
        out.loc[idx, "q_BH77"] = bh77(out.loc[idx, "p"])
    return out


def freeze_contract(cpgs: pd.DataFrame) -> dict[str, object]:
    inputs = [
        REGISTRY / "fixed_probes.json",
        SOURCE_119 / "registry" / "reviewer" / "probe_context.tsv",
        SOURCE_119 / "data" / "derived" / "Colonomics_beta.tsv.gz",
        SOURCE_119 / "data" / "derived" / "Colonomics_samples.tsv",
        SOURCE_119 / "data" / "derived" / "expression_Colonomics_gene.tsv",
        SOURCE_119 / "data" / "derived" / "expression_Colonomics_samples.tsv",
        SOURCE_119 / "data" / "raw" / "GSE101764_fixed77_beta.tsv.gz",
        SOURCE_119 / "data" / "derived" / "expression_GSE101764_samples_raw.tsv",
        SOURCE_119 / "data" / "derived" / "expression_GSE106582_gene.tsv",
        SOURCE_119 / "data" / "derived" / "expression_GSE106582_samples.tsv",
        SOURCE_119 / "data" / "derived" / "GSE164811_beta.tsv.gz",
        SOURCE_119 / "data" / "derived" / "GSE164811_samples.tsv",
        SOURCE_115_RAW / "Colonomics_CLX_ClinicalData.tab",
    ]
    contract = {
        "created_by": "scripts/analyze_public_cpg_context.py",
        "status": "frozen_before_individual_cpg_public_context_computation",
        "analysis_unit": "individual CpG probe",
        "fixed_cpg_count": int(len(cpgs)),
        "promoter_cpg_count": int(cpgs["promoter_match"].sum()),
        "gene_counts": {gene: int((cpgs["gene"] == gene).sum()) for gene in GENES},
        "forbidden_aggregation": ["score_beta gene mean", "panel_mean", "cross-CpG methylation averaging", "80% gene coverage score"],
        "allowed_aggregation": "verified technical replicates may be collapsed only within the same CpG, patient, and tissue",
        "bh_family_size": 77,
        "bootstrap_draws": N_BOOT,
        "seed": SEED,
        "inputs": [{"path": str(path.relative_to(ROOT.parent)) if path.is_relative_to(ROOT.parent) else str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in inputs if path.exists()],
    }
    write_json(CONTRACT, contract)
    return contract


def run(n_boot: int = N_BOOT) -> dict[str, object]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cpgs = fixed_cpg_table()
    cpgs.to_csv(RESULTS / "fixed77_cpg_annotation.tsv", sep="\t", index=False)
    contract = freeze_contract(cpgs)
    expr, tech_audit = expression_rows(cpgs, n_boot)
    context = colonomics_context_rows(cpgs, n_boot)
    match = match_cms_rows(cpgs, n_boot)
    expr.to_csv(RESULTS / "expression_cpg_associations.tsv", sep="\t", index=False)
    tech_audit.to_csv(RESULTS / "GSE101764_same_cpg_technical_replicate_audit.tsv", sep="\t", index=False)
    context.to_csv(RESULTS / "colonomics_cpg_context.tsv", sep="\t", index=False)
    match.to_csv(RESULTS / "match_cms_cpg_contrasts.tsv", sep="\t", index=False)
    combined = pd.concat([expr, context, match], ignore_index=True, sort=False)
    combined.to_csv(RESULTS / "all_public_cpg_context.tsv", sep="\t", index=False)
    summary = {
        "n_bootstrap_requested": int(n_boot),
        "fixed_cpg_rows": int(len(cpgs)),
        "promoter_cpg_rows": int(cpgs["promoter_match"].sum()),
        "expression_rows": int(len(expr)),
        "context_rows": int(len(context)),
        "match_rows": int(len(match)),
        "combined_rows": int(len(combined)),
        "estimated_by_output": {
            "|".join(map(str, key)): int(value)
            for key, value in combined.groupby(["cohort", "model", "analysis"])["status"].apply(lambda x: int(x.eq("estimated").sum())).to_dict().items()
        },
        "minimum_q_by_output": {
            "|".join(map(str, key)): float(value)
            for key, value in combined.groupby(["cohort", "model", "analysis"])["q_BH77"].min().dropna().to_dict().items()
        },
        "top_cpg_by_output": combined.loc[combined["q_BH77"].notna()].sort_values(["cohort", "model", "analysis", "q_BH77", "p"])[["cohort", "model", "analysis", "gene", "cpg", "effect", "p", "q_BH77"]].groupby(["cohort", "model", "analysis"]).head(1).to_dict("records"),
        "contract_sha256": sha256(CONTRACT),
    }
    outputs = [RESULTS / name for name in ["fixed77_cpg_annotation.tsv", "expression_cpg_associations.tsv", "GSE101764_same_cpg_technical_replicate_audit.tsv", "colonomics_cpg_context.tsv", "match_cms_cpg_contrasts.tsv", "all_public_cpg_context.tsv"]]
    summary["outputs"] = [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs]
    write_json(RESULTS / "run_summary.json", summary)
    contract["outputs"] = summary["outputs"] + [{"path": str((RESULTS / "run_summary.json").relative_to(ROOT)), "sha256": sha256(RESULTS / "run_summary.json"), "bytes": (RESULTS / "run_summary.json").stat().st_size}]
    write_json(CONTRACT, contract)
    summary["contract_sha256"] = sha256(CONTRACT)
    write_json(RESULTS / "run_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()
    summary = run(args.n_boot)
    print(json.dumps({k: summary[k] for k in ["fixed_cpg_rows", "promoter_cpg_rows", "combined_rows", "contract_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
