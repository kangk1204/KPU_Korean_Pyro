#!/usr/bin/env python3
"""Promoter-restricted fixed-probe methylation-expression analyses."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analyze_expression as ae
import common

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
RAW = ROOT / "data" / "raw"
REGISTRY = ROOT / "registry"
REVIEWER_REGISTRY = REGISTRY / "reviewer"
RESULTS = ROOT / "results"
REVIEWER_RESULTS = RESULTS / "reviewer"
SUPPLEMENT = ROOT / "supplement"

N_BOOT = 5000
SEED = 20260906
PROMOTER_GROUPS = {"TSS200", "TSS1500"}


def sha256(path: Path) -> str:
    return common.sha256(path)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stable_seed(*parts: object) -> int:
    return common.stable_seed("promoter", *parts)


def ensure_probe_context() -> pd.DataFrame:
    path = REVIEWER_REGISTRY / "probe_context.tsv"
    required = {
        "fixed_gene",
        "probe",
        "chr",
        "pos",
        "Relation_to_Island",
        "UCSC_RefGene_Name",
        "UCSC_RefGene_Accession",
        "UCSC_RefGene_Group",
        "semicolon_lengths_aligned",
        "promoter_match",
    }
    if not path.exists():
        helper = ROOT / "scripts" / "export_promoter_probe_context.R"
        subprocess.run(["Rscript", str(helper), str(ROOT)], check=True)
    context = pd.read_csv(path, sep="\t")
    missing = required - set(context.columns)
    if missing:
        helper = ROOT / "scripts" / "export_promoter_probe_context.R"
        subprocess.run(["Rscript", str(helper), str(ROOT)], check=True)
        context = pd.read_csv(path, sep="\t")
        missing = required - set(context.columns)
    if missing:
        raise ValueError(f"Probe context is missing required columns: {sorted(missing)}")
    if not context["semicolon_lengths_aligned"].astype(bool).all():
        bad = context.loc[~context["semicolon_lengths_aligned"].astype(bool), ["fixed_gene", "probe"]]
        raise ValueError(f"Semicolon-aligned RefGene fields failed for {bad.to_dict(orient='records')}")
    return context


def promoter_probes(context: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for gene in common.GENES:
        rows = context.loc[context["fixed_gene"].eq(gene) & context["promoter_match"].astype(bool)]
        out[gene] = rows["probe"].tolist()
    return out


def score_beta(beta: pd.DataFrame, probes_by_gene: dict[str, list[str]], cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if beta.index.duplicated().any() or beta.columns.duplicated().any():
        raise ValueError("Duplicate probe or sample identifiers")
    values = beta.to_numpy(float)
    if np.any(np.isfinite(values) & ((values < 0) | (values > 1))):
        raise ValueError("Beta values outside [0,1]")
    scores = pd.DataFrame(index=beta.columns)
    scores.index.name = "sample"
    coverage = []
    for gene in common.GENES:
        targets = probes_by_gene[gene]
        minimum = math.ceil(0.8 * len(targets))
        vals = beta.reindex(targets).T
        n_valid = vals.notna().sum(axis=1)
        scores[gene] = vals.mean(axis=1).where((len(targets) > 0) & (n_valid >= minimum))
        coverage.append(
            {
                "cohort": cohort,
                "gene": gene,
                "n_promoter_fixed": len(targets),
                "n_promoter_present": int(sum(probe in beta.index for probe in targets)),
                "minimum_valid": minimum,
                "n_samples": len(beta.columns),
                "n_valid_scores": int(scores[gene].notna().sum()),
                "promoter_probes": ";".join(targets),
                "missing_promoter_probes": ";".join(probe for probe in targets if probe not in beta.index),
                "status": "estimated" if len(targets) > 0 and sum(probe in beta.index for probe in targets) >= minimum else "not_estimable",
            }
        )
    scores["panel_mean"] = scores[common.GENES].mean(axis=1).where(scores[common.GENES].notna().all(axis=1))
    return scores[common.GENES + ["panel_mean"]], pd.DataFrame(coverage)


def bh(pvals: list[float]) -> list[float]:
    return common.bh(pvals).tolist()


def residualized_rank_spearman(df: pd.DataFrame, x_col: str, y_col: str, covars: list[str]) -> dict[str, object]:
    keep = [x_col, y_col] + covars
    sub = df[keep].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 6:
        return {"rho": math.nan, "p": math.nan, "n": int(len(sub)), "df_model": 0, "df_resid": math.nan, "status": "not_estimable"}
    x_rank = stats.rankdata(sub[x_col].to_numpy(float))
    y_rank = stats.rankdata(sub[y_col].to_numpy(float))
    cov = sub[covars] if covars else pd.DataFrame(index=sub.index)
    rx, rank = ae.residualize(x_rank, cov)
    ry, _ = ae.residualize(y_rank, cov)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return {"rho": math.nan, "p": math.nan, "n": int(len(sub)), "df_model": rank, "df_resid": math.nan, "status": "not_estimable"}
    rho = float(np.corrcoef(rx, ry)[0, 1])
    df_resid = int(len(sub) - rank - 1)
    if df_resid <= 0 or abs(rho) >= 1:
        return {"rho": rho, "p": math.nan, "n": int(len(sub)), "df_model": rank, "df_resid": df_resid, "status": "not_estimable"}
    tval = rho * math.sqrt(df_resid / max(1e-12, 1 - rho * rho))
    pval = float(2 * stats.t.sf(abs(tval), df_resid))
    return {"rho": rho, "p": pval, "n": int(len(sub)), "df_model": rank, "df_resid": df_resid, "status": "estimated"}


def unadjusted_spearman(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, object]:
    sub = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 6 or sub[x_col].nunique() <= 1 or sub[y_col].nunique() <= 1:
        return {"rho": math.nan, "p": math.nan, "n": int(len(sub)), "df_model": 0, "df_resid": math.nan, "status": "not_estimable"}
    rho, pval = stats.spearmanr(sub[x_col], sub[y_col])
    return {"rho": float(rho), "p": float(pval), "n": int(len(sub)), "df_model": 0, "df_resid": int(len(sub) - 2), "status": "estimated"}


def bootstrap_ci(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    covars: list[str],
    adjusted: bool,
    seed: int,
    n_boot: int = N_BOOT,
    unit_col: str | None = None,
) -> tuple[float, float, int]:
    keep = [x_col, y_col] + covars + ([unit_col] if unit_col else [])
    base = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(base) < 6:
        return math.nan, math.nan, 0
    if unit_col and base[unit_col].duplicated().any():
        raise ValueError(f"Bootstrap unit is not unique: {unit_col}")
    model_base = base.drop(columns=[unit_col]) if unit_col else base
    rng = np.random.default_rng(seed)
    vals = []
    if adjusted:
        x = model_base[x_col].to_numpy(float)
        y = model_base[y_col].to_numpy(float)
        cov = model_base[covars] if covars else pd.DataFrame(index=model_base.index)
        design_parts = []
        for col in cov.columns:
            if pd.api.types.is_numeric_dtype(cov[col]):
                design_parts.append(pd.to_numeric(cov[col], errors="coerce").to_numpy(float).reshape(-1, 1))
            else:
                dummies = pd.get_dummies(cov[col].astype(str), prefix=col, drop_first=True, dtype=float)
                if len(dummies.columns):
                    design_parts.append(dummies.to_numpy(float))
        xmat = np.column_stack([np.ones(len(model_base)), *design_parts]) if design_parts else np.ones((len(model_base), 1))
        for _ in range(n_boot):
            idx = rng.integers(0, len(model_base), len(model_base))
            xb = stats.rankdata(x[idx])
            yb = stats.rankdata(y[idx])
            db = xmat[idx, :]
            rank = int(np.linalg.matrix_rank(db))
            df_resid = len(idx) - rank - 1
            if df_resid <= 0:
                continue
            xcoef, *_ = np.linalg.lstsq(db, xb, rcond=None)
            ycoef, *_ = np.linalg.lstsq(db, yb, rcond=None)
            rx = xb - db @ xcoef
            ry = yb - db @ ycoef
            if np.std(rx) == 0 or np.std(ry) == 0:
                continue
            rho = float(np.corrcoef(rx, ry)[0, 1])
            if math.isfinite(rho) and abs(rho) < 1:
                vals.append(rho)
        if not vals:
            return math.nan, math.nan, 0
        lo, hi = np.quantile(vals, [0.025, 0.975])
        return float(lo), float(hi), len(vals)
    x = model_base[x_col].to_numpy(float)
    y = model_base[y_col].to_numpy(float)
    for _ in range(n_boot):
        idx = rng.integers(0, len(model_base), len(model_base))
        xb = stats.rankdata(x[idx])
        yb = stats.rankdata(y[idx])
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        rho = float(np.corrcoef(xb, yb)[0, 1])
        if math.isfinite(rho):
            vals.append(rho)
    if not vals:
        return math.nan, math.nan, 0
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi), len(vals)


def estimate_pair(
    df: pd.DataFrame,
    promoter_col: str,
    full_col: str,
    expr_col: str,
    covars: list[str],
    adjusted: bool,
    seed_parts: tuple[object, ...],
    n_boot: int = N_BOOT,
    unit_col: str = "patient",
) -> dict[str, object]:
    keep = [promoter_col, full_col, expr_col] + covars + [unit_col]
    matched = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if matched[unit_col].duplicated().any():
        dup = matched.loc[matched[unit_col].duplicated(), unit_col].astype(str).unique().tolist()
        raise ValueError(f"Analysis rows are not unique by {unit_col}: {dup[:5]}")
    model_matched = matched.drop(columns=[unit_col])
    promoter = residualized_rank_spearman(model_matched, promoter_col, expr_col, covars) if adjusted else unadjusted_spearman(model_matched, promoter_col, expr_col)
    full = residualized_rank_spearman(model_matched, full_col, expr_col, covars) if adjusted else unadjusted_spearman(model_matched, full_col, expr_col)
    p_lo, p_hi, p_valid = bootstrap_ci(matched, promoter_col, expr_col, covars, adjusted, stable_seed(*seed_parts, "promoter"), n_boot=n_boot, unit_col=unit_col)
    f_lo, f_hi, f_valid = bootstrap_ci(matched, full_col, expr_col, covars, adjusted, stable_seed(*seed_parts, "full"), n_boot=n_boot, unit_col=unit_col)
    return {
        "n_complete_case": int(len(matched)),
        "promoter_rho": promoter["rho"],
        "promoter_p": promoter["p"],
        "promoter_ci_low": p_lo,
        "promoter_ci_high": p_hi,
        "promoter_bootstrap_valid_draws": p_valid,
        "promoter_status": promoter["status"],
        "full_matched_rho": full["rho"],
        "full_matched_p": full["p"],
        "full_matched_ci_low": f_lo,
        "full_matched_ci_high": f_hi,
        "full_matched_bootstrap_valid_draws": f_valid,
        "full_matched_status": full["status"],
        "df_model": promoter["df_model"],
        "df_resid": promoter["df_resid"],
    }


def long_rows_from_pair(row: dict[str, object]) -> list[dict[str, object]]:
    common_fields = {
        "cohort": row["cohort"],
        "analysis": row["model"],
        "gene": row["gene"],
    }
    return [
        {
            **common_fields,
            "score_set": "promoter",
            "n": row["n_complete_case"],
            "rho": row["promoter_rho"],
            "p": row["promoter_p"],
            "q": row["promoter_q_BH10"],
            "ci_low": row["promoter_ci_low"],
            "ci_high": row["promoter_ci_high"],
            "valid_bootstrap_draws": row["promoter_bootstrap_valid_draws"],
            "status": row["promoter_status"],
        },
        {
            **common_fields,
            "score_set": "full",
            "n": row["n_complete_case"],
            "rho": row["full_matched_rho"],
            "p": row.get("full_original_p", math.nan),
            "q": row.get("full_original_q_BH", math.nan),
            "ci_low": row["full_matched_ci_low"],
            "ci_high": row["full_matched_ci_high"],
            "valid_bootstrap_draws": row["full_matched_bootstrap_valid_draws"],
            "status": row["full_matched_status"],
        },
    ]


def colonomics_frame(promoter_scores: pd.DataFrame) -> pd.DataFrame:
    expr = pd.read_csv(DERIVED / "expression_Colonomics_gene.tsv", sep="\t", index_col=0)
    expr_meta = pd.read_csv(DERIVED / "expression_Colonomics_samples.tsv", sep="\t")
    methyl_samples = pd.read_csv(DERIVED / "Colonomics_samples.tsv", sep="\t")
    full_scores = pd.read_csv(DERIVED / "Colonomics_scores.tsv", sep="\t").set_index("sample")
    if not expr_meta.loc[expr_meta["tissue"].eq("T"), "author_id"].is_unique:
        raise ValueError("Colonomics expression tumor author_id is not unique")
    methyl = methyl_samples.merge(full_scores, left_on="sample", right_index=True, how="left", validate="one_to_one")
    methyl = methyl.merge(promoter_scores.add_suffix("_promoter"), left_on="sample", right_index=True, how="left", validate="one_to_one")
    expr_t = expr_meta.loc[expr_meta["tissue"].eq("T")].copy()
    meth_t = methyl.loc[methyl["tissue"].eq("T") & (~methyl["excluded"].astype(bool))].copy()
    if not meth_t["author_id"].is_unique:
        raise ValueError("Colonomics methylation tumor author_id is not unique")
    merged = expr_t.merge(meth_t, on="author_id", how="inner", suffixes=("_expr_meta", "_meth_meta"), validate="one_to_one")
    if not merged["patient_meth_meta"].is_unique:
        raise ValueError("Colonomics matched tumors are not unique by patient")
    merged["patient"] = merged["patient_meth_meta"]
    gene_values = expr.reindex(merged["sample_expr_meta"])
    gene_values.index = merged.index
    for gene in common.GENES:
        merged[f"{gene}_expr"] = gene_values[gene]
        merged = merged.rename(columns={gene: f"{gene}_full"})
    merged = merged.rename(columns={"age_meth_meta": "age", "sex_meth_meta": "sex", "site_meth_meta": "site"})
    return merged


def colocare_frame(promoter_scores: pd.DataFrame) -> pd.DataFrame:
    expr = pd.read_csv(DERIVED / "expression_GSE106582_gene.tsv", sep="\t", index_col=0)
    expr_meta = pd.read_csv(DERIVED / "expression_GSE106582_samples.tsv", sep="\t")
    full_scores = pd.read_csv(DERIVED / "expression_GSE101764_methylation_scores.tsv", sep="\t")
    promoter_collapsed, _ = ae.collapse_verified_technical_replicates(
        promoter_scores,
        pd.read_csv(DERIVED / "expression_GSE101764_samples_raw.tsv", sep="\t"),
        "GSE101764_promoter",
    )
    promoter_gene_cols = {gene: f"{gene}_promoter" for gene in common.GENES}
    promoter_collapsed = promoter_collapsed.rename(columns=promoter_gene_cols)
    full_gene_cols = {gene: f"{gene}_full" for gene in common.GENES}
    full_scores = full_scores.rename(columns=full_gene_cols)
    if full_scores.duplicated(["patient", "tissue"]).any():
        raise ValueError("ColoCare full methylation scores are not unique by patient/tissue")
    if promoter_collapsed.duplicated(["patient", "tissue"]).any():
        raise ValueError("ColoCare promoter methylation scores are not unique by patient/tissue")
    meth = full_scores.merge(
        promoter_collapsed[["patient", "tissue", "sample", "author_id", *promoter_gene_cols.values()]],
        on=["patient", "tissue"],
        how="inner",
        suffixes=("", "_promoter_meta"),
        validate="one_to_one",
    )
    expr_t = expr_meta.loc[expr_meta["tissue"].eq("T")].copy()
    if expr_t.duplicated(["patient", "tissue"]).any():
        raise ValueError("ColoCare expression tumors are not unique by patient/tissue")
    meth_t = meth.loc[meth["tissue"].eq("T")].copy()
    if meth_t.duplicated(["patient", "tissue"]).any():
        raise ValueError("ColoCare methylation tumors are not unique by patient/tissue")
    merged = expr_t.merge(meth_t, on=["patient", "tissue"], how="inner", suffixes=("_expr_meta", "_meth_meta"), validate="one_to_one")
    if not merged["patient"].is_unique:
        raise ValueError("ColoCare matched tumors are not unique by patient")
    gene_values = expr.reindex(merged["sample_expr_meta"])
    gene_values.index = merged.index
    for gene in common.GENES:
        merged[f"{gene}_expr"] = gene_values[gene]
    return merged


def original_full_lookup() -> dict[tuple[str, str, str], dict[str, object]]:
    out: dict[tuple[str, str, str], dict[str, object]] = {}
    assoc = pd.read_csv(RESULTS / "expression_associations.tsv", sep="\t")
    for _, row in assoc.iterrows():
        model = "unadjusted"
        out[(row["cohort"], model, row["gene"])] = {
            "full_original_n": row["n"],
            "full_original_rho": row["rho"],
            "full_original_p": row["p"],
            "full_original_q_BH": row["q_BH"],
        }
    cov = pd.read_csv(RESULTS / "expression_covariate_sensitivity.tsv", sep="\t")
    cov = cov.loc[cov["model"].eq("rank_residual_age_sex_site_stroma")]
    for _, row in cov.iterrows():
        out[(row["cohort"], "rank_residual_age_sex_site_stroma", row["gene"])] = {
            "full_original_n": row["n"],
            "full_original_rho": row["rho"],
            "full_original_p": row["p"],
            "full_original_q_BH": row["q_BH"],
        }
    return out


def analyze() -> pd.DataFrame:
    REVIEWER_REGISTRY.mkdir(parents=True, exist_ok=True)
    REVIEWER_RESULTS.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)

    context = ensure_probe_context()
    probes = promoter_probes(context)
    col_beta = pd.read_csv(DERIVED / "Colonomics_beta.tsv.gz", sep="\t", index_col=0)
    col_promoter_scores, col_coverage = score_beta(col_beta, probes, "Colonomics")
    cc_beta = pd.read_csv(RAW / "GSE101764_fixed77_beta.tsv.gz", sep="\t", index_col=0)
    cc_promoter_scores, cc_coverage = score_beta(cc_beta, probes, "ColoCare_GSE101764")

    col_coverage.to_csv(REVIEWER_RESULTS / "promoter_probe_coverage.tsv", sep="\t", index=False)
    pd.concat([col_coverage, cc_coverage], ignore_index=True).to_csv(REVIEWER_RESULTS / "promoter_score_coverage.tsv", sep="\t", index=False)
    col_promoter_scores.to_csv(REVIEWER_RESULTS / "promoter_Colonomics_scores.tsv", sep="\t")
    cc_promoter_scores.to_csv(REVIEWER_RESULTS / "promoter_GSE101764_scores.tsv", sep="\t")

    frames = {
        "Colonomics": colonomics_frame(col_promoter_scores),
        "ColoCare (discovery overlap)": colocare_frame(cc_promoter_scores),
    }
    original = original_full_lookup()
    specs = [
        ("Colonomics", "unadjusted", [], False),
        ("Colonomics", "rank_residual_age_sex_site_stroma", ["age", "sex", "site", "stromal_score"], True),
        ("ColoCare (discovery overlap)", "unadjusted", [], False),
    ]
    rows = []
    for cohort, model, covars, adjusted in specs:
        frame = frames[cohort]
        for gene in common.GENES:
            est = estimate_pair(
                frame,
                f"{gene}_promoter",
                f"{gene}_full",
                f"{gene}_expr",
                covars,
                adjusted,
                (cohort, model, gene),
            )
            row = {
                "cohort": cohort,
                "model": model,
                "gene": gene,
                "n_promoter_fixed": len(probes[gene]),
                **est,
                **original.get((cohort, model, gene), {}),
            }
            rows.append(row)
    wide = pd.DataFrame(rows)
    for cohort, model, _, _ in specs:
        mask = wide["cohort"].eq(cohort) & wide["model"].eq(model)
        wide.loc[mask, "promoter_q_BH10"] = bh(wide.loc[mask, "promoter_p"].tolist())
    result = pd.DataFrame([long for _, row in wide.iterrows() for long in long_rows_from_pair(row.to_dict())])
    result = result[
        [
            "cohort",
            "analysis",
            "gene",
            "score_set",
            "n",
            "rho",
            "p",
            "q",
            "ci_low",
            "ci_high",
            "valid_bootstrap_draws",
            "status",
        ]
    ]
    result.to_csv(REVIEWER_RESULTS / "promoter_associations.tsv", sep="\t", index=False)
    wide.to_csv(REVIEWER_RESULTS / "promoter_associations_wide_audit.tsv", sep="\t", index=False)

    manifest_paths = [
        REGISTRY / "fixed_probes.json",
        REVIEWER_REGISTRY / "probe_context.tsv",
        DERIVED / "Colonomics_beta.tsv.gz",
        DERIVED / "Colonomics_samples.tsv",
        DERIVED / "Colonomics_scores.tsv",
        DERIVED / "expression_Colonomics_gene.tsv",
        DERIVED / "expression_Colonomics_samples.tsv",
        RAW / "GSE101764_fixed77_beta.tsv.gz",
        DERIVED / "expression_GSE101764_samples_raw.tsv",
        DERIVED / "expression_GSE101764_methylation_scores.tsv",
        DERIVED / "expression_GSE106582_gene.tsv",
        DERIVED / "expression_GSE106582_samples.tsv",
        RESULTS / "expression_associations.tsv",
        RESULTS / "expression_covariate_sensitivity.tsv",
    ]
    outputs = [
        REVIEWER_RESULTS / "promoter_probe_coverage.tsv",
        REVIEWER_RESULTS / "promoter_score_coverage.tsv",
        REVIEWER_RESULTS / "promoter_Colonomics_scores.tsv",
        REVIEWER_RESULTS / "promoter_GSE101764_scores.tsv",
        REVIEWER_RESULTS / "promoter_associations.tsv",
        REVIEWER_RESULTS / "promoter_associations_wide_audit.tsv",
    ]
    summary = {
        "created_by": "scripts/analyze_promoter.py",
        "n_bootstrap_requested": N_BOOT,
        "promoter_probe_total": int(sum(len(v) for v in probes.values())),
        "promoter_probe_counts": {gene: len(probes[gene]) for gene in common.GENES},
        "families": [
            "Colonomics unadjusted BH10",
            "Colonomics age/sex/site/stromal adjusted BH10",
            "ColoCare unadjusted BH10",
        ],
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in manifest_paths],
        "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs],
    }
    write_report(result, summary)
    supplement_path = SUPPLEMENT / "promoter_methods_results.md"
    summary["outputs"].append(
        {"path": str(supplement_path.relative_to(ROOT)), "sha256": sha256(supplement_path), "bytes": supplement_path.stat().st_size}
    )
    write_json(REVIEWER_RESULTS / "promoter_manifest.json", summary)
    return result


def write_report(result: pd.DataFrame, summary: dict[str, object]) -> None:
    show = result[
        [
            "cohort",
            "analysis",
            "gene",
            "score_set",
            "n",
            "rho",
            "ci_low",
            "ci_high",
            "p",
            "q",
            "valid_bootstrap_draws",
            "status",
        ]
    ].copy()
    lines = [
        "# Promoter-restricted methylation-expression analysis",
        "",
        "Fixed 77 probes were annotated with IlluminaHumanMethylation450kanno.ilmn12.hg19 v0.6.1. Only transcript-aligned UCSC RefGene TSS200/TSS1500 entries matching the fixed target gene were retained for promoter scores.",
        "",
        f"- Promoter probes retained: {summary['promoter_probe_total']} across 10 genes.",
        "- Score rule: mean beta when at least ceil(0.8 * fixed promoter probes for that gene) were observed.",
        "- Comparisons: promoter and full fixed-probe estimates use identical complete cases within each gene/cohort/model.",
        "- Multiplicity: BH adjustment was applied separately to the three prespecified promoter p-value families of ten tests.",
        "- Bootstrap: 5000 patient resamples; adjusted Colonomics models refit age, sex, site, and stromal residualization in each draw.",
        "- Full rows use identical complete-case rho and CI for comparison; their p and q are the original full fixed-probe expression-analysis values, not a new selected full-probe family.",
        "",
        "## Results",
        "",
        show.to_markdown(index=False, floatfmt=".4g"),
        "",
    ]
    (SUPPLEMENT / "promoter_methods_results.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
