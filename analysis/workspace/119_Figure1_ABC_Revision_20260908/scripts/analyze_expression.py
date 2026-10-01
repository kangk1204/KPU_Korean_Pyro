#!/usr/bin/env python3
"""Colonomics methylation-expression analyses for the public biology extension."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "results"
REGISTRY = ROOT / "registry"

GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
SEED = 20260905
N_BOOT = 5000

URLS = {
    "GSE44076_series_matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE44nnn/GSE44076/matrix/GSE44076_series_matrix.txt.gz",
    "GPL13667_family_soft": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13667/soft/GPL13667_family.soft.gz",
    "GSE106582_series_matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/GSE106582/matrix/GSE106582_series_matrix.txt.gz",
    "GPL10558_annot": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/GPL10558/annot/GPL10558.annot.gz",
    "GSE101764_series_matrix": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE101nnn/GSE101764/matrix/GSE101764_series_matrix.txt.gz",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def download(url: str, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".tmp")
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp.replace(path)
    return {"url": url, "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def parse_series_matrix_metadata(path: Path) -> pd.DataFrame:
    rows: dict[str, list[list[str]]] = {}
    with gzip.open(path, "rt", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for fields in reader:
            if not fields:
                continue
            if fields[0] == "!series_matrix_table_begin":
                break
            if fields[0].startswith("!Sample_"):
                rows.setdefault(fields[0], []).append(fields[1:])
    accessions = rows["!Sample_geo_accession"][0]
    meta = pd.DataFrame({"sample": accessions})
    for key in ["!Sample_title", "!Sample_source_name_ch1", "!Sample_description"]:
        if key in rows:
            meta[key.replace("!Sample_", "").lower()] = rows[key][0]
    chars = {sample: {} for sample in accessions}
    for line in rows.get("!Sample_characteristics_ch1", []):
        for i, val in enumerate(line):
            if ":" not in val:
                continue
            key, value = val.split(":", 1)
            chars[accessions[i]][key.strip().lower()] = value.strip()
    for key in sorted({k for d in chars.values() for k in d}):
        meta[key] = [chars[sample].get(key, "") for sample in accessions]
    return meta


def parse_series_matrix_table(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    in_table = False
    with gzip.open(path, "rt", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for fields in reader:
            if not fields:
                continue
            if fields[0] == "!series_matrix_table_begin":
                in_table = True
                continue
            if fields[0] == "!series_matrix_table_end":
                break
            if in_table:
                rows.append(fields)
    if not rows:
        raise ValueError(f"No expression table found in {path}")
    header, data = rows[0], rows[1:]
    expr = pd.DataFrame(data, columns=header)
    expr = expr.set_index("ID_REF")
    return expr.apply(pd.to_numeric, errors="coerce")


def normalize_symbol(symbol: object) -> str:
    if pd.isna(symbol):
        return ""
    text = str(symbol).strip()
    if not text or text == "---":
        return ""
    parts = re.split(r"\s*(?:///|//|;|,|\|)\s*", text)
    parts = [p.strip() for p in parts if p.strip() and p.strip() != "---"]
    unique = sorted(set(parts))
    return unique[0] if len(unique) == 1 else ""


def parse_gpl13667_target_annotation(path: Path, genes: list[str]) -> pd.DataFrame:
    target = set(genes)
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    in_table = False
    with gzip.open(path, "rt", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for fields in reader:
            if not fields:
                continue
            if fields[0] == "!platform_table_begin":
                in_table = True
                continue
            if fields[0] == "!platform_table_end":
                break
            if not in_table:
                continue
            if header is None:
                header = fields
                continue
            rec = dict(zip(header, fields))
            symbol = normalize_symbol(rec.get("Gene Symbol", ""))
            if symbol in target:
                rows.append(
                    {
                        "ID": rec.get("ID", ""),
                        "Gene Symbol": rec.get("Gene Symbol", ""),
                        "normalized_symbol": symbol,
                        "Gene Title": rec.get("Gene Title", ""),
                        "Entrez Gene": rec.get("Entrez Gene", ""),
                        "RefSeq Transcript ID": rec.get("RefSeq Transcript ID", ""),
                    }
                )
    return pd.DataFrame(rows)


def parse_gpl_annotation(path: Path, id_col: str, symbol_col: str, genes: list[str]) -> pd.DataFrame:
    rows: list[list[str]] = []
    in_table = False
    with gzip.open(path, "rt", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for fields in reader:
            if not fields:
                continue
            if fields[0] == "!platform_table_begin":
                in_table = True
                continue
            if fields[0] == "!platform_table_end":
                break
            if in_table:
                rows.append(fields)
    if not rows:
        raise ValueError(f"No platform table found in {path}")
    annot = pd.DataFrame(rows[1:], columns=rows[0])
    annot["normalized_symbol"] = annot[symbol_col].map(normalize_symbol)
    return annot.loc[annot["normalized_symbol"].isin(genes), [id_col, symbol_col, "normalized_symbol"]].rename(columns={id_col: "ID"})


def expression_meta_gse44076(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    tissue_map = {"Tumor": "T", "Normal": "N", "Mucosa": "M"}
    out["tissue"] = out["sample type"].map(tissue_map)
    out["patient"] = out["individual id"]
    out["author_id"] = out["patient"] + "_" + out["tissue"]
    out["sex"] = out["gender"].map({"Male": "M", "Female": "F"}).fillna(out["gender"])
    out["site"] = out["location"]
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    cols = ["sample", "author_id", "patient", "tissue", "age", "sex", "site", "title"]
    return out[cols]


def expression_meta_gse106582(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    sample_type = out.get("tissue", out.get("sample type", pd.Series("", index=out.index))).astype(str).str.lower()
    out["tissue"] = sample_type.map({"tumor": "T", "mucosa": "M"}).fillna(sample_type)
    out["patient"] = out.get("patientid", out.get("patient id", out.get("individual id", pd.Series("", index=out.index)))).astype(str)
    out["author_id"] = out["patient"] + "_" + out["tissue"].astype(str)
    keep = [c for c in ["sample", "author_id", "patient", "tissue", "age", "sex", "gender", "title"] if c in out.columns]
    return out[keep]


def methylation_meta_gse101764(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    tissue = out.get("tissue", pd.Series("", index=out.index)).astype(str).str.lower()
    out["tissue"] = tissue.map({"tumor": "T", "mucosa": "M"}).fillna(tissue)
    out["patient"] = out.get("patientid", pd.Series("", index=out.index)).astype(str)
    out["sex"] = out.get("gender", pd.Series("", index=out.index)).astype(str)
    out["age"] = pd.to_numeric(out.get("age", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["author_id"] = out["patient"] + "_" + out["tissue"].astype(str)
    cols = ["sample", "author_id", "patient", "tissue", "age", "sex", "title"]
    return out[cols]


def fixed_probes() -> dict[str, list[str]]:
    with (REGISTRY / "fixed_probes.json").open() as handle:
        return json.load(handle)


def extract_fixed_methylation_rows(src: Path, dst: Path, probes: set[str]) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    kept = 0
    in_table = False
    with gzip.open(src, "rt", errors="replace", newline="") as handle, gzip.open(tmp, "wt", newline="") as out:
        reader = csv.reader(handle, delimiter="\t")
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        for fields in reader:
            if not fields:
                continue
            if fields[0] == "!series_matrix_table_begin":
                in_table = True
                continue
            if fields[0] == "!series_matrix_table_end":
                break
            if not in_table:
                continue
            if fields[0] == "ID_REF" or fields[0] in probes:
                writer.writerow(fields)
                if fields[0] != "ID_REF":
                    kept += 1
    if kept == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"No fixed methylation probes extracted from {src}")
    tmp.replace(dst)


def methylation_gene_scores(beta: pd.DataFrame, probes_by_gene: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scores = pd.DataFrame(index=beta.columns)
    for gene in GENES:
        targets = probes_by_gene[gene]
        present = [p for p in targets if p in beta.index]
        min_valid = math.ceil(0.8 * len(targets))
        vals = beta.reindex(present).T
        counts = vals.notna().sum(axis=1)
        score = vals.mean(axis=1, skipna=True).where(counts >= min_valid)
        scores[gene] = score
        rows.append(
            {
                "gene": gene,
                "n_target_probes": len(targets),
                "n_present_probes": len(present),
                "min_valid_probes_per_sample": min_valid,
                "coverage_fraction_original_targets": len(present) / len(targets),
                "status": "pass" if len(present) >= min_valid else "platform_low_coverage",
                "present_probes": ";".join(present),
                "missing_probes": ";".join([p for p in targets if p not in present]),
            }
        )
    scores["panel_mean"] = scores[GENES].mean(axis=1, skipna=False)
    return scores, pd.DataFrame(rows)


def collapse_verified_technical_replicates(scores: pd.DataFrame, meta: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = meta.merge(scores, left_on="sample", right_index=True, how="left")
    audit_rows = []
    out_rows = []
    group_cols = ["patient", "tissue"]
    for (patient, tissue), group in joined.groupby(group_cols, dropna=False):
        same_clinical = group[["age", "sex"]].drop_duplicates().shape[0] == 1
        status = "single_biological_sample" if len(group) == 1 else ("collapsed_verified_technical_replicates" if same_clinical else "duplicate_not_collapsed")
        audit_rows.append(
            {
                "dataset": dataset,
                "patient": patient,
                "tissue": tissue,
                "n_samples": len(group),
                "samples": ";".join(group["sample"].astype(str)),
                "titles": ";".join(group["title"].astype(str)),
                "status": status,
            }
        )
        if len(group) == 1 or same_clinical:
            row = group.iloc[0][["patient", "tissue", "age", "sex"]].to_dict()
            row["sample"] = ";".join(group["sample"].astype(str))
            row["author_id"] = f"{patient}_{tissue}"
            row["n_technical_replicates"] = len(group)
            for gene in GENES + ["panel_mean"]:
                row[gene] = group[gene].mean(skipna=True)
            out_rows.append(row)
    return pd.DataFrame(out_rows), pd.DataFrame(audit_rows)


def collapse_to_genes(expr: pd.DataFrame, annot: pd.DataFrame, id_col: str = "ID") -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = annot[[id_col, "normalized_symbol"]].dropna().copy()
    mapped[id_col] = mapped[id_col].astype(str)
    mapped = mapped[mapped[id_col].isin(expr.index)]
    blocks = []
    qc_rows = []
    for gene in GENES:
        probes = mapped.loc[mapped["normalized_symbol"] == gene, id_col].drop_duplicates().tolist()
        present = [probe for probe in probes if probe in expr.index]
        qc_rows.append({"gene": gene, "n_annotated_probesets": len(probes), "n_present_probesets": len(present), "probesets": ";".join(present)})
        if present:
            block = expr.loc[present].median(axis=0, skipna=True)
        else:
            block = pd.Series(np.nan, index=expr.columns)
        block.name = gene
        blocks.append(block)
    gene_expr = pd.DataFrame(blocks).T
    gene_expr.index.name = "sample"
    return gene_expr, pd.DataFrame(qc_rows)


def bh(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[ok])]
    ranked = p[order] * len(p) / np.arange(1, len(idx) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(ranked, 1.0)
    return out.tolist()


def residualize(y: np.ndarray, covars: pd.DataFrame) -> tuple[np.ndarray, int]:
    x = covars.copy()
    numeric = []
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            numeric.append(pd.to_numeric(x[col], errors="coerce").rename(col))
        else:
            dummies = pd.get_dummies(x[col].astype(str), prefix=col, drop_first=True, dtype=float)
            for dcol in dummies.columns:
                numeric.append(dummies[dcol])
    if numeric:
        xmat = pd.concat(numeric, axis=1)
        xmat = xmat.loc[:, xmat.nunique(dropna=True) > 1]
    else:
        xmat = pd.DataFrame(index=covars.index)
    mat = np.column_stack([np.ones(len(y)), xmat.to_numpy(float)]) if len(xmat.columns) else np.ones((len(y), 1))
    coef, *_ = np.linalg.lstsq(mat, y, rcond=None)
    return y - mat @ coef, int(np.linalg.matrix_rank(mat))


def partial_rank_spearman(df: pd.DataFrame, x_col: str, y_col: str, covars: list[str]) -> tuple[float, float, int, int]:
    keep = [x_col, y_col] + covars
    sub = df[keep].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub.loc[:, ~sub.columns.duplicated()]
    if len(sub) < 6:
        return math.nan, math.nan, len(sub), 0
    x_rank = stats.rankdata(sub[x_col].to_numpy(float))
    y_rank = stats.rankdata(sub[y_col].to_numpy(float))
    cov = sub[covars] if covars else pd.DataFrame(index=sub.index)
    rx, px = residualize(x_rank, cov)
    ry, _ = residualize(y_rank, cov)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return math.nan, math.nan, len(sub), px
    r = float(np.corrcoef(rx, ry)[0, 1])
    dfree = len(sub) - px - 1
    if dfree <= 0 or abs(r) >= 1:
        return r, math.nan, len(sub), px
    tval = r * math.sqrt(dfree / max(1e-12, 1 - r * r))
    pval = float(2 * stats.t.sf(abs(tval), dfree))
    return r, pval, len(sub), px


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        if np.nanstd(xb) == 0 or np.nanstd(yb) == 0:
            continue
        vals.append(stats.spearmanr(xb, yb).statistic)
    if not vals:
        return math.nan, math.nan
    return tuple(np.nanpercentile(vals, [2.5, 97.5]).tolist())


def association_rows(cohort: str, merged: pd.DataFrame, meth_suffix: str = "", expr_suffix: str = "_expr") -> pd.DataFrame:
    rows = []
    for gene in GENES:
        meth_col = f"{gene}{meth_suffix}"
        expr_col = f"{gene}{expr_suffix}"
        sub = merged[[meth_col, expr_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) >= 3 and sub[meth_col].nunique() > 1 and sub[expr_col].nunique() > 1:
            rho, pval = stats.spearmanr(sub[meth_col], sub[expr_col])
            ci_low, ci_high = bootstrap_spearman(sub[meth_col].to_numpy(float), sub[expr_col].to_numpy(float))
            status = "estimated"
        else:
            rho, pval, ci_low, ci_high = math.nan, math.nan, math.nan, math.nan
            status = "not_estimable"
        rows.append(
            {
                "cohort": cohort,
                "gene": gene,
                "n": int(len(sub)),
                "rho": float(rho),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p": float(pval),
                "status": status,
            }
        )
    result = pd.DataFrame(rows)
    result["q_BH"] = bh(result["p"].tolist())
    return result[["cohort", "gene", "n", "rho", "ci_low", "ci_high", "p", "q_BH", "status"]]


def analyze_colonomics_expression() -> dict[str, object]:
    expr = pd.read_csv(DERIVED / "expression_Colonomics_gene.tsv", sep="\t", index_col=0)
    expr_meta = pd.read_csv(DERIVED / "expression_Colonomics_samples.tsv", sep="\t")
    methyl_samples = pd.read_csv(DERIVED / "Colonomics_samples.tsv", sep="\t")
    methyl_scores = pd.read_csv(DERIVED / "Colonomics_scores.tsv", sep="\t").set_index("sample")
    methyl = methyl_samples.merge(methyl_scores, left_on="sample", right_index=True, how="left", suffixes=("", "_meth"))

    expr_t = expr_meta.loc[expr_meta["tissue"] == "T"].copy()
    meth_t = methyl.loc[(methyl["tissue"] == "T") & (~methyl["excluded"].astype(bool))].copy()
    merged = expr_t.merge(meth_t, on="author_id", how="inner", suffixes=("_expr", "_meth"))
    alias_rows = []
    join_audit_rows = []
    expr_author_ids = set(expr_t["author_id"])
    meth_author_ids = set(meth_t["author_id"])
    for _, row in expr_t.iterrows():
        if row["author_id"] not in meth_author_ids:
            candidates = meth_t.loc[meth_t["patient"].astype(str) == str(row["patient"]), "author_id"].tolist()
            if candidates:
                alias_rows.append({"expression_author_id": row["author_id"], "methylation_author_id_candidates": ";".join(candidates)})
            join_audit_rows.append(
                {
                    "side": "expression_unmatched",
                    "sample": row["sample"],
                    "author_id": row["author_id"],
                    "patient": row["patient"],
                    "candidate_same_patient_author_ids": ";".join(candidates),
                }
            )
    for _, row in meth_t.iterrows():
        if row["author_id"] not in expr_author_ids:
            candidates = expr_t.loc[expr_t["patient"].astype(str) == str(row["patient"]), "author_id"].tolist()
            join_audit_rows.append(
                {
                    "side": "methylation_unmatched",
                    "sample": row["sample"],
                    "author_id": row["author_id"],
                    "patient": row["patient"],
                    "candidate_same_patient_author_ids": ";".join(candidates),
                }
            )
    gene_values = expr.reindex(merged["sample_expr"])
    gene_values.index = merged.index
    for gene in GENES:
        merged[f"{gene}_expr"] = gene_values[gene]

    rows = []
    for gene in GENES:
        sub = merged[[gene, f"{gene}_expr", "age_meth", "sex_meth", "site_meth", "stromal_score"]].replace([np.inf, -np.inf], np.nan).dropna(subset=[gene, f"{gene}_expr"])
        if len(sub) >= 3 and sub[gene].nunique() > 1 and sub[f"{gene}_expr"].nunique() > 1:
            rho, pval = stats.spearmanr(sub[gene], sub[f"{gene}_expr"])
            ci_low, ci_high = bootstrap_spearman(sub[gene].to_numpy(float), sub[f"{gene}_expr"].to_numpy(float))
        else:
            rho, pval, ci_low, ci_high = math.nan, math.nan, math.nan, math.nan
        r_cov, p_cov, n_cov, k_cov = partial_rank_spearman(sub.rename(columns={"age_meth": "age", "sex_meth": "sex", "site_meth": "site"}), gene, f"{gene}_expr", ["age", "sex", "site"])
        r_stroma, p_stroma, n_stroma, k_stroma = partial_rank_spearman(
            sub.rename(columns={"age_meth": "age", "sex_meth": "sex", "site_meth": "site"}).dropna(subset=["stromal_score"]),
            gene,
            f"{gene}_expr",
            ["age", "sex", "site", "stromal_score"],
        )
        rows.append(
            {
                "cohort": "Colonomics",
                "gene": gene,
                "n": int(len(sub)),
                "spearman_rho": float(rho),
                "spearman_p": float(pval),
                "spearman_boot_ci_low": float(ci_low),
                "spearman_boot_ci_high": float(ci_high),
                "partial_age_sex_site_r": r_cov,
                "partial_age_sex_site_p": p_cov,
                "partial_age_sex_site_n": n_cov,
                "partial_age_sex_site_df_model": k_cov,
                "partial_age_sex_site_stroma_r": r_stroma,
                "partial_age_sex_site_stroma_p": p_stroma,
                "partial_age_sex_site_stroma_n": n_stroma,
                "partial_age_sex_site_stroma_df_model": k_stroma,
            }
        )
    result = pd.DataFrame(rows)
    result["spearman_q_bh10"] = bh(result["spearman_p"].tolist())
    result["partial_age_sex_site_q_bh10"] = bh(result["partial_age_sex_site_p"].tolist())
    result["partial_age_sex_site_stroma_q_bh10"] = bh(result["partial_age_sex_site_stroma_p"].tolist())
    result.to_csv(RESULTS / "expression_colonomics_correlations.tsv", sep="\t", index=False)
    primary = result.rename(
        columns={
            "spearman_rho": "rho",
            "spearman_boot_ci_low": "ci_low",
            "spearman_boot_ci_high": "ci_high",
            "spearman_p": "p",
            "spearman_q_bh10": "q_BH",
        }
    )[["cohort", "gene", "n", "rho", "ci_low", "ci_high", "p", "q_BH"]].copy()
    primary["status"] = np.where(primary[["rho", "p", "q_BH"]].notna().all(axis=1), "estimated", "not_estimable")
    cov_rows = []
    for _, row in result.iterrows():
        cov_rows.append(
            {
                "cohort": row["cohort"],
                "gene": row["gene"],
                "model": "rank_residual_age_sex_site",
                "n": row["partial_age_sex_site_n"],
                "df_model": row["partial_age_sex_site_df_model"],
                "rho": row["partial_age_sex_site_r"],
                "p": row["partial_age_sex_site_p"],
                "q_BH": row["partial_age_sex_site_q_bh10"],
                "status": "estimated" if pd.notna(row["partial_age_sex_site_p"]) else "not_estimable",
            }
        )
        cov_rows.append(
            {
                "cohort": row["cohort"],
                "gene": row["gene"],
                "model": "rank_residual_age_sex_site_stroma",
                "n": row["partial_age_sex_site_stroma_n"],
                "df_model": row["partial_age_sex_site_stroma_df_model"],
                "rho": row["partial_age_sex_site_stroma_r"],
                "p": row["partial_age_sex_site_stroma_p"],
                "q_BH": row["partial_age_sex_site_stroma_q_bh10"],
                "status": "estimated" if pd.notna(row["partial_age_sex_site_stroma_p"]) else "not_estimable",
            }
        )
    pd.DataFrame(cov_rows).to_csv(RESULTS / "expression_covariate_sensitivity.tsv", sep="\t", index=False)
    merged.to_csv(DERIVED / "expression_Colonomics_methylation_matched.tsv", sep="\t", index=False)
    pd.DataFrame(alias_rows, columns=["expression_author_id", "methylation_author_id_candidates"]).to_csv(
        REGISTRY / "expression_colonomics_alias_audit.tsv", sep="\t", index=False
    )
    pd.DataFrame(join_audit_rows, columns=["side", "sample", "author_id", "patient", "candidate_same_patient_author_ids"]).to_csv(
        REGISTRY / "expression_colonomics_join_audit.tsv", sep="\t", index=False
    )
    return {
        "primary_associations": primary,
        "matched_tumors": int(len(merged)),
        "expression_tumors": int(len(expr_t)),
        "methylation_tumors": int(len(meth_t)),
        "literal_author_id_join": int(len(merged)),
        "unmatched_expression_tumors_with_patient_candidate": len(alias_rows),
    }


def analyze_colocare_expression() -> dict[str, object]:
    expr = pd.read_csv(DERIVED / "expression_GSE106582_gene.tsv", sep="\t", index_col=0)
    expr_meta = pd.read_csv(DERIVED / "expression_GSE106582_samples.tsv", sep="\t")
    meth_scores = pd.read_csv(DERIVED / "expression_GSE101764_methylation_scores.tsv", sep="\t")
    meth_meta = pd.read_csv(DERIVED / "expression_GSE101764_samples.tsv", sep="\t")
    meth = meth_meta.merge(meth_scores, on=["sample", "author_id", "patient", "tissue"], how="inner", suffixes=("", "_score"))

    expr_t = expr_meta.loc[expr_meta["tissue"] == "T"].copy()
    meth_t = meth.loc[meth["tissue"] == "T"].copy()
    merged = expr_t.merge(meth_t, on=["patient", "tissue"], how="inner", suffixes=("_expr_meta", "_meth_meta"))
    gene_values = expr.reindex(merged["sample_expr_meta"])
    gene_values.index = merged.index
    for gene in GENES:
        merged[f"{gene}_expr"] = gene_values[gene]
    associations = association_rows("ColoCare (discovery overlap)", merged, meth_suffix="", expr_suffix="_expr")
    associations.to_csv(RESULTS / "expression_colocare_associations.tsv", sep="\t", index=False)
    merged.to_csv(DERIVED / "expression_GSE101764_GSE106582_matched.tsv", sep="\t", index=False)
    overlap = {
        "matched_tumors": int(len(merged)),
        "expression_tumors": int(len(expr_t)),
        "methylation_tumors_after_technical_collapse": int(len(meth_t)),
        "unique_matched_patients": int(merged["patient"].nunique()),
    }
    write_json(REGISTRY / "expression_colocare_join_summary.json", overlap)
    return {"primary_associations": associations, **overlap}


def prepare() -> dict[str, object]:
    RAW.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    REGISTRY.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {"created_by": "scripts/analyze_expression.py", "urls": URLS, "inputs": []}
    inputs = manifest["inputs"]  # type: ignore[assignment]

    gse44076_path = RAW / "GSE44076_series_matrix.txt.gz"
    inputs.append(download(URLS["GSE44076_series_matrix"], gse44076_path))
    meta_44076 = expression_meta_gse44076(parse_series_matrix_metadata(gse44076_path))
    expr_44076 = parse_series_matrix_table(gse44076_path)
    gpl13667_target = load_gpl13667_target_annotation()
    inputs.append(
        {
            "url": URLS["GPL13667_family_soft"],
            "path": str((RAW / "GPL13667_target_annotation.tsv").relative_to(ROOT)),
            "sha256": sha256(RAW / "GPL13667_target_annotation.tsv"),
            "bytes": (RAW / "GPL13667_target_annotation.tsv").stat().st_size,
            "note": "Target-gene platform rows parsed from the GEO GPL13667 family SOFT stream; full family SOFT is not stored because it is multi-GB and platform table precedes sample sections.",
        }
    )
    gene_44076, qc_44076 = collapse_to_genes(expr_44076, gpl13667_target)
    meta_44076.to_csv(DERIVED / "expression_Colonomics_samples.tsv", sep="\t", index=False)
    gene_44076.to_csv(DERIVED / "expression_Colonomics_gene.tsv", sep="\t")
    qc_44076.to_csv(DERIVED / "expression_Colonomics_probe_mapping.tsv", sep="\t", index=False)

    support = {"GSE106582_GSE101764": {"status": "not_attempted"}}
    try:
        gse106582_path = RAW / "GSE106582_series_matrix.txt.gz"
        gpl10558_path = RAW / "GPL10558.annot.gz"
        gse101764_path = RAW / "GSE101764_series_matrix.txt.gz"
        inputs.append(download(URLS["GSE106582_series_matrix"], gse106582_path))
        inputs.append(download(URLS["GPL10558_annot"], gpl10558_path))
        inputs.append(download(URLS["GSE101764_series_matrix"], gse101764_path))
        meta_106582 = expression_meta_gse106582(parse_series_matrix_metadata(gse106582_path))
        expr_106582 = parse_series_matrix_table(gse106582_path)
        annot_10558 = parse_gpl_annotation(gpl10558_path, "ID", "Gene symbol", GENES)
        gene_106582, qc_106582 = collapse_to_genes(expr_106582, annot_10558)
        meta_106582.to_csv(DERIVED / "expression_GSE106582_samples.tsv", sep="\t", index=False)
        gene_106582.to_csv(DERIVED / "expression_GSE106582_gene.tsv", sep="\t")
        qc_106582.to_csv(DERIVED / "expression_GSE106582_probe_mapping.tsv", sep="\t", index=False)

        probes_by_gene = fixed_probes()
        fixed_rows_path = RAW / "GSE101764_fixed77_beta.tsv.gz"
        extract_fixed_methylation_rows(gse101764_path, fixed_rows_path, {p for probes in probes_by_gene.values() for p in probes})
        inputs.append(
            {
                "source_path": str(gse101764_path.relative_to(ROOT)),
                "path": str(fixed_rows_path.relative_to(ROOT)),
                "sha256": sha256(fixed_rows_path),
                "bytes": fixed_rows_path.stat().st_size,
                "note": "Fixed 77 methylation beta rows extracted from GSE101764 series matrix for supportive ColoCare analysis.",
            }
        )
        meta_101764 = methylation_meta_gse101764(parse_series_matrix_metadata(gse101764_path))
        beta_101764 = pd.read_csv(fixed_rows_path, sep="\t", index_col=0)
        beta_101764 = beta_101764.apply(pd.to_numeric, errors="coerce")
        meth_scores_raw, meth_qc = methylation_gene_scores(beta_101764, probes_by_gene)
        meth_scores_collapsed, tech_audit = collapse_verified_technical_replicates(meth_scores_raw, meta_101764, "GSE101764")
        meth_scores_collapsed.to_csv(DERIVED / "expression_GSE101764_methylation_scores.tsv", sep="\t", index=False)
        meta_101764.to_csv(DERIVED / "expression_GSE101764_samples_raw.tsv", sep="\t", index=False)
        meth_scores_collapsed[["sample", "author_id", "patient", "tissue", "age", "sex", "n_technical_replicates"]].to_csv(
            DERIVED / "expression_GSE101764_samples.tsv", sep="\t", index=False
        )
        meth_qc.to_csv(DERIVED / "expression_GSE101764_probe_coverage.tsv", sep="\t", index=False)
        tech_audit.to_csv(REGISTRY / "expression_GSE101764_technical_replicates.tsv", sep="\t", index=False)
        support["GSE106582_GSE101764"] = {
            "status": "prepared_for_supportive_correlation",
            "samples": int(gene_106582.shape[0]),
            "genes_with_expression": int(gene_106582.notna().any(axis=0).sum()),
            "methylation_samples_after_technical_collapse": int(len(meth_scores_collapsed)),
            "methylation_tumors_after_technical_collapse": int((meth_scores_collapsed["tissue"] == "T").sum()),
            "note": "Supportive only: GSE101764 was used in discovery, so this is not independent external validation.",
        }
    except Exception as exc:  # noqa: BLE001
        support["GSE106582_GSE101764"] = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}

    write_json(REGISTRY / "expression_manifest.json", manifest)
    write_json(REGISTRY / "expression_supportive_status.json", support)
    return {"manifest": manifest, "support": support}


def parse_gpl13667_target_annotation_from_url(url: str, genes: list[str]) -> pd.DataFrame:
    target = set(genes)
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    in_table = False
    with urllib.request.urlopen(url, timeout=60) as response, gzip.GzipFile(fileobj=response) as gz, io.TextIOWrapper(gz, errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for fields in reader:
            if not fields:
                continue
            if fields[0] == "!platform_table_begin":
                in_table = True
                continue
            if fields[0] == "!platform_table_end":
                break
            if not in_table:
                continue
            if header is None:
                header = fields
                continue
            rec = dict(zip(header, fields))
            symbol = normalize_symbol(rec.get("Gene Symbol", ""))
            if symbol in target:
                rows.append(
                    {
                        "ID": rec.get("ID", ""),
                        "Gene Symbol": rec.get("Gene Symbol", ""),
                        "normalized_symbol": symbol,
                        "Gene Title": rec.get("Gene Title", ""),
                        "Entrez Gene": rec.get("Entrez Gene", ""),
                        "RefSeq Transcript ID": rec.get("RefSeq Transcript ID", ""),
                    }
                )
    return pd.DataFrame(rows)


def load_gpl13667_target_annotation() -> pd.DataFrame:
    cached = RAW / "GPL13667_target_annotation.tsv"
    full_soft = RAW / "GPL13667_family.soft.gz"
    if cached.exists():
        annot = pd.read_csv(cached, sep="\t")
        if {"ID", "normalized_symbol"}.issubset(annot.columns) and set(GENES).issubset(set(annot["normalized_symbol"])):
            return annot
    if full_soft.exists():
        annot = parse_gpl13667_target_annotation(full_soft, GENES)
    else:
        annot = parse_gpl13667_target_annotation_from_url(URLS["GPL13667_family_soft"], GENES)
    annot.to_csv(cached, sep="\t", index=False)
    return annot


def write_report(summary: dict[str, object]) -> None:
    assoc_path = RESULTS / "expression_associations.tsv"
    assoc = pd.read_csv(assoc_path, sep="\t") if assoc_path.exists() else pd.DataFrame()
    lines = [
        "# Expression analysis registry",
        "",
        "Primary analysis: tumor-only Colonomics fixed-gene methylation versus matched gene expression.",
        "Supportive analysis: tumor-only ColoCare GSE101764 methylation versus GSE106582 expression; discovery-overlap, not independent validation.",
        "",
        f"- Colonomics matched tumors: {summary.get('colonomics_matched_tumors', 'NA')}",
        f"- ColoCare matched tumors: {summary.get('colocare_matched_tumors', 'NA')}",
        "- Correlation: Spearman rho by fixed gene; q values are BH-adjusted across the 10 genes within each cohort.",
        "- Bootstrap: 5000 patient resamples, seed 20260905.",
        "- Colonomics sensitivity: rank residual correlation after age, sex, site; then age, sex, site, stromal score.",
        "",
    ]
    if not assoc.empty:
        show = assoc[["cohort", "gene", "n", "rho", "p", "q_BH", "ci_low", "ci_high", "status"]]
        lines.extend(["## Results", "", show.to_markdown(index=False, floatfmt=".4g"), ""])
    support_path = REGISTRY / "expression_supportive_status.json"
    if support_path.exists():
        support = json.loads(support_path.read_text())
        lines.extend(["## Supportive expression datasets", "", "```json", json.dumps(support, indent=2, sort_keys=True), "```", ""])
    (REGISTRY / "expression_analysis.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true", help="Download and derive expression files without correlation analysis.")
    args = parser.parse_args(argv)
    prepare()
    summary: dict[str, object] = {}
    if not args.prepare_only:
        colonomics = analyze_colonomics_expression()
        colocare = analyze_colocare_expression()
        combined = pd.concat([colonomics["primary_associations"], colocare["primary_associations"]], ignore_index=True)
        combined.to_csv(RESULTS / "expression_associations.tsv", sep="\t", index=False)
        summary = {
            "colonomics_matched_tumors": colonomics["matched_tumors"],
            "colocare_matched_tumors": colocare["matched_tumors"],
            "colonomics_expression_tumors": colonomics["expression_tumors"],
            "colonomics_methylation_tumors": colonomics["methylation_tumors"],
            "colocare_expression_tumors": colocare["expression_tumors"],
            "colocare_methylation_tumors_after_technical_collapse": colocare["methylation_tumors_after_technical_collapse"],
        }
        write_json(REGISTRY / "expression_summary.json", summary)
    write_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
