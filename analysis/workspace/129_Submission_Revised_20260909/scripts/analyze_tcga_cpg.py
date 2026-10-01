#!/usr/bin/env python3
"""TCGA individual-CpG support analysis for the fixed 77-probe panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
REGISTRY = ROOT / "registry"
RESULTS = ROOT / "results" / "tcga_support"
REVIEW = ROOT / "review"
DERIVED = ROOT / "data" / "derived"
FIXED_PROBES = REGISTRY / "fixed_probes.json"
TCGA_ARCHIVE = PROJECT / "102_ML_revised_20260902" / "external" / "data" / "tcga" / "beta_archived.zip"
TCGA_SAMPLE_MAP = PROJECT / "102_ML_revised_20260902" / "external" / "data" / "tcga" / "gdc_sample_map.json"
ORIGINAL_SELECTION = PROJECT / "119_Figure1_ABC_Revision_20260908" / "registry" / "original_gene_selection.json"
CONTRACT = REGISTRY / "tcga_cpg_contract.json"
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "GFRA1", "UNC5C", "RALYL", "BEND5"]
EXPRESSION_PATTERNS = [
    "*tcga*expression*",
    "*expression*tcga*",
    "*tcga*rna*",
    "*rna*tcga*",
    "*tcga*tpm*",
    "*tpm*tcga*",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_json(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    return obj


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), indent=2, sort_keys=True) + "\n")


def bh_fixed(p_values: Iterable[float], denominator: int = 77) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return q
    ordered = finite[np.argsort(p[finite])]
    raw = p[ordered] * denominator / np.arange(1, len(finite) + 1)
    q[ordered] = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    return q


def fixed_cpg_table() -> pd.DataFrame:
    probes_by_gene = json.loads(FIXED_PROBES.read_text())
    rows = []
    for gene in GENES:
        for order, cpg in enumerate(probes_by_gene[gene], start=1):
            rows.append({"gene": gene, "cpg": cpg, "fixed_probe_order_within_gene": order})
    out = pd.DataFrame(rows)
    if len(out) != 77 or out["cpg"].duplicated().any():
        raise ValueError("fixed_probes.json must define 77 unique CpG probes")
    return out


def load_sample_map(path: Path = TCGA_SAMPLE_MAP) -> pd.DataFrame:
    raw = pd.DataFrame(json.loads(path.read_text()))
    required = {"file_id", "project", "case", "sample", "sample_type"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"gdc_sample_map.json missing columns: {sorted(missing)}")
    raw["file_id"] = raw["file_id"].astype(str)
    raw["archive_member"] = raw["file_id"] + ".txt"
    raw["tissue"] = raw["sample_type"].map({"Primary Tumor": "T", "Solid Tissue Normal": "N"})
    raw = raw.loc[raw["tissue"].isin(["T", "N"])].copy()
    if raw["file_id"].duplicated().any():
        raise ValueError("gdc_sample_map.json has duplicate file_id rows")
    return raw


def stream_fixed_probe_values(zip_path: Path, sample_map: pd.DataFrame, cpgs: pd.DataFrame) -> pd.DataFrame:
    target = set(cpgs["cpg"])
    target_bytes = {cpg.encode(): cpg for cpg in target}
    order = cpgs.set_index("cpg").reset_index().set_index("cpg")
    meta_by_member = sample_map.set_index("archive_member").to_dict("index")
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing_members = sorted(set(meta_by_member) - names)
        if missing_members:
            raise FileNotFoundError(f"archive missing mapped beta files: {missing_members[:5]}")
        for member in sample_map["archive_member"]:
            found: dict[str, float] = {}
            for raw in zf.read(member).splitlines():
                if not raw:
                    continue
                parts = raw.split(b"\t", 1)
                cpg = target_bytes.get(parts[0])
                if cpg is None or len(parts) < 2:
                    continue
                try:
                    found[cpg] = float(parts[1])
                except ValueError:
                    found[cpg] = math.nan
                if len(found) == len(target):
                    break
            meta = meta_by_member[member]
            for cpg in cpgs["cpg"]:
                rec = order.loc[cpg].to_dict()
                rows.append(
                    {
                        "file_id": meta["file_id"],
                        "case": meta["case"],
                        "sample": meta["sample"],
                        "project": meta["project"],
                        "sample_type": meta["sample_type"],
                        "tissue": meta["tissue"],
                        "gene": rec["gene"],
                        "cpg": cpg,
                        "fixed_probe_order_within_gene": int(rec["fixed_probe_order_within_gene"]),
                        "beta": found.get(cpg, math.nan),
                        "probe_present_in_file": cpg in found,
                    }
                )
    out = pd.DataFrame(rows)
    values = out["beta"].to_numpy(float)
    if np.any(np.isfinite(values) & ((values < 0) | (values > 1))):
        raise ValueError("TCGA beta values outside [0,1]")
    return out


def collapse_case_tissue(long_beta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["case", "project", "tissue", "gene", "cpg", "fixed_probe_order_within_gene"]
    collapsed = long_beta.groupby(group_cols, dropna=False, as_index=False).agg(
        beta=("beta", "mean"),
        n_files=("file_id", "nunique"),
        n_present_files=("probe_present_in_file", "sum"),
    )
    sample_audit = (
        long_beta[["case", "project", "tissue", "file_id", "sample", "sample_type"]]
        .drop_duplicates()
        .groupby(["case", "project", "tissue", "sample_type"], dropna=False, as_index=False)
        .agg(n_files=("file_id", "nunique"), samples=("sample", lambda x: ";".join(sorted(map(str, x)))))
    )
    sample_audit["status"] = np.where(sample_audit["n_files"].gt(1), "collapsed_case_tissue_technical_or_aliquot_replicates", "single_file")
    return collapsed, sample_audit


def paired_cpg_stats(collapsed: pd.DataFrame, cpgs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in cpgs.to_dict("records"):
        sub = collapsed.loc[collapsed["cpg"].eq(rec["cpg"]), ["case", "project", "tissue", "beta"]].dropna(subset=["beta"])
        tumor = sub.loc[sub["tissue"].eq("T"), ["case", "beta"]].rename(columns={"beta": "tumor_beta"})
        normal = sub.loc[sub["tissue"].eq("N"), ["case", "beta"]].rename(columns={"beta": "normal_beta"})
        paired = tumor.merge(normal, on="case", how="inner", validate="one_to_one")
        diff = paired["tumor_beta"].to_numpy(float) - paired["normal_beta"].to_numpy(float)
        if len(diff) >= 2 and np.nanstd(diff) > 0:
            t_p = float(stats.ttest_rel(paired["tumor_beta"], paired["normal_beta"], nan_policy="omit").pvalue)
            try:
                w_p = float(stats.wilcoxon(diff).pvalue)
            except ValueError:
                w_p = math.nan
        else:
            t_p = math.nan
            w_p = math.nan
        rows.append(
            {
                **rec,
                "analysis": "tcga_case_paired_tumor_minus_normal_beta",
                "n_pairs": int(len(paired)),
                "n_tumor_cases": int(tumor["case"].nunique()),
                "n_normal_cases": int(normal["case"].nunique()),
                "tumor_mean_beta": float(tumor["tumor_beta"].mean()) if len(tumor) else math.nan,
                "normal_mean_beta": float(normal["normal_beta"].mean()) if len(normal) else math.nan,
                "mean_delta_beta": float(np.mean(diff)) if len(diff) else math.nan,
                "median_delta_beta": float(np.median(diff)) if len(diff) else math.nan,
                "fraction_tumor_gt_normal": float(np.mean(diff > 0)) if len(diff) else math.nan,
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
                "status": "estimated" if len(paired) >= 2 else "not_estimable",
            }
        )
    out = pd.DataFrame(rows)
    out["paired_t_q_BH77"] = bh_fixed(out["paired_t_p"])
    out["wilcoxon_q_BH77"] = bh_fixed(out["wilcoxon_p"])
    return out


def find_tcga_expression_candidates() -> list[dict[str, object]]:
    roots = [
        PROJECT / "102_ML_revised_20260902",
        PROJECT / "114_ML_DataDriven_20260905",
        PROJECT / "115_Public_Biology_20260905",
        PROJECT / "119_Figure1_ABC_Revision_20260908",
    ]
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in EXPRESSION_PATTERNS:
            for path in root.rglob(pattern):
                if path.is_file() and ".venv" not in path.parts:
                    rows.append(
                        {
                            "path": str(path.relative_to(PROJECT)),
                            "bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                    )
    rows = sorted({row["path"]: row for row in rows}.values(), key=lambda x: str(x["path"]))
    return rows


def expression_gap_rows(cpgs: pd.DataFrame, candidates: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    status = "source_unavailable" if not candidates else "candidate_source_unreviewed"
    note = (
        "No TCGA RNA expression matrix or case-matched TPM/count table was found under the existing 102/114/115/119 workspaces."
        if not candidates
        else "Potential TCGA expression-named files exist, but no reviewed case-matched expression table was accepted by this script."
    )
    for scope in ["tumor_only", "pooled_tumor_normal"]:
        for rec in cpgs.to_dict("records"):
            rows.append(
                {
                    **rec,
                    "analysis": "tcga_methylation_expression_spearman",
                    "scope": scope,
                    "n": 0,
                    "rho": math.nan,
                    "p": math.nan,
                    "q_BH77": math.nan,
                    "status": status,
                    "source_note": note,
                }
            )
    return pd.DataFrame(rows)


def write_review(summary: dict[str, object], candidates: list[dict[str, object]]) -> None:
    original = json.loads(ORIGINAL_SELECTION.read_text()) if ORIGINAL_SELECTION.exists() else {}
    evidence_limits = original.get("evidence_limits", [])
    raster_sources = original.get("sources", [])
    text = [
        "# TCGA Individual-CpG Source Audit",
        "",
        "## Result",
        "",
        "TCGA support was rebuilt at the fixed 77-CpG level from the archived GDC beta files without fully unzipping the archive. The new methylation result is case-paired tumor-minus-normal beta per CpG. TCGA methylation-expression correlations were not recomputed because no reviewed TCGA RNA expression matrix with case barcode mapping was present in the existing workspace.",
        "",
        "## Inputs",
        "",
        f"- Fixed probes: `{FIXED_PROBES.relative_to(PROJECT)}`",
        f"- TCGA beta archive: `{TCGA_ARCHIVE.relative_to(PROJECT)}`",
        f"- TCGA sample map: `{TCGA_SAMPLE_MAP.relative_to(PROJECT)}`",
        f"- Original Figure 1 source registry: `{ORIGINAL_SELECTION.relative_to(PROJECT)}`",
        "",
        "## Output Counts",
        "",
        f"- Archive members streamed: {summary['archive_members_streamed']}",
        f"- Fixed CpG rows: {summary['fixed_cpg_rows']}",
        f"- Long beta rows: {summary['long_beta_rows']}",
        f"- Case-tissue-CpG rows: {summary['case_tissue_cpg_rows']}",
        f"- Paired CpG statistic rows: {summary['paired_rows']}",
        f"- Estimated paired CpG rows: {summary['estimated_paired_rows']}",
        f"- Not-estimable paired CpG rows: {summary['not_estimable_paired_rows']}",
        f"- Expression correlation placeholder rows: {summary['expression_rows']}",
        f"- Paired normal cases per CpG: {summary['paired_n_min']} to {summary['paired_n_max']}",
        "",
        "## Expression Source Audit",
        "",
    ]
    if candidates:
        text.append("Files with TCGA/expression/RNA/TPM-like names were found, but none were accepted as a reviewed case-matched TCGA RNA matrix by this script:")
        text.extend(f"- `{row['path']}`" for row in candidates)
    else:
        text.append("No TCGA expression/RNA/TPM file was found under `102_ML_revised_20260902`, `114_ML_DataDriven_20260905`, `115_Public_Biology_20260905`, or `119_Figure1_ABC_Revision_20260908`.")
    text.extend(
        [
            "",
            "## Historical Figure 1B Limit",
            "",
            "The preserved original registry states that Figure 1B used pooled tumor and normal TCGA methylation-expression plots as supporting context. The original raster/source registry is retained, but the exact per-CpG or gene-mean plotting inputs for that raster are not reproducibly rebuildable from the available TCGA expression sources.",
        ]
    )
    if evidence_limits:
        text.append("")
        text.append("Registry evidence limits:")
        text.extend(f"- {item}" for item in evidence_limits if "Figure 1B" in item or "TCGA" in item)
    if raster_sources:
        text.append("")
        text.append("Original source records:")
        text.extend(f"- `{item.get('path')}` sha256 `{item.get('sha256')}`" for item in raster_sources)
    text.append("")
    (REVIEW / "tcga_source_audit.md").write_text("\n".join(text), encoding="utf-8")


def run() -> dict[str, object]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    cpgs = fixed_cpg_table()
    sample_map = load_sample_map()
    long_beta = stream_fixed_probe_values(TCGA_ARCHIVE, sample_map, cpgs)
    collapsed, sample_audit = collapse_case_tissue(long_beta)
    paired = paired_cpg_stats(collapsed, cpgs)
    candidates = find_tcga_expression_candidates()
    expr = expression_gap_rows(cpgs, candidates)

    cpgs.to_csv(DERIVED / "TCGA_fixed77_cpgs.tsv", sep="\t", index=False)
    sample_map.to_csv(DERIVED / "TCGA_sample_map.tsv", sep="\t", index=False)
    long_beta.to_csv(DERIVED / "TCGA_fixed77_beta_long.tsv", sep="\t", index=False)
    collapsed.to_csv(DERIVED / "TCGA_fixed77_case_tissue_beta.tsv", sep="\t", index=False)
    sample_audit.to_csv(RESULTS / "tcga_sample_collapse_audit.tsv", sep="\t", index=False)
    paired.to_csv(RESULTS / "tcga_cpg_tumor_normal_paired.tsv", sep="\t", index=False)
    expr.to_csv(RESULTS / "tcga_cpg_expression_correlations.tsv", sep="\t", index=False)
    if candidates:
        pd.DataFrame(candidates).to_csv(RESULTS / "tcga_expression_source_candidates.tsv", sep="\t", index=False)
    else:
        pd.DataFrame(columns=["path", "bytes", "sha256"]).to_csv(RESULTS / "tcga_expression_source_candidates.tsv", sep="\t", index=False)

    summary = {
        "created_by": "scripts/analyze_tcga_cpg.py",
        "status": "complete_with_expression_source_gap",
        "fixed_cpg_rows": int(len(cpgs)),
        "archive_members_streamed": int(sample_map["archive_member"].nunique()),
        "long_beta_rows": int(len(long_beta)),
        "case_tissue_cpg_rows": int(len(collapsed)),
        "paired_rows": int(len(paired)),
        "paired_n_min": int(paired["n_pairs"].min()),
        "paired_n_max": int(paired["n_pairs"].max()),
        "estimated_paired_rows": int(paired["status"].eq("estimated").sum()),
        "not_estimable_paired_rows": int(paired["status"].ne("estimated").sum()),
        "expression_rows": int(len(expr)),
        "estimated_expression_rows": 0,
        "tcga_expression_candidates": candidates,
        "inputs": [
            {"path": str(FIXED_PROBES.relative_to(PROJECT)), "sha256": sha256(FIXED_PROBES), "bytes": FIXED_PROBES.stat().st_size},
            {"path": str(TCGA_ARCHIVE.relative_to(PROJECT)), "sha256": sha256(TCGA_ARCHIVE), "bytes": TCGA_ARCHIVE.stat().st_size},
            {"path": str(TCGA_SAMPLE_MAP.relative_to(PROJECT)), "sha256": sha256(TCGA_SAMPLE_MAP), "bytes": TCGA_SAMPLE_MAP.stat().st_size},
            {"path": str(ORIGINAL_SELECTION.relative_to(PROJECT)), "sha256": sha256(ORIGINAL_SELECTION), "bytes": ORIGINAL_SELECTION.stat().st_size} if ORIGINAL_SELECTION.exists() else None,
        ],
    }
    summary["inputs"] = [x for x in summary["inputs"] if x is not None]
    primary_outputs = [
        DERIVED / "TCGA_fixed77_cpgs.tsv",
        DERIVED / "TCGA_sample_map.tsv",
        DERIVED / "TCGA_fixed77_beta_long.tsv",
        DERIVED / "TCGA_fixed77_case_tissue_beta.tsv",
        RESULTS / "tcga_sample_collapse_audit.tsv",
        RESULTS / "tcga_cpg_tumor_normal_paired.tsv",
        RESULTS / "tcga_cpg_expression_correlations.tsv",
        RESULTS / "tcga_expression_source_candidates.tsv",
        REVIEW / "tcga_source_audit.md",
    ]
    output_manifest = lambda paths: [
        {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in paths
        if path.exists()
    ]
    write_review(summary, candidates)
    contract = {
        **{k: v for k, v in summary.items() if k != "contract_sha256"},
        "analysis_unit": "individual CpG probe",
        "bh_family_size": 77,
        "beta_archive_access": "zipfile streaming, no full archive extraction",
        "case_pairing_rule": "paired by TCGA case barcode after case+tissue beta averaging",
        "expression_rule": "do not infer or recreate historical Figure 1B expression mapping without a reviewed TCGA RNA source",
        "outputs": output_manifest(primary_outputs),
    }
    write_json(CONTRACT, contract)
    summary["contract_sha256"] = sha256(CONTRACT)
    summary["outputs"] = output_manifest(primary_outputs + [CONTRACT])
    write_json(RESULTS / "tcga_cpg_run_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    summary = run()
    print(json.dumps(clean_json({k: summary[k] for k in ["fixed_cpg_rows", "archive_members_streamed", "paired_rows", "paired_n_min", "paired_n_max", "estimated_expression_rows"]}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
