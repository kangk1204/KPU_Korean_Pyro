#!/usr/bin/env python3
"""Reproducible public-cohort methylation reanalysis.

Run once with --snapshot while the historical external workspace is available.
Subsequent runs use only 113_DataDriven_20260905/data/public and do not depend
on 102_ML_revised_20260902.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GENES, N_BOOT, PROJECT, ROOT, SEED, bh, sha256, write_json  # noqa: E402

OLD_EXTERNAL = PROJECT / "102_ML_revised_20260902" / "external"
PUBLIC = ROOT / "data" / "public"
PUBLIC_RESULTS = PUBLIC / "processed"
PROVENANCE = PUBLIC / "provenance"
RES = ROOT / "results" / "external"
SUPP = ROOT / "supplement"
TABLE1_SNAPSHOT = PUBLIC / "table1_probes.json"


def copy_if_needed(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    if src.exists() and sha256(src) != sha256(dst):
        raise ValueError(f"Snapshot hash mismatch after copy: {src} -> {dst}")
    return {
        "source": str(src.relative_to(PROJECT)) if src.is_absolute() and PROJECT in src.parents else str(src),
        "snapshot": str(dst.relative_to(ROOT)),
        "sha256": sha256(dst),
        "bytes": dst.stat().st_size,
    }


def read_table1() -> dict[str, list[str]]:
    with TABLE1_SNAPSHOT.open() as f:
        return json.load(f)


def all_table1_probes(table1: dict[str, list[str]]) -> list[str]:
    return sorted({probe for probes in table1.values() for probe in probes})


def parse_series_matrix(path: Path) -> pd.DataFrame:
    rows: dict[str, list[list[str]]] = {}
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            fields = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")]
            rows.setdefault(fields[0], []).append(fields[1:])
    gsms = rows["!Sample_geo_accession"][0]
    out = pd.DataFrame({"geo_accession": gsms})
    for key in ["!Sample_title", "!Sample_description", "!Sample_source_name_ch1"]:
        if key in rows:
            out[key.replace("!Sample_", "").lower()] = rows[key][0]
    chars: dict[str, dict[str, str]] = {g: {} for g in gsms}
    for line in rows.get("!Sample_characteristics_ch1", []):
        for i, val in enumerate(line):
            if ":" in val:
                k, v = val.split(":", 1)
                chars[gsms[i]][k.strip()] = v.strip()
    for key in sorted({k for d in chars.values() for k in d}):
        out[key] = [chars[g].get(key, "") for g in gsms]
    return out


def snapshot_gse119526_target_rows(src: Path, dst: Path, probes: set[str]) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        first = True
        with gzip.open(dst, "wt") as out:
            for chunk in pd.read_csv(src, chunksize=100_000):
                sub = chunk[chunk["ID_REF"].astype(str).isin(probes)]
                if len(sub):
                    sub.to_csv(out, sep="\t", index=False, header=first)
                    first = False
    frozen = pd.read_csv(dst, sep="\t", usecols=["ID_REF"])
    return {
        "source": str(src.relative_to(PROJECT)),
        "snapshot": str(dst.relative_to(ROOT)),
        "source_sha256": sha256(src),
        "source_bytes": src.stat().st_size,
        "sha256": sha256(dst),
        "bytes": dst.stat().st_size,
        "target_rows": int(len(frozen)),
        "note": "Only fixed Table 1 probe rows were frozen from the large signal-intensity matrix.",
    }


def validate_public_manifest() -> None:
    manifest_path = ROOT / "registry" / "external_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    for item in manifest.get("inputs", []):
        snap = ROOT / item["snapshot"]
        if not snap.exists():
            raise FileNotFoundError(f"Frozen external input missing: {snap}")
        actual = sha256(snap)
        if actual != item["sha256"]:
            raise ValueError(f"Frozen external input hash mismatch: {snap}")


def snapshot_inputs() -> None:
    if not OLD_EXTERNAL.exists():
        if TABLE1_SNAPSHOT.exists():
            print("Historical external folder absent; existing data/public snapshot will be used.")
            return
        raise FileNotFoundError("Cannot create first snapshot: 102_ML_revised_20260902/external is absent.")
    PUBLIC_RESULTS.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"created_by": "scripts/analyze_external.py --snapshot", "inputs": []}
    inputs: list[dict] = manifest["inputs"]  # type: ignore[assignment]
    inputs.append(copy_if_needed(OLD_EXTERNAL / "table1_probes.json", TABLE1_SNAPSHOT))
    table1 = read_table1()
    probes = set(all_table1_probes(table1))
    for rel in [
        "build_cohorts.py",
        "analyze_external.py",
        "process_arrays.R",
        "PROTOCOL.md",
        "results/colonomics_beta_selected.tsv",
        "results/colonomics_probe_annotation.tsv",
        "results/colonomics_sample_qc.tsv",
        "data/gse131013/CLX_ClinicalData.tab",
        "data/gse131013/GSE131013_series_matrix.txt.gz",
        "results/tcga_coadread_features.tsv",
        "results/tcga_coad_features.tsv",
        "results/tcga_read_features.tsv",
    ]:
        src = OLD_EXTERNAL / rel
        dst = (PROVENANCE if src.suffix in {".py", ".R", ".md"} else PUBLIC_RESULTS) / src.name
        inputs.append(copy_if_needed(src, dst))
    gse119_series = OLD_EXTERNAL / "data/gse119526/GSE119526_series_matrix.txt.gz"
    gse119_meta = parse_series_matrix(gse119_series)
    meta_dst = PUBLIC_RESULTS / "gse119526_sample_metadata.tsv"
    gse119_meta.to_csv(meta_dst, sep="\t", index=False)
    inputs.append(
        {
            "source": str(gse119_series.relative_to(PROJECT)),
            "snapshot": str(meta_dst.relative_to(ROOT)),
            "source_sha256": sha256(gse119_series),
            "sha256": sha256(meta_dst),
            "bytes": meta_dst.stat().st_size,
            "note": "Parsed sample metadata snapshot; the large GEO series matrix was not copied.",
        }
    )
    inputs.append(
        snapshot_gse119526_target_rows(
            OLD_EXTERNAL / "data/gse119526/GSE119526_Matrix_signal_intensities.csv.gz",
            PUBLIC_RESULTS / "gse119526_table1_signal_intensities.tsv.gz",
            probes,
        )
    )
    write_json(ROOT / "registry" / "external_manifest.json", manifest)
    print(f"Snapshot complete: {len(inputs)} frozen inputs")


def canonical_sample_id(col: str) -> str:
    m = re.search(r"([A-Z]\d+_[NTM])(?:_\d+)?$", col)
    if not m:
        raise ValueError(f"Cannot parse Colonomics sample id from {col}")
    return m.group(1)


def required_count(n_targets: int) -> int:
    return int(math.ceil(0.8 * n_targets))


def gene_scores(beta: pd.DataFrame, table1: dict[str, list[str]], cohort: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    scores = pd.DataFrame(index=beta.columns)
    valid_counts = pd.DataFrame(index=beta.columns)
    for gene in GENES:
        targets = table1[gene]
        present = [p for p in targets if p in beta.index]
        minimum = required_count(len(targets))
        vals = beta.reindex(present).T
        counts = vals.notna().sum(axis=1) if present else pd.Series(0, index=beta.columns)
        score = vals.mean(axis=1, skipna=True) if present else pd.Series(np.nan, index=beta.columns)
        scores[gene] = score.where(counts >= minimum)
        valid_counts[gene] = counts
        rows.append(
            {
                "cohort": cohort,
                "gene": gene,
                "n_target_probes": len(targets),
                "n_available_probes": len(present),
                "min_valid_probes_per_sample": minimum,
                "coverage_fraction": len(present) / len(targets),
                "probe_source": "fixed_table1_only",
                "used_probes": ";".join(present),
                "missing_probes": ";".join([p for p in targets if p not in present]),
            }
        )
    return scores, valid_counts, pd.DataFrame(rows)


def build_colonomics(table1: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    beta = pd.read_csv(PUBLIC_RESULTS / "colonomics_beta_selected.tsv", sep="\t", index_col=0)
    if beta.index.name != "probe":
        beta = beta.set_index(beta.columns[0])
    qc = pd.read_csv(PUBLIC_RESULTS / "colonomics_sample_qc.tsv", sep="\t")
    excluded = set(qc.loc[qc["excluded"].astype(bool), "sample"])
    beta = beta.drop(columns=[c for c in beta.columns if c in excluded], errors="ignore")
    beta.columns = [canonical_sample_id(c) for c in beta.columns]
    if beta.columns.duplicated().any():
        beta = beta.T.groupby(level=0).mean().T
    clin = pd.read_csv(PUBLIC_RESULTS / "CLX_ClinicalData.tab", sep=",")
    clin = clin.set_index("id_clx")
    meta = clin.reindex(beta.columns)
    tissue = meta["type"].map({"Tumor": "T", "Normal": "N", "Mucosa": "M"})
    scores, counts, coverage = gene_scores(beta, table1, "colonomics")
    out = pd.DataFrame(
        {
            "cohort": "colonomics",
            "sample": beta.columns,
            "patient": meta["id_clx_individual"].values,
            "tissue": tissue.values,
            "stage": meta["stage"].values,
            "recur": np.where(tissue.values == "T", pd.to_numeric(meta["event_free"], errors="coerce"), np.nan),
            "time": np.where(tissue.values == "T", pd.to_numeric(meta["time_free"], errors="coerce"), np.nan),
            "sex": meta["sex"].values,
            "age": meta["age"].values,
            "site": meta["site"].values,
            "cms": meta["CMS"].values,
        }
    )
    for gene in GENES:
        out[f"{gene}_beta"] = scores[gene].values
        out[f"{gene}_valid_probes"] = counts[gene].values
    return out, coverage, beta


def build_gse119526(table1: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(PUBLIC_RESULTS / "gse119526_table1_signal_intensities.tsv.gz", sep="\t")
    ids = sorted({c.removesuffix("_methylated_signal") for c in raw.columns if c.endswith("_methylated_signal")})
    m = raw[[f"{i}_methylated_signal" for i in ids]].to_numpy(float)
    u = raw[[f"{i}_unmethylated_signal" for i in ids]].to_numpy(float)
    p = raw[[f"{i}_detection_pval" for i in ids]].to_numpy(float)
    beta = pd.DataFrame(m / (m + u + 100), index=raw["ID_REF"].astype(str), columns=ids).mask(p >= 0.01)
    meta = pd.read_csv(PUBLIC_RESULTS / "gse119526_sample_metadata.tsv", sep="\t") if (PUBLIC_RESULTS / "gse119526_sample_metadata.tsv").exists() else None
    if meta is None:
        meta = parse_series_matrix(PUBLIC_RESULTS / "GSE119526_series_matrix.txt.gz")
        meta.to_csv(PUBLIC_RESULTS / "gse119526_sample_metadata.tsv", sep="\t", index=False)
    gsm_by_author = dict(zip(meta["description"], meta["geo_accession"]))
    char = meta.set_index("geo_accession")
    rows = []
    for sid in ids:
        gsm = gsm_by_author.get(sid)
        if gsm is None:
            raise ValueError(f"GSE119526 author id missing from series metadata: {sid}")
        tissue = "T" if sid.endswith("T") else "N"
        recurrent_status = char.loc[gsm, "recurrent_status"]
        rows.append(
            {
                "cohort": "gse119526",
                "sample": gsm,
                "author_id": sid,
                "patient": sid[:-1],
                "tissue": tissue,
                "stage": "early-stage",
                "recur": np.nan if tissue == "N" else (1.0 if recurrent_status == "Recurrence" else 0.0),
                "time": np.nan,
                "sex": char.loc[gsm, "Sex"],
                "recurrent_status_field": recurrent_status,
            }
        )
    out = pd.DataFrame(rows)
    pair_check = out.groupby("patient").agg(n=("tissue", "size"), tissues=("tissue", lambda x: "".join(sorted(x))))
    bad = pair_check[(pair_check["n"] != 2) | (pair_check["tissues"] != "NT")]
    if len(bad):
        raise ValueError(f"GSE119526 inconsistent T/N pairs: {bad.index.tolist()[:5]}")
    scores, counts, coverage = gene_scores(beta, table1, "gse119526")
    scores = scores.reindex(out["author_id"])
    counts = counts.reindex(out["author_id"])
    for gene in GENES:
        out[f"{gene}_beta"] = scores[gene].values
        out[f"{gene}_valid_probes"] = counts[gene].values
    return out, coverage, beta


def build_tcga(table1: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for name, cohort in [("tcga_coadread_features.tsv", "tcga_coadread"), ("tcga_coad_features.tsv", "tcga_coad"), ("tcga_read_features.tsv", "tcga_read")]:
        path = PUBLIC_RESULTS / name
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            df.insert(0, "cohort", cohort)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    cov = []
    for cohort in out["cohort"].unique():
        for gene in GENES:
            present = table1[gene]
            cov.append(
                {
                    "cohort": cohort,
                    "gene": gene,
                    "n_target_probes": len(table1[gene]),
                    "n_available_probes": np.nan,
                    "min_valid_probes_per_sample": required_count(len(table1[gene])),
                    "coverage_fraction": np.nan,
                    "probe_source": "historical_tcga_feature_snapshot_probe_availability_not_reverified",
                    "used_probes": ";".join(present),
                    "missing_probes": np.nan,
                }
            )
    return out, pd.DataFrame(cov)


def paired_stats(features: pd.DataFrame, coverage: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    values = []
    missing = []
    feat = features[(features["cohort"] == cohort) & features["tissue"].isin(["T", "N"])].copy()
    for gene in GENES:
        cols = ["patient", "sample", "tissue", f"{gene}_beta"]
        sub = feat[cols].dropna(subset=["patient"])
        t = sub[sub["tissue"] == "T"][["patient", "sample", f"{gene}_beta"]].rename(columns={"sample": "tumor_sample", f"{gene}_beta": "tumor_beta"})
        n = sub[sub["tissue"] == "N"][["patient", "sample", f"{gene}_beta"]].rename(columns={"sample": "normal_sample", f"{gene}_beta": "normal_beta"})
        paired = t.merge(n, on="patient", how="inner")
        paired = paired.dropna(subset=["tumor_beta", "normal_beta"])
        all_patients = set(t["patient"].dropna()) | set(n["patient"].dropna())
        for pid in sorted(all_patients - set(paired["patient"])):
            missing.append({"cohort": cohort, "gene": gene, "patient": pid, "reason": "missing_tumor_or_normal_or_gene_score"})
        diff = paired["tumor_beta"].to_numpy(float) - paired["normal_beta"].to_numpy(float)
        cohort_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(cohort))
        bca_lo, bca_hi = bca_ci(diff, np.mean, SEED + GENES.index(gene) + 101 * cohort_seed, n_resamples=N_BOOT)
        if len(diff) >= 2:
            t_p = stats.ttest_rel(paired["tumor_beta"], paired["normal_beta"], nan_policy="omit").pvalue
            try:
                w_p = stats.wilcoxon(diff).pvalue
            except ValueError:
                w_p = np.nan
        else:
            t_p = np.nan
            w_p = np.nan
        cov = coverage[(coverage["cohort"] == cohort) & (coverage["gene"] == gene)].iloc[0].to_dict()
        rows.append(
            {
                "cohort": cohort,
                "gene": gene,
                "n_pairs": int(len(paired)),
                "tumor_mean_beta": float(paired["tumor_beta"].mean()),
                "normal_mean_beta": float(paired["normal_beta"].mean()),
                "mean_delta_beta": float(np.mean(diff)),
                "bca95_low": bca_lo,
                "bca95_high": bca_hi,
                "median_delta_beta": float(np.median(diff)),
                "fraction_tumor_gt_normal": float(np.mean(diff > 0)),
                "paired_t_p": float(t_p) if np.isfinite(t_p) else np.nan,
                "wilcoxon_p": float(w_p) if np.isfinite(w_p) else np.nan,
                "probe_source": cov["probe_source"],
                "n_target_probes": int(cov["n_target_probes"]),
                "n_available_probes": int(cov["n_available_probes"]) if pd.notna(cov["n_available_probes"]) else np.nan,
                "coverage_fraction": float(cov["coverage_fraction"]),
            }
        )
        for record in paired.assign(cohort=cohort, gene=gene, delta_beta=diff)[
            ["cohort", "gene", "patient", "tumor_sample", "normal_sample", "tumor_beta", "normal_beta", "delta_beta"]
        ].to_dict("records"):
            values.append(record)
    out = pd.DataFrame(rows)
    out["paired_t_q_BH10"] = bh(out["paired_t_p"])
    out["wilcoxon_q_BH10"] = bh(out["wilcoxon_p"])
    return out, pd.DataFrame(values), pd.DataFrame(missing)


def bca_ci(x: np.ndarray, stat_func, seed: int, n_resamples: int = 5000) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    theta = stat_func(x)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    boots = np.apply_along_axis(stat_func, 1, x[idx])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return (np.nan, np.nan)
    prop = np.clip(np.mean(boots < theta), 1 / (2 * len(boots)), 1 - 1 / (2 * len(boots)))
    z0 = stats.norm.ppf(prop)
    jack = np.array([stat_func(np.delete(x, i)) for i in range(len(x))])
    jmean = jack.mean()
    denom = 6.0 * (np.sum((jmean - jack) ** 2) ** 1.5)
    acc = np.sum((jmean - jack) ** 3) / denom if denom > 0 else 0.0
    qs = []
    for alpha in (0.025, 0.975):
        za = stats.norm.ppf(alpha)
        adj = stats.norm.cdf(z0 + (z0 + za) / (1 - acc * (z0 + za)))
        qs.append(float(np.quantile(boots, np.clip(adj, 0, 1))))
    return qs[0], qs[1]


def cohort_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, df in features.groupby("cohort", sort=False):
        tn = df[df["tissue"].isin(["T", "N"])]
        pair_tab = tn.pivot_table(index="patient", columns="tissue", values="sample", aggfunc="size")
        n_pairs = int(((pair_tab.get("T", 0) > 0) & (pair_tab.get("N", 0) > 0)).sum())
        rows.append(
            {
                "cohort": cohort,
                "n_rows": int(len(df)),
                "n_patients": int(df["patient"].nunique()),
                "n_tumor": int((df["tissue"] == "T").sum()),
                "n_normal": int((df["tissue"] == "N").sum()),
                "n_other_or_missing_tissue": int((~df["tissue"].isin(["T", "N"])).sum()),
                "n_tn_pairs_by_patient": n_pairs,
                "n_tumor_with_outcome": int(((df["tissue"] == "T") & df["recur"].notna()).sum()) if "recur" in df else 0,
                "events_in_tumor_rows": int(df.loc[df["tissue"] == "T", "recur"].fillna(0).sum()) if "recur" in df else 0,
            }
        )
    return pd.DataFrame(rows)


def assay_mapping(table1: dict[str, list[str]], coverage: pd.DataFrame) -> pd.DataFrame:
    primer = pd.read_csv(ROOT / "data" / "derived" / "assay_primers.tsv", sep="\t")
    ann_path = PUBLIC_RESULTS / "colonomics_probe_annotation.tsv"
    ann = pd.read_csv(ann_path, sep="\t") if ann_path.exists() else pd.DataFrame()
    rows = []
    for gene in GENES:
        a = ann[ann["probe"].isin(table1[gene])] if len(ann) else pd.DataFrame()
        pos = ";".join(a.assign(coord=a["chr"].astype(str) + ":" + a["pos"].astype(str))["coord"].tolist()) if len(a) else ""
        pr = primer[primer["gene"] == gene].iloc[0].to_dict()
        rows.append(
            {
                "gene": gene,
                "psq_amplicon_bp": pr["amplicon_bp"],
                "forward_primer": pr["forward_primer"],
                "reverse_primer": pr["reverse_primer"],
                "sequencing_primer": pr["sequencing_primer"],
                "array_table1_probes": ";".join(table1[gene]),
                "array_probe_coordinates_from_colonomics_annotation": pos,
                "psq_array_mapping_level": "gene_or_region_level",
                "exact_psq_cpg_overlap_verified": False,
                "mapping_note": "No amplicon genomic coordinates, local human genome, or bisulfite aligner were available in this workspace; exact PSQ CpG overlap was not inferred from degenerate primer strings.",
            }
        )
    return pd.DataFrame(rows)


def write_methods(cohort_summary_df: pd.DataFrame, coverage: pd.DataFrame) -> None:
    lines = [
        "# External cohort methods and limitations",
        "",
        "Public cohort analyses were rerun from frozen inputs under `data/public` after one snapshot step.",
        "Gene scores used only the fixed Table 1 probe lists from the prior analysis. No promoter fallback or significance-based probe replacement was used.",
        "For Colonomics and GSE119526, the sample-level gene score was the mean beta value over available target probes only when at least 80% of the originally listed probes had non-missing beta values.",
        "Colonomics used previously Noob-normalized 450K beta values. The historical processing masked detection p >= 0.01 and excluded a sample if more than 5% of probes failed detection. The current snapshot contained no sample exclusions among 240 samples; paired tissue comparisons used the 92 patient IDs with both tumor and adjacent normal records.",
        "GSE119526 used fixed EPIC probe rows from the supplied methylated (M) and unmethylated (U) signal matrix. Beta was computed as M/(M+U+100), and detection p >= 0.01 was masked. This step did not repeat whole-array IDAT preprocessing. Exact probe identifiers and missing probes are listed in the coverage workbook.",
        "Tumor-normal comparisons used paired patient joins, not row order. P values were adjusted by Benjamini-Hochberg within each cohort across the ten fixed genes.",
        "Confidence intervals for paired mean delta beta were patient-pair BCa bootstrap intervals with 5,000 resamples. Deterministic cohort/gene seeds were derived from the base seed 20260905 in the analysis script.",
        "",
        "TCGA is retained as supportive, discovery-overlapping evidence. It is reported separately and is not pooled with the two independent public patient series.",
        "For TCGA, the current package uses frozen historical gene-score feature tables. Because raw TCGA beta files were not frozen into this new package, TCGA probe availability and per-sample 80% probe coverage are marked as not reverified.",
        "Public outcomes were not analyzed in this script; recurrence modeling is handled by the private clinical cohort analysis.",
        "",
        "## Cohort counts",
        "",
        cohort_summary_df.to_markdown(index=False),
        "",
        "## Probe coverage",
        "",
        coverage[["cohort", "gene", "n_target_probes", "n_available_probes", "coverage_fraction", "probe_source"]].to_markdown(index=False),
        "",
        "## Assay mapping limit",
        "",
        "PSQ primer sequences and amplicon lengths were recovered from the prior supplementary table. Exact genomic amplicon coordinates or assayed CpG positions were not available. Therefore PSQ-array agreement is reported at the gene level; exact CpG or within-region correspondence was not established.",
        "",
    ]
    (SUPP / "external_methods.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis() -> None:
    if not TABLE1_SNAPSHOT.exists():
        snapshot_inputs()
    validate_public_manifest()
    table1 = read_table1()
    RES.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    col, col_cov, col_beta = build_colonomics(table1)
    gse, gse_cov, gse_beta = build_gse119526(table1)
    tcga, tcga_cov = build_tcga(table1)
    features = pd.concat([col, gse, tcga], ignore_index=True, sort=False)
    coverage = pd.concat([col_cov, gse_cov, tcga_cov], ignore_index=True, sort=False)
    primary_rows = []
    source_values = []
    missing_rows = []
    for cohort in ["colonomics", "gse119526", "tcga_coadread", "tcga_coad", "tcga_read"]:
        if cohort not in set(features["cohort"]):
            continue
        stats_df, vals_df, miss_df = paired_stats(features, coverage, cohort)
        primary_rows.append(stats_df)
        source_values.append(vals_df)
        missing_rows.append(miss_df)
    external = pd.concat(primary_rows, ignore_index=True)
    values = pd.concat(source_values, ignore_index=True)
    missing = pd.concat(missing_rows, ignore_index=True) if missing_rows else pd.DataFrame(columns=["cohort", "gene", "patient", "reason"])
    summary = cohort_summary(features)
    mapping = assay_mapping(table1, coverage)
    features.to_csv(RES / "public_gene_scores.tsv", sep="\t", index=False)
    coverage.to_csv(RES / "probe_coverage.tsv", sep="\t", index=False)
    summary.to_csv(RES / "cohort_summary.tsv", sep="\t", index=False)
    external.to_csv(RES / "paired_public_effects.tsv", sep="\t", index=False)
    values.to_csv(RES / "paired_source_values.tsv", sep="\t", index=False)
    missing.to_csv(RES / "missing_paired_ids.tsv", sep="\t", index=False)
    mapping.to_csv(SUPP / "assay_mapping.tsv", sep="\t", index=False)
    write_methods(summary, coverage)
    registry = {
        "analysis": "external_public_cohorts",
        "seed": SEED,
        "n_boot": N_BOOT,
        "uses_only_data_public_after_snapshot": True,
        "cohorts_primary_public": ["colonomics", "gse119526"],
        "tcga_supportive_only": ["tcga_coadread", "tcga_coad", "tcga_read"],
        "outputs": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(list(RES.glob("*")) + list(SUPP.glob("assay_mapping.tsv")))},
        "limitations": [
            "Exact PSQ CpG overlap with array probes was not verifiable from local assay documentation.",
            "GSE119526 was rerun from frozen Table 1 target signal-intensity rows, not full IDAT rerun.",
            "TCGA is supportive and discovery-overlapping; it is not an untouched validation cohort. TCGA probe availability and per-sample 80% probe coverage were not reverified from raw beta files in this script.",
        ],
    }
    manifest_path = ROOT / "registry" / "external_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["analysis_registry"] = registry
    write_json(manifest_path, manifest)
    print(summary.to_string(index=False))
    print(external[external["cohort"].isin(["colonomics", "gse119526"])][["cohort", "gene", "n_pairs", "mean_delta_beta", "bca95_low", "bca95_high", "paired_t_q_BH10"]].round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="store_true", help="Freeze required public inputs from the historical external workspace.")
    args = parser.parse_args()
    if args.snapshot:
        snapshot_inputs()
    run_analysis()


if __name__ == "__main__":
    main()
