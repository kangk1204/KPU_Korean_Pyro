#!/usr/bin/env python3
"""Prepare fixed-probe GEO methylation inputs for the public-biology extension.

This script only downloads public files, extracts the frozen 77 CpG rows, and
curates sample metadata. It deliberately does not estimate effects, draw figures,
or write manuscript text.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DER = ROOT / "data" / "derived"
REG = ROOT / "registry"


@dataclass(frozen=True)
class Resource:
    accession: str
    url: str
    raw_name: str
    role: str


SERIES = {
    "GSE48684": Resource("GSE48684", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE48nnn/GSE48684/matrix/GSE48684_series_matrix.txt.gz", "GSE48684_series_matrix.txt.gz", "series_beta"),
    "GSE42752": Resource("GSE42752", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42752/matrix/GSE42752_series_matrix.txt.gz", "GSE42752_series_matrix.txt.gz", "series_beta"),
    "GSE193535": Resource("GSE193535", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE193nnn/GSE193535/matrix/GSE193535_series_matrix.txt.gz", "GSE193535_series_matrix.txt.gz", "series_beta"),
    "GSE77718": Resource("GSE77718", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77718/matrix/GSE77718_series_matrix.txt.gz", "GSE77718_series_matrix.txt.gz", "series_beta"),
    "GSE77954": Resource("GSE77954", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77954/matrix/GSE77954_series_matrix.txt.gz", "GSE77954_series_matrix.txt.gz", "series_normalized_m_values"),
    "GSE164811": Resource("GSE164811", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164811/matrix/GSE164811_series_matrix.txt.gz", "GSE164811_series_matrix.txt.gz", "series_metadata"),
}

SUPPLEMENT = {
    "GSE164811": Resource("GSE164811", "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164811/suppl/GSE164811_BMIQ-norm-beta-matrix.txt.gz", "GSE164811_BMIQ-norm-beta-matrix.txt.gz", "supplement_beta_detection"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            for _ in iter(lambda: f.read(1024 * 1024), b""):
                pass
        return True
    except Exception:
        return False


def fixed_probe_map() -> dict[str, list[str]]:
    return json.loads((REG / "fixed_probes.json").read_text())


def all_fixed_probes() -> set[str]:
    return {probe for probes in fixed_probe_map().values() for probe in probes}


def download(resource: Resource) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / resource.raw_name
    if path.exists() and path.suffix == ".gz" and not valid_gzip(path):
        path.unlink()
    if not path.exists() or path.stat().st_size == 0:
        tmp = path.with_suffix(path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        with urllib.request.urlopen(resource.url, timeout=180) as r, tmp.open("wb") as w:
            shutil.copyfileobj(r, w)
        tmp.replace(path)
    if path.suffix == ".gz" and not valid_gzip(path):
        path.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded gzip failed integrity check: {resource.url}")
    return {
        "cohort": resource.accession,
        "url": resource.url,
        "path": str(path.relative_to(ROOT)),
        "role": resource.role,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def parse_series_matrix(path: Path, wanted: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows: dict[str, list[list[str]]] = {}
    beta_rows: list[list[str]] = []
    header: list[str] | None = None
    in_table = False
    with gzip.open(path, "rt", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for fields in reader:
            if not fields:
                continue
            if fields[0] == "!series_matrix_table_begin":
                in_table = True
                continue
            if fields[0] == "!series_matrix_table_end":
                break
            if not in_table:
                if fields[0].startswith("!Sample_"):
                    meta_rows.setdefault(fields[0].replace("!Sample_", ""), []).append([x.strip().strip('"') for x in fields[1:]])
                continue
            if header is None:
                header = [x.strip().strip('"') for x in fields]
                continue
            probe = fields[0].strip().strip('"')
            if probe in wanted:
                beta_rows.append([probe] + [x.strip().strip('"') for x in fields[1:]])
    if "geo_accession" not in meta_rows:
        raise ValueError(f"missing GEO sample accessions in {path}")
    samples = meta_rows["geo_accession"][0]
    meta = pd.DataFrame({"sample": samples})
    for key, groups in meta_rows.items():
        if key == "geo_accession":
            continue
        if key == "characteristics_ch1":
            for group in groups:
                for i, value in enumerate(group):
                    if ":" in value:
                        k, v = value.split(":", 1)
                        meta.loc[i, k.strip()] = v.strip()
            continue
        for idx, group in enumerate(groups):
            if len(group) == len(samples):
                col = key if idx == 0 else f"{key}_{idx + 1}"
                meta[col] = group
    if header is None:
        return meta, pd.DataFrame()
    beta = pd.DataFrame(beta_rows, columns=[header[0]] + samples).set_index(header[0]).apply(pd.to_numeric, errors="coerce")
    return meta, beta


def normalize_sex(value: object) -> str:
    value = str(value).strip().lower()
    if value in {"f", "female"}:
        return "F"
    if value in {"m", "male"}:
        return "M"
    return ""


def normalize_site(value: object) -> str:
    s = str(value).strip().lower()
    if not s or s in {"na", "nan", "unknown"}:
        return ""
    if any(x in s for x in ["right", "ascending", "cecum", "caecum", "hepatic", "proximal"]):
        return "Right"
    if any(x in s for x in ["left", "descending", "sigmoid", "rect", "distal", "splenic"]):
        return "Left"
    return "Other"


def numeric_exact(row: pd.Series, *fields: str) -> float:
    for field in fields:
        if field in row and pd.notna(row[field]):
            m = re.search(r"[-+]?\d+(?:\.\d+)?", str(row[field]))
            if m:
                return float(m.group())
    return np.nan


def tissue_and_patient(accession: str, row: pd.Series) -> tuple[str, str, str]:
    sample = str(row["sample"])
    if accession == "GSE48684":
        status = str(row.get("disease status", "")).strip().lower()
        tissue = {"normal-h": "H", "normal-c": "N", "adenoma": "A", "cancer": "T"}.get(status, "")
        return tissue, sample, "GSE48684 disease status; unpaired sample-as-patient"
    if accession == "GSE42752":
        tissue_field = str(row.get("tissue", "")).strip().lower()
        cancer_patient = str(row.get("cancer patient", "")).strip().lower()
        sample_name = str(row.get("sample name", "")).strip()
        if "adenocarcinoma" in tissue_field:
            tissue = "T"
        elif "normal" in tissue_field and cancer_patient == "yes":
            tissue = "N"
        elif "normal" in tissue_field and cancer_patient == "no":
            tissue = "H"
        else:
            tissue = ""
        patient = re.sub(r"p[12]$", "", sample_name, flags=re.I) if sample_name else sample
        return tissue, patient or sample, "GSE42752 tissue/cancer patient; sample name p1/p2"
    if accession == "GSE193535":
        state = str(row.get("disease state", "")).strip().lower()
        source = str(row.get("source_name_ch1", "")).strip().lower()
        title = str(row.get("title", "")).strip()
        tissue = {"tumor": "T", "normal": "N"}.get(state, "")
        if not tissue and source == "crc tissue":
            tissue = "T"
        if not tissue and source == "normal tissue":
            tissue = "N"
        patient = re.sub(r"[TN]$", "", title, flags=re.I)
        return tissue, patient or sample, "GSE193535 disease state; title numeric T/N"
    if accession == "GSE77718":
        state = str(row.get("disease state", "")).strip().lower()
        title = str(row.get("title", "")).strip()
        tissue = {"tumour": "T", "tumor": "T", "normal": "N"}.get(state, "")
        m = re.search(r"\[(\d+)[NT]\]", title)
        return tissue, (m.group(1) if m else sample), "GSE77718 disease state; bracket patient id"
    if accession == "GSE77954":
        tt = str(row.get("tissue type", "")).strip().lower()
        if tt == "adenoma":
            tissue = "A"
        elif tt == "carcinoma":
            tissue = "T"
        elif "normal colon adjacent to  carcinoma" in tt or "normal colon adjacent to carcinoma" in tt:
            tissue = "N"
        elif "normal colon adjacent to" in tt:
            tissue = "NA"
        else:
            tissue = "excluded"
        return tissue, sample, "GSE77954 tissue type; liver/metastasis excluded"
    if accession == "GSE164811":
        cms = str(row.get("cms class", "")).strip()
        return "T", sample, f"GSE164811 tumor CMS{cms} public subset"
    return "", sample, "no accession rule"


def curate_samples(accession: str, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in meta.iterrows():
        tissue, patient, rule = tissue_and_patient(accession, row)
        if accession == "GSE48684":
            age = np.nan
            sex = normalize_sex(row.get("gender", ""))
            site = normalize_site(row.get("colon region", ""))
        elif accession == "GSE42752":
            age = np.nan
            sex = ""
            site = ""
        elif accession == "GSE77954":
            age = numeric_exact(row, "age (yrs)")
            sex = normalize_sex(row.get("gender", ""))
            site = normalize_site(row.get("site", ""))
        elif accession == "GSE164811":
            age = np.nan
            sex = normalize_sex(row.get("gender", ""))
            site = normalize_site(row.get("tumor location", ""))
        else:
            age = np.nan
            sex = normalize_sex(row.get("gender", ""))
            site = normalize_site(row.get("colon subdivision", ""))
        cms = ""
        if accession == "GSE164811":
            cms_value = str(row.get("cms class", "")).strip()
            cms = f"CMS{cms_value}" if cms_value in {"1", "2", "3", "4"} else ""
        rows.append({
            "cohort": accession,
            "sample": str(row["sample"]),
            "patient": str(patient),
            "tissue": tissue,
            "pair_verified": False,
            "pair_rule": rule,
            "age": age,
            "sex": sex,
            "site": site,
            "cms": cms,
            "stage": str(row.get("Stage", row.get("tumor stage", ""))).strip(),
            "title": str(row.get("title", "")),
            "source_name_ch1": str(row.get("source_name_ch1", "")),
            "description": str(row.get("description", "")),
            "source_declared_country": str(row.get("contact_country", "")).strip(),
        })
    out = pd.DataFrame(rows)
    if accession in {"GSE193535", "GSE77718", "GSE42752"}:
        counts = out[out["tissue"].isin(["N", "T"])].groupby(["patient", "tissue"]).size().unstack(fill_value=0)
        good = counts.index[(counts.get("N", 0) == 1) & (counts.get("T", 0) == 1)]
        out["pair_verified"] = out["patient"].isin(set(good)) & out["tissue"].isin(["N", "T"])
    return out


def audit_beta(beta: pd.DataFrame, accession: str, source: str) -> tuple[pd.DataFrame, dict]:
    arr = beta.to_numpy(float)
    finite = arr[np.isfinite(arr)]
    min_v = float(finite.min()) if finite.size else np.nan
    max_v = float(finite.max()) if finite.size else np.nan
    n_low = int(np.sum(arr < 0))
    n_high = int(np.sum(arr > 1))
    decision = "as_downloaded_beta"
    out = beta.copy()
    if n_low or n_high:
        if min_v >= -0.1 and max_v <= 1.1 and source in {"series_beta", "supplement_beta_detection"}:
            out = out.mask((out < 0) | (out > 1))
            decision = "out_of_range_values_set_missing_small_normalized_beta_artifact"
        else:
            raise ValueError(f"{accession} beta range unsupported: {min_v}..{max_v} from {source}")
    return out, {"min_beta": min_v, "max_beta": max_v, "n_below_0": n_low, "n_above_1": n_high, "decision": decision}


def m_values_to_beta(m_values: pd.DataFrame) -> pd.DataFrame:
    odds = np.power(2.0, m_values.astype(float))
    return odds / (1.0 + odds)


def parse_gse164811_supplement(path: Path, wanted: set[str], meta: pd.DataFrame) -> pd.DataFrame:
    title_to_gsm = dict(zip(meta["title"].astype(str).str.split("_").str[0], meta["sample"].astype(str)))
    rows = []
    with gzip.open(path, "rt", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        sample_ids = [header[i].strip() for i in range(1, len(header), 2)]
        columns = [title_to_gsm.get(s, s) for s in sample_ids]
        for fields in reader:
            probe = fields[0]
            if probe not in wanted:
                continue
            vals = []
            for i in range(1, len(header), 2):
                b = pd.to_numeric(fields[i], errors="coerce")
                p = pd.to_numeric(fields[i + 1], errors="coerce")
                vals.append(float(b) if np.isfinite(b) and np.isfinite(p) and p < 0.01 else np.nan)
            rows.append([probe] + vals)
    return pd.DataFrame(rows, columns=["probe"] + columns).set_index("probe")


def align_beta(beta: pd.DataFrame, samples: pd.DataFrame, accession: str) -> pd.DataFrame:
    missing = sorted(set(samples["sample"]) - set(beta.columns))
    if missing:
        raise ValueError(f"{accession} beta matrix missing metadata samples: {missing[:5]}")
    return beta.loc[:, samples["sample"].tolist()]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    DER.mkdir(parents=True, exist_ok=True)
    probes = all_fixed_probes()
    manifest = []
    curation_lines = ["# GEO ingestion curation", ""]
    beta_audits = []
    for accession, resource in SERIES.items():
        print(f"Preparing {accession}", flush=True)
        rec = download(resource)
        meta, beta = parse_series_matrix(RAW / resource.raw_name, probes)
        if accession == "GSE77954":
            beta = m_values_to_beta(beta)
            beta_source = "series_normalized_m_values_plogis_log2"
        elif accession == "GSE164811":
            supp = download(SUPPLEMENT[accession])
            manifest.append(supp)
            beta = parse_gse164811_supplement(RAW / SUPPLEMENT[accession].raw_name, probes, meta)
            beta_source = "supplement_beta_detection"
        else:
            beta_source = "series_beta"
        beta, audit = audit_beta(beta, accession, beta_source)
        samples = curate_samples(accession, meta)
        beta = align_beta(beta, samples, accession)
        beta.to_csv(DER / f"{accession}_beta.tsv.gz", sep="\t", compression="gzip")
        samples.to_csv(DER / f"{accession}_samples.tsv", sep="\t", index=False)
        counts = samples["tissue"].value_counts(dropna=False).to_dict()
        pair_samples = int(samples["pair_verified"].sum())
        pair_patients = int(samples.loc[samples["pair_verified"], "patient"].nunique())
        rec.update({
            "derived_beta": f"data/derived/{accession}_beta.tsv.gz",
            "derived_samples": f"data/derived/{accession}_samples.tsv",
            "n_samples": int(len(samples)),
            "n_beta_rows": int(len(beta)),
            "n_beta_columns": int(beta.shape[1]),
            "tissue_counts": counts,
            "verified_pair_samples": pair_samples,
            "verified_pair_patients": pair_patients,
            "beta_source": beta_source,
            **audit,
        })
        manifest.append(rec)
        beta_audits.append({"cohort": accession, **audit})
        curation_lines.append(f"## {accession}")
        curation_lines.append(f"- Tissue counts: {counts}")
        curation_lines.append(f"- Verified paired patients: {pair_patients}; verified paired samples: {pair_samples}")
        curation_lines.append(f"- Patient/tissue rule: {samples['pair_rule'].iloc[0] if len(samples) else 'NA'}")
        curation_lines.append(f"- Beta source: {beta_source}; range decision: {audit['decision']}")
        if accession == "GSE77718":
            curation_lines.append("- GEO matrix contains 192 samples (96 normal, 96 tumour). Metadata yield 95 complete N/T patient pairs; the paper reports 94 paired cases, so paired status is recorded but no outcome-based sample exclusion is made here.")
        if accession == "GSE77954":
            curation_lines.append("- Adenoma-adjacent colon is coded `NA`; liver, metastasis, and normal liver samples are coded `excluded`.")
            curation_lines.append("- GEO series data_processing states normalized M-values after methylumi/lumi background correction and quantile normalization; the derived beta file applies beta = 2^M/(1+2^M). The non-normalized supplementary signal file is not used for the primary derived beta.")
        if accession == "GSE164811":
            curation_lines.append("- The 9 KB series matrix contains metadata only. Fixed probes were extracted from GSE164811_BMIQ-norm-beta-matrix.txt.gz.")
        curation_lines.append("")
    (REG / "geo_downloads.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (REG / "geo_curation.md").write_text("\n".join(curation_lines))
    pd.DataFrame(beta_audits).to_csv(REG / "geo_beta_range_audit.tsv", sep="\t", index=False)
    print(json.dumps({m["cohort"]: m.get("tissue_counts") for m in manifest if "tissue_counts" in m}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
