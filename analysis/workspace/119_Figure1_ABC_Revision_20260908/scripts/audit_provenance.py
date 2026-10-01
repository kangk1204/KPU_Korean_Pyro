#!/usr/bin/env python3
"""Probe, cohort, CIMP, and endpoint provenance audit for public115."""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, fixed_probes, sha256, write_json

RAW = ROOT / "data" / "raw"
REG = ROOT / "registry"

ZHOU_BASE = "https://zhouserver.research.chop.edu/InfiniumAnnotation/current/HM450"
ZHOU_FILES = {
    "manifest": f"{ZHOU_BASE}/HM450.hg19.manifest.tsv.gz",
}

COHORTS = [
    {
        "accession": "KPU_87",
        "label": "Korean cohort",
        "country_or_source": "Korea",
        "country_type": "author-confirmed cohort context",
        "reported_population": "Korean, author-confirmed",
        "analysis_role": "local paired pyrosequencing",
        "source_quote": "User/author confirmed that the local cohort is Korean.",
        "source_basis": "author confirmation",
    },
    {
        "accession": "Colonomics/GSE131013/GSE44076/GSE166427",
        "label": "Colonomics",
        "country_or_source": "Spain",
        "country_type": "recruitment/source cohort",
        "reported_population": "not reported as ancestry in GEO",
        "analysis_role": "public methylation/expression/subtype context; one cohort",
        "source_quote": "Colonomics clinical table and GEO metadata identify the Spanish Colonomics cohort; ancestry is not reported.",
        "source_basis": "Colonomics baseline clinical/GEO metadata",
    },
    {
        "accession": "GSE48684",
        "label": "Luo et al. colorectal 450K",
        "country_or_source": "USA",
        "country_type": "GEO contact/source context, not proven ancestry",
        "reported_population": "not reported as ancestry in GEO",
        "analysis_role": "healthy/adjacent/adenoma/cancer",
        "source_quote": "GEO `Series_contact_country` and all `Sample_contact_country` entries are USA; article record places multiple authors at Fred Hutchinson Cancer Research Center, Seattle, Washington.",
        "source_basis": "GSE48684 series matrix plus Europe PMC record for PMID 24793120",
    },
    {
        "accession": "GSE42752",
        "label": "Naumov et al. colorectal 450K",
        "country_or_source": "Russia",
        "country_type": "GEO contact country/source laboratory",
        "reported_population": "not reported as ancestry in GEO",
        "analysis_role": "healthy/adjacent/cancer; metadata-paired p1/p2 tumor-normal samples where sample-name pairs verify",
        "source_quote": "GEO `Series_contact_country` and all `Sample_contact_country` entries are Russia; PubMed PMID 23867710 lists Naumov affiliation as Moscow, Russia.",
        "source_basis": "GSE42752 series matrix plus PubMed PMID 23867710 search record",
    },
    {
        "accession": "GSE193535",
        "label": "Malaysia CRC EWAS",
        "country_or_source": "Malaysia",
        "country_type": "GEO contact/source context",
        "reported_population": "recruited in Malaysia; ethnicity not inferred",
        "analysis_role": "paired tumor/normal replication",
        "source_quote": "GEO `Series_contact_country` and all `Sample_contact_country` entries are Malaysia.",
        "source_basis": "GSE193535 series matrix",
    },
    {
        "accession": "GSE77718",
        "label": "McInnes et al. CRC methylome",
        "country_or_source": "New Zealand",
        "country_type": "GEO contact country and article author country",
        "reported_population": "recruitment/source country recorded as New Zealand; ethnicity not inferred",
        "analysis_role": "paired tumor/normal replication after ID audit",
        "source_quote": "GEO `Series_contact_country` and all `Sample_contact_country` entries are New Zealand; article affiliations are University of Otago, Dunedin, New Zealand.",
        "source_basis": "GSE77718 series matrix plus BMC Cancer 2017 XML",
    },
    {
        "accession": "GSE77954",
        "label": "adenoma/CRC colon methylation",
        "country_or_source": "USA",
        "country_type": "GEO specimen collection/source context",
        "reported_population": "not reported as ancestry in GEO",
        "analysis_role": "adenoma/primary CRC/adjacent colon only",
        "source_quote": "GEO overall design states tissues were collected by the Department of Pathology at the University of Virginia; `Series_contact_country` and all `Sample_contact_country` entries are USA.",
        "source_basis": "GSE77954 series matrix; GEO-linked PMID is 27270421, not PMC4963574",
    },
    {
        "accession": "GSE164811",
        "label": "MATCH methylation subset",
        "country_or_source": "Netherlands",
        "country_type": "recruitment/source cohort",
        "reported_population": "not reported as ancestry in GEO",
        "analysis_role": "CMS2/CMS3 subtype only",
        "source_quote": "GEO overall design states patients were selected from the MATCH study in seven hospitals in the Rotterdam region, the Netherlands.",
        "source_basis": "GSE164811 series matrix",
    },
]


def download(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with urllib.request.urlopen(url, timeout=90) as r, path.open("wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    return {"url": url, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def export_r_annotation(path: Path) -> dict:
    r_script = RAW / "export_probe_annotations.R"
    out = {"path": str(path.relative_to(ROOT)), "status": "not_run"}
    if not path.exists():
        r_script.write_text(
            "suppressPackageStartupMessages(library(IlluminaHumanMethylation450kanno.ilmn12.hg19))\n"
            "data(list='IlluminaHumanMethylation450kanno.ilmn12.hg19')\n"
            "ann <- getAnnotation(IlluminaHumanMethylation450kanno.ilmn12.hg19)\n"
            "keep <- intersect(c('Name','chr','pos','UCSC_RefGene_Name','Probe_rs','CpG_rs','SBE_rs','Probe_maf','CpG_maf','SBE_maf'), colnames(ann))\n"
            f"write.table(as.data.frame(ann[, keep]), file='{path}', sep='\\t', quote=FALSE, row.names=FALSE)\n"
        )
        subprocess.check_call(["Rscript", str(r_script)], cwd=ROOT)
    out.update(status="available", bytes=path.stat().st_size, sha256=sha256(path))
    return out




def normalize_chr(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "nan", "none", "unknown"}:
        return None
    if text.lower().startswith("chr"):
        text = text[3:]
    return text.upper()


def parse_int(value: object) -> int | None:
    if value is None:
        return None
    num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(num):
        return None
    return int(num)


def coordinate_agrees(illumina_chr: object, illumina_pos_1based: object, zhou_chr: object, zhou_cpg_beg_0based: object) -> bool | str:
    """Compare Illumina hg19 1-based position with Zhou CpG_beg 0-based coordinate."""
    ichr = normalize_chr(illumina_chr)
    zchr = normalize_chr(zhou_chr)
    ipos = parse_int(illumina_pos_1based)
    zbeg = parse_int(zhou_cpg_beg_0based)
    if ichr is None or zchr is None or ipos is None or zbeg is None:
        return "unknown"
    return ichr == zchr and ipos == zbeg + 1

def bool_from_snp_row(row: dict) -> bool:
    if not row:
        return False
    rs_cols = [k for k in row if k.lower() in {"rs", "probe_rs", "cpg_rs", "sbe_rs"}]
    for col in rs_cols:
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in {"na", "nan", "none"}:
            return True
    maf_cols = [k for k in row if "maf" in k.lower()]
    for col in maf_cols:
        val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.notna(val) and val > 0:
            return True
    return False


def bool_from_mask_value(value: object) -> bool | str:
    value = str(value).strip().upper()
    if value in {"TRUE", "T", "1"}:
        return True
    if value in {"FALSE", "F", "0"}:
        return False
    return "unknown"


def build_probe_qc(downloads: list[dict]) -> pd.DataFrame:
    r_ann_path = RAW / "probe_annotations_450k_r.tsv"
    downloads.append(export_r_annotation(r_ann_path))
    for key, url in ZHOU_FILES.items():
        downloads.append(download(url, RAW / f"zhou_{Path(url).name}"))
    r_ann = pd.read_csv(r_ann_path, sep="\t", dtype=str).set_index("Name", drop=False)
    manifest = pd.read_csv(RAW / "zhou_HM450.hg19.manifest.tsv.gz", sep="\t", dtype=str)
    manifest = manifest.rename(columns={"probeID": "probe"})
    manifest = manifest.set_index("probe", drop=False)
    rows = []
    for gene, probes in fixed_probes().items():
        for probe in probes:
            rr = r_ann.loc[probe].to_dict() if probe in r_ann.index else {}
            mm = manifest.loc[probe].to_dict() if probe in manifest.index else {}
            illumina_snp = bool_from_snp_row(rr)
            zhou_snp = any(bool_from_mask_value(mm.get(col)) is True for col in ["MASK_snp5_common", "MASK_snp5_GMAF1p", "MASK_extBase"])
            mapping_flag = bool_from_mask_value(mm.get("MASK_mapping"))
            general_mask = bool_from_mask_value(mm.get("MASK_general"))
            technical = general_mask if general_mask != "unknown" else (True if illumina_snp or zhou_snp or mapping_flag is True else "unknown")
            illumina_chr = rr.get("chr", mm.get("CpG_chrm", "unknown"))
            illumina_pos = rr.get("pos", "unknown")
            zhou_chr = mm.get("CpG_chrm", "unknown")
            zhou_cpg_beg = mm.get("CpG_beg", "unknown")
            zhou_pos_1based = parse_int(zhou_cpg_beg)
            zhou_pos_1based = zhou_pos_1based + 1 if zhou_pos_1based is not None else "unknown"
            rows.append({
                "probe": probe,
                "fixed_gene": gene,
                "annotated_gene": rr.get("UCSC_RefGene_Name", mm.get("gene_HGNC", mm.get("gene", "unknown"))),
                "chr": illumina_chr,
                "pos": illumina_pos,
                "illumina_hg19_chr": illumina_chr,
                "illumina_pos_1based": illumina_pos,
                "zhou_hg19_chr": zhou_chr,
                "zhou_cpg_beg_0based": zhou_cpg_beg,
                "zhou_pos_1based": zhou_pos_1based,
                "coordinate_rule": "Illumina pos is 1-based; Zhou CpG_beg is 0-based; agreement requires normalized chromosome equality and Illumina pos = CpG_beg + 1",
                "coordinate_agreement": coordinate_agrees(rr.get("chr"), rr.get("pos"), mm.get("CpG_chrm"), mm.get("CpG_beg")),
                "platform450k": True,
                "platform_epic": "unknown",
                "cross_reactive": mapping_flag,
                "mask_mapping": mapping_flag,
                "mask_snp5_common": bool_from_mask_value(mm.get("MASK_snp5_common")),
                "mask_snp5_gmaf1p": bool_from_mask_value(mm.get("MASK_snp5_GMAF1p")),
                "mask_extbase": bool_from_mask_value(mm.get("MASK_extBase")),
                "mask_general": general_mask,
                "snp_flag": bool(illumina_snp or zhou_snp),
                "technical_flag": technical,
                "flag_source": "IlluminaHumanMethylation450kanno.ilmn12.hg19 plus Zhou-lab InfiniumAnnotation current HM450.hg19.manifest.tsv.gz; MASK_mapping recorded separately, MASK_general used as the fixed technical mask",
            })
    out = pd.DataFrame(rows)
    out.to_csv(REG / "probe_qc.tsv", sep="\t", index=False)
    return out


def write_cohort_sources() -> pd.DataFrame:
    out = pd.DataFrame(COHORTS)
    out["country_is_ancestry"] = False
    out.to_csv(REG / "cohort_sources.tsv", sep="\t", index=False)
    return out


def write_cimp_endpoint_audits() -> None:
    cimp_lines = [
        "# CIMP Endpoint Audit",
        "",
        "| Resource | Checked item | Status | Decision |",
        "|---|---|---|---|",
        "| GSE77718 | Sample-mapped standard CIMP labels | Not verified in GEO series matrix during provenance audit | Do not analyze CIMP |",
        "| TCGA | Published/supportive CIMP labels | Supportive historical resource, not an untouched public validation cohort for this fixed panel | Do not mix into primary public replication |",
        "| Fixed 10-gene panel | Derive CIMP internally | Not a standard CIMP definition | Prohibited |",
        "",
        "Heatmap labels, unstored supplementary labels, or the ten-gene panel itself were not used to infer CIMP.",
    ]
    (REG / "cimp_endpoint_audit.md").write_text("\n".join(cimp_lines) + "\n")
    endpoint = pd.DataFrame([
        {
            "resource": "Colonomics Sci Data 2022 / UB-DD clinical data",
            "field": "event_free",
            "status": "definition_verified",
            "definition": "binary variable indicating recurrence of colon cancer, 1 yes and 0 no",
            "time_origin": "surgery",
            "time_unit": "not explicitly stated in the retrieved definition",
            "censoring": "not explicitly defined; event_free=0 identifies no recurrence, but last-follow-up/censoring rule is not described in the retrieved text",
            "death_included": "not supported; event_free is recurrence-specific, while event_global/time_global are separate survival fields",
            "source_evidence": "Scientific Data 2022 Data Records defines event_free as recurrence of colon cancer (1/0) and time_free as time from surgery to recurrence; Dataverse Readme lists CLX_ClinicalData as patient clinical annotation.",
            "decision": "optional univariable Cox can only be described as recurrence-free/recurrence endpoint with surgery origin; do not call it DFS including death, and state units/censoring limitation unless further dictionary is found",
        },
        {
            "resource": "Colonomics Sci Data 2022 / UB-DD clinical data",
            "field": "time_free",
            "status": "definition_verified_partial",
            "definition": "continuous variable indicating time from surgery to recurrence of colon cancer",
            "time_origin": "surgery",
            "time_unit": "not explicitly stated in the retrieved definition",
            "censoring": "not explicitly defined for non-recurrence rows",
            "death_included": "not supported",
            "source_evidence": "Scientific Data 2022 Data Records variable dictionary; UB-DD Dataverse Readme and CLX_ClinicalData.tab downloaded during audit",
            "decision": "usable only with transparent limitation on time unit/censoring definition",
        },
        {
            "resource": "GSE119526",
            "field": "binary recurrent status",
            "status": "binary status only in baseline snapshot",
            "definition": "not time-to-event",
            "time_origin": "not available",
            "time_unit": "not available",
            "censoring": "not available",
            "death_included": "not assessable",
            "source_evidence": "baseline snapshot",
            "decision": "no survival/time-to-event analysis",
        },
        {
            "resource": "Local Korean 87",
            "field": "recurrence",
            "status": "small event count and markers not recurrence-selected",
            "definition": "local recurrence coding outside this public provenance audit",
            "time_origin": "not reassessed here",
            "time_unit": "not reassessed here",
            "censoring": "not reassessed here",
            "death_included": "not reassessed here",
            "source_evidence": "prior local audit context",
            "decision": "do not position recurrence ML as main finding",
        },
    ])
    endpoint.to_csv(REG / "recurrence_endpoint_audit.tsv", sep="\t", index=False)


def write_markdown(probe_qc: pd.DataFrame, downloads: list[dict]) -> None:
    flagged = int(probe_qc["technical_flag"].eq(True).sum())
    lines = [
        "# Provenance Audit",
        "",
        f"Fixed probe records: {len(probe_qc)}.",
        f"Zhou MASK_general technical-mask probes: {flagged}.",
        "",
        "Probe locations and Illumina SNP annotations were obtained from the installed IlluminaHumanMethylation450kanno.ilmn12.hg19 package. Mapping, SNP, extension-base, and general technical mask fields were decoded directly from Zhou-lab InfiniumAnnotation `HM450.hg19.manifest.tsv.gz`. `coordinate_agreement` requires normalized chromosome equality and numeric Illumina hg19 `pos` (1-based) = Zhou `CpG_beg` (0-based) + 1. `MASK_mapping` is recorded separately as the cross-reactive/mapping flag, while `MASK_general` is used as the fixed downstream technical exclusion flag.",
        "",
        "No probe was removed or retained because of a methylation effect, clinical outcome, or model performance. Any downstream exclusion sensitivity must use `probe_qc.tsv` as a fixed technical filter.",
        "",
        "Country labels in `cohort_sources.tsv` describe recruitment/source context only. The local cohort is Korean by author confirmation; public country labels are not treated as ancestry.",
        "",
        "Country/source corrections made in this audit: GSE48684 is recorded as USA based on GEO contact/sample country and Fred Hutchinson/Seattle article metadata, not Germany; GSE42752 is recorded as Russia based on GEO contact/sample country and Naumov Moscow affiliation, with no Denmark claim retained; GSE77718 is recorded as New Zealand; GSE77954 is recorded as USA based on the GEO statement that tissues were collected by the Department of Pathology at the University of Virginia. For GSE77954, the GEO-linked PMID is 27270421, whereas PMC4963574 refers to a different Wei et al. marker study and is not used as provenance for this accession.",
        "",
        "Endpoint provenance: Colonomics Scientific Data 2022 explicitly defines `event_free` as recurrence of colon cancer coded 1/0 and `time_free` as time from surgery to recurrence. This supports a recurrence-specific endpoint with surgery as time origin, but the retrieved definition does not state that death is included, does not define censoring for event-free patients, and does not explicitly state the time unit. The Dataverse Readme and clinical table were also downloaded; they list the clinical file but do not add a censoring dictionary.",
        "",
        "Downloaded source files are recorded in `provenance_downloads.json` with bytes and SHA256.",
    ]
    (REG / "provenance_audit.md").write_text("\n".join(lines) + "\n")
    write_json(REG / "provenance_downloads.json", {"downloads": downloads})


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    REG.mkdir(parents=True, exist_ok=True)
    downloads: list[dict] = []
    probe_qc = build_probe_qc(downloads)
    write_cohort_sources()
    write_cimp_endpoint_audits()
    write_markdown(probe_qc, downloads)
    print(json.dumps({"probe_rows": len(probe_qc), "downloads": len(downloads)}, indent=2))


if __name__ == "__main__":
    main()
