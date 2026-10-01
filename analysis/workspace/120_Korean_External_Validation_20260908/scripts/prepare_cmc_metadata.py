#!/usr/bin/env python3
"""Prepare public CMCBSN RNA/CMS metadata for Korean external validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "cmc_rna"
DERIVED = ROOT / "data" / "derived"
REVIEW = ROOT / "review" / "cmc_metadata"


ZENODO_RECORD = REVIEW / "zenodo_8333650.json"
CLINICAL_XLSX = RAW / "Supplementary_table1.xlsx"
CMS_XLSX = RAW / "Supplementary_table5.xlsx"
MATRIXES = {
    "381": RAW / "CMCBSN_log2cpm_381.txt",
    "374": RAW / "CMCBSN_log2cpm_374.txt",
    "342": RAW / "CMCBSN_log2cpm_342.txt",
}
ARTICLE_XML = RAW / "PMC11004400.xml"
ARTICLE_TXT = RAW / "PMC11004400.txt"


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def matrix_samples(path: Path) -> list[str]:
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
    if len(header) < 2:
        raise ValueError(f"{path} has no sample columns")
    return header[1:]


def normalize_sample_id(value: object) -> str:
    return str(value).strip().replace(".", "-")


def tissue_label(nt: object) -> str:
    if nt == "T":
        return "tumor"
    if nt == "N":
        return "adjacent_normal"
    return "unknown"


def load_zenodo_files() -> dict[str, dict[str, object]]:
    record = json.loads(ZENODO_RECORD.read_text())
    return {f["key"]: f for f in record["files"]}


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)

    zenodo_files = load_zenodo_files()
    headers = {name: matrix_samples(path) for name, path in MATRIXES.items()}
    header_sets = {name: set(samples) for name, samples in headers.items()}

    for name, samples in headers.items():
        (REVIEW / f"CMCBSN_log2cpm_{name}.samples.txt").write_text(
            "\n".join(samples) + "\n"
        )

    clinical = pd.read_excel(CLINICAL_XLSX, sheet_name="clinical")
    cms = pd.read_excel(CMS_XLSX, sheet_name="information")

    clinical["sample_id"] = clinical["Unnamed: 0"].map(normalize_sample_id)
    clinical["rna_sample_id"] = clinical["sample_id"]
    clinical["patient_id"] = clinical["Patient_ID"]
    clinical["sex"] = clinical["Sex"]
    clinical["site"] = clinical["Location"]
    clinical["tissue"] = clinical["NT"].map(tissue_label)
    clinical["cohort"] = "CMCBSN"
    clinical["platform"] = "RNA-seq"
    clinical["source_file"] = "Zenodo 8333650 Supplementary table1.xlsx"
    clinical["source_url"] = zenodo_files["Supplementary table1.xlsx"]["links"]["self"]
    clinical["sample_in_log2cpm_381"] = clinical["sample_id"].isin(header_sets["381"])
    clinical["sample_in_log2cpm_374"] = clinical["sample_id"].isin(header_sets["374"])
    clinical["sample_in_log2cpm_342"] = clinical["sample_id"].isin(header_sets["342"])
    clinical["rna_filter_status"] = "in_381_only"
    clinical.loc[clinical["sample_in_log2cpm_374"], "rna_filter_status"] = "passed_mapping_qc_374"
    clinical.loc[clinical["sample_in_log2cpm_342"], "rna_filter_status"] = "passed_mixed_tissue_filter_342"

    cms["sample_id"] = cms["Unnamed: 0"].map(normalize_sample_id)
    cms_source = cms[["sample_id", "CMS"]].copy()
    cms_source = cms_source.rename(columns={"CMS": "CMS_source"})
    cms_source["CMS_confidence"] = pd.NA
    cms_source["CMS_label_status"] = cms_source["CMS_source"].map(
        lambda value: "assigned" if pd.notna(value) else "missing_in_source_table"
    )
    cms_source["CMS_source_file"] = "Zenodo 8333650 Supplementary table5.xlsx"
    cms_source["CMS_source_url"] = zenodo_files["Supplementary table5.xlsx"]["links"][
        "self"
    ]

    merged = clinical.merge(cms_source, on="sample_id", how="left", validate="1:1")
    patient_tissues = (
        merged.groupby("Patient_ID")["tissue"].agg(lambda x: sorted(set(x))).to_dict()
    )
    merged["rna_tumor_normal_pair_available"] = merged["Patient_ID"].map(
        lambda pid: patient_tissues[pid] == ["adjacent_normal", "tumor"]
    )
    merged["methylation_sample_id_candidate"] = merged["sample_id"]
    merged["methylation_link_status"] = (
        "candidate_same_assay_id_unverified_until_KAP240422_manifest_join"
    )
    merged["methylation_accession_note"] = (
        "Public methylation route reported separately as KAP240422/KSE101261; "
        "this script does not read beta matrices."
    )
    merged["stromal_score"] = pd.NA
    merged["stromal_score_status"] = "not_found_in_public_supplement_tables_1_to_5"
    merged["CIMP_source"] = pd.NA
    merged["CIMP_source_status"] = "not_found_in_CMCBSN_RNA_public_metadata"

    out_cols = [
        "cohort",
        "patient_id",
        "sample_id",
        "rna_sample_id",
        "tissue",
        "NT",
        "sex",
        "age",
        "site",
        "Stage",
        "MSI",
        "CMS_source",
        "CMS_confidence",
        "stromal_score",
        "sample_in_log2cpm_381",
        "sample_in_log2cpm_374",
        "sample_in_log2cpm_342",
        "rna_filter_status",
        "rna_tumor_normal_pair_available",
        "methylation_sample_id_candidate",
        "methylation_link_status",
        "platform",
        "source_file",
        "source_url",
        "CMS_source_file",
        "CMS_source_url",
        "stromal_score_status",
        "CIMP_source",
        "CIMP_source_status",
        "methylation_accession_note",
    ]
    merged[out_cols].to_csv(DERIVED / "CMCBSN_rna_samples.tsv", sep="\t", index=False)

    cms_labels = merged.loc[merged["tissue"] == "tumor", [
        "cohort",
        "patient_id",
        "sample_id",
        "rna_sample_id",
        "CMS_source",
        "CMS_confidence",
        "CMS_label_status",
        "sample_in_log2cpm_381",
        "sample_in_log2cpm_374",
        "sample_in_log2cpm_342",
        "rna_filter_status",
        "methylation_sample_id_candidate",
        "methylation_link_status",
        "CMS_source_file",
        "CMS_source_url",
    ]].copy()
    cms_labels["CMS_label_status"] = cms_labels["CMS_label_status"].fillna(
        "absent_from_public_CMS_source_table"
    )
    cms_labels.to_csv(REVIEW / "CMCBSN_cms_source_labels.tsv", sep="\t", index=False)
    cms.assign(
        sample_id=cms["sample_id"],
        CMS_label_status=cms["CMS"].map(
            lambda value: "assigned" if pd.notna(value) else "missing_in_source_table"
        ),
        methylation_sample_id_candidate=cms["sample_id"],
        methylation_link_status="candidate_same_assay_id_unverified_until_KAP240422_manifest_join",
    ).to_csv(REVIEW / "CMCBSN_public_table5_cms_rows.tsv", sep="\t", index=False)

    linkage = []
    for size in ["381", "374", "342"]:
        linkage.append(
            {
                "matrix": f"CMCBSN_log2cpm_{size}.txt",
                "sample_columns": len(headers[size]),
                "clinical_overlap": int(merged["sample_id"].isin(header_sets[size]).sum()),
                "clinical_missing_from_matrix": int(
                    (~merged["sample_id"].isin(header_sets[size])).sum()
                ),
            }
        )
    pd.DataFrame(linkage).to_csv(REVIEW / "sample_linkage_audit.tsv", sep="\t", index=False)

    exclusions = []
    for before, after, reason in [
        ("381", "374", "low_quality_mapping_reads"),
        ("374", "342", "kmeans_mixed_normal_tumor"),
        ("381", "342", "all_expression_filtering"),
    ]:
        for sample_id in sorted(header_sets[before] - header_sets[after]):
            exclusions.append(
                {
                    "from_matrix": before,
                    "to_matrix": after,
                    "sample_id": sample_id,
                    "published_reason": reason,
                }
            )
    pd.DataFrame(exclusions).to_csv(REVIEW / "rna_expression_exclusions.tsv", sep="\t", index=False)

    cms_counts = (
        merged.loc[merged["tissue"] == "tumor", "CMS_source"]
        .fillna("missing_or_unclassified")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    tumor_342 = merged[(merged["tissue"] == "tumor") & merged["sample_in_log2cpm_342"]]
    cms_public_rows = int(cms["sample_id"].nunique())
    cms_public_assigned = int(cms["CMS"].notna().sum())
    cms_public_missing = int(cms["CMS"].isna().sum())
    cms_rows_in_342 = int(cms["sample_id"].isin(header_sets["342"]).sum())
    tumor_342_not_in_cms = sorted(set(tumor_342["sample_id"]) - set(cms["sample_id"]))
    cms_not_in_tumor_342 = sorted(set(cms["sample_id"]) - set(tumor_342["sample_id"]))
    tumor_universe = merged[merged["tissue"] == "tumor"]
    tumor_not_in_public_cms = sorted(set(tumor_universe["sample_id"]) - set(cms["sample_id"]))
    public_cms_na_samples = sorted(cms.loc[cms["CMS"].isna(), "sample_id"])
    article_table2_inferred_counts = {"CMS1": 20, "CMS2": 43, "CMS3": 40, "CMS4": 57}
    source_files = []
    for key, path in [
        ("Supplementary table1.xlsx", CLINICAL_XLSX),
        ("Supplementary table2.xlsx", RAW / "Supplementary_table2.xlsx"),
        ("Supplementary table3.xlsx", RAW / "Supplementary_table3.xlsx"),
        ("Supplementary table4.xlsx", RAW / "Supplementary_table4.xlsx"),
        ("Supplementary table5.xlsx", CMS_XLSX),
        ("CMCBSN_log2cpm_381.txt", MATRIXES["381"]),
        ("CMCBSN_log2cpm_374.txt", MATRIXES["374"]),
        ("CMCBSN_log2cpm_342.txt", MATRIXES["342"]),
        ("PMC11004400.xml", ARTICLE_XML),
        ("PMC11004400.txt", ARTICLE_TXT),
    ]:
        observed_md5 = file_md5(path)
        has_source_checksum = key in zenodo_files
        expected_md5 = (
            zenodo_files[key]["checksum"].split(":", 1)[1]
            if has_source_checksum
            else observed_md5
        )
        source_files.append(
            {
                "source_key": key,
                "local_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "md5": observed_md5,
                "source_expected_md5": expected_md5,
                "checksum_reference": "Zenodo API" if has_source_checksum else "local staged copy",
                "md5_ok": observed_md5 == expected_md5,
                "source_url": zenodo_files[key]["links"]["self"]
                if has_source_checksum
                else "https://pmc.ncbi.nlm.nih.gov/articles/PMC11004400/",
            }
        )
    pd.DataFrame(source_files).to_csv(REVIEW / "source_file_manifest.tsv", sep="\t", index=False)
    (REVIEW / "download_manifest.json").write_text(
        json.dumps(source_files, indent=2, ensure_ascii=False) + "\n"
    )

    status = {
        "cohort": "CMCBSN",
        "metadata_universe": {
            "clinical_rows": int(len(merged)),
            "patients": int(merged["patient_id"].nunique()),
            "tumor_samples": int((merged["tissue"] == "tumor").sum()),
            "adjacent_normal_samples": int((merged["tissue"] == "adjacent_normal").sum()),
            "clinical_rows_in_log2cpm_381": int(merged["sample_in_log2cpm_381"].sum()),
            "clinical_rows_in_log2cpm_374": int(merged["sample_in_log2cpm_374"].sum()),
            "clinical_rows_in_log2cpm_342": int(merged["sample_in_log2cpm_342"].sum()),
        },
        "cms": {
            "source_labels_available": True,
            "source_label_scope": "tumor samples only",
            "cms_confidence_available": False,
            "tumor_cms_counts": cms_counts,
            "public_cms_source_rows": cms_public_rows,
            "public_cms_assigned_rows": cms_public_assigned,
            "public_cms_missing_rows": cms_public_missing,
            "public_cms_rows_in_filtered_342_tumors": cms_rows_in_342,
            "tumor_samples_absent_from_public_cms_table": len(tumor_not_in_public_cms),
            "article_table2_inferred_counts": article_table2_inferred_counts,
            "cms_method_from_article": "CMScaller on RNA-seq data",
        },
        "expression_matrix_interpretation": {
            "381": "complete RNA-seq clinical sample universe; 214 patients, 212 tumors, 169 normals",
            "374": "after excluding 7 low-quality mapping-read samples",
            "342": "after additionally excluding 32 samples with k-means normal/tumor mixing; use for filtered tumor-normal expression analyses",
        },
        "methylation_linkage": {
            "candidate_key": "same normalized assay sample ID",
            "status": "unverified until KAP240422/KSE101261 methylation manifest is joined",
            "do_not_treat_as_pairing": True,
        },
        "stromal_or_estimate_score": {
            "available_in_public_metadata": False,
            "status": "no sample-level stromal/ESTIMATE score found in Zenodo tables 1-5 or PMC11004400 text",
        },
        "cimp": {
            "available_in_public_rna_metadata": False,
            "status": "CIMP labels not found in CMCBSN RNA public metadata",
        },
        "source_files": source_files,
    }
    discrepancy = {
        "paper_reported_cms_or_subtyping_tumors": 187,
        "public_filtered_342_tumors_from_matrix_and_clinical_join": int(len(tumor_342)),
        "public_zenodo_supplementary_table5_information_rows": cms_public_rows,
        "public_zenodo_supplementary_table5_assigned_cms_rows": cms_public_assigned,
        "public_zenodo_supplementary_table5_missing_cms_rows": cms_public_missing,
        "public_table5_rows_in_filtered_342_tumors": cms_rows_in_342,
        "complete_tumor_universe_from_381": int(len(tumor_universe)),
        "complete_tumor_universe_absent_from_public_table5": len(tumor_not_in_public_cms),
        "complete_tumor_universe_absent_from_public_table5_ids": tumor_not_in_public_cms,
        "public_table5_missing_cms_ids": public_cms_na_samples,
        "filtered_342_tumors_absent_from_public_table5": tumor_342_not_in_cms,
        "public_table5_rows_absent_from_filtered_342_tumors": cms_not_in_tumor_342,
        "article_table2_inferred_counts_from_printed_categories": article_table2_inferred_counts,
        "article_table2_inferred_total": sum(article_table2_inferred_counts.values()),
        "interpretation": (
            "Public files are internally consistent at 185 filtered tumor samples: "
            "the log2cpm_342 matrix contains 185 tumor columns after joining to clinical metadata, "
            "and Zenodo Supplementary table5 information has the same 185 tumor sample IDs. "
            "Only 159 of those 185 rows have CMS1-4 labels; 26 rows are public CMS NA. "
            "The article reports N=187 for CMScaller/NMF tumor subtyping, while Figure 2E reports "
            "Tumor-weak N=30 and remaining tumor N=155, matching the public 185 tumor set. "
            "Article Table 2 printed category counts imply 160 classified samples "
            "(CMS1=20, CMS2=43, CMS3=40, CMS4=57), which also differs from the public "
            "source-label table counts (20, 41, 41, 57; total 159). "
            "Treat 187 as an unresolved article/public-data discrepancy unless the journal-only "
            "supplement or authors provide the missing two samples."
        ),
    }
    (REVIEW / "cms_label_discrepancy_audit.json").write_text(
        json.dumps(discrepancy, indent=2, ensure_ascii=False) + "\n"
    )
    (REVIEW / "cmc_rna_metadata_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n"
    )

    report = [
        "# CMCBSN RNA/CMS metadata status",
        "",
        "Scope: public RNA metadata and CMS labels only. No methylation beta matrix or panel outcome was read.",
        "",
        f"- Clinical RNA universe: {len(merged)} samples from {merged['patient_id'].nunique()} patients; "
        f"{(merged['tissue'] == 'tumor').sum()} tumors and {(merged['tissue'] == 'adjacent_normal').sum()} adjacent normals.",
        f"- Matrix membership: 381 complete, 374 after 7 low-quality mapping-read exclusions, 342 after 32 additional k-means mixed-tissue exclusions.",
        f"- CMS labels: sample-level public labels found for tumor samples in Zenodo Supplementary table5; counts {cms_counts}.",
        f"- CMS source-label table: {cms_public_rows} tumor rows, {cms_public_assigned} assigned CMS1-4 labels, {cms_public_missing} rows with CMS missing.",
        f"- 187 discrepancy: article reports N = 187 for CMScaller/NMF tumor subtyping, but public log2cpm_342 plus Supplementary table5 resolve to {len(tumor_342)} matching tumor IDs; no two extra source labels are present in the public files.",
        "- Article Table 2 discrepancy: printed clinical-category counts imply CMS1=20, CMS2=43, CMS3=40, CMS4=57 (total 160), while the public source-label file gives CMS1=20, CMS2=41, CMS3=41, CMS4=57 (total 159). Use sample-level public labels only.",
        "- CMS confidence: not present in the public table; left blank rather than reconstructed.",
        "- Stroma/ESTIMATE: no sample-level stromal or ESTIMATE score found in public Zenodo tables 1-5 or article text.",
        "- CIMP: not found in CMCBSN RNA public metadata.",
        "- RNA-to-methylation linkage: sample IDs are staged as candidate same-assay IDs only; confirmed linkage must wait for the KAP240422/KSE101261 methylation manifest.",
        "",
        "Primary sources: Zenodo record 8333650 and PMC11004400 article text. Journal media retrieval through PMC media paths was attempted and recorded, but the useful public workbooks were available through Zenodo.",
    ]
    (REVIEW / "cmc_rna_metadata_status.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
