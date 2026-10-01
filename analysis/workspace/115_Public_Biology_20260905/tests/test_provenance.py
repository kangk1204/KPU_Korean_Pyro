from pathlib import Path
import importlib.util
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("audit_provenance", ROOT / "scripts" / "audit_provenance.py")
audit_provenance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_provenance)


def test_probe_qc_has_fixed_77_and_decoded_mask_fields():
    probes = json.loads((ROOT / "registry" / "fixed_probes.json").read_text())
    expected = {p for values in probes.values() for p in values}
    qc = pd.read_csv(ROOT / "registry" / "probe_qc.tsv", sep="\t")
    assert set(qc["probe"]) == expected
    assert len(qc) == 77
    assert qc["probe"].is_unique
    assert set(qc["fixed_gene"]) == set(probes)
    assert qc["cross_reactive"].isin([True, False]).all()
    assert qc["mask_general"].isin([True, False]).all()
    assert qc["mask_mapping"].isin([True, False]).all()
    assert qc["snp_flag"].isin([True, False]).all()
    assert qc["technical_flag"].isin([True, False]).all()
    assert qc["flag_source"].str.contains("HM450.hg19.manifest.tsv.gz").all()
    assert qc["coordinate_agreement"].eq(True).all()
    assert qc["coordinate_rule"].str.contains("CpG_beg.*0-based", regex=True).all()
    assert (pd.to_numeric(qc["illumina_pos_1based"]) == pd.to_numeric(qc["zhou_cpg_beg_0based"]) + 1).all()


def test_coordinate_agreement_requires_chromosome_and_position_system_match():
    assert audit_provenance.coordinate_agrees("chr1", "101", "1", "100") is True
    assert audit_provenance.coordinate_agrees("chr1", "102", "1", "100") is False
    assert audit_provenance.coordinate_agrees("chr2", "101", "1", "100") is False
    assert audit_provenance.coordinate_agrees("chr1", "unknown", "1", "100") == "unknown"


def test_cohort_sources_distinguish_country_from_ancestry():
    src = pd.read_csv(ROOT / "registry" / "cohort_sources.tsv", sep="\t")
    assert "KPU_87" in set(src["accession"])
    assert src["country_is_ancestry"].eq(False).all()
    kpu = src.loc[src["accession"].eq("KPU_87")].iloc[0]
    assert "Korean" in kpu["reported_population"]
    assert "Colonomics/GSE131013/GSE44076/GSE166427" in set(src["accession"])
    gse77718 = src.loc[src["accession"].eq("GSE77718")].iloc[0]
    assert gse77718["country_or_source"] == "New Zealand"
    assert src.loc[src["accession"].eq("GSE48684"), "country_or_source"].iloc[0] == "USA"
    gse42752 = src.loc[src["accession"].eq("GSE42752")].iloc[0]
    assert gse42752["country_or_source"] == "Russia"
    assert "metadata-paired p1/p2" in gse42752["analysis_role"]
    assert src.loc[src["accession"].eq("GSE77954"), "country_or_source"].iloc[0] == "USA"
    assert src["source_quote"].notna().all()
    assert src["source_basis"].notna().all()


def test_endpoint_audits_block_unverified_cimp_and_bound_colonomics_endpoint():
    cimp = (ROOT / "registry" / "cimp_endpoint_audit.md").read_text()
    recurrence = pd.read_csv(ROOT / "registry" / "recurrence_endpoint_audit.tsv", sep="\t")
    audit_md = (ROOT / "registry" / "provenance_audit.md").read_text()

    assert "Do not analyze CIMP" in cimp
    assert "Prohibited" in cimp
    assert "Endpoint provenance" in audit_md

    event_free = recurrence.loc[recurrence["field"].eq("event_free")].iloc[0]
    assert event_free["status"] == "definition_verified"
    assert "recurrence of colon cancer" in event_free["definition"]
    assert event_free["time_origin"] == "surgery"
    assert "not explicitly stated" in event_free["time_unit"]
    assert "not explicitly defined" in event_free["censoring"]
    assert "not supported" in event_free["death_included"]
    assert "do not call it DFS including death" in event_free["decision"]

    time_free = recurrence.loc[recurrence["field"].eq("time_free")].iloc[0]
    assert time_free["status"] == "definition_verified_partial"
    assert time_free["time_origin"] == "surgery"
    assert "censoring definition" in time_free["decision"]


def test_download_manifest_contains_hashed_primary_annotation_sources():
    manifest = json.loads((ROOT / "registry" / "provenance_downloads.json").read_text())
    records = manifest["downloads"]
    assert len(records) >= 2
    for rec in records:
        assert rec["bytes"] > 0
        assert len(rec["sha256"]) == 64
    urls = " ".join(rec.get("url", "") for rec in records)
    assert "InfiniumAnnotation" in urls
    assert "HM450.hg19.manifest.tsv.gz" in urls
