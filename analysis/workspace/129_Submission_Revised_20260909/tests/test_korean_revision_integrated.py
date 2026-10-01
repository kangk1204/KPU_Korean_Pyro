from pathlib import Path
import json
import sys

import pandas as pd
import pathlib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from add_qc_coordination_sensitivity import read_fixed, sha256_file  # noqa: E402



def _require_retained(path, what):
    """Skip when a retained input/evidence file is absent (clean archive checkouts; see reproducibility/INPUT_REQUIREMENTS.tsv)."""
    if not pathlib.Path(path).exists():
        pytest.skip(f"{what} not present in this checkout: {path}")

def test_pairing_evidence_counts_are_source_label_row_counts():
    evidence = json.loads((ROOT / "registry" / "pairing_evidence.json").read_text())
    expected = {"CMCBSN": (103, 206, 127, "PRJKA2086326", "KAP240422"), "SNUH": (142, 284, 153, "PRJKA2086323", "KAP240419"), "ASAN": (128, 256, 44, "PRJKA2086325", "KAP240421")}
    for cohort, (pairs, true_rows, false_rows, accession, browse) in expected.items():
        got = evidence["cohorts"][cohort]
        assert got["paired_patient_count"] == pairs
        assert got["paired_sample_count"] == true_rows
        assert got["pair_verified_true_rows"] == true_rows
        assert got["pair_verified_false_rows"] == false_rows
        assert got["registry_accession"] == accession
        assert got["browse_accession"] == browse
        assert "not genotype-confirmed" in got["pair_verification_semantics"]


def test_probe_coverage_and_qc_mask_registry_are_bounded_to_evidence():
    summary = json.loads((ROOT / "registry" / "korean_probe_coverage_summary.json").read_text())
    assert summary["fixed77"] == 77
    assert summary["epic_annotation_absent"] == 6
    assert summary["processed_present_all_three"] == 56
    assert summary["processed_absent_all_three"] == 21
    assert summary["epic_annotated_but_absent_processed"] == 15
    assert set(summary["technical_mask_ids"]) == {"cg20680720", "cg04481096"}
    assert "does not assign an upstream processing cause" in summary["interpretation_limit"]

    zhou = pd.read_csv(ROOT / "registry" / "probe_qc_original_zhou_flagged.tsv", sep="\t")
    assert set(zhou["probeID"]) == {"cg20680720", "cg04481096"}
    assert zhou["MASK_general"].astype(str).str.upper().eq("TRUE").all()


def test_korean_exclude2_sensitivity_preserves_core_effect_and_coordination_claims():
    effects = pd.read_csv(ROOT / "results" / "qc_sensitivity" / "korean_exclude2_effect_summary.tsv", sep="\t")
    assert set(effects["cohort"]) == {"CMCBSN", "SNUH", "ASAN"}
    assert (effects["n_effect_rows"] == 54).all()
    assert (effects["n_positive_mean_delta"] == 54).all()
    assert (effects["n_paired_t_q77_lt_005"] == 54).all()
    assert effects["mean_delta_min_pp"].min() > 14

    coord = pd.read_csv(ROOT / "results" / "coordination_sensitivity" / "korean_exclude2_coordination54.tsv", sep="\t")
    assert (coord["n_available_cpgs"] == 54).all()
    assert (coord["n_between_gene_pairs"] == 1282).all()
    assert (coord["n_positive_between_gene_pairs"] == 1282).all()
    assert coord.set_index("cohort").loc["CMCBSN", "median_between_gene_rho_54"] == pytest.approx(0.6672615648201028)
    assert coord.set_index("cohort").loc["SNUH", "median_between_gene_rho_54"] == pytest.approx(0.574036392136482)
    assert coord.set_index("cohort").loc["ASAN", "median_between_gene_rho_54"] == pytest.approx(0.6837346639809558)
    assert coord["full_ci_or_p_recomputed"].astype(str).str.lower().eq("false").all()


def test_public_ml75_and_tumor_content_sensitivity_outputs_are_available():
    public_ml = pd.read_csv(ROOT / "results" / "qc_sensitivity" / "public_ml75_before_after_summary.tsv", sep="\t")
    assert set(public_ml["cohort"]) == {"colonomics", "gse119526"}
    assert (public_ml["sensitivity_features"] == 75).all()
    assert public_ml.set_index("cohort").loc["colonomics", "exclude_auc"] == pytest.approx(0.9767840264650284)
    assert public_ml.set_index("cohort").loc["gse119526", "exclude_auc"] == pytest.approx(0.9918619791666664)

    stroma = pd.read_csv(ROOT / "results" / "coordination_sensitivity" / "colonomics_stromal_rank_coordination.tsv", sep="\t")
    row = stroma.iloc[0]
    assert row["n_paired_patients_with_tumor_stromal_score"] == 90
    assert row["median_between_gene_rho_unadjusted"] == pytest.approx(0.5993686437415201)
    assert row["median_between_gene_rho_stromal_rank_adjusted"] == pytest.approx(0.5570551568884964)

    pc1 = pd.read_csv(ROOT / "results" / "coordination_sensitivity" / "korean_delta_pc1_fraction.tsv", sep="\t").set_index("cohort")
    assert pc1.loc["CMCBSN", "pc1_explained_variance_fraction"] == pytest.approx(0.6972509611120057)
    assert pc1.loc["SNUH", "pc1_explained_variance_fraction"] == pytest.approx(0.6047968649367312)
    assert pc1.loc["ASAN", "pc1_explained_variance_fraction"] == pytest.approx(0.7073018426150439)


def test_fixed_registry_still_has_original_family_size():
    probes, mapping = read_fixed()
    assert len(probes) == 77
    assert len(mapping) == 77
    assert {"cg20680720", "cg04481096"}.issubset(mapping)


def test_public_exclude2_aggregate_uses_current_124_results_and_allowlist_excludes_audit_prose():
    _require_retained(ROOT / "review" / "evidence" / "korean_revision" / "aggregate_allowlist_paths.tsv", "retained allowlist evidence")
    agg = pd.read_csv(ROOT / "results" / "qc_sensitivity" / "exclude2_aggregate_sensitivity.tsv", sep="\t")
    tn = agg[(agg["analysis"].eq("public_meta")) & (agg["group"].eq("T-N"))].iloc[0]
    # Primary T-N pool = paired cohorts with >=20 pairs (rule fixed 2026-09-09; 68/77);
    # the five-cohort pool that adds GSE77954 (76/77) is the sensitivity analysis.
    assert tn["original_significant"] == 68
    assert tn["sensitivity_significant"] == 66

    gse77954 = agg[(agg["analysis"].eq("public_contrast_cohort")) & (agg["group"].eq("GSE77954|T-N"))].iloc[0]
    assert gse77954["retained_estimable"] == 75
    assert gse77954["sensitivity_significant"] == 2

    allow = pd.read_csv(ROOT / "review" / "evidence" / "korean_revision" / "aggregate_allowlist_paths.tsv", sep="\t")
    paths = set(allow["path"])
    assert "results/coordination_sensitivity/colonomics_stromal_rank_coordination.tsv" in paths
    assert "results/coordination_sensitivity/korean_delta_pc1_fraction.tsv" in paths
    assert "review/evidence/korean_revision/prose_ready_counts.md" not in paths
    assert "review/decisions/korean_revision.md" not in paths


def test_physical_gzip_hash_matches_registered_zhou_manifest_hash():
    _require_retained(ROOT.parent / "115_Public_Biology_20260905" / "data" / "raw" / "zhou_HM450.hg19.manifest.tsv.gz", "Zhou HM450 annotation archive")
    _require_retained(ROOT / "review" / "evidence" / "korean_revision" / "run_manifest.json", "retained run manifest evidence")
    manifest_path = ROOT.parent / "115_Public_Biology_20260905" / "data" / "raw" / "zhou_HM450.hg19.manifest.tsv.gz"
    assert sha256_file(manifest_path) == "88e995655e68b2867105e02924ad9f3ca6d51357e187676053bba935bf1fbc65"
    run_manifest = json.loads((ROOT / "review" / "evidence" / "korean_revision" / "run_manifest.json").read_text())
    assert run_manifest["input_hashes"]["zhou_hm450_manifest_tsv_gz_physical_bytes"] == "88e995655e68b2867105e02924ad9f3ca6d51357e187676053bba935bf1fbc65"


def test_public_ml75_refit_manifest_is_124_local_and_aggregate_only():
    manifest = json.loads((ROOT / "results" / "qc_sensitivity" / "public_ml75_run_manifest.json").read_text())
    assert manifest["script"] == "scripts/refit_public_ml75_sensitivity.py"
    assert manifest["source_builder"] == "scripts/analyze_public_cpg_ml.py"
    assert manifest["private_outputs_written"] is False
    assert manifest["conditional_patient_bootstrap"] is True
    assert set(manifest["cohorts"]) == {"colonomics", "gse119526"}
    assert "123_Independent" not in json.dumps(manifest)
