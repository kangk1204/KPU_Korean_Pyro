#!/usr/bin/env python3
"""Add bounded Korean QC, coverage, coordination, and provenance sensitivity outputs.

All outputs stay inside 124_Integrated_Revision_20260909. The script uses the
already verified derived matrices/results plus 119/120 source registries; it does
not reread the 11.8 GB Korean raw processed beta inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
P119 = PROJECT / "119_Figure1_ABC_Revision_20260908"
P120 = PROJECT / "120_Korean_External_Validation_20260908"
ZHOU = PROJECT / "115_Public_Biology_20260905" / "data" / "raw" / "zhou_HM450.hg19.manifest.tsv.gz"
FLAGGED = ["cg20680720", "cg04481096"]
COHORTS = ["CMCBSN", "SNUH", "ASAN"]
FAMILY_SIZE = 77


def sha256_file(path: Path) -> str:
    """Hash the physical file bytes, including compressed bytes for .gz inputs."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.nan_to_num(np.asarray(values, dtype=float), nan=1.0)
    order = np.argsort(p)
    q = np.empty(len(p), dtype=float)
    q[order] = np.minimum(1.0, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    return q


def read_fixed() -> tuple[list[str], dict[str, str]]:
    fixed = json.loads((ROOT / "registry" / "fixed_probes.json").read_text())
    probes = [probe for probes in fixed.values() for probe in probes]
    mapping = {probe: gene for gene, probes in fixed.items() for probe in probes}
    if len(probes) != FAMILY_SIZE or len(mapping) != FAMILY_SIZE:
        raise AssertionError("fixed_probes.json must define 77 unique probes")
    return probes, mapping


def bh_adjust_with_planned_family(p_values: pd.Series, family_size: int = FAMILY_SIZE) -> pd.Series:
    p = p_values.astype(float).to_numpy()
    q = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return pd.Series(q, index=p_values.index)
    ordered = finite[np.argsort(p[finite])]
    raw = p[ordered] * family_size / np.arange(1, len(finite) + 1)
    q[ordered] = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    return pd.Series(q, index=p_values.index)


def write_probe_qc_and_coverage(probes: list[str]) -> dict[str, object]:
    registry = ROOT / "registry"
    qc = pd.read_csv(P119 / "registry" / "probe_qc.tsv", sep="\t")
    qc = qc.set_index("probe").loc[probes].reset_index()
    qc["technical_flag_reason"] = np.where(
        qc["probe"].eq("cg20680720"),
        "Zhou HM450 MASK_general via SNP-related masks; sensitivity exclusion only",
        np.where(qc["probe"].eq("cg04481096"), "Zhou HM450 MASK_general via cross-reactive mapping; sensitivity exclusion only", ""),
    )
    qc.to_csv(registry / "probe_qc_fixed77.tsv", sep="\t", index=False)

    zhou_cols = ["probeID", "MASK_mapping", "MASK_snp5_common", "MASK_snp5_GMAF1p", "MASK_extBase", "MASK_general"]
    zhou = pd.read_csv(ZHOU, sep="\t", compression="gzip", usecols=zhou_cols)
    zhou_rows = zhou[zhou["probeID"].isin(FLAGGED)].copy()
    zhou_rows.to_csv(registry / "probe_qc_original_zhou_flagged.tsv", sep="\t", index=False)

    epic = pd.read_csv(ROOT / "registry" / "epic_platform_probe_audit.tsv", sep="\t")
    epic_bool = epic.set_index("probe")["epic_annotation_present"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    rows = []
    for probe in probes:
        row = {"probe": probe, "fixed_gene": qc.set_index("probe").loc[probe, "fixed_gene"], "epic_annotation_present": bool(epic_bool.loc[probe])}
        for cohort in COHORTS:
            beta = pd.read_csv(ROOT / "data" / "derived" / f"{cohort}_beta.tsv", sep="\t", index_col=0, usecols=[0])
            row[f"{cohort}_processed_present"] = probe in set(beta.index)
        present_all = all(row[f"{cohort}_processed_present"] for cohort in COHORTS)
        if present_all:
            status = "measured_in_all_three_processed_korean_matrices"
        elif not row["epic_annotation_present"]:
            status = "absent_from_EPIC_annotation"
        else:
            status = "EPIC_annotated_but_absent_from_deposited_processed_korean_matrices"
        row["coverage_status"] = status
        row["technical_flag"] = bool(qc.set_index("probe").loc[probe, "technical_flag"])
        row["coverage_interpretation_limit"] = "No processing-cause inference is made beyond annotation presence and deposited processed-matrix presence."
        rows.append(row)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(registry / "korean_probe_coverage_provenance.tsv", sep="\t", index=False)
    summary = {
        "fixed77": len(coverage),
        "epic_annotation_present": int(coverage["epic_annotation_present"].sum()),
        "epic_annotation_absent": int((~coverage["epic_annotation_present"]).sum()),
        "processed_present_all_three": int((coverage["coverage_status"] == "measured_in_all_three_processed_korean_matrices").sum()),
        "processed_absent_all_three": int((coverage["coverage_status"] != "measured_in_all_three_processed_korean_matrices").sum()),
        "epic_annotated_but_absent_processed": int((coverage["coverage_status"] == "EPIC_annotated_but_absent_from_deposited_processed_korean_matrices").sum()),
        "technical_mask_ids": FLAGGED,
        "interpretation_limit": "6 fixed probes lack EPIC annotation and 15 are EPIC-annotated but absent from the deposited processed Korean matrices; this registry does not assign an upstream processing cause.",
    }
    (registry / "korean_probe_coverage_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def recompute_korean_qc_sensitivity(mapping: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    (ROOT / "results" / "qc_sensitivity").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "coordination_sensitivity").mkdir(parents=True, exist_ok=True)
    effect_rows = []
    coord_rows = []
    retained_by_cohort = {}
    for cohort in COHORTS:
        delta = pd.read_csv(ROOT / "results" / "individual_cpg" / cohort / "paired_cpg_deltas.tsv", sep="\t", index_col=0)
        retained = [p for p in mapping if p in delta.columns and p not in FLAGGED]
        retained_by_cohort[cohort] = retained
        if len(retained) != 54:
            raise AssertionError(f"{cohort}: expected 54 retained observed CpGs, got {len(retained)}")
        for probe in retained:
            arr = delta[probe].dropna().to_numpy(dtype=float)
            t_p = float(stats.ttest_1samp(arr, 0).pvalue) if not np.all(arr == 0) else 1.0
            w_p = float(stats.wilcoxon(arr, zero_method="wilcox", alternative="two-sided").pvalue) if not np.all(arr == 0) else 1.0
            effect_rows.append({
                "cohort": cohort,
                "probe": probe,
                "gene": mapping[probe],
                "n_pairs": len(arr),
                "mean_delta_pp": float(arr.mean() * 100),
                "sd_delta_pp": float(arr.std(ddof=1) * 100),
                "se_delta_pp": float(arr.std(ddof=1) * 100 / np.sqrt(len(arr))),
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
                "n_tumor_higher": int((arr > 0).sum()),
                "fraction_tumor_higher": float((arr > 0).mean()),
            })
        x = delta[retained].dropna().to_numpy(dtype=float)
        ranks = np.apply_along_axis(stats.rankdata, 0, x)
        corr = np.corrcoef(ranks, rowvar=False)
        groups = np.array([mapping[p] for p in retained])
        ii, jj = np.triu_indices(len(retained), 1)
        between = groups[ii] != groups[jj]
        vals = corr[ii[between], jj[between]]
        original = pd.read_csv(ROOT / "results" / "individual_cpg" / cohort / "coordination_summary.tsv", sep="\t").iloc[0]
        coord_rows.append({
            "cohort": cohort,
            "scope": "exclude cg20680720 and cg04481096; median and sign recomputed only; no full bootstrap/permutation rerun",
            "n_complete_pairs": len(x),
            "n_available_cpgs": len(retained),
            "n_genes": len(set(groups)),
            "n_between_gene_pairs": len(vals),
            "n_positive_between_gene_pairs": int(np.sum(vals > 0)),
            "median_between_gene_rho_54": float(np.median(vals)),
            "original_median_between_gene_rho_56": float(original["median_between_gene_rho"]),
            "median_change_54_minus_56": float(np.median(vals) - float(original["median_between_gene_rho"])),
            "full_ci_or_p_recomputed": False,
        })
    effects = pd.DataFrame(effect_rows)
    effects["paired_t_q77"] = effects.groupby("cohort", group_keys=False)["paired_t_p"].apply(bh_adjust_with_planned_family)
    effects["wilcoxon_q77"] = effects.groupby("cohort", group_keys=False)["wilcoxon_p"].apply(bh_adjust_with_planned_family)
    summary = effects.groupby("cohort").agg(
        n_effect_rows=("probe", "size"),
        n_positive_mean_delta=("mean_delta_pp", lambda s: int((s > 0).sum())),
        n_paired_t_q77_lt_005=("paired_t_q77", lambda s: int((s < 0.05).sum())),
        mean_delta_min_pp=("mean_delta_pp", "min"),
        mean_delta_max_pp=("mean_delta_pp", "max"),
        max_paired_t_q77=("paired_t_q77", "max"),
    ).reset_index()
    effects.to_csv(ROOT / "results" / "qc_sensitivity" / "korean_exclude2_paired_cpg_effects54.tsv", sep="\t", index=False)
    summary.to_csv(ROOT / "results" / "qc_sensitivity" / "korean_exclude2_effect_summary.tsv", sep="\t", index=False)
    coord = pd.DataFrame(coord_rows)
    coord.to_csv(ROOT / "results" / "coordination_sensitivity" / "korean_exclude2_coordination54.tsv", sep="\t", index=False)
    return summary, coord



def recompute_public_exclude2_aggregate() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def summarize(df: pd.DataFrame, groups: list[str], probe_col: str, p_col: str, q_col: str, effect_col: str, label: str) -> None:
        for key, group in df.groupby(groups, sort=False):
            keys = key if isinstance(key, tuple) else (key,)
            pvals = pd.to_numeric(group[p_col], errors="coerce").to_numpy(copy=True)
            observed = np.isfinite(pvals) & group["status"].eq("estimated").to_numpy()
            mask = group[probe_col].isin(FLAGGED).to_numpy()
            pvals[mask] = 1.0
            qvals = bh(pvals)
            retained = observed & ~mask
            effect = pd.to_numeric(group[effect_col], errors="coerce").to_numpy()
            original = pd.to_numeric(group[q_col], errors="coerce").lt(0.05).to_numpy()
            directional = not ("analysis" in group and group["analysis"].eq("CMS_global_Kruskal").any())
            rows.append({
                "analysis": label,
                "group": "|".join(map(str, keys)),
                "planned_tests": int(len(group)),
                "flagged_observed": int((mask & observed).sum()),
                "retained_estimable": int(retained.sum()),
                "original_significant": int(original.sum()),
                "sensitivity_significant": int((retained & (qvals < 0.05)).sum()),
                "sensitivity_positive_significant": int((retained & (qvals < 0.05) & (effect > 0)).sum()) if directional else None,
                "sensitivity_negative_significant": int((retained & (qvals < 0.05) & (effect < 0)).sum()) if directional else None,
            })

    meta = pd.read_csv(ROOT / "results" / "public_contrasts" / "public_probe_meta_analysis.tsv", sep="\t")
    summarize(meta, ["contrast"], "probe", "p", "q_BH_contrast_77", "effect_pp", "public_meta")

    context = pd.read_csv(ROOT / "results" / "public_context" / "all_public_cpg_context.tsv", sep="\t")
    summarize(context, ["cohort", "model", "analysis"], "cpg", "p", "q_BH77", "effect", "public_context")

    contrasts = pd.read_csv(ROOT / "results" / "public_contrasts" / "public_probe_contrasts.tsv", sep="\t")
    summarize(contrasts, ["family"], "probe", "p", "q_BH_family", "effect_pp", "public_contrast_families")
    for _family, family_group in contrasts.groupby("family", sort=False):
        family_group = family_group.copy()
        pvals = pd.to_numeric(family_group["p"], errors="coerce").to_numpy(copy=True)
        pvals[family_group["probe"].isin(FLAGGED).to_numpy()] = 1.0
        family_group["exclude2_family_q"] = bh(pvals)
        for key, cohort_group in family_group.groupby(["cohort", "contrast"], sort=False):
            observed = pd.to_numeric(cohort_group["p"], errors="coerce").notna() & cohort_group["status"].eq("estimated")
            retained = observed & ~cohort_group["probe"].isin(FLAGGED)
            sig = retained & cohort_group["exclude2_family_q"].lt(0.05)
            rows.append({
                "analysis": "public_contrast_cohort",
                "group": "|".join(map(str, key)),
                "planned_tests": int(len(family_group)),
                "flagged_observed": int((cohort_group["probe"].isin(FLAGGED) & observed).sum()),
                "retained_estimable": int(retained.sum()),
                "original_significant": int(pd.to_numeric(cohort_group["q_BH_family"], errors="coerce").lt(0.05).sum()),
                "sensitivity_significant": int(sig.sum()),
                "sensitivity_positive_significant": int((sig & pd.to_numeric(cohort_group["effect_pp"], errors="coerce").gt(0)).sum()),
                "sensitivity_negative_significant": int((sig & pd.to_numeric(cohort_group["effect_pp"], errors="coerce").lt(0)).sum()),
            })

    expr = pd.read_csv(ROOT / "results" / "context" / "expression_cpg_associations.tsv", sep="\t")
    expr["analysis_group"] = "CMCBSN"
    summarize(expr, ["analysis_group"], "cpg", "p", "bh_q", "rho", "cmc_expression")
    summarize(expr, ["analysis_group"], "cpg", "partial_p_age_sex_site", "partial_bh_q_age_sex_site", "partial_rho_age_sex_site", "cmc_expression_adjusted")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "qc_sensitivity" / "exclude2_aggregate_sensitivity.tsv", sep="\t", index=False)
    return out

def ensure_public_ml75_aggregate() -> None:
    summary = ROOT / "results" / "qc_sensitivity" / "public_ml75_before_after_summary.tsv"
    manifest = ROOT / "results" / "qc_sensitivity" / "public_ml75_run_manifest.json"
    if not summary.exists() or not manifest.exists():
        raise FileNotFoundError("publicML75 aggregate sensitivity files must be present in 124 results/qc_sensitivity")
    table = pd.read_csv(summary, sep="\t")
    if set(table["cohort"]) != {"colonomics", "gse119526"} or not (table["sensitivity_features"] == 75).all():
        raise AssertionError("publicML75 aggregate sensitivity summary has unexpected cohorts or feature counts")

def colonomics_stromal_coordination(mapping: dict[str, str]) -> pd.DataFrame:
    beta = pd.read_csv(P119 / "data" / "derived" / "Colonomics_beta.tsv.gz", sep="\t", index_col=0)
    samples = pd.read_csv(P119 / "data" / "derived" / "Colonomics_samples.tsv", sep="\t")
    t = samples[(samples["tissue"] == "T") & (samples["pair_verified"] == True) & (samples["excluded"] == False)].set_index("patient")
    n = samples[(samples["tissue"] == "N") & (samples["pair_verified"] == True) & (samples["excluded"] == False)].set_index("patient")
    patients = sorted(set(t.index) & set(n.index))
    patients = [p for p in patients if pd.notna(t.loc[p, "stromal_score"])]
    probes = [p for p in mapping if p in beta.index]
    delta = beta.loc[probes, t.loc[patients, "sample"].tolist()].to_numpy(dtype=float).T - beta.loc[probes, n.loc[patients, "sample"].tolist()].to_numpy(dtype=float).T
    groups = np.array([mapping[p] for p in probes])
    ii, jj = np.triu_indices(len(probes), 1)
    between = groups[ii] != groups[jj]
    ranks = np.apply_along_axis(stats.rankdata, 0, delta)
    original = np.corrcoef(ranks, rowvar=False)[ii[between], jj[between]]
    stroma_rank = stats.rankdata(t.loc[patients, "stromal_score"].astype(float).to_numpy())
    x = np.column_stack([np.ones(len(stroma_rank)), stroma_rank])
    resid = ranks - x @ np.linalg.lstsq(x, ranks, rcond=None)[0]
    adjusted = np.corrcoef(resid, rowvar=False)[ii[between], jj[between]]
    rows = [{
        "cohort": "Colonomics",
        "analysis": "paired CpG delta between-gene coordination with tumor stromal-score rank adjustment",
        "n_paired_patients_with_tumor_stromal_score": len(patients),
        "n_cpgs": len(probes),
        "n_between_gene_pairs": int(between.sum()),
        "median_between_gene_rho_unadjusted": float(np.median(original)),
        "median_between_gene_rho_stromal_rank_adjusted": float(np.median(adjusted)),
        "median_change_adjusted_minus_unadjusted": float(np.median(adjusted) - np.median(original)),
        "interpretation_limit": "Stromal-rank adjustment attenuates but does not remove coordination; this is not a causal purity decomposition and CIMP labels are not treated as available for a primary CIMP analysis.",
    }]
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "review" / "evidence" / "korean_revision" / "colonomics_stromal_rank_coordination.tsv", sep="\t", index=False)
    out.to_csv(ROOT / "results" / "coordination_sensitivity" / "colonomics_stromal_rank_coordination.tsv", sep="\t", index=False)
    return out


def korean_pc1_fractions() -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        delta = pd.read_csv(ROOT / "results" / "individual_cpg" / cohort / "paired_cpg_deltas.tsv", sep="\t", index_col=0)
        x = delta.to_numpy(dtype=float)
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        rows.append({"cohort": cohort, "n_paired_patients": len(x), "n_cpgs": x.shape[1], "pc1_explained_variance_fraction": float(PCA().fit(x).explained_variance_ratio_[0])})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "review" / "evidence" / "korean_revision" / "korean_delta_pc1_fraction.tsv", sep="\t", index=False)
    out.to_csv(ROOT / "results" / "coordination_sensitivity" / "korean_delta_pc1_fraction.tsv", sep="\t", index=False)
    return out


def write_revision_registries(coverage_summary: dict[str, object], qc_summary: pd.DataFrame, coord: pd.DataFrame, stroma: pd.DataFrame, pc1: pd.DataFrame) -> None:
    cohort_manifest = pd.read_csv(P120 / "registry" / "cohort_manifest.tsv", sep="\t", dtype=str)
    korean = cohort_manifest[cohort_manifest["cohort"].isin(COHORTS)][["cohort", "accession", "current_accession", "secondary_accession", "institution", "platform", "methylation_access"]]
    provenance = {
        "date": "2026-09-09",
        "scope": "124 integrated Korean correction registries",
        "no_raw_11_8gb_reread": True,
        "raw_processed_matrix_verification_basis": "Prior independent raw audit streamed 11,801,439,241 bytes and matched selected CpG hashes; 124 uses derived/result files unless raw re-verification is explicitly requested.",
        "korean_accessions": korean.to_dict(orient="records"),
        "pairing_registry": "registry/pairing_evidence.json",
        "coverage_registry": "registry/korean_probe_coverage_provenance.tsv",
        "qc_registry": "registry/probe_qc_fixed77.tsv",
        "interpretation_limits": [
            "Pairing is exact source-label pairing, not genotype fingerprint identity confirmation.",
            "EPIC annotation absence and processed-matrix absence are reported separately without inferring an upstream filtering cause.",
            "Tumor-content/CIMP sensitivity is descriptive and does not establish causal purity or direct CIMP effects."
        ]
    }
    (ROOT / "registry" / "korean_revision_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    qc_registry = {
        "date": "2026-09-09",
        "excluded_in_sensitivity_only": FLAGGED,
        "fixed_family_size_retained_for_bh": FAMILY_SIZE,
        "technical_flag_source": "registry/probe_qc_fixed77.tsv and registry/probe_qc_original_zhou_flagged.tsv",
        "korean_qc_sensitivity_outputs": [
            "results/qc_sensitivity/korean_exclude2_paired_cpg_effects54.tsv",
            "results/qc_sensitivity/korean_exclude2_effect_summary.tsv",
            "results/coordination_sensitivity/korean_exclude2_coordination54.tsv"
        ],
        "public_ml75_refit_outputs": ["results/qc_sensitivity/public_ml75_before_after_summary.tsv", "results/qc_sensitivity/public_ml75_run_manifest.json"],
        "public_exclude2_aggregate_output": "results/qc_sensitivity/exclude2_aggregate_sensitivity.tsv",
        "submission_caveat": "This lane did not edit package scripts or submission contents; private OOF artifacts remain outside this new submission-facing registry.",
        "korean_effect_summary": qc_summary.to_dict(orient="records"),
        "korean_coordination_summary": coord.to_dict(orient="records"),
        "colonomics_stromal_sensitivity": stroma.to_dict(orient="records"),
        "korean_pc1_fractions": pc1.to_dict(orient="records"),
    }
    (ROOT / "registry" / "qc_sensitivity_registry.json").write_text(json.dumps(qc_registry, indent=2, ensure_ascii=False) + "\n")


def write_review_notes(coverage_summary: dict[str, object], qc_summary: pd.DataFrame, coord: pd.DataFrame, stroma: pd.DataFrame, pc1: pd.DataFrame) -> None:
    review = ROOT / "review"
    (review / "evidence" / "korean_revision").mkdir(parents=True, exist_ok=True)
    (review / "decisions").mkdir(parents=True, exist_ok=True)
    prose = f"""# Korean revision prose-ready verified wording\n\nPairing wording: Pairing was defined by exact matrix-column to public KBDS sample-title matching, source patient codes, and explicit tumor/nontumor labels. The paired sample row counts are CMCBSN 206, SNUH 284, and ASAN 256, corresponding to 103, 142, and 128 source-label verified tumor-normal pairs; this is not genotype-confirmed identity verification.\n\nCoverage wording: Among 77 fixed CpGs, 71 were present in the EPIC annotation and 6 were absent from that annotation. The deposited processed Korean matrices contained 56 fixed CpGs in all three cohorts. The remaining 21 were absent from the processed matrices; 15 of these were EPIC-annotated. This supports reporting platform/processed-matrix coverage without assigning an upstream processing cause.\n\nProbe-QC wording: Two measured fixed CpGs carried general technical-mask flags in the Zhou HM450 annotation: `cg20680720` in ZNF568 through SNP-related masks and `cg04481096` in HOXA2 through cross-reactive mapping. Excluding those two probes retained 54 observed Korean CpGs. All 54 remained tumor-hypermethylated at paired t q77<0.05 in CMCBSN, SNUH, and ASAN. The 54-CpG median between-gene delta correlations were 0.667, 0.574, and 0.684, compared with the original 56-CpG medians of 0.661, 0.569, and 0.675.\n\nTumor-content/CIMP caveat wording: In Colonomics paired array data with tumor stromal scores available for 90 paired patients, the fixed-CpG between-gene delta-correlation median was 0.599 before and 0.557 after stromal-score rank adjustment. In the three Korean paired arrays, the first principal component explained 0.697, 0.605, and 0.707 of standardized CpG-delta variance. These findings are consistent with a shared tumor-content or CIMP-like component contributing to coordination, but they do not establish a causal purity effect and they do not replace direct CIMP analysis. Standard CIMP endpoint labels were not verified for a primary CIMP analysis in the available files.\n"""
    (review / "evidence" / "korean_revision" / "prose_ready_counts.md").write_text(prose)
    decision = {
        "date": "2026-09-09",
        "scope": "124 integrated Korean revision; manuscript/figures/tables/package scripts intentionally untouched by this lane",
        "pairing_decision": "Fix stale pair_verified counts in registry/pairing_evidence.json to row counts from source-label verified pairs; retain explicit non-genotype caveat.",
        "coverage_decision": coverage_summary,
        "qc_sensitivity_decision": qc_summary.to_dict(orient="records"),
        "coordination_sensitivity_decision": coord.to_dict(orient="records"),
        "tumor_content_caveat_decision": {
            "colonomics_stromal_rank_coordination": stroma.to_dict(orient="records"),
            "korean_pc1_fractions": pc1.to_dict(orient="records"),
            "limit": "descriptive sensitivity only; no causal purity claim and no primary CIMP analysis",
        },
    }
    (review / "decisions" / "korean_revision.md").write_text("# Korean revision decisions\n\n```json\n" + json.dumps(decision, indent=2, ensure_ascii=False) + "\n```\n")


def write_aggregate_allowlist() -> None:
    rel_paths = [
        "registry/pairing_evidence.json",
        "registry/probe_qc_fixed77.tsv",
        "registry/probe_qc_original_zhou_flagged.tsv",
        "registry/korean_probe_coverage_provenance.tsv",
        "registry/korean_probe_coverage_summary.json",
        "registry/korean_revision_provenance.json",
        "registry/qc_sensitivity_registry.json",
        "results/qc_sensitivity/korean_exclude2_effect_summary.tsv",
        "results/qc_sensitivity/korean_exclude2_paired_cpg_effects54.tsv",
        "results/coordination_sensitivity/korean_exclude2_coordination54.tsv",
        "results/qc_sensitivity/exclude2_aggregate_sensitivity.tsv",
        "results/qc_sensitivity/public_ml75_before_after_summary.tsv",
        "results/qc_sensitivity/public_ml75_run_manifest.json",
        "results/coordination_sensitivity/colonomics_stromal_rank_coordination.tsv",
        "results/coordination_sensitivity/korean_delta_pc1_fraction.tsv",
    ]
    rows = []
    for rel in rel_paths:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(rel)
        rows.append({"path": rel, "bytes": path.stat().st_size, "allowlist_scope": "aggregate_or_registry_no_private_oof"})
    tsv = ROOT / "review" / "evidence" / "korean_revision" / "aggregate_allowlist_paths.tsv"
    tsv.write_text("path\tbytes\tallowlist_scope\n" + "\n".join(f"{r['path']}\t{r['bytes']}\t{r['allowlist_scope']}" for r in rows) + "\n")
    payload = {
        "date": "2026-09-09",
        "root": "124_Integrated_Revision_20260909",
        "excludes": ["private OOF predictions", "per-fold private train/test identifiers", "model state dumps", "bootstrap draw-level files unless separately requested", "audit prose and review decision files"],
        "paths": rows,
    }
    (ROOT / "review" / "evidence" / "korean_revision" / "aggregate_allowlist_paths.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    (ROOT / "results" / "qc_sensitivity").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "coordination_sensitivity").mkdir(parents=True, exist_ok=True)
    (ROOT / "review" / "evidence" / "korean_revision").mkdir(parents=True, exist_ok=True)
    (ROOT / "review" / "decisions").mkdir(parents=True, exist_ok=True)
    probes, mapping = read_fixed()
    coverage_summary = write_probe_qc_and_coverage(probes)
    qc_summary, coord = recompute_korean_qc_sensitivity(mapping)
    public_aggregate = recompute_public_exclude2_aggregate()
    ensure_public_ml75_aggregate()
    stroma = colonomics_stromal_coordination(mapping)
    pc1 = korean_pc1_fractions()
    write_review_notes(coverage_summary, qc_summary, coord, stroma, pc1)
    write_revision_registries(coverage_summary, qc_summary, coord, stroma, pc1)
    manifest = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "no_raw_11_8gb_reread": True,
        "input_basis": "124 derived/results plus 119/120 source registries and current 124 public results",
        "outputs": sorted(str(p.relative_to(ROOT)) for p in [
            ROOT / "registry" / "probe_qc_fixed77.tsv",
            ROOT / "registry" / "probe_qc_original_zhou_flagged.tsv",
            ROOT / "registry" / "korean_probe_coverage_provenance.tsv",
            ROOT / "registry" / "korean_probe_coverage_summary.json",
            ROOT / "registry" / "korean_revision_provenance.json",
            ROOT / "registry" / "qc_sensitivity_registry.json",
            ROOT / "results" / "qc_sensitivity" / "korean_exclude2_paired_cpg_effects54.tsv",
            ROOT / "results" / "qc_sensitivity" / "public_ml75_before_after_summary.tsv",
            ROOT / "results" / "qc_sensitivity" / "public_ml75_run_manifest.json",
            ROOT / "results" / "qc_sensitivity" / "korean_exclude2_effect_summary.tsv",
            ROOT / "results" / "qc_sensitivity" / "exclude2_aggregate_sensitivity.tsv",
            ROOT / "results" / "coordination_sensitivity" / "korean_exclude2_coordination54.tsv",
            ROOT / "results" / "coordination_sensitivity" / "colonomics_stromal_rank_coordination.tsv",
            ROOT / "results" / "coordination_sensitivity" / "korean_delta_pc1_fraction.tsv",
            ROOT / "review" / "evidence" / "korean_revision" / "colonomics_stromal_rank_coordination.tsv",
            ROOT / "review" / "evidence" / "korean_revision" / "korean_delta_pc1_fraction.tsv",
            ROOT / "review" / "evidence" / "korean_revision" / "prose_ready_counts.md",
            ROOT / "review" / "decisions" / "korean_revision.md",
        ]),
        "input_hashes": {
            "119_probe_qc": sha256_file(P119 / "registry" / "probe_qc.tsv"),
            "120_cohort_manifest": sha256_file(P120 / "registry" / "cohort_manifest.tsv"),
            "zhou_hm450_manifest_tsv_gz_physical_bytes": sha256_file(ZHOU),
        },
        "public_exclude2_aggregate_rows": int(len(public_aggregate)),
    }
    (ROOT / "review" / "evidence" / "korean_revision" / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    write_aggregate_allowlist()
    print(json.dumps({"coverage": coverage_summary, "qc_summary": qc_summary.to_dict(orient="records"), "coordination": coord.to_dict(orient="records"), "stroma": stroma.to_dict(orient="records"), "pc1": pc1.to_dict(orient="records")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
