"""Create manuscript display specifications from aggregate results only."""
import argparse
import json
from pathlib import Path
import pandas as pd


def build(aggregate_dir, output_dir):
    aggregate_dir = Path(aggregate_dir); output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score = pd.read_csv(aggregate_dir/"background_score_adjustment.tsv", sep="\t")
    s8rows = []
    labels = {"raw_balanced_median": "Unconditioned candidate correlation",
              "background_adjusted_balanced_median": "Common-score-conditioned candidate correlation",
              "adjusted_minus_raw": "Conditioned minus unconditioned correlation",
              "median_probe_background_correlation": "Median candidate-CpG/common-score correlation",
              "mean_rank_panel_background_correlation": "Mean-rank panel/common-score correlation"}
    for metric, label in labels.items():
        row = {"Measure": label}
        for cohort, count in [("ASAN", 128), ("SNUH", 142)]:
            r = score.loc[score.cohort.eq(cohort) & score.metric.eq(metric)].iloc[0]
            row[f"{cohort} ({count} pairs)"] = f"{r.estimate:.3f} ({r.ci_low:.3f}, {r.ci_high:.3f})"
        s8rows.append(row)
    tissue = pd.read_csv(aggregate_dir/"candidate_tissue_decomposition.tsv", sep="\t")
    s9a = []
    for cohort in ["CMCBSN", "SNUH", "ASAN"]:
        d = tissue.loc[tissue.cohort.eq(cohort)].set_index("measurement")
        s9a.append({"Cohort": cohort, "Pairs": str(int(d.iloc[0].pairs)),
                    "Tumor-minus-adjacent rho": f"{d.loc['delta','balanced_median_spearman']:.3f}",
                    "Tumor-only rho": f"{d.loc['tumor','balanced_median_spearman']:.3f}",
                    "Adjacent-only rho": f"{d.loc['adjacent','balanced_median_spearman']:.3f}",
                    "Median Var(adjacent)/Var(difference)": f"{d.loc['adjacent_variance_divided_by_delta_variance','median_ratio']:.3f}"})
    support = pd.read_csv(aggregate_dir/"set_effect_coordination_relationship.tsv", sep="\t").set_index("cohort")
    s9b = []
    for name, cohort, prefix in [("CMCBSN (source)", "ASAN", "source"), ("ASAN (target)", "ASAN", "target"), ("SNUH (target)", "SNUH", "target")]:
        d = support.loc[cohort]
        s9b.append({"Cohort and role": name,
                    "Candidate mean difference (pp)": f"{100*d['candidate_'+prefix+'_mean_delta']:.2f}",
                    "Reference-set median (pp)": f"{100*d[prefix+'_control_mean_delta_median']:.2f}",
                    "Reference-set minimum (pp)": f"{100*d[prefix+'_control_mean_delta_min']:.2f}",
                    "Reference-set maximum (pp)": f"{100*d[prefix+'_control_mean_delta_max']:.2f}"})
    qc = pd.read_csv(aggregate_dir/"local_qc_coordination_summary.tsv", sep="\t")
    s10 = []
    for r in qc.itertuples():
        s10.append({"Analysis": "Original cohort" if r.scenario == "original_87" else "Two complete patient pairs excluded",
                    "Pairs": str(r.pairs), "Median rho": f"{r.median_spearman_rho:.3f}",
                    "Rho range": f"{r.min_spearman_rho:.3f}–{r.max_spearman_rho:.3f}",
                    "Positive correlations": f"{r.positive_correlations}/45",
                    "BH q<0.05": f"{r.BH_q_lt_0_05}/45", "PC1 variance (%)": f"{100*r.PC1_variance_fraction:.2f}"})
    definitions = [
        ("S8", "Exploratory conditioning on a noncandidate common-change score", s8rows,
         "Values are estimates (95% percentile confidence intervals). The unconditioned and conditioned summaries give equal weight to 45 gene pairs: median Spearman correlation within each gene-pair block, followed by the median of these 45 values. Conditioned values are Pearson correlations of candidate-probe rank residuals. The common score uses 965 unique noncandidate CpGs in 151 distinct source-selected genes; probe ranks are centered and normalized to unit length before averaging over unique probes within each gene, each gene score is centered and scaled, the 151 scores are averaged equally, and this average is ranked. Candidate probes and genes are excluded. Source CMCBSN determined membership, whereas score values, ranks and standardization were calculated in each target. All score construction and regression steps were repeated in 500 complete-patient bootstrap resamples (ASAN seed 2026091619; SNUH seed 2026091633). The intervals condition on the frozen candidates, control pool, score definition and deposited values. The unconditioned intervals here use these new 500 resamples and do not replace the original primary-analysis intervals. This is neither independently measured purity/CIMP adjustment nor evidence of a panel-specific mechanism."),
        ("S9A", "A. Candidate correlation in paired differences and the two tissue types", s9a,
         "Rho is the median of 45 gene-pair-specific median Spearman correlations at the same 56 candidate CpGs. The variance ratio is the median over these CpGs. Tumor-only and adjacent-only measurements come from exactly the same source-defined patient pairs as the difference analysis. These are descriptive summaries, not tests of equivalence or longitudinal changes. The raw Pearson covariance decomposition is supplied in the aggregate data; component medians need not sum to the median total."),
        ("S9B", "B. Mean methylation-effect support in the 5,000 frozen reference sets", s9b,
         "For candidates and each reference set, mean tumor-minus-adjacent methylation is first averaged over probes within each of ten regions and then equally over regions; beta differences are multiplied by 100 to obtain percentage points (pp). All 5,000 source-frozen sets are included. Their ranges are reference-set distributions, not confidence intervals. The candidate mean exceeds the maximum reference-set mean in the source and both targets, so there is no overlapping support at the candidate value for this scalar. Set overlap precludes treating 5,000 sets as independent patients. No rematching, P value or target-directed selection was performed."),
        ("S10", "Local coordination after exclusion of two patients with recovered assay QC flags", s10,
         "The exclusions remove both tissues of the two patients identified by the existing authorized QC plan, without re-quantification or imputation. All ten genes and all 45 gene pairs are retained. Spearman tests are two-sided, with Benjamini–Hochberg (BH) correction over 45 pairs separately in each scenario. PC1 uses centered, gene-standardized paired differences; variance is expressed as a percentage. These descriptive sensitivity results do not establish assay quality in unobserved runs, and no bootstrap interval or significance test for the between-scenario change is reported.")]
    specs = {"schema_version": 1, "table_S9_title": "Candidate tissue covariance and reference-set methylation-effect support", "tables": []}
    for ident, title, records, note in definitions:
        frame = pd.DataFrame(records)
        frame.to_csv(output_dir/f"Table_{ident}_display.tsv", sep="\t", index=False)
        specs["tables"].append({"id": ident, "title": title, "columns": list(frame.columns),
                                "rows": frame.values.tolist(), "footnote": note,
                                "display_file": f"Table_{ident}_display.tsv"})
    (output_dir/"table_specs.json").write_text(json.dumps(specs, ensure_ascii=False, indent=2)+"\n")
    return specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    build(args.aggregate_dir, args.output_dir)
