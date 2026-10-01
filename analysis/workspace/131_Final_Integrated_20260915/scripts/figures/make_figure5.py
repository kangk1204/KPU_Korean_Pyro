"""Draw the reference-set comparison and exploratory common-score sensitivity.

All inputs are aggregate statistics. No individual patient data are required.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import style as s

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[2]/"figures/source_data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2]/"figures")
    args = parser.parse_args()
    paths = [args.input_dir/n for n in ("Figure5_reference_summary.csv", "Figure5_reference_sets.csv", "Figure5_common_score.csv")]
    ref, sets, common = [pd.read_csv(p) for p in paths]
    cohorts = ["ASAN", "SNUH"]
    if not (set(ref.cohort) == set(common.cohort) == set(cohorts)):
        raise AssertionError
    fig = s.figure(180, 105)
    ax = s.axes(fig, 20, 20, 66, 57)
    bx = s.axes(fig, 111, 20, 65, 57)
    for j, cohort in enumerate(cohorts):
        y = 1-j
        row = ref.set_index("cohort").loc[cohort]
        values = sets.loc[sets.cohort.eq(cohort), "gene_pair_balanced_median_rho"].to_numpy()
        if not (len(values) == int(row.full_reference_n) == 5000):
            raise AssertionError
        if not (np.isfinite(values).all()):
            raise AssertionError
        if not (np.isclose(np.median(values), row.full_reference_median_rho)):
            raise AssertionError
        if not (int(np.count_nonzero(values >= row.candidate_rho)) == int(row.background_sets_ge_candidate_count)):
            raise AssertionError
        violin = ax.violinplot([values], positions=[y], vert=False, widths=.45, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(s.LIGHT); body.set_edgecolor(s.GRAY); body.set_alpha(.85); body.set_linewidth(.5)
        ax.plot([row.full_reference_q025, row.full_reference_q975], [y, y], color=s.GRAY, lw=1.1)
        ax.plot(row.full_reference_median_rho, y, marker="|", color=s.GRAY, ms=7, mew=1.1)
        ax.errorbar(row.candidate_rho, y+.16,
                    xerr=[[row.candidate_rho-row.candidate_ci_low], [row.candidate_ci_high-row.candidate_rho]],
                    fmt="o", color=s.BLUE, ms=4, capsize=2, lw=1)
        ax.text(.99, y-.36, f"{int(row.background_sets_ge_candidate_count):,} of 5,000 sets ≥ candidate", ha="right", va="center", fontsize=7)
        for metric, offset, color, marker in [("raw_balanced_median", .13, s.BLUE, "o"),
                                               ("background_adjusted_balanced_median", -.13, s.GOLD, "s")]:
            estimate = common.loc[common.cohort.eq(cohort) & common.metric.eq(metric)].iloc[0]
            if not (estimate.ci_low <= estimate.estimate <= estimate.ci_high):
                raise AssertionError
            bx.errorbar(estimate.estimate, y+offset,
                        xerr=[[estimate.estimate-estimate.ci_low], [estimate.ci_high-estimate.estimate]],
                        fmt=marker, color=color, ms=4, capsize=2, lw=1)
            bx.text(estimate.estimate, y+offset+.10, f"{estimate.estimate:.3f}", ha="center", va="bottom", color=color, fontsize=7)
    for axis in (ax,bx):
        axis.set_xlim(0,1); axis.set_ylim(-.56,1.56)
        axis.set_xticks([0,.25,.5,.75,1]); axis.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
        axis.tick_params(axis="y", length=0)
        axis.spines["left"].set_visible(False)
        axis.set_xlabel("Gene-pair-balanced median correlation", labelpad=4)
    ax.set_yticks([1,0], [f"{c}\n{int(ref.set_index('cohort').loc[c,'paired_patients'])} pairs" for c in cohorts])
    bx.set_yticks([1,0], cohorts)
    s.label(fig,"A",2,6); s.label(fig,"B",94,6)
    s.text(fig,"Source-matched reference sets",20,7,fontsize=8)
    s.text(fig,"Common-score adjustment",111,7,fontsize=8)
    ax.legend(handles=[Line2D([0],[0],color=s.LIGHT,lw=5,label="Reference-set distribution"),
                       Line2D([0],[0],color=s.BLUE,marker="o",ms=4,lw=1,label="Candidate with 95% CI")],
              loc="lower left",bbox_to_anchor=(-.02,1.01),ncol=1,borderaxespad=0,handlelength=1.6,labelspacing=.3)
    bx.legend(handles=[Line2D([0],[0],color=s.BLUE,marker="o",ms=4,lw=1,label="Raw"),
                       Line2D([0],[0],color=s.GOLD,marker="s",ms=4,lw=1,label="Score-adjusted")],
              loc="lower left",bbox_to_anchor=(-.02,1.01),ncol=2,borderaxespad=0,handlelength=1.6,columnspacing=1.2)
    s.text(fig,"Reference sets retained lower methylation increases than the candidates.",20,91,fontsize=7)
    s.text(fig,"The common score used 151 noncandidate genes. It was not a measured purity or CIMP covariate.",20,97,fontsize=7)
    result = s.save(fig,args.output_dir,"Figure_5",paths,
                    "A retains the original frozen reference comparison and its patient-bootstrap intervals. "
                    "B shows raw and partial rank correlations after removal of the ranked common score. "
                    "Both analyses use 500 complete-patient bootstrap resamples, with distinct draws. "
                    "The reference-set central 95% range is not a confidence interval. Empirical counts are not P values.",
                    {"n_patients":{"ASAN":128,"SNUH":142},"n_reference_sets_per_cohort":5000,
                     "n_background_genes":151,"n_background_unique_probes":965,"patient_bootstrap_resamples":500})
    print(result["figure"], result["outputs"])

if __name__ == "__main__":
    main()
