#!/usr/bin/env python3
"""Plot annotation-scope CpG effects with patient-bootstrap median intervals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style as s


def require(condition, message="Integrity check failed"):
    if not condition:
        raise RuntimeError(message)
BASE = Path(__file__).resolve().parents[1]
COHORTS = ["CMCBSN", "SNUH", "ASAN"]
LOCATIONS = ["promoter", "body", "other"]
COLORS = {True: "#276187", False: "#7C8286"}


def jitter(probe):
    """Fixed non-data vertical offset; x coordinates retain the measured effect."""
    v=int(hashlib.sha256(probe.encode()).hexdigest()[:8],16)/0xffffffff
    return (v-.5)*.20


def read_inputs(effects_path, groups_path):
    effects=pd.read_csv(effects_path)
    groups=pd.read_csv(groups_path)
    require(len(effects)==1173 and not effects.duplicated(["cohort","probe"]).any(), 'Integrity check failed: len(effects)==1173 and not effects.duplicated(["cohort","probe"]).any()')
    require(set(effects.cohort)==set(COHORTS), 'Integrity check failed: set(effects.cohort)==set(COHORTS)')
    require(set(effects.pooled_location)==set(LOCATIONS), 'Integrity check failed: set(effects.pooled_location)==set(LOCATIONS)')
    selected_sets=[]
    checks=[]
    for cohort in COHORTS:
        sub=effects[effects.cohort==cohort]
        selected_sets.append(set(sub.loc[sub.selected,"probe"]))
        require(len(sub)==391 and sub.selected.sum()==77, 'Integrity check failed: len(sub)==391 and sub.selected.sum()==77')
        require(sub.is_cpg.sum()==388 and (sub.probe_type=="ch").sum()==3, 'Integrity check failed: sub.is_cpg.sum()==388 and (sub.probe_type=="ch").sum()==3')
        require(sub.is_cpg.equals(sub.probe.str.startswith("cg")), 'Integrity check failed: sub.is_cpg.equals(sub.probe.str.startswith("cg"))')
        require(sub.loc[sub.selected,"is_cpg"].all(), 'Integrity check failed: sub.loc[sub.selected,"is_cpg"].all()')
        require(sub.present.sum()==282, 'Integrity check failed: sub.present.sum()==282')
        require((sub.present & sub.is_cpg).sum()==281, 'Integrity check failed: (sub.present & sub.is_cpg).sum()==281')
        require((sub.present & sub.MASK_general).sum()==12, 'Integrity check failed: (sub.present & sub.MASK_general).sum()==12')
        require((sub.MASK_general).sum()==18, 'Integrity check failed: (sub.MASK_general).sum()==18')
        eligible=sub[sub.inferential_eligible]
        require(len(eligible)==269 and eligible.selected.sum()==54, 'Integrity check failed: len(eligible)==269 and eligible.selected.sum()==54')
        require(eligible.is_cpg.all(), 'Integrity check failed: eligible.is_cpg.all()')
        require(not eligible.MASK_general.any() and eligible.present.all(), 'Integrity check failed: not eligible.MASK_general.any() and eligible.present.all()')
        require(eligible.n_complete_pairs.min()>=3, 'Integrity check failed: eligible.n_complete_pairs.min()>=3')
        require(np.isfinite(eligible.mean_delta_pp).all(), 'Integrity check failed: np.isfinite(eligible.mean_delta_pp).all()')
        for location in LOCATIONS:
            for selected in [True,False]:
                row=eligible[(eligible.pooled_location==location)&(eligible.selected==selected)]
                summary=groups[(groups.cohort==cohort)&(groups.location==location)&
                               (groups.selection==("selected" if selected else "unselected"))]
                require(len(summary)==1, 'Integrity check failed: len(summary)==1')
                summary=summary.iloc[0]
                require(len(row)==summary.n_analyzed, 'Integrity check failed: len(row)==summary.n_analyzed')
                require(np.isclose(row.mean_delta_pp.median(),summary.median_delta_pp,rtol=0,atol=1e-10), 'Integrity check failed: np.isclose(row.mean_delta_pp.median(),summary.median_delta_pp,rtol=0,atol=1e-10)')
                require(summary.bootstrap_valid==2000, 'Integrity check failed: summary.bootstrap_valid==2000')
                require(np.isfinite([summary.median_ci_low_pp,summary.median_ci_high_pp]).all(), 'Integrity check failed: np.isfinite([summary.median_ci_low_pp,summary.median_ci_high_pp]).all()')
                require(summary.median_ci_low_pp<=summary.median_delta_pp<=summary.median_ci_high_pp, 'Integrity check failed: summary.median_ci_low_pp<=summary.median_delta_pp<=summary.median_ci_high_pp')
                checks.append({"cohort":cohort,"location":location,
                               "selection":"selected" if selected else "unselected",
                               "n":len(row),"median_delta_pp":float(summary.median_delta_pp),
                               "ci_low_pp":float(summary.median_ci_low_pp),
                               "ci_high_pp":float(summary.median_ci_high_pp)})
    require(selected_sets[0]==selected_sets[1]==selected_sets[2], 'Integrity check failed: selected_sets[0]==selected_sets[1]==selected_sets[2]')
    return effects,groups,checks


def render(effects_path,groups_path,output):
    effects,groups,checks=read_inputs(effects_path,groups_path)
    plotted=effects[effects.inferential_eligible].copy()
    fig=s.figure(180,109)
    handles=[Line2D([],[],ls="",marker="o",ms=3,color=COLORS[True],label="Selected"),
             Line2D([],[],ls="",marker="o",ms=3,color=COLORS[False],label="Unselected"),
             Line2D([],[],lw=.8,marker="D",ms=3,color=s.INK,label="Median and 95% CI")]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.13,.999),
               ncol=3,columnspacing=1.6,handletextpad=.5)
    all_bounds=np.r_[plotted.mean_delta_pp,groups.median_ci_low_pp,groups.median_ci_high_pp]
    xmin=np.floor((all_bounds.min()-3)/10)*10
    xmax=np.ceil((all_bounds.max()+3)/10)*10
    for i,cohort in enumerate(COHORTS):
        left=26+i*51
        ax=s.axes(fig,left,27,40,64)
        n_pairs=int(effects.loc[effects.cohort==cohort,"n_cohort_pairs"].iloc[0])
        s.label(fig,"ABC"[i],left,12)
        s.text(fig,cohort,left+5,12.3,fontweight="bold",fontsize=8)
        s.text(fig,f"{n_pairs} patient pairs",left,18,fontsize=7,color=s.GRAY)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(5.15,-.55)
        ax.axvline(0,color="#9DA5AA",lw=.7,zorder=0)
        ax.set_xticks([x for x in [-20,0,20,40] if xmin<=x<=xmax])
        ax.set_yticks([0.3,2.3,4.3])
        ax.set_yticklabels(["Promoter","Body","Other"] if i==0 else ["","",""])
        ax.tick_params(axis="y",length=0,pad=5)
        ax.spines['left'].set_visible(False)
        for split in [1.3,3.3]:
            ax.axhline(split,color="#E1E5E7",lw=.55,zorder=0)
        for li,location in enumerate(LOCATIONS):
            for selected,offset in [(True,0),(False,.60)]:
                y=li*2+offset
                subset=plotted[(plotted.cohort==cohort)&(plotted.pooled_location==location)&
                               (plotted.selected==selected)].sort_values("probe")
                row=groups[(groups.cohort==cohort)&(groups.location==location)&
                           (groups.selection==("selected" if selected else "unselected"))].iloc[0]
                yy=y+np.array([jitter(p) for p in subset.probe])
                ax.scatter(subset.mean_delta_pp,yy,s=6,color=COLORS[selected],
                           alpha=.65,edgecolors="none",zorder=2)
                median=float(row.median_delta_pp)
                ax.errorbar(median,y,xerr=[[median-row.median_ci_low_pp],
                                          [row.median_ci_high_pp-median]],
                            fmt="D",markersize=3.2,color=s.INK,elinewidth=.9,
                            capsize=2,capthick=.75,zorder=4)
                ax.text(xmax+2,y,f"n={len(subset)}",ha="left",va="center",fontsize=7,
                        color=COLORS[selected],clip_on=False)
    s.text(fig,"Mean paired tumor minus adjacent mucosal methylation difference (percentage points)",
           91,98,ha="center",fontsize=7)
    s.text(fig,"Each point represents one CpG. Numbers beside rows are CpG counts.",
           26,104,fontsize=7,color=s.GRAY)
    details={"registered_unique_probes":391,"registered_unique_cpgs":388,
             "registered_non_cpg_probes":3,"measured_cpgs_per_cohort":281,
             "eligible_cpgs_per_cohort":269,"plotted_probe_effects":len(plotted),
             "patient_bootstrap_draws":2000,"group_checks":checks,
             "exclusions":"Non-CpG, absent, annotation-masked or fewer than three complete pairs; no effect-based exclusions.",
             "x_range_pp":[float(xmin),float(xmax)],
             "point_unit":"CpG mean paired effect", "patient_is_biological_replicate":True,
             "offset_rule":"SHA-256 of probe ID, visual y offset only", "selected_unselected_test":None}
    return s.save(fig,output,"Figure_S6",[effects_path,groups_path],
                  "Descriptive annotation-scope sensitivity. Median intervals resample complete patient vectors, not CpGs. No selected-versus-unselected significance test.",details)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--effects",type=Path,default=BASE/"source_data/FigureS6_probe_effects.csv")
    p.add_argument("--groups",type=Path,default=BASE/"source_data/FigureS6_group_summary.csv")
    p.add_argument("--output-dir",type=Path,default=BASE)
    a=p.parse_args()
    print(json.dumps(render(a.effects,a.groups,a.output_dir),indent=2))


if __name__=="__main__":
    main()
