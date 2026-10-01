"""Cohort-local tissue-classification AUCs, unchanged from the former Figure 9."""
from pathlib import Path
import argparse
import pandas as pd
from matplotlib.lines import Line2D
import style as s

ROWS = [("Local PSQ, 87 patients","ridge","Local pyrosequencing","Ridge, 10 genes, 87 patients",s.BLUE),
        ("Local PSQ, 87 patients","inner-selected single gene","Local pyrosequencing","Inner-selected single gene, 87 patients",s.BLUE),
        ("Colonomics, 92 patients","ridge","Colonomics","Ridge, 77 CpGs, 92 patients", "#AF5D34"),
        ("GSE119526, 48 patients","ridge","GSE119526","Ridge, 71 CpGs, 48 patients", "#AF5D34")]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path(__file__).resolve().parents[2]/"figures/source_data/FigureS1_tissue_auc.csv")
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parents[2]/"figures")
    args=parser.parse_args()
    data=pd.read_csv(args.input)
    if not (len(data)==4):
        raise AssertionError
    fig=s.figure(180,66)
    ax=s.axes(fig,69,13,64,40)
    left=s.axes(fig,2,13,64,40); right=s.axes(fig,135,13,42,40)
    for axis in (left,right):
        axis.set_axis_off(); axis.set_xlim(0,1); axis.set_ylim(3.6,-.6)
    ax.set_xlim(.85,1); ax.set_ylim(3.6,-.6)
    ax.set_xticks([.85,.90,.95,1]); ax.set_xticklabels(["0.85","0.90","0.95","1.00"])
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.grid(axis="x",color="#E4E7E8",lw=.5); ax.set_axisbelow(True)
    ax.set_xlabel("Area under the ROC curve (AUC)",labelpad=4)
    right.text(1,-.63,"AUC (95% CI)",ha="right",va="bottom",fontsize=8)
    for i,(cohort,model,title,subtitle,color) in enumerate(ROWS):
        row=data.loc[data.cohort.eq(cohort)&data.model.eq(model)].iloc[0]
        if not (.85 <= row.ci_low <= row.auc <= row.ci_high <= 1):
            raise AssertionError
        ax.errorbar(row.auc,i,xerr=[[row.auc-row.ci_low],[row.ci_high-row.auc]],fmt="o",color=color,ms=4,capsize=2,lw=1)
        left.text(1,i-.19,title,ha="right",va="center",fontsize=8)
        left.text(1,i+.19,subtitle,ha="right",va="center",fontsize=7,color=s.GRAY)
        right.text(1,i,f"{row.auc:.3f} ({row.ci_low:.3f} to {row.ci_high:.3f})",ha="right",va="center",fontsize=8)
    fig.legend(handles=[Line2D([0],[0],color=s.BLUE,marker="o",ms=4,lw=1,label="Local pyrosequencing"),
                        Line2D([0],[0],color="#AF5D34",marker="o",ms=4,lw=1,label="Public arrays")],
               loc="upper left",bbox_to_anchor=(.02,.99),ncol=2,borderaxespad=0,fontsize=8)
    result=s.save(fig,args.output_dir,"Figure_S1",[args.input],
                  "The same four cohort-local nested-cross-validation estimates as the former Figure 9. "
                  "Intervals are conditional 95% patient-cluster bootstrap intervals from 2,000 resamples. "
                  "No refitting or re-estimation was performed.",{"n_rows":4,"patient_bootstrap_resamples":2000})
    print(result["figure"], result["outputs"])

if __name__=="__main__":
    main()
