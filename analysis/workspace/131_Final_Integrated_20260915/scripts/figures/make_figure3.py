"""Methylation differences, correlations and PCA for the original 87 patients.

The private input directory must contain the archived delta and PCA score files.
Patient-level inputs are read only and are never copied into public source data.
No event or recurrence column is loaded or used by this figure.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm, Normalize
import style as s

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-input-dir",type=Path,required=True,
                        help="Directory with archived delta and PCA score TSVs. Not supplied in the public archive.")
    parser.add_argument("--tissue-input",type=Path,
                        help="Optional authorized full-precision tissue TSV for independent correlation verification.")
    parser.add_argument("--aggregate-dir",type=Path,default=Path(__file__).resolve().parents[2]/"figures/source_data")
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parents[2]/"figures")
    args=parser.parse_args()
    paths=[args.private_input_dir/"F3_delta_matrix_source_data.tsv",args.private_input_dir/"F3_pca_scores_source_data.tsv",
           args.aggregate_dir/"Figure3_correlations.csv",args.aggregate_dir/"Figure3_loadings.csv",args.aggregate_dir/"Figure3_summary.json"]
    delta=pd.read_csv(paths[0],sep="\t",usecols=["study_id"]+s.GENES)
    scores=pd.read_csv(paths[1],sep="\t",usecols=["study_id","PC1","PC2"])
    corr=pd.read_csv(paths[2]).set_index("gene").loc[s.GENES,s.GENES].to_numpy(float)
    load=pd.read_csv(paths[3]).set_index("gene").loc[s.GENES]
    summary=json.loads(paths[4].read_text())
    if not (len(delta)==len(scores)==summary["n_pairs"]==87):
        raise AssertionError
    if not (delta.study_id.equals(scores.study_id) and delta.study_id.is_unique):
        raise AssertionError
    x=delta[s.GENES].to_numpy(float)
    if not (np.isfinite(x).all() and np.all(x.std(axis=0)>0)):
        raise AssertionError
    z=(x-x.mean(axis=0))/x.std(axis=0)
    rounded_delta_correlation_difference=float(np.max(abs(spearmanr(x,axis=0).statistic-corr)))
    full_precision_difference=None
    if args.tissue_input:
        tissue=pd.read_csv(args.tissue_input,sep="\t",usecols=["study_id","tissue"]+s.GENES)
        tumor=tissue.loc[tissue.tissue.eq("T")].set_index("study_id").loc[delta.study_id,s.GENES]
        mucosa=tissue.loc[tissue.tissue.eq("N")].set_index("study_id").loc[delta.study_id,s.GENES]
        full_delta=tumor.to_numpy()-mucosa.to_numpy()
        if not (np.allclose(full_delta,x,atol=1e-8)):
            raise AssertionError
        full_precision_difference=float(np.max(abs(spearmanr(full_delta,axis=0).statistic-corr)))
        if not (full_precision_difference<1e-8):
            raise AssertionError
        paths.append(args.tissue_input)
    _,sv,vt=np.linalg.svd(z/np.sqrt(len(delta)-1),full_matrices=False)
    evr=sv**2/(sv**2).sum()
    pc1pct=summary["pca_explained_variance_ratio"]["PC1"]*100
    pc2pct=summary["pca_explained_variance_ratio"]["PC2"]*100
    if not (np.allclose(evr[:2]*100,[pc1pct,pc2pct],atol=1e-6)):
        raise AssertionError
    own=z@vt.T
    for k in (0,1):
        sign=np.sign(np.dot(own[:,k],scores[f"PC{k+1}"]))
        if not (np.allclose(sign*own[:,k],scores[f"PC{k+1}"],atol=1e-6)):
            raise AssertionError
    sign=np.sign(np.dot(vt[0],load.PC1_loading))
    if not (np.allclose(sign*vt[0],load.PC1_loading,atol=1e-6)):
        raise AssertionError
    pc1=scores.PC1.to_numpy(); pc2=scores.PC2.to_numpy()
    order=np.argsort(pc1,kind="stable")
    clipped=int((abs(z)>3).sum())
    if not (clipped==2):
        raise AssertionError
    fig=s.figure(180,110)
    ax=s.axes(fig,20,8,124,29)
    norm=TwoSlopeNorm(vcenter=0,vmin=-3,vmax=3)
    ax.imshow(np.clip(z[order].T,-3,3),cmap=s.DIV,norm=norm,aspect="auto",interpolation="nearest",
              extent=(0,len(delta),len(s.GENES),0))
    ax.set_yticks(np.arange(10)+.5,s.GENES,fontstyle="italic")
    ax.set_xticks([]); ax.tick_params(axis="y",length=0,pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(.4)
    s.text(fig,"87 patients ordered by increasing PC1 score",82,39,ha="center",fontsize=7)
    cbax=s.axes(fig,151,12,25,2)
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=s.DIV),cax=cbax,orientation="horizontal")
    cb.set_ticks([-3,0,3]); cb.set_label("z score",labelpad=2)
    cb.ax.tick_params(labelsize=7,length=1.5,pad=1); cb.outline.set_linewidth(.4)
    s.text(fig,"Display limits ±3\n2 of 870 values",150,28,fontsize=7,linespacing=1.5)
    s.label(fig,"A",2,5)
    bx=s.axes(fig,19,49,67.5,45)
    tri=np.full((9,9),np.nan)
    for i in range(1,10):
        for j in range(i):
            tri[i-1,j]=corr[i,j]
    bx.imshow(np.ma.masked_invalid(tri),cmap=s.CORR,vmin=0,vmax=1,aspect="auto",interpolation="nearest",extent=(0,9,9,0))
    for i in range(1,10):
        for j in range(i):
            bx.text(j+.5,i-.5,f"{corr[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if corr[i,j]>.6 else s.INK)
    bx.set_xticks(np.arange(9)+.5,s.GENES[:-1],rotation=45,ha="right",rotation_mode="anchor",fontstyle="italic")
    bx.set_yticks(np.arange(9)+.5,s.GENES[1:],fontstyle="italic")
    bx.tick_params(length=0,pad=1.5)
    for spine in bx.spines.values():spine.set_visible(False)
    cbax=s.axes(fig,48,51,29,2)
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=s.CORR),cax=cbax,orientation="horizontal")
    cb.set_ticks([0,.5,1]); cb.set_label("Spearman ρ",labelpad=2); cb.outline.set_linewidth(.4)
    cb.ax.tick_params(labelsize=7,length=1.5,pad=1)
    off=corr[np.triu_indices(10,1)]
    s.text(fig,f"45 gene pairs\nρ {off.min():.2f} to {off.max():.2f}",57,61,fontsize=7,linespacing=1.5)
    s.label(fig,"B",2,46)
    cx=s.axes(fig,104,50,70,40.385)
    cx.axhline(0,color=s.LIGHT,lw=.6,ls="--",zorder=0)
    cx.axvline(0,color=s.LIGHT,lw=.6,ls="--",zorder=0)
    cx.plot(pc1,pc2,linestyle="none",marker="o",markersize=3.5,markerfacecolor=s.GRAY,
            markeredgecolor="white",markeredgewidth=.3)
    cx.set_xlim(-5.2,5.2);cx.set_ylim(-3,3)
    if not ((abs(pc1)<5.2).all() and (abs(pc2)<3).all()):
        raise AssertionError
    cx.set_xticks([-4,-2,0,2,4]);cx.set_yticks([-2,0,2])
    cx.set_xlabel(f"PC1 ({pc1pct:.1f}% of variance)",labelpad=3)
    cx.set_ylabel(f"PC2 ({pc2pct:.1f}% of variance)",labelpad=3)
    s.text(fig,"Each point is one patient",139,98,ha="center",fontsize=7)
    s.label(fig,"C",93,46)
    result=s.save(fig,args.output_dir,"Figure_3",paths,
                  "All 87 patients and 10 gene regions are retained. Heatmap uses per-gene population-SD z scores, "
                  "ordered by the archived PC1 score. Exactly two of 870 heatmap cells are clipped at ±3 for display. "
                  "The archived correlation matrix is used directly. Rounded delta exports can change tied ranks, "
                  "so they are not used to recompute its correlations. The optional tissue input verifies correlations "
                  "from the original precision. PCA scores, variance and loadings reproduce the archived values. No clinical outcome "
                  "column is read, used for sorting, or displayed. PCA axes have equal score-unit scale.",
                  {"n_patients":87,"n_genes":10,"n_correlations":45,"clipped_heatmap_cells":2,
                   "pca_variance_ratio":evr[:2].tolist(),"patient_data_exported":False,
                   "rounded_delta_max_correlation_difference":rounded_delta_correlation_difference,
                   "full_precision_max_correlation_difference":full_precision_difference})
    print(result["figure"],result["outputs"])

if __name__=="__main__":
    main()
