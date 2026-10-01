"""Reproducible individual-CpG effects and delta-correlation figure."""
from pathlib import Path
import json,shutil,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'registry/fixed_probes.json').exists())
COHORTS=['CMCBSN','SNUH','ASAN']
GENE_ORDER=['EYA4','ZNF568','ZNF793','SFMBT2','ADHFE1','HOXA2','BEND5','UNC5C','RALYL','GFRA1']
EXPORT_SUFFIXES=('.png','.pdf','.svg')
def main():
 out=ROOT/'figures/Korean_CpG';out.mkdir(parents=True,exist_ok=True)
 fixed=json.loads((ROOT/'registry/fixed_probes.json').read_text());mapping={p:g for g,ps in fixed.items() for p in ps}
 ann=pd.read_csv(ROOT.parent/'120_Korean_External_Validation_20260908/registry/probe_context_hg19.tsv',sep='\t').set_index('probe')
 probes=[p for gene in GENE_ORDER for p in sorted(fixed[gene],key=lambda p:int(ann.loc[p,'pos']))]
 effects=pd.concat([pd.read_csv(ROOT/'results/individual_cpg'/c/'paired_cpg_effects.tsv',sep='\t') for c in COHORTS],ignore_index=True)
 effects.to_csv(out/'effects_source.tsv',sep='\t',index=False)
 coords=pd.concat([pd.read_csv(ROOT/'results/individual_cpg'/c/'coordination_summary.tsv',sep='\t') for c in COHORTS],ignore_index=True);coords.to_csv(out/'coordination_source.tsv',sep='\t',index=False)
 correlations=[]
 for c in COHORTS:
  d=pd.read_csv(ROOT/'results/individual_cpg'/c/'cpg_delta_correlations.tsv',sep='\t');correlations.append(d)
 pd.concat(correlations,ignore_index=True).to_csv(out/'correlations_source.tsv',sep='\t',index=False)
 ann.loc[probes].reset_index().to_csv(out/'probe_annotation_source.tsv',sep='\t',index=False)
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans','sans-serif'],'font.size':6.0,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig=plt.figure(figsize=(7.05,9.35));gs=gridspec.GridSpec(3,2,figure=fig,width_ratios=[1.55,1.0],hspace=.52,wspace=.48)
 ax=fig.add_subplot(gs[:,0]);matrix=effects.pivot(index='probe',columns='cohort',values='mean_delta_pp').reindex(index=probes,columns=COHORTS)
 cmap=plt.colormaps['RdBu_r'].copy();cmap.set_bad('#e3e3e3')
 im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=-50,vmax=50)
 ax.set_yticks(range(77),probes,fontsize=5.0);ax.set_xticks(range(3),['CMCBSN\n103 pairs','SNUH\n142 pairs','ASAN\n128 pairs'],fontsize=6.0)
 ax.tick_params(length=0);ax.set_title('A  CpG-specific methylation changes',loc='left',fontweight='bold',fontsize=7.2,pad=8)
 offset=0
 for gene in GENE_ORDER:
  ps=fixed[gene];mid=offset+(len(ps)-1)/2;ax.text(-0.22,mid,gene,transform=ax.get_yaxis_transform(),ha='right',va='center',fontstyle='italic',fontsize=5.8,clip_on=False)
  offset+=len(ps)
  if offset<77:ax.axhline(offset-.5,color='white',lw=1.2)
 cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.025,pad=.060);cb.set_label('Tumor - adjacent mucosa (percentage points)',fontsize=5.8);cb.ax.tick_params(labelsize=5.0)
 for i,c in enumerate(COHORTS):
  a=fig.add_subplot(gs[i,1]);d=correlations[i];available=[p for p in probes if effects[(effects.cohort==c)&(effects.probe==p)].status.iloc[0]=='estimated']
  m=pd.DataFrame(np.eye(len(available)),index=available,columns=available)
  for row in d.itertuples():
   if row.probe_a in m.index and row.probe_b in m.index:m.loc[row.probe_a,row.probe_b]=m.loc[row.probe_b,row.probe_a]=row.rho
  m.to_csv(out/f'{c}_correlation_matrix.tsv',sep='\t')
  im2=a.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest')
  positions=[];labels=[];start=0
  for gene in GENE_ORDER:
   n=sum(mapping[p]==gene for p in available);positions.append(start+(n-1)/2);labels.append(gene);start+=n
   if start<len(available):a.axhline(start-.5,color='white',lw=.5);a.axvline(start-.5,color='white',lw=.5)
  a.set_xticks(positions,labels,rotation=90,rotation_mode='anchor',ha='right',fontsize=5);a.set_yticks(positions,labels,fontsize=5)
  for label in a.get_xticklabels()+a.get_yticklabels():label.set_fontstyle('italic')
  a.tick_params(length=0);r=coords[coords.cohort==c].iloc[0]
  a.set_title(f'{chr(66+i)}  {c}\nMedian between-gene rho = {r.median_between_gene_rho:.2f}',loc='left',fontsize=6.5,fontweight='bold',pad=5)
 cax=fig.add_axes([.72,.035,.2,.010]);fig.colorbar(im2,cax=cax,orientation='horizontal').set_label('Spearman rho',fontsize=5.8);cax.tick_params(labelsize=5.0)
 fig.subplots_adjust(left=.27,right=.98,top=.965,bottom=.10)
 for suffix in EXPORT_SUFFIXES:
  target=out/f'Korean_CpG{suffix}';fig.savefig(target,dpi=600 if suffix=='.png' else None,facecolor='white')
  if suffix=='.svg':target.write_text(re.sub(r'<!DOCTYPE[^>]*>','',target.read_text(),count=1,flags=re.S))
 plt.close(fig)
 (out/'legend.md').write_text('''Individual CpG methylation changes in three Korean colorectal cancer cohorts. (A) Mean within-patient tumor-minus-adjacent-mucosa differences at the 77 fixed candidate positions, grouped by gene and genomic position. Gray cells indicate the 21 positions absent from each deposited processed matrix. All 56 measured positions were hypermethylated with BH-adjusted q<0.05 in each cohort. (B–D) Spearman correlations between patient-level methylation changes at the 56 measured CpGs. Labels delimit gene loci; each matrix row and column represents one CpG. The displayed summary is the median correlation between CpGs assigned to different genes. No methylation values were averaged across CpGs. Sample sizes were 103, 142 and 128 patient pairs. Per-CpG confidence intervals and exact statistical results are supplied in the source tables.\n''')
 if Path(__file__).resolve()!=(out/'plot_korean_cpg.py').resolve():shutil.copy2(Path(__file__),out/'plot_korean_cpg.py')
 print(out.resolve())
if __name__=='__main__':main()
