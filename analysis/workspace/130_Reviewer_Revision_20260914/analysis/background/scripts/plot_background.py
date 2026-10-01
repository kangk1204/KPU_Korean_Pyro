"""Figure S5: conditional reference comparison and shared-patient uncertainty.
All observations are aggregate outputs; no subject-level data are loaded here.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
T=pd.read_csv(ROOT/'results/Table_S7_matched_background.tsv',sep='\t')
blue='#276187';gray='#C9CED2';dark='#56626B'
fig,(ax,bx)=plt.subplots(1,2,figsize=(7.08661417323,3.54330708661),gridspec_kw={'width_ratios':[1.6,1]})
fig.subplots_adjust(left=.14,right=.97,bottom=.31,top=.80,wspace=.37)
source=[]
for j,row in T.iterrows():
 y=1-j
 data=pd.read_csv(ROOT/f'results/{row.cohort}_set_statistics.tsv',sep='\t')
 values=data.loc[data.set_id.ne('FIXED10'),'gene_pair_balanced_median_rho'].to_numpy()
 violin=ax.violinplot([values],positions=[y],vert=False,widths=.50,showextrema=False)
 for body in violin['bodies']:body.set_facecolor(gray);body.set_edgecolor(dark);body.set_alpha(.8);body.set_linewidth(.5)
 ax.plot([row.full_reference_q025,row.full_reference_q975],[y,y],color=dark,lw=1)
 ax.plot(row.full_reference_median_rho,y,marker='|',color=dark,ms=7,mew=1.3)
 ax.errorbar(row.candidate_rho,y+.12,xerr=[[row.candidate_rho-row.candidate_ci_low],[row.candidate_ci_high-row.candidate_rho]],fmt='o',color=blue,ms=4,capsize=2,lw=1)
 bx.errorbar(row.bootstrap_reference_point_difference,y,xerr=[[row.bootstrap_reference_point_difference-row.bootstrap_reference_difference_ci_low],[row.bootstrap_reference_difference_ci_high-row.bootstrap_reference_point_difference]],fmt='o',color=blue,ms=4,capsize=2,lw=1.1)
 ax.text(.99,y-.32,f'{int(row.background_sets_ge_candidate_count):,}/{int(row.full_reference_n):,} sets ≥ candidate',ha='right',va='center',fontsize=7)
 source.append({'cohort':row.cohort,'paired_patients':int(row.paired_patients),'background_sets':len(values),'candidate_rho':row.candidate_rho,'candidate_95CI':f'{row.candidate_ci_low:.6f};{row.candidate_ci_high:.6f}','full_reference_point_difference':row.full_reference_point_difference_no_ci,'bootstrap_reference_point_difference':row.bootstrap_reference_point_difference,'bootstrap_difference_95CI':f'{row.bootstrap_reference_difference_ci_low:.6f};{row.bootstrap_reference_difference_ci_high:.6f}'})
ax.set_xlim(0,1);ax.set_ylim(-.53,1.53);bx.set_ylim(ax.get_ylim())
ax.set_yticks([1,0],[f'{r.cohort}\n{int(r.paired_patients)} pairs' for r in T.itertuples()]);bx.set_yticks([1,0],['ASAN','SNUH'])
ax.set_xlabel('Gene-pair-balanced median Spearman ρ');bx.set_xlabel('Candidate − reference median ρ')
ax.set_title('Matched-set distribution',loc='left',pad=15);bx.set_title('Patient-bootstrap difference',loc='left',pad=15)
bx.set_xlim(-.05,.25);bx.set_xticks([0,.1,.2]);bx.axvline(0,color='#929A9F',linestyle='--',linewidth=.8,zorder=0)
for axis in (ax,bx):axis.tick_params(axis='y',length=0);axis.spines['left'].set_visible(False)
fig.text(.022,.86,'A',fontweight='bold',fontsize=9);fig.text(.63,.86,'B',fontweight='bold',fontsize=9)
fig.legend(handles=[Line2D([0],[0],color=gray,lw=7,label='5,000 matched sets (distribution)'),Line2D([0],[0],color=blue,marker='o',ms=4,lw=1,label='Candidate / difference (95% patient-bootstrap CI)')],loc='lower left',bbox_to_anchor=(.12,.105),ncol=1,borderaxespad=0,handlelength=1.8)
fig.text(.14,.065,'Matching used CMCBSN only. Residual methylation-effect imbalance remains.',fontsize=7)
fig.text(.14,.018,'Panel B uses a fixed 256-set reference subset. Empirical set fractions are not P values.',fontsize=7)
fig.savefig(OUT/'Figure_S5_matched_background.svg')
fig.savefig(OUT/'Figure_S5_matched_background.pdf')
fig.savefig(OUT/'Figure_S5_matched_background.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(OUT/'Figure_S5_matched_background.png',dpi=600)
pd.DataFrame(source).to_csv(OUT/'Figure_S5_source_summary.tsv',sep='\t',index=False)
plt.close(fig)
