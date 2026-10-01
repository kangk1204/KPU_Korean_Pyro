"""Render public extension figures directly from frozen result and source tables."""
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,GENES,stable_seed

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
    'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'crc_public_20260905',
    'axes.spines.top':False,'axes.spines.right':False,
    'savefig.dpi':300,'figure.dpi':120})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
TISSUE_COLORS={'H':'#0072B2','N':'#56B4E9','A':'#E69F00','T':'#D55E00'}
TISSUE_LABELS={'H':'Healthy donor','N':'Cancer-adjacent','A':'Adenoma','T':'Primary cancer'}


def letter(ax,s):
    ax.text(-.13,1.08,s,transform=ax.transAxes,fontsize=13,fontweight='bold',va='top')


def forest(ax,frame,group_col,groups,xlabel,scale=100,legend=True):
    offsets=np.linspace(-.23,.23,len(groups)) if len(groups)>1 else [0]
    for j,g in enumerate(groups):
        sub=frame[frame[group_col]==g].set_index('gene').reindex(GENES)
        if sub['effect'].notna().any():
            ax.scatter([],[],c=COLORS[j%len(COLORS)],s=17,label=g)
        for i,gene in enumerate(GENES):
            r=sub.loc[gene]
            if pd.isna(r.get('effect')):continue
            y=i+offsets[j];val=float(r.effect)*scale
            lo=float(r.ci_low)*scale;hi=float(r.ci_high)*scale
            ax.plot([lo,hi],[y,y],color=COLORS[j%len(COLORS)],lw=1.1)
            ax.scatter(val,y,c=COLORS[j%len(COLORS)],s=17,zorder=3)
    ax.axvline(0,color='#777777',lw=.8,ls='--')
    ax.set_yticks(range(len(GENES)),GENES,fontstyle='italic');ax.invert_yaxis()
    ax.set_xlabel(xlabel);ax.grid(axis='x',color='#eeeeee',lw=.7);ax.set_axisbelow(True)
    if legend:ax.legend(frameon=False,fontsize=7,loc='best')


def boxpoints(ax,frame,groupcol,groups,labels,colors):
    vals=[frame.loc[frame[groupcol].astype(str)==str(g),'panel_mean'].dropna().values*100 for g in groups]
    bp=ax.boxplot(vals,positions=np.arange(len(groups)),widths=.5,showfliers=False,patch_artist=True,
                  medianprops={'color':'black','lw':1.2})
    rng=np.random.default_rng(stable_seed(groupcol,'points'))
    for i,(v,c) in enumerate(zip(vals,colors)):
        bp['boxes'][i].set(facecolor=c,alpha=.25,edgecolor=c)
        ax.scatter(i+rng.uniform(-.15,.15,len(v)),v,s=9,c=c,alpha=.6,edgecolors='none')
    ax.set_xticks(np.arange(len(groups)),[f'{s}\n(n={len(v)})' for s,v in zip(labels,vals)])
    ax.set_ylabel('Mean of ten gene scores (%)');ax.set_ylim(0,100)
    ax.grid(axis='y',color='#eeeeee',lw=.7);ax.set_axisbelow(True)


def save(fig,stem,sources):
    fig.savefig(ROOT/f'figures/{stem}.png',bbox_inches='tight')
    fig.savefig(ROOT/f'figures/{stem}.pdf',bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(ROOT/f'figures/{stem}.svg',bbox_inches='tight',metadata={'Date':None})
    svg=ROOT/f'figures/{stem}.svg'
    # Matplotlib's external SVG1.1 DTD is unnecessary for this self-contained file.
    svg.write_text(re.sub(r'<!DOCTYPE svg PUBLIC.*?>\s*','',svg.read_text(),count=1,flags=re.S))
    plt.close(fig)
    with pd.ExcelWriter(ROOT/f'figures/source_data/{stem}.xlsx',engine='openpyxl') as w:
        for name,df in sources.items():df.to_excel(w,sheet_name=name[:31],index=False)


def main():
    samples=pd.read_csv(ROOT/'data/derived/all_public_analysis.tsv',sep='\t')
    effects=pd.read_csv(ROOT/'results/tissue_contrasts.tsv',sep='\t')
    meta=pd.read_csv(ROOT/'results/meta_analysis.tsv',sep='\t')
    context=pd.read_csv(ROOT/'results/molecular_context.tsv',sep='\t')
    expr=pd.read_csv(ROOT/'results/expression_associations.tsv',sep='\t')
    # Figure 8: keep measured healthy and adjacent mucosa distinct.
    fig,axs=plt.subplots(1,2,figsize=(11.8,5.5),gridspec_kw={'width_ratios':[1.15,1]},layout='constrained')
    col=samples[samples.cohort=='Colonomics']
    means=col.groupby('tissue')[GENES].mean().reindex(['H','N','T'])*100
    im=axs[0].imshow(means.T,vmin=0,vmax=80,aspect='auto',cmap='cividis')
    axs[0].set_yticks(range(10),GENES,fontstyle='italic')
    axs[0].set_xticks(range(3),['Healthy\n(n=48)','Adjacent\n(n=96)','Tumor\n(n=96)'])
    for i in range(10):
        for j in range(3):
            val=means.iloc[j,i];axs[0].text(j,i,f'{val:.1f}',ha='center',va='center',fontsize=8,color='white' if val<40 else 'black')
    axs[0].set_title('Colonomics: mean gene methylation');fig.colorbar(im,ax=axs[0],label='Mean beta × 100 (%)',shrink=.65)
    healthy=effects[effects.contrast=='N-H']
    forest(axs[1],healthy,'cohort',['Colonomics','GSE48684','GSE42752'],'Adjacent minus healthy (percentage points)')
    axs[1].set_title('Healthy-reference differences across cohorts')
    for ax,l in zip(axs,'AB'):letter(ax,l)
    save(fig,'F8_healthy_reference',{'A_means':means.reset_index(),'A_individuals':col[['sample','patient','tissue']+GENES],
                                  'B_effects':healthy,'meta_N-H':meta[meta.contrast=='N-H']})
    # Figure 9: cross-sectional lesions and independent paired replication.
    fig,axs=plt.subplots(2,2,figsize=(11.8,9.6),layout='constrained')
    a=samples[samples.cohort=='GSE48684'];b=samples[samples.cohort=='GSE77954']
    boxpoints(axs[0,0],a,'tissue',['H','N','A','T'],['Healthy','Adjacent','Adenoma','Cancer'],[TISSUE_COLORS[x] for x in ['H','N','A','T']])
    axs[0,0].set_title('GSE48684: cross-sectional tissue groups')
    boxpoints(axs[0,1],b,'tissue',['A','T'],['Adenoma','Primary cancer'],[TISSUE_COLORS[x] for x in ['A','T']])
    axs[0,1].set_title('GSE77954: lesion comparison')
    lesion=effects[effects.contrast=='T-A']
    forest(axs[1,0],lesion,'cohort',['GSE48684','GSE77954'],'Cancer minus adenoma (percentage points)')
    axs[1,0].set_title('Gene-specific lesion differences')
    paired=effects[(effects.contrast=='T-N')&effects.paired.astype(str).str.lower().eq('true')].copy()
    pooled=meta[meta.contrast=='T-N'].copy();pooled['cohort']='REML summary'
    fp=pd.concat([paired,pooled],ignore_index=True)
    forest(axs[1,1],fp,'cohort',list(paired.cohort.unique())+(['REML summary'] if len(pooled) else []),'Tumor minus adjacent (percentage points)')
    axs[1,1].set_title('Paired array effects; Korean PSQ analyzed separately')
    for ax,l in zip(axs.flat,'ABCD'):letter(ax,l)
    save(fig,'F9_lesions_replication',{'A_individuals':a[['sample','patient','tissue','panel_mean']],
        'B_individuals':b[['sample','patient','tissue','panel_mean']],'C_effects':lesion,'D_effects':fp})
    # Figure 10: within-tumor biology; no pooled tumor-normal correlation.
    fig,axs=plt.subplots(2,2,figsize=(11.8,9.6),layout='constrained')
    ep=expr.copy()
    if 'effect' not in ep:ep['effect']=ep['rho']
    forest(axs[0,0],ep,'cohort',list(ep.cohort.unique()),'Tumor-only methylation–expression rho',scale=1)
    axs[0,0].set_xlim(-1,1);axs[0,0].set_title('Matched tumor methylation and expression')
    st=context[context.analysis=='stromal_spearman']
    forest(axs[0,1],st,'cohort',['Colonomics'],'Methylation–stromal score rho',scale=1,legend=False)
    axs[0,1].set_xlim(-1,1);axs[0,1].set_title('Tumor tissue composition association')
    c=col[col.tissue=='T'].copy()
    boxpoints(axs[1,0],c,'cms',['CMS1','CMS2','CMS3','CMS4'],['CMS1','CMS2','CMS3','CMS4'],COLORS[:4])
    axs[1,0].set_title('Colonomics: reported CMS groups')
    m=samples[(samples.cohort=='GSE164811')&(samples.tissue=='T')].copy()
    m['cms']=m.cms.astype(str).str.replace('CMS','',regex=False).str.replace('.0','',regex=False)
    match=context[context.analysis=='CMS3-CMS2']
    forest(axs[1,1],match,'cohort',['GSE164811'],'CMS3 minus CMS2 (percentage points)',legend=False)
    axs[1,1].set_title('MATCH: CMS3 (n=22) versus CMS2 (n=124)')
    axs[1,1].text(.98,.655,'Not estimable: <80% valid probes',transform=axs[1,1].transAxes,
        ha='right',va='center',fontsize=7,color='#555555')
    for ax,l in zip(axs.flat,'ABCD'):letter(ax,l)
    save(fig,'F10_expression_context',{'A_expression':ep,'B_stroma':st,'C_CMS':c[['sample','patient','cms','panel_mean']],
        'D_MATCH_effects':match,'D_MATCH_individuals':m[['sample','patient','cms']+GENES]})
    print('Rendered F8–F10 as PNG/PDF/SVG with exact source workbooks.')


if __name__=='__main__':main()
