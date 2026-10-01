"""Plot all prespecified reviewer comparisons from aggregate result tables."""
from pathlib import Path
import hashlib
import json
import matplotlib

def require(condition, message="Integrity check failed"):
    if not condition:
        raise RuntimeError(message)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/reviewer'
GENES = ['EYA4','ZNF568','ZNF793','SFMBT2','ADHFE1','HOXA2','BEND5','UNC5C','RALYL','GFRA1']
BLUE = '#226A95'
GRAY = '#888888'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
    'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})


def interval(ax, row, y, color, marker):
    value = float(row['rho'] if 'rho' in row.index else row['estimate'])
    low, high = float(row.ci_low), float(row.ci_high)
    if np.isfinite([value, low, high]).all():
        require(low <= high, 'Integrity check failed: low <= high')
        ax.plot([low,high],[y,y],color=color,lw=.9,zorder=2)
        ax.scatter([value],[y],c=color,marker=marker,s=15,zorder=3)
    else:
        ax.text(.02,y,'NE',transform=ax.get_yaxis_transform(),va='center',fontsize=7)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{stem}.pdf', dpi=300, facecolor='white')
    fig.savefig(OUT / f'{stem}.svg', dpi=300, facecolor='white')
    fig.savefig(OUT / f'{stem}.png', dpi=300, facecolor='white')
    plt.close(fig)


def promoter():
    data = pd.read_csv(ROOT/'results/reviewer/promoter_associations.tsv',sep='\t')
    ctx = pd.read_csv(ROOT/'registry/reviewer/probe_context.tsv',sep='\t')
    counts = ctx.loc[ctx.promoter_match].groupby('fixed_gene').size()
    require(len(data) == 60 and set(data.gene) == set(GENES), 'Integrity check failed: len(data) == 60 and set(data.gene) == set(GENES)')
    specs = [('Colonomics','unadjusted','Colonomics\nUnadjusted'),
             ('Colonomics','rank_residual_age_sex_site_stroma','Colonomics\nClinical and stromal adjustment'),
             ('ColoCare (discovery overlap)','unadjusted','ColoCare\nSupportive; discovery overlap')]
    fig, axs = plt.subplots(1,3,figsize=(7.204724,4.566929),sharey=True)
    fig.subplots_adjust(left=.16,right=.99,bottom=.17,top=.83,wspace=.19)
    for j,(cohort,analysis,title) in enumerate(specs):
        ax=axs[j]; sub=data.loc[data.cohort.eq(cohort)&data.analysis.eq(analysis)]
        require(len(sub) == 20, 'Integrity check failed: len(sub) == 20')
        for i,gene in enumerate(GENES):
            rows=sub.loc[sub.gene.eq(gene)]
            require(rows.n.nunique() == 1, 'Integrity check failed: rows.n.nunique() == 1')
            for score,offset,color,marker in [('full',.13,GRAY,'s'),('promoter',-.13,BLUE,'o')]:
                row=rows.loc[rows.score_set.eq(score)].iloc[0]
                interval(ax,row,i+offset,color,marker)
        ax.axvline(0,color='#bbbbbb',lw=.65,ls='--',zorder=0)
        require(sub.ci_low.min() >= -0.85 and sub.ci_high.max() <= 0.6, 'Integrity check failed: sub.ci_low.min() >= -0.85 and sub.ci_high.max() <= 0.6')
        ax.set_xlim(-.85,.6); ax.set_xticks([-.8,-.4,0,.4])
        ax.set_title(title,pad=12)
        ax.set_xlabel('Rank correlation with expression',labelpad=7)
        ax.text(-.05,1.1,'abc'[j],transform=ax.transAxes,fontweight='bold',fontsize=9)
        ax.set_yticks(range(10))
        ax.set_yticklabels([f'{g} ({counts[g]})' for g in GENES])
        ax.tick_params(axis='y',length=0)
    axs[0].invert_yaxis()
    fig.legend(handles=[Line2D([],[],color=GRAY,marker='s',label='All fixed probes',lw=.9,markersize=3.5),
                        Line2D([],[],color=BLUE,marker='o',label='Promoter probes',lw=.9,markersize=3.5)],
               loc='lower center',bbox_to_anchor=(.56,.015),ncol=2,fontsize=7)
    save(fig,'S1_promoter_expression')


def stage():
    metrics=pd.read_csv(ROOT/'results/reviewer/stage_metrics.tsv',sep='\t')
    diffs=pd.read_csv(ROOT/'results/reviewer/stage_differences.tsv',sep='\t')
    metrics=metrics.loc[metrics.metric.eq('uno_c')]
    diffs=diffs.loc[diffs.metric.eq('uno_c')]
    require(len(metrics) == 6 and len(diffs) == 3, 'Integrity check failed: len(metrics) == 6 and len(diffs) == 3')
    fig,axs=plt.subplots(1,2,figsize=(7.204724,3.346457),gridspec_kw={'width_ratios':[1.1,1]})
    fig.subplots_adjust(left=.16,right=.97,bottom=.27,top=.76,wspace=.4)
    labels=['Recorded stage','TNM-derived stage','Stage omitted']
    for i,key in enumerate(['provider','tnm','none']):
        for block,offset,color,marker in [('clinical',.12,GRAY,'s'),('combined',-.12,BLUE,'o')]:
            row=metrics.loc[metrics.stage_definition.eq(key)&metrics.block.eq(block)].iloc[0]
            interval(axs[0],row,i+offset,color,marker)
        interval(axs[1],diffs.loc[diffs.stage_definition.eq(key)].iloc[0],i,BLUE,'o')
    for j,ax in enumerate(axs):
        ax.set_yticks(range(3)); ax.set_ylim(2.5,-.5)
        ax.set_yticklabels(labels if j==0 else [])
        ax.tick_params(axis='y',length=0)
        ax.text(-.07,1.16,'ab'[j],transform=ax.transAxes,fontweight='bold',fontsize=9)
    axs[0].set_xlim(.2,.9); axs[0].set_xticks([.2,.4,.6,.8]); axs[0].set_xlabel('Five-year Uno concordance index')
    axs[0].set_title('Clinical and combined models',pad=14)
    lower=min(-.3,float(diffs.ci_low.min())-.03)
    upper=max(.3,float(diffs.ci_high.max())+.03)
    axs[1].set_xlim(lower,upper); axs[1].axvline(0,color='#bbbbbb',ls='--',lw=.65)
    axs[1].set_title('Change after adding methylation',pad=14)
    axs[1].set_xlabel('Combined minus clinical Uno C')
    fig.legend(handles=[Line2D([],[],color=GRAY,marker='s',label='Clinical',lw=.9,markersize=3.5),
                        Line2D([],[],color=BLUE,marker='o',label='Clinical + methylation',lw=.9,markersize=3.5)],
               loc='lower center',bbox_to_anchor=(.55,.04),ncol=2,fontsize=7)
    save(fig,'S2_stage_sensitivity')


def main():
    promoter(); stage()
    sources=['results/reviewer/promoter_associations.tsv','registry/reviewer/probe_context.tsv',
             'results/reviewer/stage_metrics.tsv','results/reviewer/stage_differences.tsv']
    manifest={'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('S*')},
              'width_mm':183,'minimum_font_pt':7,'raster_dpi':300,
              'exclusions':'None of the prespecified ten genes or three stage definitions is omitted.'}
    (ROOT/'verification/reviewer_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
