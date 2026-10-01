"""Patient-paired individual-CpG analysis. Never average methylation across CpGs."""
from pathlib import Path
from itertools import combinations
import json,hashlib
import numpy as np
import pandas as pd
from scipy import stats
from analyze_korean_replication import load_beta,load_metadata,bh_adjust,stable_seed,_bca_mean_ci

def require(condition, message="Integrity check failed"):
    if not condition:
        raise RuntimeError(message)

ROOT=Path(__file__).resolve().parents[1]


def paired_deltas(beta,meta):
 verified=meta[meta.pair_verified]
 tumor=verified[verified.tissue.eq('T')].set_index('patient_id')
 normal=verified[verified.tissue.eq('N')].set_index('patient_id')
 patients=sorted(set(tumor.index)&set(normal.index))
 return pd.DataFrame(beta[tumor.loc[patients,'sample_id']].to_numpy().T-beta[normal.loc[patients,'sample_id']].to_numpy().T,index=patients,columns=beta.index)

def paired_effects(delta,probe_gene,cohort,n_boot=5000,seed=20260908):
 rows=[]
 for probe,gene in probe_gene.items():
  a=delta[probe].dropna().to_numpy() if probe in delta else np.array([])
  row={'cohort':cohort,'probe':probe,'gene':gene,'n_pairs':len(a),'mean_delta_pp':np.nan,'sd_delta_pp':np.nan,'ci_low_pp':np.nan,'ci_high_pp':np.nan,'paired_t_p':np.nan,'wilcoxon_p':np.nan,'n_tumor_higher':0,'fraction_tumor_higher':np.nan,'ci_method':'not_estimable','status':'missing_probe' if probe not in delta else 'insufficient_pairs'}
  if len(a)>=3:
   rng=np.random.default_rng(stable_seed(seed,cohort,probe,'paired_cpg'))
   lo,hi,method,_=_bca_mean_ci(a,rng,n_boot)
   if np.all(a==0):tp,wp=1.,1.
   else:
    tp=float(stats.ttest_1samp(a,0).pvalue)
    wp=float(stats.wilcoxon(a,zero_method='wilcox',alternative='two-sided').pvalue)
   row.update(mean_delta_pp=float(a.mean()*100),sd_delta_pp=float(a.std(ddof=1)*100),ci_low_pp=lo*100,ci_high_pp=hi*100,paired_t_p=tp,wilcoxon_p=wp,n_tumor_higher=int((a>0).sum()),fraction_tumor_higher=float((a>0).mean()),ci_method=method,status='estimated')
  rows.append(row)
 out=pd.DataFrame(rows)
 out['se_delta_pp']=out.sd_delta_pp/np.sqrt(out.n_pairs.replace(0,np.nan))
 out['paired_t_q77']=bh_adjust(out.paired_t_p);out['wilcoxon_q77']=bh_adjust(out.wilcoxon_p)
 return out

def corr_matrix(x):
 ranks=stats.rankdata(x,axis=0,method='average')
 return np.corrcoef(ranks,rowvar=False)

def block_permutation(x,groups,rng):
 out=x.copy()
 for group in dict.fromkeys(groups):
  cols=np.flatnonzero(np.asarray(groups)==group)
  out[:,cols]=x[rng.permutation(len(x))[:,None],cols]
 return out

def global_coordination(delta,probe_gene,cohort,n_boot=5000,n_perm=10000,seed=20260908):
 available=[p for p in probe_gene if p in delta and delta[p].notna().sum()>=3 and delta[p].std()>0]
 x=delta[available].dropna().to_numpy();groups=[probe_gene[p] for p in available]
 row={'cohort':cohort,'scope':'individual CpGs, between-gene correlations; no methylation averaging','n_complete_pairs':len(x),'n_available_cpgs':len(available),'n_genes':len(set(groups)),'median_between_gene_rho':np.nan,'ci_low':np.nan,'ci_high':np.nan,'permutation_p':np.nan,'n_boot':n_boot,'n_permutation':n_perm,'status':'insufficient_data'}
 if len(x)<3 or len(set(groups))<2:return row
 ii,jj=np.triu_indices(len(available),1);mask=np.asarray(groups)[ii]!=np.asarray(groups)[jj];ii,jj=ii[mask],jj[mask]
 observed=corr_matrix(x)[ii,jj];rng=np.random.default_rng(stable_seed(seed,cohort,'cpg_coordination'))
 boot=[]
 for _ in range(n_boot):
  r=corr_matrix(x[rng.integers(0,len(x),size=len(x))])[ii,jj]
  if np.isfinite(r).all():boot.append(float(np.median(r)))
 ranks=stats.rankdata(x,axis=0,method='average');null=[]
 for _ in range(n_perm):
  r=np.corrcoef(block_permutation(ranks,groups,rng),rowvar=False)[ii,jj];null.append(float(np.median(r)))
 theta=float(np.median(observed));lo,hi=np.quantile(boot,[.025,.975])
 row.update(median_between_gene_rho=theta,ci_low=float(lo),ci_high=float(hi),permutation_p=float((1+np.sum(np.asarray(null)>=theta))/(1+len(null))),n_between_gene_pairs=len(observed),n_positive_between_gene_pairs=int((observed>0).sum()),valid_bootstraps=len(boot),status='estimated')
 return row

def correlations(delta,probe_gene,cohort):
 rows=[]
 for a,b in combinations(probe_gene,2):
  d=delta[[a,b]].dropna() if a in delta and b in delta else pd.DataFrame()
  rho=p=np.nan
  if len(d)>=3 and d[a].std()>0 and d[b].std()>0:rho,p=stats.spearmanr(d[a],d[b])
  rows.append({'cohort':cohort,'probe_a':a,'gene_a':probe_gene[a],'probe_b':b,'gene_b':probe_gene[b],'family':'within_gene' if probe_gene[a]==probe_gene[b] else 'between_gene','n_pairs':len(d),'rho':rho,'p':p})
 out=pd.DataFrame(rows);out['q']=np.nan
 for family,ids in out.groupby('family').groups.items():out.loc[ids,'q']=bh_adjust(out.loc[ids,'p'])
 return out

def main():
 fixed=json.loads((ROOT/'registry/fixed_probes.json').read_text());mapping={p:g for g,ps in fixed.items() for p in ps}
 contract=json.loads((ROOT/'registry/cpg_analysis_contract.json').read_text());require(len(mapping)==77, 'CpG mapping must contain 77 candidate probes')
 for cohort in ['CMCBSN','SNUH','ASAN']:
  beta=load_beta(ROOT/'data/derived'/f'{cohort}_beta.tsv');meta=load_metadata(ROOT/'data/derived'/f'{cohort}_samples.tsv',beta.columns)
  delta=paired_deltas(beta,meta);out=ROOT/'results/individual_cpg'/cohort;out.mkdir(parents=True,exist_ok=True)
  effects=paired_effects(delta,mapping,cohort,contract['n_boot'],contract['seed']);effects.to_csv(out/'paired_cpg_effects.tsv',sep='\t',index=False)
  delta.index.name='patient_id';delta.to_csv(out/'paired_cpg_deltas.tsv',sep='\t')
  correlations(delta,mapping,cohort).to_csv(out/'cpg_delta_correlations.tsv',sep='\t',index=False)
  summary=global_coordination(delta,mapping,cohort,contract['n_boot'],contract['n_permutation'],contract['seed']);pd.DataFrame([summary]).to_csv(out/'coordination_summary.tsv',sep='\t',index=False)
  desc=[]
  for gene,d in effects.groupby('gene',sort=False):
   e=d[d.status.eq('estimated')]
   desc.append({'cohort':cohort,'gene':gene,'n_target_cpgs':len(d),'n_measured_cpgs':len(e),'n_positive_delta':int(e.mean_delta_pp.gt(0).sum()),'n_positive_q77_lt_005':int((e.mean_delta_pp.gt(0)&e.paired_t_q77.lt(.05)).sum()),'n_negative_q77_lt_005':int((e.mean_delta_pp.lt(0)&e.paired_t_q77.lt(.05)).sum()),'min_delta_pp':e.mean_delta_pp.min(),'max_delta_pp':e.mean_delta_pp.max(),'summary_type':'descriptive counts, not a gene-level hypothesis test'})
  pd.DataFrame(desc).to_csv(out/'gene_locus_descriptive_summary.tsv',sep='\t',index=False)
  print(json.dumps({'cohort':cohort,'estimated':int(effects.status.eq('estimated').sum()),'positive_significant':int((effects.mean_delta_pp.gt(0)&effects.paired_t_q77.lt(.05)).sum()),'coordination':summary}),flush=True)
if __name__=='__main__':main()
