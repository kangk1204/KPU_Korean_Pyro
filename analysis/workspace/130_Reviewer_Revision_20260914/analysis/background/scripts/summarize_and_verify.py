"""Aggregate matching diagnostics, manuscript table and independent arithmetic checks."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from matched_background import ROOT,REG,RESULTS,PRIVATE,OLD,FIXED,sha,FEATURES,SCALE

def main():
 c=pd.read_csv(REG/'candidate_regions.tsv',sep='\t').set_index('slot');p=pd.read_csv(REG/'control_region_pool.tsv',sep='\t');s=pd.read_csv(REG/'frozen_control_sets.tsv',sep='\t');eligible=pd.read_csv(REG/'matching_eligibility.tsv',sep='\t').set_index('slot')
 freq=s[list(FIXED)].melt().value.value_counts();p['times_used_in_5000_sets']=p.region_id.map(freq).fillna(0).astype(int)
 p[['slot','region_id','gene','times_used_in_5000_sets']].to_csv(RESULTS/'control_region_usage.tsv',sep='\t',index=False)
 geneusage=p.groupby('gene').times_used_in_5000_sets.sum().sort_values(ascending=False);geneusage.rename('set_occurrences').to_csv(RESULTS/'control_gene_usage.tsv',sep='\t')
 balance=[];setbal=[];verifications={}
 targets={}
 for cohort in ['ASAN','SNUH']:
  z=np.load(PRIVATE/f'{cohort}_cache.npz');targets[cohort]=(dict(zip(z['probes'],range(len(z['probes'])))),z['delta'])
 for slot in FIXED:
  a=p[p.slot.eq(slot)].copy();t=c.loc[slot];w=a.times_used_in_5000_sets.to_numpy(float);w/=w.sum()
  rec={'candidate_gene':slot,'n_probes':int(t.n_probes),'large_gap_clusters':int(t.cluster_count),'matching_tier':int(eligible.loc[slot,'selected_tier']),'control_genes':len(a),'candidate_source_mean_delta_pp':100*t.mean_delta,'control_source_mean_delta_pp':100*np.dot(w,a.mean_delta),'control_minus_candidate_source_delta_pp':100*(np.dot(w,a.mean_delta)-t.mean_delta),'candidate_source_mean_normal_pp':100*t.mean_normal,'control_source_mean_normal_pp':100*np.dot(w,a.mean_normal),'control_minus_candidate_normal_pp':100*(np.dot(w,a.mean_normal)-t.mean_normal),'candidate_span_bp':np.exp(t.log_span),'control_span_median_bp':np.exp(a.log_span.median()),'control_to_candidate_span_ratio_median':np.exp(a.log_span.median()-t.log_span),'candidate_max_gap_bp':np.exp(t.log_max_gap),'control_max_gap_median_bp':np.exp(a.log_max_gap.median()),'candidate_median_gap_bp':np.exp(t.log_median_gap),'control_median_gap_median_bp':np.exp(a.log_median_gap.median())}
  for feat in ['island','shore','shelf','promoter','design_I','mask']:
   rec[f'candidate_{feat}_fraction']=t[feat];rec[f'control_{feat}_fraction']=np.dot(w,a[feat]);rec[f'control_minus_candidate_{feat}_fraction']=np.dot(w,a[feat])-t[feat]
  for cohort,(idx,d) in targets.items():
   td=d[[idx[x] for x in t.probes.split(';')]];means=[];sds=[]
   for row in a.itertuples():
    ad=d[[idx[x] for x in row.probes.split(';')]];means.append(ad.mean());sds.append(ad.std(axis=1,ddof=1).mean())
   rec[f'{cohort}_candidate_mean_delta_pp']=100*td.mean();rec[f'{cohort}_control_mean_delta_pp']=100*np.dot(w,means);rec[f'{cohort}_control_minus_candidate_delta_pp']=100*(np.dot(w,means)-td.mean());rec[f'{cohort}_candidate_mean_probe_sd_delta_pp']=100*td.std(axis=1,ddof=1).mean();rec[f'{cohort}_control_mean_probe_sd_delta_pp']=100*np.dot(w,sds)
  balance.append(rec)
 b=pd.DataFrame(balance);b.to_csv(RESULTS/'source_and_target_balance_by_slot.tsv',sep='\t',index=False)
 pidx=p.set_index('region_id')
 for row in s.itertuples(index=False):
  a=pidx.loc[list(row[1:])];x=a[FEATURES].to_numpy()-c.loc[list(FIXED),FEATURES].to_numpy();scaled=x/SCALE
  setbal.append({'set_id':row.set_id,'source_mean_delta_difference_pp_gene_balanced':float(x[:,8].mean()*100),'source_mean_normal_difference_pp_gene_balanced':float(x[:,7].mean()*100),'rms_caliper_scaled_imbalance':float(np.sqrt(np.mean(scaled**2))),'largest_absolute_scaled_difference':float(np.abs(scaled).max())})
 sb=pd.DataFrame(setbal);sb.to_csv(RESULTS/'source_balance_by_set.tsv',sep='\t',index=False)
 frozen=json.loads((REG/'SELECTION_FREEZE.json').read_text())
 checks={'frozen_hashes_match':all(sha(REG/f)==h for f,h in frozen['files'].items()),'n_sets':len(s),'unique_sets':not s[list(FIXED)].duplicated().any(),'unique_control_genes':p.gene.nunique(),'unique_control_probes':len(set(x for ps in p.probes for x in ps.split(';'))),'selection_frozen_before_targets':True,'all_sets_valid':True,'n_cpG_pairs':1381,'gene_pair_blocks':45}
 for row in s.itertuples(index=False):
  rows=pidx.loc[list(row[1:])];probes=[x for ps in rows.probes for x in ps.split(';')]
  if not (len(probes)==len(set(probes))==56 and rows.gene.nunique()==10 and not set(rows.gene)&set(FIXED)):
      raise AssertionError
  if not (list(rows.n_probes)==list(c.loc[list(FIXED),'n_probes'])):
      raise AssertionError
  if not (list(rows.cluster_count)==list(c.loc[list(FIXED),'cluster_count'])):
      raise AssertionError
 summaries=[]
 for cohort,(idx,d) in targets.items():
  result=json.loads((RESULTS/f'{cohort}_summary.json').read_text());raw=json.loads((REG/f'{cohort}_raw_audit.json').read_text())
  if not (raw['time_utc']>frozen['time_utc']):
      raise AssertionError
  pub=pd.read_csv(OLD/f'results/individual_cpg/{cohort}/cpg_delta_correlations.tsv',sep='\t');pub=pub[pub.family.eq('between_gene')&pub.rho.notna()]
  rhos=[]
  for row in pub.itertuples():rhos.append(spearmanr(d[idx[row.probe_a]],d[idx[row.probe_b]]).statistic)
  maxerror=np.max(np.abs(np.array(rhos)-pub.rho.to_numpy()))
  if not (maxerror<1e-12 and len(pub)==1381):
      raise AssertionError
  balanced=float(pub.groupby(['gene_a','gene_b']).rho.median().median())
  if not (abs(balanced-result['primary_gene_pair_balanced']['candidate'])<1e-12):
      raise AssertionError
  setsstats=pd.read_csv(RESULTS/f'{cohort}_set_statistics.tsv',sep='\t')
  if not (len(setsstats)==5001):
      raise AssertionError
  boots=pd.read_csv(RESULTS/f'{cohort}_patient_bootstrap_statistics.tsv',sep='\t');ci=np.quantile(boots.difference,[.025,.975])
  if not (np.allclose(ci,result['patient_bootstrap']['difference_ci'])):
      raise AssertionError
  verifications[cohort]={'raw_hash_matches_historical':raw['historical_sha256_matches'],'n_pairwise_correlations_recomputed_individually_with_scipy':len(pub),'max_abs_difference_from_historical_rho':float(maxerror),'gene_pair_balanced_matches_independent_historical_regroup':True,'bootstrap_ci_recomputed':True,'source_freeze_precedes_target_read_complete':raw['time_utc']>frozen['time_utc'],'evaluable_sets':result['evaluable_sets'],'pairs':result['pairs']}
  st=result['primary_gene_pair_balanced'];bo=result['patient_bootstrap'];summaries.append({'cohort':cohort,'paired_patients':result['pairs'],'candidate_rho':st['candidate'],'candidate_ci_low':bo['candidate_ci'][0],'candidate_ci_high':bo['candidate_ci'][1],'full_reference_n':result['evaluable_sets'],'full_reference_median_rho':st['background_median'],'full_reference_q025':st['background_q025'],'full_reference_q975':st['background_q975'],'full_reference_point_difference_no_ci':st['candidate_minus_background_median'],'background_sets_ge_candidate_count':int(round(st['fraction_background_ge_candidate']*result['evaluable_sets'])),'empirical_upper_fraction_not_p':st['fraction_background_ge_candidate'],'empirical_candidate_percentile':st['candidate_midrank_percentile'],'bootstrap_reference_n':bo['background_reference_sets'],'bootstrap_reference_median_rho':bo['reference_subset_median_point'],'bootstrap_reference_point_difference':bo['difference_to_reference_subset_point'],'bootstrap_reference_difference_ci_low':bo['difference_ci'][0],'bootstrap_reference_difference_ci_high':bo['difference_ci'][1],'patient_bootstrap_resamples':bo['n_resamples'],'probe_pair_weighted_candidate_rho':result['sensitivity_probe_pair_median']['candidate'],'qc_clean_candidate_rho':result['sensitivity_exclude_mask_general']['candidate']})
 pd.DataFrame(summaries).to_csv(RESULTS/'Table_S7_matched_background.tsv',sep='\t',index=False)
 info={'n_regions':len(p),'n_unique_genes':int(p.gene.nunique()),'n_unique_control_probes':checks['unique_control_probes'],'tier1_slots':b.loc[b.matching_tier.eq(1),'candidate_gene'].tolist(),'tier2_slots':b.loc[b.matching_tier.eq(2),'candidate_gene'].tolist(),'tier3_slots':b.loc[b.matching_tier.eq(3),'candidate_gene'].tolist(),'source_delta_gap_pp_range':[float(b.control_minus_candidate_source_delta_pp.min()),float(b.control_minus_candidate_source_delta_pp.max())],'source_normal_gap_pp_range':[float(b.control_minus_candidate_normal_pp.min()),float(b.control_minus_candidate_normal_pp.max())],'source_set_mean_delta_gap_pp_median':float(sb.source_mean_delta_difference_pp_gene_balanced.median()),'source_set_mean_delta_gap_pp_q025_q975':np.quantile(sb.source_mean_delta_difference_pp_gene_balanced,[.025,.975]).tolist(),'interpretability':'limited: systematic lower control hypermethylation remains, despite source-frozen annotation/geometry/distribution matching; higher target correlations cannot distinguish panel specificity from residual methylation-magnitude/distribution differences'}
 (RESULTS/'balance_summary.json').write_text(json.dumps(info,indent=2)+'\n');(RESULTS/'VERIFICATION.json').write_text(json.dumps({'checks':checks,'cohorts':verifications},indent=2)+'\n')
 print(json.dumps({'verification':checks,'balance':info},indent=2))

if __name__=='__main__':main()
