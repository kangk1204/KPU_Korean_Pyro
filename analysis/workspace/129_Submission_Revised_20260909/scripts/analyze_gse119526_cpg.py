"""Supplemental individual-CpG paired analysis from the same public ML inputs."""
from pathlib import Path
import json
import pandas as pd
from analyze_public_cpg_ml import read_fixed_probes,load_gse119526
from analyze_korean_cpg import paired_effects
ROOT=Path(__file__).resolve().parents[1]
def main():
 fixed=read_fixed_probes();frame,qc,sources=load_gse119526(fixed)
 features=fixed.feature.tolist();normal=frame[frame.y.eq(0)].set_index('study_id')[features];tumor=frame[frame.y.eq(1)].set_index('study_id')[features]
 delta=tumor-normal;delta.columns=fixed.cpg
 result=paired_effects(delta,dict(zip(fixed.cpg,fixed.gene)),'GSE119526',5000,20260908)
 out=ROOT/'results/gse119526_support';out.mkdir(parents=True,exist_ok=True)
 result.to_csv(out/'paired_cpg_effects.tsv',sep='\t',index=False)
 delta.to_csv(out/'paired_cpg_deltas.tsv',sep='\t')
 (out/'method.json').write_text(json.dumps({'unit':'individual CpG','purpose':'replace historical gene-average paired tissue comparison','n_boot':5000,'seed':20260908,'family':77,'multiple_testing':'BH77, absent tests p1 only inside family','interval':'patient-paired BCa bootstrap','posthoc_scope':'retrospective CpG unit amendment; effects not independently preregistered','input_sources':sources},indent=2))
 print(result.groupby('status').size().to_dict());print('positive significant',int((result.mean_delta_pp.gt(0)&result.paired_t_q77.lt(.05)).sum()))
if __name__=='__main__':main()
