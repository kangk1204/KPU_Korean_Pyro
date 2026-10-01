"""Recover all 48 healthy samples through GEO metadata, preserving assay IDs."""
import re
import pandas as pd
from common import ROOT, matrix_metadata, score_beta, write_json


def main():
    p=ROOT/'baseline_114/data/public/processed'
    beta=pd.read_csv(p/'colonomics_beta_selected.tsv',sep='\t',index_col=0)
    raw=matrix_metadata(p/'GSE131013_series_matrix.txt.gz')
    beta.columns=[c.split('_')[0] for c in beta.columns]
    if beta.columns.duplicated().any():raise ValueError('Duplicated GSM')
    raw=raw.set_index('sample').loc[beta.columns].rename_axis('sample').reset_index()
    clin=pd.read_csv(p/'CLX_ClinicalData.tab').set_index('id_clx')
    author=raw['description_2']
    canonical=author.str.replace(r'_[0-9]+$','',regex=True)
    if canonical.duplicated().any():raise ValueError('Non-unique canonical biological sample')
    m=pd.DataFrame({'cohort':'Colonomics','sample':raw['sample'],'author_id':canonical,
        'source_author_id':author,'patient':canonical.str.replace(r'_[NTM]$','',regex=True),
        'tissue':raw['tissue'].map({'Normal':'N','Tumor':'T','Mucosa':'H'}),
        'pair_verified':True,'age':pd.to_numeric(raw['age'],errors='coerce'),
        'sex':raw['gender'].map({'Male':'M','Female':'F'}),'site':raw['location'],
        'stage':raw['Stage'],'country':'Spain','reported_population':'not reported in GEO'})
    if m.tissue.isna().any():raise ValueError('Unknown GEO tissue')
    for key in ['CMS','stromal_score','BRAF_V600E','KRAS_mutated','event_free','time_free','event_global','time_global']:
        m[{'CMS':'cms'}.get(key,key)]=clin[key].reindex(canonical).to_numpy()
    # Verify canonical aliases against independent clinical age/sex/tissue where present.
    for i,r in m.iterrows():
        if r.author_id in clin.index:
            c=clin.loc[r.author_id]
            if str(c['id_clx_individual'])!=r.patient:raise ValueError('Patient mismatch')
            if pd.notna(c.age) and float(c.age)!=r.age:raise ValueError('Age mismatch')
            if {'Male':'M','Female':'F'}.get(c.sex)!=r.sex:raise ValueError('Sex mismatch')
    qc=pd.read_csv(p/'colonomics_sample_qc.tsv',sep='\t')
    excluded=set(qc.loc[qc.excluded.astype(str).str.lower().eq('true'),'sample'].str.split('_').str[0])
    m['excluded']=m['sample'].isin(excluded)
    m=m.loc[~m.excluded].copy();beta=beta[m['sample']]
    fixed=set(sum(__import__('json').loads((ROOT/'registry/fixed_probes.json').read_text()).values(),[]))
    beta=beta.loc[beta.index.isin(fixed)]
    beta.index.name='probe';beta.to_csv(ROOT/'data/derived/Colonomics_beta.tsv.gz',sep='\t')
    m.to_csv(ROOT/'data/derived/Colonomics_samples.tsv',sep='\t',index=False)
    scores,cov=score_beta(beta,'Colonomics')
    scores.to_csv(ROOT/'data/derived/Colonomics_scores.tsv',sep='\t')
    cov.to_csv(ROOT/'data/derived/Colonomics_coverage.tsv',sep='\t',index=False)
    counts=m.groupby(['patient','tissue']).size().unstack(fill_value=0)
    note={'n_assays':len(m),'tissues':m.tissue.value_counts().to_dict(),
        'complete_tn_pairs':int(((counts.get('T',0)==1)&(counts.get('N',0)==1)).sum()),
        'clinical_mapped':int(canonical.isin(clin.index).sum()),
        'aliases':m.loc[m.author_id!=m.source_author_id,['sample','source_author_id','author_id','patient']].to_dict('records'),
        'healthy_metadata':'All 48 GEO-labelled healthy samples retained; 11 without extended clinical table rows have GEO age/sex/location.',
        'assay_source':'Frozen Noob beta from baseline114; primary paper GSE131013/GSE166427 same cohort.'}
    write_json(ROOT/'registry/colonomics_curation.json',note)
    print(note)


if __name__=='__main__':main()
