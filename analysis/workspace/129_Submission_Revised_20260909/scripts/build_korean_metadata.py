"""Link matrix columns to independently retrieved KBDS patient/tissue titles."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT.parent/'120_Korean_External_Validation_20260908'
EXPECTED={'CMCBSN':(333,228,105,103),'SNUH':(437,294,143,142),'ASAN':(300,172,128,128)}
RAW_HEADERS={'CMCBSN':'000_CMCBSN_catholic_beta/processed_beta.txt','SNUH':'000_SNUH_seoul_beta/processed_beta.txt','ASAN':'000_ASAN_seoul_beta/processed_beta.txt'}

def build_metadata(cohort,sample_ids,titles):
 titles=titles.copy();titles['sample_id']=titles.sample_title.str.split('#').str[-1].str.strip()
 if titles.sample_id.duplicated().any():raise ValueError('Duplicate source sample title')
 if len(sample_ids)!=len(set(sample_ids)):raise ValueError('Duplicate matrix sample IDs')
 if set(sample_ids)!=set(titles.sample_id):raise ValueError('Matrix and public source title inventory differ')
 # The tissue label is read from the source title, independently checked against its sample suffix.
 titles['tissue']=titles.sample_title.map(lambda x:'N' if x.startswith('Colon, nontumor, patient #') else 'T' if x.startswith('Colon, tumor, patient #') else None)
 if titles.tissue.isna().any():raise ValueError('Unrecognized source tissue description')
 titles['patient_id']=titles.patient_code.str.strip()
 expected_prefix=titles.patient_id+'-'+titles.tissue+'-'
 if not all(sid.startswith(prefix) and sid[len(prefix):] in {'A1','A2'} for sid,prefix in zip(titles.sample_id,expected_prefix)):raise ValueError('Source patient/tissue fields contradict source sampleID')
 if titles.duplicated(['patient_id','tissue']).any():raise ValueError('Duplicate patient/tissue')
 paired=titles.groupby('patient_id').tissue.agg(lambda x:set(x)=={'T','N'})
 titles['pair_verified']=titles.patient_id.map(paired)
 titles['cohort']=cohort;titles['platform']='Illumina MethylationEPIC';titles['pair_evidence']='exact matrix column to KBDS sample title join; source patient code and explicit tumor/nontumor labels; unique T/N pair'
 titles['pair_status']=titles.pair_verified.map({True:'source_label_verified_pair',False:'unmatched_source_labeled_sample'})
 titles['source_file']=f'data/raw/kbds/{cohort}_browse_sample_titles.tsv in120'
 cols=['cohort','patient_id','sample_id','tissue','pair_verified','pair_status','platform','sample_title','pair_evidence','source_file']
 return titles[cols].set_index('sample_id').loc[sample_ids].reset_index()
def _read_cohort_manifest():
 manifest=PRIOR/'registry/cohort_manifest.tsv'
 if not manifest.exists():
  return {}
 table=pd.read_csv(manifest,sep='\t',dtype=str).set_index('cohort')
 return table.to_dict(orient='index')

def write_pairing_evidence(records,meta_all):
 manifest=_read_cohort_manifest()
 cohorts={}
 for record,m in zip(records,meta_all):
  cohort=record['cohort'];paired_samples=int(m.pair_verified.sum());unpaired_samples=int((~m.pair_verified).sum())
  paired_patients=int(record['source_verified_pairs']);unique_patients=int(record['unique_patients'])
  info=manifest.get(cohort,{})
  cohorts[cohort]={
   'registry_accession':info.get('accession'),
   'browse_accession':info.get('current_accession'),
   'secondary_accession':info.get('secondary_accession'),
   'institution':info.get('institution'),
   'platform':info.get('platform'),
   'header_count':int(record['matrix_samples']),
   'browse_title_count':int(record['matrix_samples']),
   'exact_header_title_matches':int(record['sample_title_exact_matches']),
   'header_only_ids':[],
   'browse_only_ids':[],
   'paired_patient_count':paired_patients,
   'unpaired_patient_count':unique_patients-paired_patients,
   'paired_sample_count':paired_samples,
   'unpaired_sample_count':unpaired_samples,
   'pair_verified_true_rows':paired_samples,
   'pair_verified_false_rows':unpaired_samples,
   'pair_verification_semantics':'source-label verified tumor/nontumor pair from exact matrix-column to KBDS sample-title join; not genotype-confirmed identity',
   'sample_title_source':str((PRIOR/'data/derived'/f'{cohort}_browse_sample_titles.tsv').relative_to(ROOT.parent)),
   'header_source':RAW_HEADERS[cohort],
  }
 pairwise={}
 for i,a in enumerate(meta_all):
  for b in meta_all[i+1:]:
   ca,cb=a.cohort.iloc[0],b.cohort.iloc[0]
   pairwise[f'{ca}_vs_{cb}']={
    'exact_sample_id_overlap':len(set(a.sample_id)&set(b.sample_id)),
    'exact_patient_id_overlap':len(set(a.patient_id)&set(b.patient_id)),
    'scope':'exact public source identifier check; does not prove absence of reidentified patients'
   }
 out={
  'date':'2026-09-09',
  'scope':'provenance-only; regenerated in 124 integrated revision',
  'summary':{
   'title_header_gate':'pass',
   'notes':[
    'Public browse inventories encode human-readable sample titles; terminal sample IDs match deposited matrix header IDs exactly.',
    'pair_verified counts are row counts in data/derived/*_samples.tsv under source-label semantics, not genotype-based identity confirmation.'
   ]
  },
  'cohorts':cohorts,
  'pairwise_header_overlap':pairwise,
  'evidence_limits':[
   'Header/title identity is exact at sample-ID level, but source labels do not provide genotype fingerprinting.',
   'Pairing uses public sample title, patient code, and explicit tumor/nontumor label.',
   'No panel outcome, CIMP call, or downstream biological claim is derived from this provenance registry.'
  ],
  'source_files':sorted({c['sample_title_source'] for c in cohorts.values()}|set(RAW_HEADERS.values())|{'120_Korean_External_Validation_20260908/registry/cohort_manifest.tsv'})
 }
 (ROOT/'registry/pairing_evidence.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
 return out

def main():
 records=[];meta_all=[]
 for cohort,(n,t,normal,pairs) in EXPECTED.items():
  ids=pd.read_csv(ROOT/'data/derived'/f'{cohort}_matrix_samples.tsv',sep='\t').sample_id.tolist()
  source=PRIOR/'data/derived'/f'{cohort}_browse_sample_titles.tsv';titles=pd.read_csv(source,sep='\t',dtype=str)
  m=build_metadata(cohort,ids,titles)
  observed=(len(m),int(m.tissue.eq('T').sum()),int(m.tissue.eq('N').sum()),int(m.pair_verified.sum()/2))
  if observed!=(n,t,normal,pairs):raise ValueError(f'{cohort}: Source paper count mismatch {observed}')
  m['source_file']=str(source.relative_to(ROOT.parent))
  m.to_csv(ROOT/'data/derived'/f'{cohort}_samples.tsv',sep='\t',index=False);meta_all.append(m)
  records.append({'cohort':cohort,'matrix_samples':len(m),'tumors':observed[1],'normals':observed[2],'source_verified_pairs':observed[3],'unmatched_tumors':int(((~m.pair_verified)&m.tissue.eq('T')).sum()),'unmatched_normals':int(((~m.pair_verified)&m.tissue.eq('N')).sum()),'unique_patients':m.patient_id.nunique(),'sample_title_exact_matches':len(m),'source_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_reported_pair_count_matches':True,'pairing_limit':'source-label linkage, not genotype-based identity verification'})
 pd.DataFrame(records).to_csv(ROOT/'registry/sample_flow.tsv',sep='\t',index=False)
 overlaps=[]
 for i,a in enumerate(meta_all):
  for b in meta_all[i+1:]:
   overlaps.append({'cohort_a':a.cohort.iloc[0],'cohort_b':b.cohort.iloc[0],'exact_sample_id_overlap':len(set(a.sample_id)&set(b.sample_id)),'exact_patient_id_overlap':len(set(a.patient_id)&set(b.patient_id)),'scope':'exact source identifier check; does not prove absence of reidentified patients'})
 pd.DataFrame(overlaps).to_csv(ROOT/'registry/cohort_overlap.tsv',sep='\t',index=False)
 write_pairing_evidence(records,meta_all)
 print(pd.DataFrame(records).to_string(index=False))
if __name__=='__main__':main()
