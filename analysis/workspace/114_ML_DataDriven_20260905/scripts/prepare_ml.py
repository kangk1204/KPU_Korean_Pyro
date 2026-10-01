"""Build deidentified ML tables from the frozen, verified 113 reanalysis inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd


def require(condition, message="Integrity check failed"):
    if not condition:
        raise RuntimeError(message)

ROOT=Path(__file__).resolve().parents[1]
GENES=['EYA4','ZNF568','ZNF793','SFMBT2','ADHFE1','HOXA2','BEND5','UNC5C','RALYL','GFRA1']

def main():
 c=pd.read_csv(ROOT/'data/derived/clinical.tsv',sep='\t')
 m=pd.read_csv(ROOT/'data/derived/methylation_wide.tsv',sep='\t')
 d=c.merge(m,on=['study_id','patient_id'],validate='one_to_one')
 require(len(d) == 87 and d.event.sum() == 17 and (d.recurrence_primary.sum() == 82), 'Integrity check failed: len(d) == 87 and d.event.sum() == 17 and (d.recurrence_primary.sum() == 82)')
 require(d.loc[d.recurrence_primary.eq(1), 'event'].sum() == 14, "Integrity check failed: d.loc[d.recurrence_primary.eq(1), 'event'].sum() == 14")
 require((d.cea_ng_ml.gt(7) == d.cea_elevated.astype(bool)).all() and d.cea_elevated.sum() == 18, 'Integrity check failed: (d.cea_ng_ml.gt(7) == d.cea_elevated.astype(bool)).all() and d.cea_elevated.sum() == 18')
 out=d[['study_id','event','duration_days','recurrence_primary','stage','lvi']].copy()
 out['stage_iii']=d.stage.eq(3).astype(int);out['stage_advanced']=d.stage.ge(3).astype(int)
 out['cea_log1p']=np.log1p(d.cea_ng_ml);out['cea_binary']=d.cea_elevated
 for gene in GENES:
  for tissue in ['T','N']:out[f'{tissue}_{gene}']=d[f'{tissue}_{gene}']
  out[f'D_{gene}']=d[f'T_{gene}']-d[f'N_{gene}']
 require(out.notna().all().all() and out.duration_days.gt(0).all(), 'Integrity check failed: out.notna().all().all() and out.duration_days.gt(0).all()')
 out.to_csv(ROOT/'data/ml/recurrence.tsv',sep='\t',index=False)
 rows=[]
 for _,r in d.iterrows():
  for tissue,y in [('N',0),('T',1)]:
   rows.append({'study_id':r.study_id,'specimen_id':f'{r.study_id}_{tissue}','tissue':tissue,'y':y,**{g:r[f'{tissue}_{g}'] for g in GENES}})
 t=pd.DataFrame(rows);t.to_csv(ROOT/'data/ml/tissue.tsv',sep='\t',index=False)
 require(t.groupby('study_id').y.agg(['size', 'sum']).eq([2, 1]).all().all(), "Integrity check failed: t.groupby('study_id').y.agg(['size', 'sum']).eq([2, 1]).all().all()")
 public=pd.read_csv(ROOT/'results/external/paired_source_values.tsv',sep='\t')
 public_names=[]
 for cohort,expected in [('colonomics',92),('gse119526',48)]:
  z=public[public.cohort.eq(cohort)]
  require(len(z) == expected * 10 and (not z.duplicated(['patient', 'gene']).any()), "Integrity check failed: len(z) == expected * 10 and (not z.duplicated(['patient', 'gene']).any())")
  rows=[]
  for patient,zp in z.groupby('patient',sort=True):
   zp=zp.set_index('gene').loc[GENES]
   for tissue,y,col in [('N',0,'normal_beta'),('T',1,'tumor_beta')]:
    sid=f'{cohort}_{patient}'
    rows.append({'study_id':sid,'specimen_id':f'{sid}_{tissue}','tissue':tissue,'y':y,**zp[col].to_dict()})
  pub=pd.DataFrame(rows)
  require(len(pub) == expected * 2 and pub.notna().all().all(), 'Integrity check failed: len(pub) == expected * 2 and pub.notna().all().all()')
  name=f'public_{cohort}.tsv';public_names.append(name)
  pub.to_csv(ROOT/'data/ml'/name,sep='\t',index=False)
 inputs=[]
 for name in ['recurrence.tsv','tissue.tsv']+public_names:
  p=ROOT/'data/ml'/name;inputs.append({'file':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 summary={'n_patients':87,'n_specimens':174,'n_primary':82,'primary_events':14,'all_events':17,'cea_elevated_n':18,'genes':GENES,'inputs':inputs,'no_original_ids_or_dates':True}
 (ROOT/'registry/ml_input_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary))
if __name__=='__main__':main()
