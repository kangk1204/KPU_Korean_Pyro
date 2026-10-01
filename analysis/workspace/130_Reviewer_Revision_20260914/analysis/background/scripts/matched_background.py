"""Source-frozen matched region comparison; patient-vector bootstrap; no patient exports.
Run stages in order: source, match, target. Standard numpy/pandas/scipy, Python >=3.9.
"""
import argparse, csv, gzip, hashlib, json, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[2]
OLD = WORK / '121_Korean_Beta_Validation_20260908'
PREV = WORK / '120_Korean_External_Validation_20260908'
REG = ROOT / 'registry'; PRIVATE = ROOT / 'private'; RESULTS = ROOT / 'results'
COHORTS = {'CMCBSN': ('000_CMCBSN_catholic_beta',103), 'ASAN':('000_ASAN_seoul_beta',128), 'SNUH':('000_SNUH_seoul_beta',142)}
PROM = {'TSS200','TSS1500',"5'UTR",'1stExon'}
FIXED = json.loads((OLD/'registry/fixed_probes.json').read_text())
PROBEGENE = {p:g for g,ps in FIXED.items() for p in ps}
FEATURES = ['island','shore','shelf','open_sea','promoter','design_I','mask','mean_normal','mean_delta','log_sd_normal','log_sd_tumor','log_sd_delta','log_span','log_median_gap','log_max_gap']
SCALE = np.array([.35]*6+[.20,.15,.15]+[np.log(2)]*3+[np.log(4)]*3)
WIDE = np.array([.50]*6+[.35,.20,.20]+[np.log(3)]*3+[np.log(10)]*3)

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def writejson(p,obj): Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def base_manifest():
 return {'time_utc':now(),'plan_sha256':sha(ROOT/'PLAN.md'),'script_sha256':sha(__file__),'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__}
def validate_plan():
 if not ((REG/'PLAN.sha256').read_text().split()[0]==sha(ROOT/'PLAN.md')):
     raise AssertionError('Plan changed after freeze')

def annotation():
 a=pd.read_csv(REG/'hg19_annotation.tsv',sep='\t',keep_default_na=False).set_index('Name')
 z=pd.read_csv(WORK/'115_Public_Biology_20260905/data/raw/zhou_HM450.hg19.manifest.tsv.gz',sep='\t',usecols=['probeID','MASK_general'],dtype=str).set_index('probeID')
 a['mask']=z.MASK_general.reindex(a.index).eq('TRUE').astype(float)
 rows=[]
 for p,r in a.iterrows():
  genes=r.UCSC_RefGene_Name.split(';') if r.UCSC_RefGene_Name else []
  groups=r.UCSC_RefGene_Group.split(';') if r.UCSC_RefGene_Group else []
  distinct=set(genes)
  fixed=PROBEGENE.get(p)
  single=len(distinct)==1 and '' not in distinct
  gene=fixed if fixed else (genes[0] if single else '')
  if not gene or r.chr not in {f'chr{x}' for x in range(1,23)}:continue
  if fixed is None and (not single or distinct & set(FIXED)):continue
  if len(genes)!=len(groups):continue
  promoter=any(g==gene and group in PROM for g,group in zip(genes,groups))
  rel=r.Relation_to_Island
  context='island' if rel=='Island' else 'shore' if 'Shore' in rel else 'shelf' if 'Shelf' in rel else 'open_sea'
  rows.append({'probe':p,'gene':gene,'chr':r.chr,'pos':int(r.pos),'promoter':float(promoter),'design_I':float(r.Type=='I'),'mask':float(r['mask']),**{key:float(key==context) for key in ['island','shore','shelf','open_sea']},'is_fixed':fixed is not None,'annotation_genes':r.UCSC_RefGene_Name})
 return pd.DataFrame(rows).set_index('probe')

def pairing(cohort, columns):
 meta=pd.read_csv(OLD/f'data/derived/{cohort}_samples.tsv',sep='\t',dtype=str)
 source=pd.read_csv(PREV/f'data/derived/{cohort}_browse_sample_titles.tsv',sep='\t',dtype=str)
 if not (len(columns)==len(set(columns)) and set(columns)==set(meta.sample_id)):
     raise AssertionError
 if not (not meta.sample_id.duplicated().any()):
     raise AssertionError
 ids=source.sample_title.str.split('#').str[-1].str.strip()
 if not (not ids.duplicated().any() and set(ids)==set(columns)):
     raise AssertionError
 source=source.assign(sample_id=ids).set_index('sample_id')
 for r in meta.itertuples():
  sr=source.loc[r.sample_id]
  label='N' if sr.sample_title.startswith('Colon, nontumor, patient #') else 'T' if sr.sample_title.startswith('Colon, tumor, patient #') else None
  if not (label==r.tissue and sr.patient_code.strip()==r.patient_id):
      raise AssertionError
 if not (not meta.duplicated(['patient_id','tissue']).any()):
     raise AssertionError
 valid=meta[meta.pair_verified.str.lower().eq('true')]
 wide=valid.pivot(index='patient_id',columns='tissue',values='sample_id').sort_index()
 if not (wide.notna().all().all() and len(wide)==COHORTS[cohort][1]):
     raise AssertionError
 lookup={x:i for i,x in enumerate(columns)}
 return np.array([lookup[x] for x in wide['N']]),np.array([lookup[x] for x in wide['T']]),{'patient_pairs':len(wide),'matrix_samples':len(columns),'pairing_source':'existing metadata independently rejoined to source patient and explicit tissue titles','metadata_sha256':sha(OLD/f'data/derived/{cohort}_samples.tsv'),'source_titles_sha256':sha(PREV/f'data/derived/{cohort}_browse_sample_titles.tsv'),'identity_limit':'source labels, not genotype identity verification'}

def stream(cohort, requested, stats=False):
 """Hash full source once; validate all row IDs/widths; numeric QC requested paired data."""
 path=WORK/COHORTS[cohort][0]/'processed_beta.txt';before=path.stat();h=hashlib.sha256();seen=set();good=[];D=[];marg=[];bad=[]
 start=time.time()
 with path.open('rb') as f:
  header=f.readline();h.update(header);fields=header.decode('utf-8-sig').rstrip('\r\n').split('\t')
  if not (fields[0]=='ProbeID'):
      raise AssertionError
  ni,ti,pa=pairing(cohort,fields[1:])
  for rowno,line in enumerate(f,1):
   h.update(line)
   if not (line.count(b'\t')==len(fields)-1):
       raise AssertionError(('ragged',rowno))
   ident,_,raw=line.partition(b'\t');p=ident.decode('ascii')
   if not (p not in seen):
       raise AssertionError(('duplicate',p))
   seen.add(p)
   if p not in requested:continue
   values=np.fromstring(raw.decode('ascii'),sep='\t')
   if len(values)!=len(fields)-1:bad.append((p,'numeric_parse'));continue
   n=values[ni];t=values[ti]
   if not (np.isfinite(n).all() and np.isfinite(t).all() and np.all((n>=0)&(n<=1)) and np.all((t>=0)&(t<=1))):bad.append((p,'invalid_or_missing_pair_beta'));continue
   d=t-n
   if np.std(d,ddof=1)<=0:bad.append((p,'constant_delta'));continue
   if stats and (np.std(n,ddof=1)<=0 or np.std(t,ddof=1)<=0):bad.append((p,'constant_tissue_source'));continue
   good.append(p);D.append(d)
   if stats:marg.append([n.mean(),d.mean(),np.std(n,ddof=1),np.std(t,ddof=1),np.std(d,ddof=1)])
   if rowno % 100000==0:print(f'{cohort}: {rowno:,} rows processed',flush=True)
 if not (path.stat().st_size==before.st_size and path.stat().st_mtime_ns==before.st_mtime_ns):
     raise AssertionError
 expected=json.loads((OLD/'registry/input_matrix_audit.json').read_text())[cohort]
 if not (h.hexdigest()==expected['sha256']):
     raise AssertionError(f'{cohort}: previous source hash mismatch')
 info={**base_manifest(),**pa,'source':str(path.relative_to(WORK)),'size_bytes':before.st_size,'sha256':h.hexdigest(),'historical_sha256_matches':True,'source_probe_rows':len(seen),'requested_probes':len(requested),'valid_probes':len(good),'requested_absent':sorted(set(requested)-seen),'requested_invalid':[{'probe':p,'reason':r} for p,r in bad],'wall_seconds':time.time()-start,'validation_scope':'full file bytes, row widths and probe uniqueness; numeric/range/variance QC only requested paired values','no_patient_identifiers_in_cache':True}
 D=np.array(D,dtype=np.float64)
 np.savez_compressed(PRIVATE/f'{cohort}_cache.npz',probes=np.array(good),delta=D)
 os.chmod(PRIVATE/f'{cohort}_cache.npz',0o600)
 writejson(REG/f'{cohort}_raw_audit.json',info)
 return good,np.array(marg),D,info

def source():
 validate_plan();a=annotation();a.to_csv(REG/'eligible_annotation.tsv',sep='\t')
 p,m,_,audit=stream('CMCBSN',set(a.index),stats=True)
 out=a.loc[p].copy();out[['mean_normal','mean_delta','sd_normal','sd_tumor','sd_delta']]=m
 out.to_csv(REG/'source_probe_descriptors.tsv.gz',sep='\t',compression='gzip')
 writejson(REG/'source_stage.json',{**base_manifest(),'annotation_sha256':sha(REG/'hg19_annotation.tsv'),'fixed_registry_sha256':sha(OLD/'registry/fixed_probes.json'),'source_eligible_probes':len(out),'source_eligible_control_genes':out.loc[~out.is_fixed,'gene'].nunique()})
 print(json.dumps({'source_complete':True,'eligible_probes':len(out),'source_hash':audit['sha256']}),flush=True)

def window_features(frame,n):
 """Rows sorted by genomic position. Same geometry for fixed and control blocks."""
 vals=frame[['island','shore','shelf','open_sea','promoter','design_I','mask','mean_normal','mean_delta']].to_numpy(float)
 vals=np.column_stack([vals,frame[['sd_normal','sd_tumor','sd_delta']].to_numpy(float)])
 cs=np.vstack([np.zeros(vals.shape[1]),np.cumsum(vals,axis=0)])
 means=(cs[n:]-cs[:-n])/n
 means[:,9:12]=np.log(means[:,9:12])
 positions=frame.pos.to_numpy(float);gaps=np.diff(positions)
 window_gaps=np.lib.stride_tricks.sliding_window_view(gaps,n-1)
 geom=np.column_stack([positions[n-1:]-positions[:len(positions)-n+1],np.median(window_gaps,axis=1),window_gaps.max(axis=1)])
 geom=np.maximum(geom,1.0)
 features=np.column_stack([means,np.log(geom)])
 clusters=1+(window_gaps>5000).sum(axis=1)
 return features,clusters

def match():
 validate_plan()
 if not (not any((REG/f'{c}_raw_audit.json').exists() for c in ['ASAN','SNUH'])):
     raise AssertionError('Targets already read; cannot refreeze matching')
 a=pd.read_csv(REG/'source_probe_descriptors.tsv.gz',sep='\t',index_col=0)
 target=[];counts={};bycount={};targets={}
 for g,ps in FIXED.items():
  row=a.loc[a.index.isin(ps)].sort_values('pos');n=len(row)
  if not (n>=2):
      raise AssertionError
  f,c=window_features(row,n)
  if not (len(f)==1):
      raise AssertionError
  record={'slot':g,'gene':g,'n_probes':n,'cluster_count':int(c[0]),'chr':row.chr.iloc[0],'start':int(row.pos.min()),'end':int(row.pos.max()),'probes':';'.join(row.index),**dict(zip(FEATURES,f[0]))}
  target.append(record);targets[g]=record;bycount.setdefault(n,[]).append(g)
 if not (sum(t['n_probes'] for t in target)==56):
     raise AssertionError
 candidates={g:{tier:[] for tier in [1,2,3]} for g in FIXED}
 groups=a.loc[~a.is_fixed.astype(bool)].groupby('gene',sort=True)
 for i,(gene,raw) in enumerate(groups):
  if raw.chr.nunique()!=1:continue
  row=raw.sort_values(['pos'],kind='stable')
  for n,slots in bycount.items():
   if len(row)<n:continue
   f,cl=window_features(row,n)
   for g in slots:
    tg=targets[g];delta=f-np.array([tg[k] for k in FEATURES]);distance=np.sqrt(np.mean((delta/SCALE)**2,axis=1))
    core=(cl==tg['cluster_count'])&(f[:,8]>=.10)&np.isfinite(f).all(axis=1)
    for tier,width in [(1,SCALE),(2,WIDE),(3,np.repeat(np.inf,len(SCALE)))]:
     eligible=core&np.all(np.abs(delta)<=width+1e-12,axis=1)
     if not eligible.any():continue
     j=int(np.argmin(np.where(eligible,distance,np.inf)));window=row.iloc[j:j+n]
     candidates[g][tier].append({'slot':g,'gene':gene,'n_probes':n,'cluster_count':int(cl[j]),'chr':window.chr.iloc[0],'start':int(window.pos.min()),'end':int(window.pos.max()),'probes':';'.join(window.index),'distance':float(distance[j]),'tier':tier,**dict(zip(FEATURES,f[j]))})
  if (i+1)%4000==0:print(f'matching: {i+1:,} control genes examined',flush=True)
 pools=[];elig=[]
 for slot in FIXED:
  tier=next((t for t in [1,2] if len(candidates[slot][t])>=10),3)
  chosen=sorted(candidates[slot][tier],key=lambda r:(r['distance'],r['gene'],r['probes']))[:20]
  if not chosen:raise RuntimeError(f'No controls for {slot}')
  for j,r in enumerate(chosen):r['region_id']=f'{slot}__{j+1:02d}__{r["gene"]}'
  pools.extend(chosen);elig.append({'slot':slot,'eligible_tier1_genes':len(candidates[slot][1]),'eligible_tier2_genes':len(candidates[slot][2]),'eligible_tier3_genes':len(candidates[slot][3]),'selected_tier':tier,'selected_genes':len(chosen)})
  print(elig[-1],flush=True)
 p=pd.DataFrame(pools);pd.DataFrame(target).to_csv(REG/'candidate_regions.tsv',sep='\t',index=False);p.to_csv(REG/'control_region_pool.tsv',sep='\t',index=False);pd.DataFrame(elig).to_csv(REG/'matching_eligibility.tsv',sep='\t',index=False)
 rng=np.random.default_rng(2026091401);sets=[];seen=set();indices={g:np.where(p.slot.to_numpy()==g)[0] for g in FIXED}
 for attempt in range(500000):
  arr=[int(rng.choice(indices[g])) for g in FIXED];key=tuple(arr)
  if key in seen:continue
  rows=p.iloc[arr];genes=rows.gene.tolist()
  if len(set(genes))<10:continue
  probes=[x for s in rows.probes for x in s.split(';')]
  if len(probes)!=len(set(probes)):continue
  rr=list(rows.itertuples());overlap=any(a.chr==b.chr and max(a.start,b.start)<=min(a.end,b.end) for i,a in enumerate(rr) for b in rr[i+1:])
  if overlap:continue
  seen.add(key);sets.append({'set_id':f'BG{len(sets)+1:05d}',**{g:rid for g,rid in zip(FIXED,rows.region_id)}})
  if len(sets)==5000:break
 pd.DataFrame(sets).to_csv(REG/'frozen_control_sets.tsv',sep='\t',index=False)
 balance=[]
 for r in p.to_dict('records'):
  t=targets[r['slot']]
  for feature,scale in zip(FEATURES,SCALE):balance.append({'slot':r['slot'],'region_id':r['region_id'],'control_gene':r['gene'],'tier':r['tier'],'feature':feature,'candidate':t[feature],'control':r[feature],'difference':r[feature]-t[feature],'scaled_difference':(r[feature]-t[feature])/scale})
 pd.DataFrame(balance).to_csv(RESULTS/'source_balance_long.tsv',sep='\t',index=False)
 frozen={name:sha(REG/name) for name in ['candidate_regions.tsv','control_region_pool.tsv','matching_eligibility.tsv','frozen_control_sets.tsv']}
 writejson(REG/'SELECTION_FREEZE.json',{**base_manifest(),'seed':2026091401,'sets':len(sets),'draws':attempt+1,'files':frozen,'source_descriptor_sha256':sha(REG/'source_probe_descriptors.tsv.gz'),'target_outcomes_read':False,'target_ids_or_beta_used_in_matching':False,'selection_scope':'source marginal moments and geometry only; never covariance or target data'})
 print('SELECTION FROZEN',flush=True)

def normalized_ranks(delta):
 r=rankdata(delta,axis=1,method='average');r-=r.mean(axis=1,keepdims=True);s=np.sqrt(np.sum(r*r,axis=1));return np.divide(r,s[:,None],out=np.zeros_like(r),where=s[:,None]>0)

def summaries_from_corr(corr,regions,sets):
 """Region pair medians computed once for shared regions; set statistic balances 45 pairs."""
 n=len(regions); pair=np.full((n,n),np.nan)
 groups={}
 for i,r in enumerate(regions):
  if len(r):groups.setdefault(len(r),[]).append(i)
 sizes=list(groups)
 for ia,sa in enumerate(sizes):
  ga=np.array(groups[sa]);ra=np.array([regions[i] for i in ga])
  for sb in sizes[ia:]:
   gb=np.array(groups[sb]);rb=np.array([regions[i] for i in gb])
   blocks=corr[np.ix_(ra.ravel(),rb.ravel())].reshape(len(ga),sa,len(gb),sb).transpose(0,2,1,3).reshape(len(ga),len(gb),sa*sb)
   vals=np.median(blocks,axis=2);pair[np.ix_(ga,gb)]=vals;pair[np.ix_(gb,ga)]=vals.T
 first,second=np.triu_indices(10,k=1)
 balanced=np.median(pair[sets[:,first],sets[:,second]],axis=1)
 return balanced

def probe_weighted(corr,regions,sets):
 vals=[]
 for s in sets:
  chunks=[corr[np.ix_(regions[s[i]],regions[s[j]])].ravel() for i in range(10) for j in range(i+1,10)]
  vals.append(np.median(np.concatenate(chunks)))
 return np.array(vals)

def cohort_analysis(cohort,nboot=500):
 validate_plan();freeze=json.loads((REG/'SELECTION_FREEZE.json').read_text())
 for f,h in freeze['files'].items():
  if not (sha(REG/f)==h):
      raise AssertionError
 cand=pd.read_csv(REG/'candidate_regions.tsv',sep='\t');pool=pd.read_csv(REG/'control_region_pool.tsv',sep='\t');sets=pd.read_csv(REG/'frozen_control_sets.tsv',sep='\t')
 requested=set(p for s in list(cand.probes)+list(pool.probes) for p in s.split(';'))
 if (PRIVATE/f'{cohort}_cache.npz').exists():
  z=np.load(PRIVATE/f'{cohort}_cache.npz');probes=z['probes'].tolist();D=z['delta'];audit=json.loads((REG/f'{cohort}_raw_audit.json').read_text())
 else:probes,_,D,audit=stream(cohort,requested)
 lookup={p:i for i,p in enumerate(probes)}
 valid=pool.probes.map(lambda s:all(p in lookup for p in s.split(';')))
 region_ok=set(pool.loc[valid,'region_id']);set_ok=sets[list(FIXED)].isin(region_ok).all(axis=1);eligible=sets[set_ok].reset_index(drop=True)
 coverage=pool[['region_id','slot','gene','probes']].copy();coverage['available_probes']=pool.probes.map(lambda s:sum(p in lookup for p in s.split(';')));coverage['complete_region']=valid
 coverage.to_csv(RESULTS/f'{cohort}_control_coverage.tsv',sep='\t',index=False)
 if not len(eligible):
  writejson(RESULTS/f'{cohort}_summary.json',{**base_manifest(),'cohort':cohort,'frozen_sets':len(sets),'evaluable_sets':0,'status':'no_evaluable_frozen_background_sets','reason':'technical availability only; no reselection permitted'})
  return
 if not (all(p in lookup for s in cand.probes for p in s.split(';'))):
     raise AssertionError('Missing candidate probe')
 ids=list(cand.slot)+pool.region_id.tolist();all_regions=[np.array([lookup[p] for p in s.split(';') if p in lookup],int) for s in list(cand.probes)+list(pool.probes)]
 ridx={r:i for i,r in enumerate(ids)}
 ss=np.array([[ridx[r] for r in row] for row in eligible[list(FIXED)].to_numpy()],int)
 ss=np.vstack([np.arange(10),ss])
 # Only referenced regions/probes retained for rank computation and bootstrap.
 use_region=np.unique(ss);regionmap={int(old):new for new,old in enumerate(use_region)}
 rr=[all_regions[i] for i in use_region];use_probe=np.unique(np.concatenate(rr));pm={int(old):new for new,old in enumerate(use_probe)}
 rr=[np.array([pm[int(v)] for v in ar],int) for ar in rr];dd=D[use_probe]
 ss=np.array([[regionmap[int(v)] for v in row] for row in ss],int)
 r=normalized_ranks(dd);cor=r@r.T
 stats=summaries_from_corr(cor,rr,ss);weighted=probe_weighted(cor,rr,ss)
 pd.DataFrame({'set_id':['FIXED10']+eligible.set_id.tolist(),'gene_pair_balanced_median_rho':stats,'probe_pair_median_rho':weighted}).to_csv(RESULTS/f'{cohort}_set_statistics.tsv',sep='\t',index=False)
 offset={'ASAN':11,'SNUH':22,'CMCBSN':0}[cohort];brng=np.random.default_rng(2026091403+offset)
 chosen=np.sort(brng.choice(np.arange(1,len(ss)),size=min(256,len(ss)-1),replace=False))
 bss=ss[np.r_[0,chosen]]
 # Drop unreferenced regions and probes from bootstrap computations.
 b_use_region=np.unique(bss);brmap={int(old):new for new,old in enumerate(b_use_region)};brr=[rr[i] for i in b_use_region];bp=np.unique(np.concatenate(brr));bpmap={int(old):new for new,old in enumerate(bp)}
 brr=[np.array([bpmap[int(x)] for x in ar]) for ar in brr];bdata=dd[bp]
 bss=np.array([[brmap[int(x)] for x in row] for row in bss]);rng=np.random.default_rng(2026091402+offset)
 bvalues=[];start=time.time()
 for b in range(nboot):
  ix=rng.integers(0,bdata.shape[1],size=bdata.shape[1]);r=normalized_ranks(bdata[:,ix]);co=r@r.T
  values=summaries_from_corr(co,brr,bss);bvalues.append([values[0],np.median(values[1:]),values[0]-np.median(values[1:])])
  if (b+1)%50==0:print(f'{cohort}: bootstrap {b+1}/{nboot}; elapsed {time.time()-start:.1f}s',flush=True)
 bvalues=np.array(bvalues);interval=np.quantile(bvalues,[.025,.975],axis=0)
 pd.DataFrame(bvalues,columns=['candidate','background_subset_median','difference']).to_csv(RESULTS/f'{cohort}_patient_bootstrap_statistics.tsv',sep='\t',index=False)
 pd.DataFrame({'set_id':eligible.set_id.iloc[chosen-1]}).to_csv(REG/f'{cohort}_bootstrap_reference_set_ids.tsv',sep='\t',index=False)
 def summary(x):
  return {'candidate':float(x[0]),'background_median':float(np.median(x[1:])),'background_q025':float(np.quantile(x[1:],.025)),'background_q975':float(np.quantile(x[1:],.975)),'candidate_minus_background_median':float(x[0]-np.median(x[1:])),'fraction_background_ge_candidate':float(np.mean(x[1:]>=x[0])),'candidate_midrank_percentile':float(100*(np.mean(x[1:]<x[0])+.5*np.mean(x[1:]==x[0])))}
 out={**base_manifest(),'cohort':cohort,'source_selection_freeze_sha256':sha(REG/'SELECTION_FREEZE.json'),'pairs':dd.shape[1],'frozen_sets':len(sets),'evaluable_sets':len(eligible),'adequate_distribution_count':len(eligible)>=100,'complete_control_regions':int(valid.sum()),'frozen_control_regions':len(pool),'unique_probes_evaluated':len(use_probe),'primary_gene_pair_balanced':summary(stats),'sensitivity_probe_pair_median':summary(weighted),'patient_bootstrap':{'n_resamples':nboot,'background_reference_sets':len(chosen),'reference_subset_median_point':float(np.median(stats[chosen])),'difference_to_reference_subset_point':float(stats[0]-np.median(stats[chosen])),'candidate_ci':interval[:,0].tolist(),'background_subset_median_ci':interval[:,1].tolist(),'difference_ci':interval[:,2].tolist(),'bootstrap_seconds':time.time()-start,'unit':'paired patient vector, shared indices across all probe/region vectors; ranks recalculated'},'empirical_position_is_formal_p_value':False}
 # Probe-quality sensitivity, retaining fixed selected regions, no refitting.
 ann=pd.read_csv(REG/'eligible_annotation.tsv',sep='\t',index_col=0)
 masked=set(ann.index[ann['mask'].eq(1)])
 use_names=[probes[i] for i in use_probe];clean_rr=[np.array([j for j in ar if use_names[j] not in masked],int) for ar in rr]
 clean_valid=np.array([all(len(clean_rr[int(j)])>0 for j in s) for s in ss])
 if clean_valid[0]:
  clean_ss=ss[clean_valid];clean=summaries_from_corr(cor,clean_rr,clean_ss)
  out['sensitivity_exclude_mask_general']={**summary(clean),'candidate_probes':sum(len(clean_rr[i]) for i in ss[0]),'evaluable_sets':len(clean)-1,'candidate_removed_probes':[p for p in use_names if p in masked and p in PROBEGENE]}
  pd.DataFrame({'set_id':np.array(['FIXED10']+eligible.set_id.tolist())[clean_valid],'gene_pair_balanced_median_rho':clean}).to_csv(RESULTS/f'{cohort}_qc_sensitivity_set_statistics.tsv',sep='\t',index=False)
 writejson(RESULTS/f'{cohort}_summary.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['source','match','target']);parser.add_argument('--cohort',choices=['ASAN','SNUH']);parser.add_argument('--nboot',type=int,default=500);args=parser.parse_args()
 if args.stage=='source':source()
 elif args.stage=='match':match()
 else:cohort_analysis(args.cohort,args.nboot)
