"""Stream immutable source matrices once; retain only predeclared CpGs and audit structure."""
from pathlib import Path
import hashlib,json,csv
import pandas as pd
from analyze_korean_replication import load_beta, read_fixed_probes
ROOT=Path(__file__).resolve().parents[1]
SOURCES={'CMCBSN':('000_CMCBSN_catholic_beta',3652282903),'SNUH':('000_SNUH_seoul_beta',4849869650),'ASAN':('000_ASAN_seoul_beta',3299286688)}
def extract(source,out,targets,expected_bytes):
 if source.stat().st_size!=expected_bytes:raise ValueError(f'Unexpected source size: {source}')
 before=source.stat();seen=set();selected=[];h=hashlib.sha256();n=0
 with source.open('rb') as f:
  header=f.readline();h.update(header)
  fields=header.rstrip(b'\r\n').decode('utf-8-sig').split('\t')
  if fields[0]!='ProbeID' or len(fields[1:])!=len(set(fields[1:])):raise ValueError('Bad source header')
  for line in f:
   n+=1;h.update(line)
   if line.count(b'\t')!=len(fields)-1:raise ValueError(f'Ragged row {n+1}')
   probe=line.partition(b'\t')[0].decode('ascii')
   if probe in seen:raise ValueError(f'Duplicate probe {probe}')
   seen.add(probe)
   if probe in targets:selected.append(line)
  if n<1:raise ValueError('Empty source matrix')
 if source.stat().st_size!=before.st_size or source.stat().st_mtime_ns!=before.st_mtime_ns:raise ValueError('Source changed during read')
 out.write_bytes(header+b''.join(selected))
 return {'source_file':str(source.relative_to(ROOT.parent)),'size_bytes':before.st_size,'registered_size_matches':True,'sha256':h.hexdigest(),'source_probe_rows':n,'samples':len(fields)-1,'selected_probes':len(selected),'selected_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'missing_fixed_probes':sorted(targets-seen),'validation_scope':'All source row widths/probe uniqueness; numeric and range validation on fixed-panel values only. Registered remote checksum unavailable.'}
def main():
 probes=read_fixed_probes();target=set(sum(probes.values(),[]));audit={};coverage=[]
 for cohort,(folder,size) in SOURCES.items():
  source=ROOT.parent/folder/'processed_beta.txt';out=ROOT/'data/derived'/f'{cohort}_beta.tsv'
  info=extract(source,out,target,size);beta=load_beta(out)
  cov=pd.DataFrame([{'gene':g,'n_fixed':len(ps),'n_present':sum(p in beta.index for p in ps)} for g,ps in probes.items()])
  info['analysis_unit']='individual CpG; no across-CpG methylation mean';cov.insert(0,'cohort',cohort);coverage.append(cov);audit[cohort]=info
  pd.DataFrame({'sample_id':beta.columns}).to_csv(ROOT/'data/derived'/f'{cohort}_matrix_samples.tsv',sep='\t',index=False)
  print(json.dumps({'cohort':cohort,**info},ensure_ascii=False),flush=True)
  (ROOT/'registry/input_matrix_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 pd.concat(coverage,ignore_index=True).to_csv(ROOT/'registry/cpg_coverage_by_gene.tsv',sep='\t',index=False)
if __name__=='__main__':main()
