"""Collect provenance and machine-readable result pointers for manuscript assembly."""
import importlib.metadata
from pathlib import Path
import json
import subprocess
from common import ROOT,GENES,SEED,N_BOOT,sha256,write_json

def main():
    results={}
    for p in sorted((ROOT/'results').rglob('*')):
        if p.is_file() and p.suffix in ['.csv','.tsv','.json'] and p.name!='results_registry.json':
            results[str(p.relative_to(ROOT))]={'sha256':sha256(p),'bytes':p.stat().st_size}
    summaries={}
    for p in sorted((ROOT/'results').glob('*summary.json')):
        summaries[p.stem]=json.loads(p.read_text())
    obj={'analysis_date':'2026-09-05','exploratory_reanalysis':True,'genes':GENES,
         'seed':SEED,'patient_bootstrap_replicates':N_BOOT,'summaries':summaries,
         'result_files':results,'author_prerequisites':'registry/AUTHOR_CONFIRMATIONS.md'}
    write_json(ROOT/'registry/results_registry.json',obj)
    versions={}
    for p in ['numpy','pandas','scipy','scikit-learn','statsmodels','matplotlib','openpyxl','python-docx','Pillow','PyMuPDF','pytest','tabulate']:
        versions[p]=importlib.metadata.version(p)
    (ROOT/'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
    r=subprocess.run(['Rscript','-e','library(survival); sessionInfo()'],capture_output=True,text=True,check=True)
    (ROOT/'registry/R_sessionInfo.txt').write_text(r.stdout+r.stderr)
    print(json.dumps({'result_records':len(results),'summaries':list(summaries)},ensure_ascii=False))

if __name__=='__main__':main()
