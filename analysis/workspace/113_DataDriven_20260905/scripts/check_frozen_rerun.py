"""Recompute the scientific outputs in isolation, without historical folders."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import pandas as pd
from common import ROOT, write_json



def require(condition, message="Integrity check failed"):
    if not condition:
        raise RuntimeError(message)

def main():
    comparisons=[]
    logs=[]
    with tempfile.TemporaryDirectory(prefix='crc_frozen_recompute_') as directory:
        isolated=Path(directory)/'analysis'
        isolated.mkdir()
        for name in ['scripts','data','registry']:
            shutil.copytree(ROOT/name,isolated/name,ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT/'run_all.py',isolated/'run_all.py')
        # Canonical derived data and result outputs must be reconstructed.
        shutil.rmtree(isolated/'data/derived')
        (isolated/'data/derived').mkdir()
        env=os.environ.copy()
        env.update({'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        for stage in ['prepare','paired','external','survival','figures']:
            proc=subprocess.run([sys.executable,'run_all.py','--stage',stage],cwd=isolated,
                                env=env,capture_output=True,text=True)
            logs.append(f'{stage}\n{proc.stdout}\n{proc.stderr}')
            if proc.returncode:
                (ROOT/'verification/frozen_rerun.log').write_text('\n'.join(logs))
                raise RuntimeError(f'Isolated stage failed: {stage}')
        for base in ['data/derived','results','qc','figures/source_data']:
            for fresh in sorted((isolated/base).rglob('*')):
                if not fresh.is_file() or fresh.suffix not in {'.csv','.tsv'}:
                    continue
                relative=fresh.relative_to(isolated)
                expected=ROOT/relative
                separator='\t' if fresh.suffix=='.tsv' else ','
                a=pd.read_csv(expected,sep=separator)
                b=pd.read_csv(fresh,sep=separator)
                pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-10,atol=1e-12)
                comparisons.append({'file':str(relative),'rows':len(a),'columns':len(a.columns),'status':'pass'})
        for fresh in sorted((isolated/'results').glob('*summary.json')):
            expected=ROOT/'results'/fresh.name
            require(json.loads(expected.read_text()) == json.loads(fresh.read_text()), fresh.name)
            comparisons.append({'file':str(fresh.relative_to(isolated)),'status':'pass','comparison':'JSON exact semantic equality'})
    (ROOT/'verification/frozen_rerun.log').write_text('\n'.join(logs))
    write_json(ROOT/'verification/frozen_rerun.json',{
        'status':'pass','historical_parent_folders_present':False,
        'scope':'Frozen final workbooks and processed public snapshots to all scientific tables; does not repeat historical full IDAT preprocessing',
        'float_rtol':1e-10,'float_atol':1e-12,'comparisons':comparisons})
    print(json.dumps({'status':'pass','compared_files':len(comparisons)}))


if __name__=='__main__':main()
