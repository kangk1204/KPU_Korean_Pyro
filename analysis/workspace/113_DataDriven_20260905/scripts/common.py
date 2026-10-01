"""Shared, deterministic I/O contract for the 2026-09-05 reanalysis."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
GENES = ['EYA4','ZNF568','ZNF793','SFMBT2','ADHFE1','HOXA2','BEND5','UNC5C','RALYL','GFRA1']
SEED = 20260905
N_BOOT = 5000

def ensure_dirs():
    for name in ['registry','data/raw','data/derived','data/public','qc','results','results/external',
                 'figures/source_data','tables','manuscript','supplement','submission','verification','tests']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    good = np.flatnonzero(np.isfinite(p))
    if len(good):
        order = good[np.argsort(p[good])]
        # Keep the declared family size when some tests are not estimable.
        q = np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(good)+1))[::-1])[::-1]
        out[order] = np.minimum(q,1)
    return out
