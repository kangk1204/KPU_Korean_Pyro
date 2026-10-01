#!/usr/bin/env python3
"""One chunked pass over a processed beta matrix.

Writes the rows named in rows_to_extract.txt verbatim, and accumulates per-sample
means over the open-sea and island probe classes. The matrix is never held in memory:
chunks are summed and discarded, so peak use is one chunk.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

import os
R = Path(os.environ.get('KPU_PROJECT_ROOT', Path.home() / 'mnt/03_KPU_ML_Final'))
W = Path(os.environ.get('KPU_SPECIFICITY_WORK', Path(__file__).resolve().parents[1]))  # registry/ is shipped here; private/ is written here
cohort, src = sys.argv[1], R / sys.argv[2] / 'processed_beta.txt'

keep = set(pd.read_csv(W / 'registry/rows_to_extract.txt', header=None)[0])
osea = set(pd.read_csv(W / 'registry/purity_opensea_probes.txt', header=None)[0])
isl  = set(pd.read_csv(W / 'registry/purity_island_probes.txt', header=None)[0])

t0 = time.time()
so = ci = co = si = None
kept, nrow, n_os, n_is = [], 0, 0, 0
for ch in pd.read_csv(src, sep='\t', chunksize=40000, low_memory=False):
    idcol = ch.columns[0]
    ch = ch.set_index(idcol)
    vals = ch.apply(pd.to_numeric, errors='coerce')
    if so is None:
        so = np.zeros(vals.shape[1]); co = np.zeros(vals.shape[1])
        si = np.zeros(vals.shape[1]); ci = np.zeros(vals.shape[1])
        cols = vals.columns
    m = ch.index.isin(keep)
    if m.any():
        kept.append(ch.loc[m])
    a = vals.loc[vals.index.isin(osea)]
    if len(a):
        so += a.sum(axis=0, skipna=True).to_numpy(); co += a.notna().sum(axis=0).to_numpy(); n_os += len(a)
    b = vals.loc[vals.index.isin(isl)]
    if len(b):
        si += b.sum(axis=0, skipna=True).to_numpy(); ci += b.notna().sum(axis=0).to_numpy(); n_is += len(b)
    nrow += len(ch)

out = W / 'private'; out.mkdir(parents=True, exist_ok=True)
pd.concat(kept).rename_axis('ProbeID').to_csv(out / f'{cohort}_keep_beta.tsv', sep='\t')
pd.DataFrame({'sample_id': cols, 'n_opensea': co.astype(int), 'mean_opensea': np.divide(so, co, where=co > 0),
              'n_island': ci.astype(int), 'mean_island': np.divide(si, ci, where=ci > 0)}
             ).to_csv(out / f'{cohort}_class_means.tsv', sep='\t', index=False)
print(f'{cohort}: {nrow} rows, {n_os} open-sea, {n_is} island, {sum(len(k) for k in kept)} kept, '
      f'{len(cols)} samples, {time.time()-t0:.0f}s', flush=True)
