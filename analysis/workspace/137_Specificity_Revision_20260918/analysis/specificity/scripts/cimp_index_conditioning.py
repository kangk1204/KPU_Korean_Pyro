#!/usr/bin/env python3
"""Condition the candidate correlation on the continuous CIMP index.

The first pass conditioned on a count of markers methylated above tumor beta 0.20,
which put a threshold inside a covariate that does not need one and left two different
cutoffs in the paper. The continuous index - the mean tumor-minus-adjacent beta across
the five Weisenberger markers - carries the same information without a cutoff, so the
only threshold left in the analysis is the one used to define CIMP-high strata.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr

import os
R = Path(os.environ.get('KPU_PROJECT_ROOT', Path.home() / 'mnt/03_KPU_ML_Final'))
REG = Path(os.environ.get('KPU_COMMON_SCORE_REGISTRY', R / '135_Final_Revision_20260917/source/Supplementary_Data_1/revision_131/analysis/common_axis/public/registry'))  # Supplementary Data 1: results/common_score/public/registry
W = Path(os.environ.get('KPU_SPECIFICITY_WORK', Path(__file__).resolve().parents[1]))  # registry/ is shipped here; private/ is written here
CACHE = Path(os.environ.get('KPU_BACKGROUND_CACHE', R / '130_Reviewer_Revision_20260914/analysis/background/private'))  # restricted paired-difference caches
SAMP = R / '121_Korean_Beta_Validation_20260908/data/derived'
OUT = W / 'results'

def unit(x):
    x = np.asarray(x, float); x = x - x.mean(axis=1, keepdims=True)
    n = np.sqrt((x*x).sum(axis=1, keepdims=True)); assert (n > 0).all(); return x / n

def rankunit(x):
    return unit(rankdata(x, axis=1, method='average'))

def balanced(c, regions):
    return float(np.median([np.median(c[np.ix_(a, b)])
                            for i, a in enumerate(regions) for b in regions[i+1:]]))

cand = pd.read_csv(REG/'candidate_regions.tsv', sep='\t')
pool = pd.read_csv(REG/'control_region_pool.tsv', sep='\t')
cand_probes = [p for s in cand.probes for p in s.split(';')]
cimp_map = pd.read_csv(W/'registry/cimp_marker_probes.tsv', sep='\t')
regions, start = [], 0
for s in cand.probes:
    k = len(s.split(';')); regions.append(np.arange(start, start+k)); start += k

rows = []
for cohort in ['ASAN', 'SNUH']:
    m = pd.read_csv(SAMP/f'{cohort}_samples.tsv', sep='\t')
    m = m[m.pair_verified.astype(str).str.lower().eq('true')]
    pairs = m.pivot(index='patient_id', columns='tissue', values='sample_id').sort_index()
    keep = pd.read_csv(W/f'private/{cohort}_keep_beta.tsv', sep='\t').set_index('ProbeID')
    t_ids, n_ids = pairs['T'].to_numpy(), pairs['N'].to_numpy()
    d = {}
    for g, part in cimp_map.groupby('gene'):
        pr = [p for p in part.probe if p in keep.index]
        d[g] = (keep.loc[pr, t_ids].to_numpy(float).mean(axis=0)
                - keep.loc[pr, n_ids].to_numpy(float).mean(axis=0))
    index = pd.DataFrame(d, index=pairs.index).mean(axis=1).to_numpy()

    with np.load(CACHE/f'{cohort}_cache.npz', allow_pickle=False) as z:
        delta, names = z['delta'], z['probes'].tolist()
    lookup = {p: i for i, p in enumerate(names)}
    ci = [lookup[p] for p in cand_probes]
    ranks = rankunit(delta)
    cr = ranks[ci]
    raw = balanced(cr @ cr.T, regions)

    gene_probes = {g: sorted({p for s in part.probes for p in s.split(';')})
                   for g, part in pool.groupby('gene', sort=True)}
    gu = unit(np.array([ranks[[lookup[p] for p in gene_probes[g]]].mean(axis=0)
                        for g in sorted(gene_probes)]))
    common = rankunit(gu.mean(axis=0, keepdims=True))[0]

    s = rankunit(index[None, :])[0]
    c = cr @ s
    res = unit(cr - c[:, None]*s[None, :])
    cond = balanced(res @ res.T, regions)
    rows.append({'cohort': cohort, 'axis': 'cimp_index_continuous',
                 'raw_balanced_median': raw, 'conditioned_balanced_median': cond,
                 'change': cond - raw, 'median_probe_axis_correlation': float(np.median(c)),
                 'rho_axis_vs_common_score': float(spearmanr(index, common).statistic),
                 'rho_axis_vs_candidate_mean_delta': float(spearmanr(index, delta[ci].mean(axis=0)).statistic)})
    print(f'{cohort}: {raw:.4f} -> {cond:.4f}  (rho with common score '
          f'{rows[-1]["rho_axis_vs_common_score"]:+.3f}, with candidate mean delta '
          f'{rows[-1]["rho_axis_vs_candidate_mean_delta"]:+.3f})', flush=True)

pd.DataFrame(rows).to_csv(OUT/'A_cimp_index_conditioning.tsv', sep='\t', index=False)
print('\nwritten', OUT/'A_cimp_index_conditioning.tsv')
