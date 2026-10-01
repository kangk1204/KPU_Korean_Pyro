#!/usr/bin/env python3
"""Analysis B, refined: calibrate the de novo CIMP call before stratifying.

The first pass called a marker methylated at tumor beta >= 0.20 and CIMP-high at three
of five markers, which put 56-62% of patients in the high group. Published CIMP-high
prevalence in unselected colorectal cancer is roughly 15-20%, so that threshold is too
permissive to carry the label. This reports the index under several marker definitions,
picks the operating point by prevalence rather than by outcome, and stratifies the
candidate between-gene correlation under each.

Nothing here uses the ten candidate genes: the five Weisenberger markers (CACNA1G,
IGF2, NEUROG1, RUNX3, SOCS1) share no gene with the panel.
"""
from __future__ import annotations
import json
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
    n = np.sqrt((x * x).sum(axis=1, keepdims=True)); assert (n > 0).all(); return x / n


def rankunit(x):
    return unit(rankdata(x, axis=1, method='average'))


def balanced(corr, regions):
    return float(np.median([np.median(corr[np.ix_(a, b)])
                            for i, a in enumerate(regions) for b in regions[i + 1:]]))


cand = pd.read_csv(REG / 'candidate_regions.tsv', sep='\t')
cand_probes = [p for s in cand.probes for p in s.split(';')]
cimp_map = pd.read_csv(W / 'registry/cimp_marker_probes.tsv', sep='\t')
regions, start = [], 0
for s in cand.probes:
    k = len(s.split(';')); regions.append(np.arange(start, start + k)); start += k

prev_rows, strat_rows, idx_rows = [], [], []
for cohort in ['ASAN', 'SNUH']:
    m = pd.read_csv(SAMP / f'{cohort}_samples.tsv', sep='\t')
    m = m[m.pair_verified.astype(str).str.lower().eq('true')]
    pairs = m.pivot(index='patient_id', columns='tissue', values='sample_id').sort_index()
    keep = pd.read_csv(W / f'private/{cohort}_keep_beta.tsv', sep='\t').set_index('ProbeID')
    t_ids, n_ids = pairs['T'].to_numpy(), pairs['N'].to_numpy()

    gl_t, gl_d = {}, {}
    for g, part in cimp_map.groupby('gene'):
        pr = [p for p in part.probe if p in keep.index]
        tb = keep.loc[pr, t_ids].to_numpy(float).mean(axis=0)
        nb = keep.loc[pr, n_ids].to_numpy(float).mean(axis=0)
        gl_t[g], gl_d[g] = tb, tb - nb
    T = pd.DataFrame(gl_t, index=pairs.index)
    D = pd.DataFrame(gl_d, index=pairs.index)

    with np.load(CACHE / f'{cohort}_cache.npz', allow_pickle=False) as z:
        delta, names = z['delta'], z['probes'].tolist()
    lookup = {p: i for i, p in enumerate(names)}
    ci = [lookup[p] for p in cand_probes]
    ranks = rankunit(delta)
    raw = balanced(ranks[ci] @ ranks[ci].T, regions)

    defs = {'tumor_beta_0.20': (T, 0.20), 'tumor_beta_0.30': (T, 0.30),
            'tumor_beta_0.40': (T, 0.40), 'delta_beta_0.20': (D, 0.20),
            'delta_beta_0.30': (D, 0.30)}
    for name, (frame, thr) in defs.items():
        n_meth = (frame >= thr).sum(axis=1).to_numpy()
        for k in (3, 4, 5):
            high = n_meth >= k
            prev_rows.append({'cohort': cohort, 'marker_rule': name, 'markers_required': k,
                              'cimp_high_n': int(high.sum()), 'patients': len(pairs),
                              'cimp_high_percent': round(100 * high.mean(), 1)})

    # continuous index and a prevalence-anchored split at the published 15-20% band
    index = D.mean(axis=1).to_numpy()
    cut = np.quantile(index, 0.80)
    high = index >= cut
    idx_rows.append({'cohort': cohort, 'patients': len(pairs),
                     'index_definition': 'mean tumor-minus-adjacent beta across the five markers',
                     'top_quintile_cut': float(cut), 'cimp_high_n': int(high.sum()),
                     'rho_index_vs_candidate_mean_delta':
                         float(spearmanr(index, delta[ci].mean(axis=0)).statistic)})

    # Operating points: the prevalence-anchored top quintile, and every marker rule whose
    # CIMP-high prevalence lands in the published 10-25% band for unselected colorectal cancer.
    strata = {'index_top_quintile': high}
    for name, (frame, thr) in defs.items():
        for k in (3, 4, 5):
            h = (frame >= thr).sum(axis=1).to_numpy() >= k
            if 10 <= 100 * h.mean() <= 25:
                strata[f'{name}_ge{k}'] = h
    for label, mask in strata.items():
        for part, sel in [('high', mask), ('low', ~mask)]:
            if sel.sum() < 20:
                strat_rows.append({'cohort': cohort, 'stratification': label, 'stratum': part,
                                   'patients': int(sel.sum()), 'balanced_median': np.nan,
                                   'note': 'fewer than 20 patients'})
                continue
            rs = rankunit(delta[:, sel])[ci]
            strat_rows.append({'cohort': cohort, 'stratification': label, 'stratum': part,
                               'patients': int(sel.sum()),
                               'balanced_median': balanced(rs @ rs.T, regions),
                               'raw_all_patients': raw, 'note': ''})

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(prev_rows).to_csv(OUT / 'B_cimp_prevalence_grid.tsv', sep='\t', index=False)
pd.DataFrame(idx_rows).to_csv(OUT / 'B_cimp_index.tsv', sep='\t', index=False)
pd.DataFrame(strat_rows).to_csv(OUT / 'B_cimp_stratified_correlation.tsv', sep='\t', index=False)

p = pd.DataFrame(prev_rows)
print('=== CIMP-high prevalence (%) by rule ===')
print(p.pivot_table(index=['marker_rule', 'markers_required'], columns='cohort',
                    values='cimp_high_percent').to_string())
print('\n=== candidate between-gene correlation within strata ===')
s = pd.DataFrame(strat_rows)
print(s[['cohort', 'stratification', 'stratum', 'patients', 'balanced_median', 'raw_all_patients']].to_string(index=False))
print('\n=== continuous index ===')
print(pd.DataFrame(idx_rows).to_string(index=False))
