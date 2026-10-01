#!/usr/bin/env python3
"""Discordant adjacent-mucosa sample sensitivity (Table S2): paired effects and correlations.

The excluded patient is identified by rule, not by identifier: the patient whose mean
adjacent-mucosa methylation across the ten genes exceeds 30% (one patient, 41.8%; all
others are at most 13.6%). Uses the retained 114 paired-analysis functions, so BCa
intervals, BH families and seeds match the primary analysis. Requires the restricted
114 inputs (data/derived/*.tsv). Writes aggregate TSVs only.
"""
import itertools, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
W114 = Path(os.environ.get('KPU_WORKSPACE_114', HERE.parent / '114_ML_DataDriven_20260905'))
sys.path.insert(0, str(W114 / 'scripts'))
import analyze_paired as ap
from common import GENES, bh

clin, tumor, normal, legacy = ap._read_inputs()
flag = normal[GENES].mean(axis=1) > 30
assert int(flag.sum()) == 1, int(flag.sum())
keep = (~flag).to_numpy()
full = ap._paired_summary(clin, tumor, normal)
res = ap._paired_summary(clin[keep].reset_index(drop=True), tumor[keep].reset_index(drop=True), normal[keep].reset_index(drop=True))
out = pd.DataFrame({'gene': res.gene, 'full_mean': full.mean_difference_pp, 'full_lo': full.mean_diff_bca95_low,
                    'full_hi': full.mean_diff_bca95_high, 'x_mean': res.mean_difference_pp, 'x_lo': res.mean_diff_bca95_low,
                    'x_hi': res.mean_diff_bca95_high, 'x_t_q': res.paired_t_BH_q, 'x_w_q': res.wilcoxon_BH_q, 'x_n': res.n_pairs})
out.to_csv(HERE / 'paired_exclusion.tsv', sep='\t', index=False)
D = tumor - normal
def corr(Dm):
    r, p = [], []
    for a, b in itertools.combinations(GENES, 2):
        s = stats.spearmanr(Dm[a], Dm[b]); r.append(s.statistic); p.append(s.pvalue)
    q = bh(p); r = np.array(r)
    Z = (Dm - Dm.mean()) / Dm.std(ddof=0)
    ev = np.linalg.svd(Z.to_numpy(), compute_uv=False) ** 2
    return dict(n=len(Dm), median=float(np.median(r)), min=float(r.min()), max=float(r.max()),
                pos=int((r > 0).sum()), sig=int((np.array(q) < 0.05).sum()), pc1=float(ev[0] / ev.sum() * 100))
pd.DataFrame([dict(scenario='original_87', **corr(D)),
              dict(scenario='discordant_adjacent_sample_excluded_86', **corr(D[keep]))]
             ).to_csv(HERE / 'coordination_exclusion.tsv', sep='\t', index=False)
print(out.round(3).to_string())
