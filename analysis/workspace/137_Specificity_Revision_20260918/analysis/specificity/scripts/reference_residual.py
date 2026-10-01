#!/usr/bin/env python3
"""Analysis C: where does the candidate residual correlation sit among 5,000 matched sets?

Revision 135 reports that conditioning the candidate between-gene correlation on a
common score built from 151 noncandidate genes reduces the gene-pair-balanced median
from 0.6713 to 0.0877 in ASAN and from 0.5675 to 0.1239 in SNUH, and states that the
residual was never compared with a reference distribution. This builds that comparison.

Each of the 5,000 frozen reference sets is treated exactly as the candidate panel is:
ten regions, one per slot, conditioned on a common score and summarised by the median
of the 45 gene-pair block medians. The score for a reference set is built from the
background genes with that set's own ten genes removed, so the set is never regressed
on a score containing itself; the candidate panel's ten genes are outside the pool by
construction, so this is the matched comparison. The fixed-score variant, in which
every set is conditioned on all 151 genes, is reported alongside to show how much the
leave-own-genes-out step matters.

Statistics, ranks and residuals follow common_axis.py exactly.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import os
R = Path(os.environ.get('KPU_PROJECT_ROOT', Path.home() / 'mnt/03_KPU_ML_Final'))
REG = Path(os.environ.get('KPU_COMMON_SCORE_REGISTRY', R / '135_Final_Revision_20260917/source/Supplementary_Data_1/revision_131/analysis/common_axis/public/registry'))  # Supplementary Data 1: results/common_score/public/registry
CACHE = Path(os.environ.get('KPU_BACKGROUND_CACHE', R / '130_Reviewer_Revision_20260914/analysis/background/private'))  # restricted paired-difference caches
W = Path(os.environ.get('KPU_SPECIFICITY_WORK', Path(__file__).resolve().parents[1]))
OUT = W / 'results'
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
COHORTS = ["ASAN", "SNUH"]
II, JJ = np.triu_indices(10, 1)


def unit(x):
    x = np.asarray(x, float)
    x = x - x.mean(axis=1, keepdims=True)
    n = np.sqrt((x * x).sum(axis=1, keepdims=True))
    assert (n > 0).all(), "constant vector"
    return x / n


def rankunit(x):
    return unit(rankdata(x, axis=1, method="average"))


def balanced_from_corr(corr, regions):
    return float(np.median([np.median(corr[np.ix_(a, b)])
                            for i, a in enumerate(regions) for b in regions[i + 1:]]))


def conditioned(ranks_sub, regions, score):
    """Residual balanced median after regressing each probe's rank vector on the score."""
    c = ranks_sub @ score
    res = unit(ranks_sub - c[:, None] * score[None, :])
    return balanced_from_corr(res @ res.T, regions), float(np.median(c))


def main():
    cand = pd.read_csv(REG / 'candidate_regions.tsv', sep='\t')
    pool = pd.read_csv(REG / 'control_region_pool.tsv', sep='\t')
    frozen = pd.read_csv(REG / 'frozen_control_sets.tsv', sep='\t')
    assert len(frozen) == 5000 and frozen.set_id.is_unique
    cand_probes = [p for s in cand.probes for p in s.split(';')]

    rows, setrows = [], []
    for cohort in COHORTS:
        with np.load(CACHE / f'{cohort}_cache.npz', allow_pickle=False) as z:
            delta, names = z['delta'], z['probes'].tolist()
        lookup = {p: i for i, p in enumerate(names)}
        ranks = rankunit(delta)                      # computed once; sets only re-index it
        npat = delta.shape[1]

        # per-gene unit vectors of the background pool, and each region's probe indices
        gene_probes = {g: sorted({p for s in part.probes for p in s.split(';')})
                       for g, part in pool.groupby('gene', sort=True)}
        gene_names = sorted(gene_probes)
        assert len(gene_names) == 151
        gene_unit = unit(np.array([ranks[[lookup[p] for p in gene_probes[g]]].mean(axis=0)
                                   for g in gene_names]))
        gsum_all = gene_unit.sum(axis=0)
        gidx = {g: i for i, g in enumerate(gene_names)}
        region_probes = {rid: [lookup[p] for p in s.split(';')]
                         for rid, s in zip(pool.region_id, pool.probes)}
        region_gene = dict(zip(pool.region_id, pool.gene))

        def score_from(sum_vec, k):
            return rankunit((sum_vec / k)[None, :])[0]

        score_all = score_from(gsum_all, len(gene_names))

        # candidate panel, same code path
        cregions, start = [], 0
        for s in cand.probes:
            n = len(s.split(';')); cregions.append(np.arange(start, start + n)); start += n
        cr = ranks[[lookup[p] for p in cand_probes]]
        craw = balanced_from_corr(cr @ cr.T, cregions)
        ccond, cmed = conditioned(cr, cregions, score_all)

        # 5,000 reference sets
        setids = frozen[cand.slot].to_numpy()
        raw_v = np.empty(5000); loo_v = np.empty(5000); fix_v = np.empty(5000)
        for k in range(5000):
            rids = setids[k]
            idx, regions, start = [], [], 0
            for rid in rids:
                pr = region_probes[rid]
                regions.append(np.arange(start, start + len(pr))); start += len(pr)
                idx.extend(pr)
            sub = ranks[idx]
            raw_v[k] = balanced_from_corr(sub @ sub.T, regions)
            own = {gidx[region_gene[rid]] for rid in rids}
            assert len(own) == 10
            loo = score_from(gsum_all - gene_unit[sorted(own)].sum(axis=0), len(gene_names) - 10)
            loo_v[k] = conditioned(sub, regions, loo)[0]
            fix_v[k] = conditioned(sub, regions, score_all)[0]

        # gate: raw statistic must reproduce the frozen registry exactly
        exp = pd.read_csv(REG / f'{cohort}_set_statistics.tsv', sep='\t').set_index('set_id')
        err = float(np.max(np.abs(raw_v - exp.loc[frozen.set_id, 'gene_pair_balanced_median_rho'].to_numpy())))
        assert err < 1e-12, f'{cohort}: raw statistic drifted by {err}'

        ge_loo = int((loo_v <= ccond).sum())          # sets whose residual is at or below the candidate's
        ge_fix = int((fix_v <= ccond).sum())
        rows.append({
            'cohort': cohort, 'pairs': npat, 'sets': 5000,
            'candidate_raw_balanced_median': craw,
            'candidate_conditioned_balanced_median': ccond,
            'candidate_median_probe_score_correlation': cmed,
            'raw_statistic_max_abs_error_vs_registry': err,
            'reference_conditioned_loo_median': float(np.median(loo_v)),
            'reference_conditioned_loo_p2.5': float(np.quantile(loo_v, .025)),
            'reference_conditioned_loo_p97.5': float(np.quantile(loo_v, .975)),
            'reference_conditioned_loo_min': float(loo_v.min()),
            'reference_conditioned_loo_max': float(loo_v.max()),
            'sets_with_conditioned_at_or_below_candidate_loo': ge_loo,
            'sets_with_conditioned_at_or_above_candidate_loo': 5000 - ge_loo,
            'candidate_percentile_in_loo_reference': float((loo_v < ccond).mean() * 100),
            'reference_conditioned_fixedscore_median': float(np.median(fix_v)),
            'sets_with_conditioned_at_or_below_candidate_fixedscore': ge_fix,
            'reference_raw_median': float(np.median(raw_v)),
        })
        setrows.append(pd.DataFrame({'cohort': cohort, 'set_id': frozen.set_id,
                                     'raw_balanced_median': raw_v,
                                     'conditioned_leave_own_genes_out': loo_v,
                                     'conditioned_fixed_151_gene_score': fix_v}))
        print(f'{cohort}: candidate raw {craw:.4f} -> conditioned {ccond:.4f}; '
              f'reference conditioned median {np.median(loo_v):.4f} '
              f'(central 95% {np.quantile(loo_v,.025):.4f} to {np.quantile(loo_v,.975):.4f}); '
              f'{ge_loo} of 5000 sets at or below the candidate', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'C_reference_residual_summary.tsv', sep='\t', index=False)
    pd.concat(setrows).to_csv(OUT / 'C_reference_residual_per_set.tsv', sep='\t', index=False)
    print('\nwritten', OUT / 'C_reference_residual_summary.tsv')


if __name__ == '__main__':
    main()
