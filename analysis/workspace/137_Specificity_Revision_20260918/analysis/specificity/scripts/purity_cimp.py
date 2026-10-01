#!/usr/bin/env python3
"""Analyses A and B: tumor-content proxies and a de novo CIMP index in the Korean cohorts.

A. Tumor content. No orthogonal purity assay exists for these cohorts, so two
   methylation axes are built from probe classes that share no gene and no probe with
   the candidate panel or the 965-probe background score:
     hypo  = mean beta over 171,595 autosomal open-sea probes, tumor minus adjacent.
             Global open-sea hypomethylation increases with tumor content, so the
             proxy is its negative.
     hyper = mean beta over 143,685 autosomal island probes, tumor minus adjacent.
   The two axes move in opposite directions in beta space; if both track tumor
   fraction they must agree after the sign flip, which is the internal check.
   Neither is a calibrated purity value. They order patients, which is all the
   conditioning analysis needs.

B. CIMP. The Weisenberger five-marker panel (CACNA1G, IGF2, NEUROG1, RUNX3, SOCS1)
   shares no gene with the ten candidates, so an index derived from it is not circular
   with the panel being tested. Gene level = mean tumor beta over that gene's
   promoter-island probes; a marker counts as methylated above the stated threshold;
   CIMP-high is three or more of five.

Conditioning uses the same rank-residual machinery as common_axis.py.
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
W = Path(os.environ.get('KPU_SPECIFICITY_WORK', Path(__file__).resolve().parents[1]))  # registry/ is shipped here; private/ is written here
CACHE = Path(os.environ.get('KPU_BACKGROUND_CACHE', R / '130_Reviewer_Revision_20260914/analysis/background/private'))  # restricted paired-difference caches
SAMP = R / '121_Korean_Beta_Validation_20260908/data/derived'
OUT = W / 'results'
COHORTS = ['ASAN', 'SNUH', 'CMCBSN']
RESID_COHORTS = ['ASAN', 'SNUH']
CIMP_THRESHOLD = 0.20  # first-pass threshold, superseded; the reported 0.30 rule and strata come from cimp_refine.py
CIMP_HIGH_MIN = 3


def unit(x):
    x = np.asarray(x, float)
    x = x - x.mean(axis=1, keepdims=True)
    n = np.sqrt((x * x).sum(axis=1, keepdims=True))
    assert (n > 0).all()
    return x / n


def rankunit(x):
    return unit(rankdata(x, axis=1, method='average'))


def balanced_from_corr(corr, regions):
    return float(np.median([np.median(corr[np.ix_(a, b)])
                            for i, a in enumerate(regions) for b in regions[i + 1:]]))


def condition_on(ranks_sub, regions, axis_vec):
    s = rankunit(np.asarray(axis_vec, float)[None, :])[0]
    c = ranks_sub @ s
    res = unit(ranks_sub - c[:, None] * s[None, :])
    return balanced_from_corr(res @ res.T, regions), float(np.median(c)), s


def pairs_for(cohort):
    m = pd.read_csv(SAMP / f'{cohort}_samples.tsv', sep='\t')
    m = m[m.pair_verified.astype(str).str.lower().eq('true')]
    p = m.pivot(index='patient_id', columns='tissue', values='sample_id').sort_index()
    assert p.notna().all().all()
    return p


def main():
    cand = pd.read_csv(REG / 'candidate_regions.tsv', sep='\t')
    pool = pd.read_csv(REG / 'control_region_pool.tsv', sep='\t')
    frozen = pd.read_csv(REG / 'frozen_control_sets.tsv', sep='\t')
    cand_probes = [p for s in cand.probes for p in s.split(';')]
    cimp_map = pd.read_csv(W / 'registry/cimp_marker_probes.tsv', sep='\t')

    axis_rows, cimp_rows, cond_rows, strat_rows, setrows = [], [], [], [], []
    for cohort in COHORTS:
        pairs = pairs_for(cohort)
        means = pd.read_csv(W / f'private/{cohort}_class_means.tsv', sep='\t').set_index('sample_id')
        keep = pd.read_csv(W / f'private/{cohort}_keep_beta.tsv', sep='\t').set_index('ProbeID')

        t_ids, n_ids = pairs['T'].to_numpy(), pairs['N'].to_numpy()
        hypo_d = means.loc[t_ids, 'mean_opensea'].to_numpy() - means.loc[n_ids, 'mean_opensea'].to_numpy()
        hyper_d = means.loc[t_ids, 'mean_island'].to_numpy() - means.loc[n_ids, 'mean_island'].to_numpy()
        purity_hypo = -hypo_d                      # more tumor -> more open-sea loss
        rho_axes = spearmanr(purity_hypo, hyper_d).statistic

        # de novo CIMP from tumor beta at the five marker genes
        gene_level = {}
        for g, part in cimp_map.groupby('gene'):
            pr = [p for p in part.probe if p in keep.index]
            gene_level[g] = keep.loc[pr, t_ids].to_numpy(float).mean(axis=0)
            cimp_rows.append({'cohort': cohort, 'gene': g, 'probes_used': len(pr),
                              'mean_tumor_beta': float(np.mean(gene_level[g])),
                              'fraction_above_threshold': float(np.mean(gene_level[g] >= CIMP_THRESHOLD))})
        gl = pd.DataFrame(gene_level, index=pairs.index)
        n_meth = (gl >= CIMP_THRESHOLD).sum(axis=1).to_numpy()
        cimp_high = n_meth >= CIMP_HIGH_MIN

        axis_rows.append({'cohort': cohort, 'pairs': len(pairs),
                          'mean_opensea_delta': float(hypo_d.mean()),
                          'mean_island_delta': float(hyper_d.mean()),
                          'rho_hypoproxy_vs_islandgain': float(rho_axes),
                          'cimp_markers_median': float(np.median(n_meth)),
                          'cimp_high_n': int(cimp_high.sum()),
                          'cimp_high_fraction': float(cimp_high.mean())})
        print(f'{cohort}: open-sea delta {hypo_d.mean():+.4f}, island delta {hyper_d.mean():+.4f}, '
              f'axes rho {rho_axes:.3f}, CIMP-high {cimp_high.sum()}/{len(pairs)}', flush=True)

        if cohort not in RESID_COHORTS:
            continue

        with np.load(CACHE / f'{cohort}_cache.npz', allow_pickle=False) as z:
            delta, names = z['delta'], z['probes'].tolist()
        lookup = {p: i for i, p in enumerate(names)}
        assert delta.shape[1] == len(pairs)
        # The cache carries no identifiers, so confirm its patient order against the
        # freshly extracted matrix before any axis is aligned to it.
        recomputed = (keep.loc[cand_probes, t_ids].to_numpy(float)
                      - keep.loc[cand_probes, n_ids].to_numpy(float))
        drift = float(np.max(np.abs(delta[[lookup[p] for p in cand_probes]] - recomputed)))
        assert drift < 1e-9, f'{cohort}: cache patient order or values differ by {drift}'
        print(f'   cache/order check: max |difference| {drift:.2e}', flush=True)
        ranks = rankunit(delta)
        regions, start = [], 0
        for s in cand.probes:
            k = len(s.split(';')); regions.append(np.arange(start, start + k)); start += k
        cr = ranks[[lookup[p] for p in cand_probes]]
        raw = balanced_from_corr(cr @ cr.T, regions)

        gene_probes = {g: sorted({p for s in part.probes for p in s.split(';')})
                       for g, part in pool.groupby('gene', sort=True)}
        gene_unit = unit(np.array([ranks[[lookup[p] for p in gene_probes[g]]].mean(axis=0)
                                   for g in sorted(gene_probes)]))
        common = rankunit(gene_unit.mean(axis=0, keepdims=True))[0]

        axes = {'common_score': common, 'purity_hypo': purity_hypo,
                'island_gain': hyper_d, 'cimp_marker_count': n_meth.astype(float)}
        for name, vec in axes.items():
            cond, medc, s = condition_on(cr, regions, vec)
            cond_rows.append({'cohort': cohort, 'axis': name, 'raw_balanced_median': raw,
                              'conditioned_balanced_median': cond, 'change': cond - raw,
                              'median_probe_axis_correlation': medc,
                              'rho_axis_vs_common_score': float(spearmanr(vec, common).statistic)})
            print(f'   condition on {name:18s}: {raw:.4f} -> {cond:.4f} '
                  f'(rho with common score {spearmanr(vec, common).statistic:+.3f})', flush=True)

        # purity and common score together
        sp = rankunit(np.asarray(purity_hypo, float)[None, :])[0]
        X = np.column_stack([common, sp])
        beta, *_ = np.linalg.lstsq(X, cr.T, rcond=None)
        res2 = unit(cr - (X @ beta).T)
        cond2 = balanced_from_corr(res2 @ res2.T, regions)
        cond_rows.append({'cohort': cohort, 'axis': 'common_score_plus_purity_hypo',
                          'raw_balanced_median': raw, 'conditioned_balanced_median': cond2,
                          'change': cond2 - raw, 'median_probe_axis_correlation': np.nan,
                          'rho_axis_vs_common_score': np.nan})
        print(f'   condition on both axes         : {raw:.4f} -> {cond2:.4f}', flush=True)

        # CIMP-stratified raw correlations
        for label, mask in [('CIMP_high', cimp_high), ('CIMP_low', ~cimp_high)]:
            if mask.sum() < 20:
                strat_rows.append({'cohort': cohort, 'stratum': label, 'patients': int(mask.sum()),
                                   'balanced_median': np.nan, 'note': 'fewer than 20 patients'})
                continue
            rs = rankunit(delta[:, mask])[[lookup[p] for p in cand_probes]]
            strat_rows.append({'cohort': cohort, 'stratum': label, 'patients': int(mask.sum()),
                               'balanced_median': balanced_from_corr(rs @ rs.T, regions), 'note': ''})
            print(f'   {label}: n={int(mask.sum())}, raw balanced median '
                  f'{strat_rows[-1]["balanced_median"]:.4f}', flush=True)

        # do reference sets also survive purity conditioning better than the candidate?
        region_probes = {rid: [lookup[p] for p in s.split(';')]
                         for rid, s in zip(pool.region_id, pool.probes)}
        setids = frozen[cand.slot].to_numpy()
        vals = np.empty(5000)
        for k in range(5000):
            idx, regs, start = [], [], 0
            for rid in setids[k]:
                pr = region_probes[rid]
                regs.append(np.arange(start, start + len(pr))); start += len(pr)
                idx.extend(pr)
            vals[k] = condition_on(ranks[idx], regs, purity_hypo)[0]
        candp = next(r for r in cond_rows if r['cohort'] == cohort and r['axis'] == 'purity_hypo')
        setrows.append(pd.DataFrame({'cohort': cohort, 'set_id': frozen.set_id,
                                     'conditioned_on_purity_hypo': vals}))
        cond_rows.append({'cohort': cohort, 'axis': 'purity_hypo_reference_distribution',
                          'raw_balanced_median': float(np.median(vals)),
                          'conditioned_balanced_median': candp['conditioned_balanced_median'],
                          'change': np.nan,
                          'median_probe_axis_correlation': float((vals >= candp['conditioned_balanced_median']).sum()),
                          'rho_axis_vs_common_score': np.nan})
        print(f'   reference sets conditioned on purity: median {np.median(vals):.4f}, '
              f'{int((vals >= candp["conditioned_balanced_median"]).sum())} of 5000 at or above candidate',
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(axis_rows).to_csv(OUT / 'AB_cohort_axes_summary.tsv', sep='\t', index=False)
    pd.DataFrame(cimp_rows).to_csv(OUT / 'B_cimp_marker_levels.tsv', sep='\t', index=False)
    pd.DataFrame(cond_rows).to_csv(OUT / 'A_conditioning_comparison.tsv', sep='\t', index=False)
    pd.DataFrame(strat_rows).to_csv(OUT / 'B_cimp_stratified_correlation.tsv', sep='\t', index=False)
    if setrows:
        pd.concat(setrows).to_csv(OUT / 'A_purity_reference_per_set.tsv', sep='\t', index=False)
    json.dump({'cimp_threshold_tumor_beta': CIMP_THRESHOLD, 'cimp_high_minimum_markers': CIMP_HIGH_MIN,
               'cimp_panel': sorted(set(cimp_map.gene)),
               'purity_axes': 'open-sea mean beta (negated) and island mean beta, tumor minus adjacent',
               'axis_probe_exclusions': 'every probe annotated to a candidate or background gene, the 77-CpG registry, the 965 background probes'},
              open(OUT / 'AB_method_parameters.json', 'w'), indent=2)
    print('\nwritten', OUT)


if __name__ == '__main__':
    main()
