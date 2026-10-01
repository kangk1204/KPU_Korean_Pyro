"""Aggregate paired sensitivity from authorized workbooks and private exclusions.

No patient identifier or individual methylation percentage is embedded here.
Exclusions are mandatory, user-supplied data; no values are re-quantified.
Statistical functions preserve the original 113_DataDriven BCa/BH procedure.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
from scipy import stats

GENES = ['EYA4','ZNF568','ZNF793','SFMBT2','ADHFE1','HOXA2','BEND5','UNC5C','RALYL','GFRA1']
SEED = 20260905
N_BOOT = 5000

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

def _bca_mean_ci(values: np.ndarray, boot_idx: np.ndarray) -> dict[str, float | int]:
    n = len(values)
    theta_hat = float(np.mean(values))
    if not (boot_idx.shape == (N_BOOT, n)):
        raise AssertionError
    boot = values[boot_idx].mean(axis=1)
    jack = np.array([np.delete(values, i).mean() for i in range(n)], dtype=float)
    jack_bar = jack.mean()
    num = np.sum((jack_bar - jack) ** 3)
    den = 6.0 * (np.sum((jack_bar - jack) ** 2) ** 1.5)
    accel = float(num / den) if den > 0 else 0.0
    prop = (np.sum(boot < theta_hat) + 0.5 * np.sum(boot == theta_hat)) / N_BOOT
    prop = min(max(float(prop), 1.0 / (2 * N_BOOT)), 1 - 1.0 / (2 * N_BOOT))
    z0 = float(stats.norm.ppf(prop))
    adjusted = []
    for alpha in (0.025, 0.975):
        z_alpha = stats.norm.ppf(alpha)
        denom = 1 - accel * (z0 + z_alpha)
        adjusted_alpha = stats.norm.cdf(z0 + (z0 + z_alpha) / denom) if denom != 0 else alpha
        adjusted.append(float(min(max(adjusted_alpha, 0.0), 1.0)))
    low, high = np.quantile(boot, adjusted, method="linear")
    return {
        "mean_diff_bca95_low": float(low),
        "mean_diff_bca95_high": float(high),
        "bca_z0": z0,
        "bca_acceleration": accel,
        "bootstrap_replicates": int(N_BOOT),
    }

def normalize_id(value):
    if pd.isna(value):
        raise ValueError('Missing sample identifier')
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def main(args):
    clinical = pd.read_excel(args.clinical, sheet_name='data').set_index('ListNo.')
    tumor = pd.read_excel(args.psq, sheet_name='종양조직PSQ').set_index('Samples')
    normal = pd.read_excel(args.psq, sheet_name='정상조직PSQ').set_index('Samples')
    for frame in (clinical, tumor, normal):
        frame.index = frame.index.map(normalize_id)
        if not (frame.index.is_unique):
            raise AssertionError
    tumor = tumor.loc[clinical.index, GENES]
    normal = normal.loc[clinical.index, GENES]
    if not (len(clinical) == 87):
        raise AssertionError('This adapter targets the archived 87-patient analysis')
    for frame in (tumor, normal):
        if not (frame.notna().all().all()):
            raise AssertionError
        if not (frame.ge(0).all().all() and frame.le(100).all().all()):
            raise AssertionError
    if not (np.array_equal(tumor.to_numpy(), clinical[['CP_'+g for g in GENES]].to_numpy())):
        raise AssertionError
    if not (np.array_equal(normal.to_numpy(), clinical[['NP_'+g for g in GENES]].to_numpy())):
        raise AssertionError
    exclusions = pd.read_csv(args.exclusions, sep='\t', dtype=str)
    if not (set(exclusions.columns) >= {'gene', 'sample_id'}):
        raise AssertionError
    if not (not exclusions[['gene', 'sample_id']].duplicated().any()):
        raise AssertionError
    exclusions['sample_id'] = exclusions.sample_id.map(normalize_id)
    if not (set(exclusions.gene) <= set(GENES)):
        raise AssertionError
    if not (set(exclusions.sample_id) <= set(clinical.index)):
        raise AssertionError
    if not (len(exclusions) == 2 and exclusions.sample_id.nunique() == 2):
        raise AssertionError('The archived sensitivity omits two distinct representative patients')
    excluded_by_gene = {g: set(exclusions.loc[exclusions.gene==g, 'sample_id']) for g in GENES}
    excluded_patients = set(exclusions.sample_id)
    rows = []
    for scenario in ['original_87', 'omit_flagged_patient_gene_pairs', 'omit_two_patients_all_genes']:
        scenario_rows = []
        for gene in GENES:
            excluded = (set() if scenario=='original_87' else excluded_by_gene[gene]
                        if scenario=='omit_flagged_patient_gene_pairs' else excluded_patients)
            keep = ~tumor.index.isin(excluded)
            t = tumor.loc[keep, gene].to_numpy(float)
            n = normal.loc[keep, gene].to_numpy(float)
            d = t-n
            if not (len(d) >= 3):
                raise AssertionError
            boot_idx = np.random.default_rng(SEED).integers(0, len(d), size=(N_BOOT, len(d)))
            try:
                wil_p = float(stats.wilcoxon(t, n, zero_method='wilcox', alternative='two-sided').pvalue)
            except ValueError:
                wil_p = math.nan
            row = {'scenario': scenario, 'gene': gene, 'n_pairs': len(d),
                'tumor_mean_pct': float(t.mean()), 'normal_mean_pct': float(n.mean()),
                'mean_difference_pp': float(d.mean()), 'median_difference_pp': float(np.median(d)),
                'paired_t_p': float(stats.ttest_rel(t, n).pvalue), 'wilcoxon_p': wil_p}
            row.update(_bca_mean_ci(d, boot_idx))
            scenario_rows.append(row)
        frame = pd.DataFrame(scenario_rows)
        frame['paired_t_BH_q'] = bh(frame.paired_t_p)
        frame['wilcoxon_BH_q'] = bh(frame.wilcoxon_p)
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    baseline = out[out.scenario=='original_87'].set_index('gene')
    out['mean_difference_change_pp'] = out.apply(lambda r: r.mean_difference_pp-baseline.loc[r.gene, 'mean_difference_pp'], axis=1)
    cols = ['scenario','gene','n_pairs','tumor_mean_pct','normal_mean_pct','mean_difference_pp',
        'median_difference_pp','mean_diff_bca95_low','mean_diff_bca95_high','paired_t_p',
        'paired_t_BH_q','wilcoxon_p','wilcoxon_BH_q','bootstrap_replicates','mean_difference_change_pp']
    args.outdir.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(args.outdir/'flagged_summary_sensitivity.tsv', sep='\t', index=False)
    report = {'source': 'authorized PSQ and clinical workbooks; all 1740 methylation values agree',
        'method': 'paired t; two-sided Wilcoxon, zero_method=wilcox; BH ten genes per test/scenario; 5000 BCa bootstrap samples; seed 20260905',
        'excluded_patient_gene_pairs': len(exclusions), 'excluded_distinct_patients': len(excluded_patients),
        'all_scenarios_all_genes_positive_mean': bool((out.mean_difference_pp>0).all()),
        'all_scenarios_all_genes_positive_BCa_lower': bool((out.mean_diff_bca95_low>0).all()),
        'all_scenarios_all_genes_t_BH_below_0_05': bool((out.paired_t_BH_q<0.05).all()),
        'all_scenarios_all_genes_wilcoxon_BH_below_0_05': bool((out.wilcoxon_BH_q<0.05).all()),
        'max_absolute_mean_change_pp': float(out.mean_difference_change_pp.abs().max())}
    (args.outdir/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--psq', type=Path, required=True)
    parser.add_argument('--clinical', type=Path, required=True)
    parser.add_argument('--exclusions', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    main(parser.parse_args())
