"""Independent mean/t/BH/bootstrap verification from authorized private extracts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from freeze_registry import sha


def verify(config_path, extraction_dir, registry_dir, aggregate_dir, report_path):
    config = json.loads(config_path.read_text())
    provenance = json.loads((aggregate_dir/'run_provenance.json').read_text())
    u = pd.read_csv(registry_dir/'probe_universe.tsv',sep='\t').merge(
        pd.read_csv(registry_dir/'probe_types.tsv',sep='\t'),on='probe',validate='one_to_one')
    mapping = pd.read_csv(registry_dir/'probe_gene_mapping.tsv',sep='\t')
    effects = pd.read_csv(aggregate_dir/'probe_effects.tsv',sep='\t')
    groups = pd.read_csv(aggregate_dir/'group_distributions.tsv',sep='\t')
    report = {'status':'PASS','method':'Independent pivot pairing, analytic Student t/BH, multinomial-count matrix bootstrap',
              'cohorts':{},'scope':'No new raw-matrix stream; verify cached extraction hashes from completed raw audit'}
    for cohort,c in config['cohorts'].items():
        pr = provenance['cohorts'][cohort]
        p = extraction_dir/(cohort+'_universe_beta.tsv')
        if not (sha(p)==pr['raw']['extracted_sha256']):
            raise AssertionError
        if not (sha(Path(c['pair_metadata']))==pr['input_sha256']['pair_metadata']):
            raise AssertionError
        beta = pd.read_csv(p,sep='\t',index_col=0)
        meta = pd.read_csv(c['pair_metadata'],sep='\t',dtype=str)
        paired = meta[meta.pair_verified.str.lower().eq('true')].pivot(
            index='patient_id',columns='tissue',values='sample_id').sort_index()
        if not (paired.notna().all().all() and len(paired)==pr['n_pairs']):
            raise AssertionError
        aligned = beta.reindex(u.probe)
        x = aligned[paired['T']].to_numpy().T-aligned[paired['N']].to_numpy().T
        e = effects[effects.cohort.eq(cohort)].set_index('probe').loc[u.probe].reset_index()
        n = np.isfinite(x).sum(axis=0)
        means = np.divide(np.nansum(x,axis=0),n,out=np.full(len(u),np.nan),where=n>=3)
        eligible = u.is_cpg.to_numpy() & ~u.MASK_general.to_numpy() & (n>=3)
        np.testing.assert_array_equal(eligible,e.inferential_eligible)
        np.testing.assert_array_equal(n,e.n_complete_pairs)
        np.testing.assert_allclose(means*100,e.mean_delta_pp,rtol=0,atol=1e-10,equal_nan=True)
        se = np.std(x[:,eligible],axis=0,ddof=1)/np.sqrt(n[eligible])
        t = means[eligible]/se
        pvalues = 2*stats.t.sf(np.abs(t),df=n[eligible]-1)
        np.testing.assert_allclose(pvalues,e.loc[eligible,'paired_t_p'],rtol=2e-12,atol=1e-14)
        all_p = np.ones(391);all_p[eligible]=pvalues
        q = stats.false_discovery_control(all_p)
        np.testing.assert_allclose(q[eligible],e.loc[eligible,'paired_t_q391'],rtol=2e-12,atol=1e-14)
        if not (e.loc[~eligible,['paired_t_p','paired_t_q391','ci_low_pp','ci_high_pp']].isna().all().all()):
            raise AssertionError
        rng = np.random.default_rng(pr['seed'])
        weight = np.array([np.bincount(rng.integers(0,len(x),len(x)),minlength=len(x))
                           for _ in range(2000)],dtype=float)
        counts = weight @ np.isfinite(x).astype(float)
        boot = np.divide(weight @ np.nan_to_num(x),counts,
                         out=np.full((2000,391),np.nan),where=counts>=3)*100
        interval = np.quantile(boot[:,eligible],[.025,.975],axis=0)
        np.testing.assert_allclose(interval.T,e.loc[eligible,['ci_low_pp','ci_high_pp']],rtol=0,atol=1e-10)
        g = groups[groups.cohort.eq(cohort)]
        max_group_error = 0.0
        for r in g.itertuples(index=False):
            keep = u.is_cpg.to_numpy() & u.selected.eq(r.selection=='selected').to_numpy()
            if r.gene=='ALL':
                if r.location!='all_locations':
                    keep &= u.pooled_location.eq(r.location).to_numpy()
            else:
                m = mapping[mapping.gene.eq(r.gene)]
                if r.location!='all_locations':
                    m = m[m.location.eq(r.location)]
                keep &= u.probe.isin(m.probe).to_numpy()
            if not (int(keep.sum())==r.n_registered):
                raise AssertionError
            keep &= n>=3
            if r.scope=='annotation_unmasked':
                keep &= ~u.MASK_general.to_numpy()
            if not (int(keep.sum())==r.n_analyzed):
                raise AssertionError
            vals = means[keep]*100
            if not (int((vals>0).sum())==r.n_positive and int((vals<0).sum())==r.n_negative):
                raise AssertionError
            # Inferential counts always use the full family's eligible probe results.
            test_keep = keep&eligible
            if not (int(((means>0)&(q<.05)&test_keep).sum())==r.n_positive_q391_lt_005):
                raise AssertionError
            if not (int(((means<0)&(q<.05)&test_keep).sum())==r.n_negative_q391_lt_005):
                raise AssertionError
            if len(vals):
                np.testing.assert_allclose([np.median(vals),np.min(vals),np.max(vals),*np.quantile(vals,[.25,.75])],
                                           [r.median_delta_pp,r.min_delta_pp,r.max_delta_pp,r.q25_delta_pp,r.q75_delta_pp],rtol=0,atol=1e-10)
                if r.scope=='annotation_unmasked':
                    ci = np.quantile(np.median(boot[:,keep],axis=1),[.025,.975])
                    np.testing.assert_allclose(ci,[r.median_ci_low_pp,r.median_ci_high_pp],rtol=0,atol=1e-10)
                    max_group_error=max(max_group_error,float(np.max(np.abs(ci-[r.median_ci_low_pp,r.median_ci_high_pp]))))
            if r.scope=='all_measured_descriptive':
                if not (np.isnan(r.median_ci_low_pp) and np.isnan(r.median_ci_high_pp)):
                    raise AssertionError
        report['cohorts'][cohort] = dict(n_pairs=len(x),n_tested=int(eligible.sum()),
                                        n_group_rows_checked=len(g),n_bootstrap_draws=2000,
                                        max_mean_abs_error_pp=float(np.nanmax(np.abs(means*100-e.mean_delta_pp))),
                                        max_probe_ci_abs_error_pp=float(np.max(np.abs(interval.T-e.loc[eligible,['ci_low_pp','ci_high_pp']].to_numpy()))),
                                        max_group_ci_abs_error_pp=max_group_error,
                                        input_extraction_sha256=sha(p))
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--extraction-dir',type=Path,required=True)
    p.add_argument('--registry-dir',type=Path,required=True)
    p.add_argument('--aggregate-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a = p.parse_args()
    verify(a.config,a.extraction_dir,a.registry_dir,a.aggregate_dir,a.report)


if __name__=='__main__':
    main()
