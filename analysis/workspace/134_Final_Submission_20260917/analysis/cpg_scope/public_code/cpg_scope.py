"""Fixed annotation-scope sensitivity from explicitly authorized paired matrices.

No patient-level value or identifier is written to the aggregate output directory.
The extraction directory is private and must not be published.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from freeze_registry import GENES, parse_bool, sha
from register_probe_types import EXPECTED_CH

COHORTS = {'CMCBSN': (103, 2026091501), 'SNUH': (142, 2026091502),
           'ASAN': (128, 2026091503)}
N_BOOT = 2000
FAMILY_SIZE = 391
MISSING = ['', 'NA', 'NaN', 'nan', 'NAN', 'null', 'NULL']


def extract_registered(source, output, targets, expected_sha, expected_bytes):
    """Read all bytes once; validate structure; privately retain registered rows."""
    before = source.stat()
    if before.st_size != expected_bytes:
        raise ValueError('Raw matrix size differs from authorized source registry')
    h = hashlib.sha256()
    seen, selected = set(), []
    n_rows = 0
    with source.open('rb') as handle:
        header = handle.readline()
        h.update(header)
        fields = header.decode('utf-8-sig').rstrip('\r\n').split('\t')
        if fields[0] != 'ProbeID' or len(fields[1:]) != len(set(fields[1:])):
            raise ValueError('Invalid matrix header or duplicate sample identifier')
        for line in handle:
            h.update(line)
            n_rows += 1
            if line.count(b'\t') != len(fields) - 1:
                raise ValueError('Ragged source row')
            probe = line.partition(b'\t')[0].decode('ascii')
            if probe in seen:
                raise ValueError('Duplicate source probe identifier')
            seen.add(probe)
            if probe in targets:
                selected.append(line)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Raw matrix changed during streaming read')
    if h.hexdigest() != expected_sha:
        raise ValueError('Raw matrix SHA256 differs from authorized registry')
    output.write_bytes(header + b''.join(selected))
    output.chmod(0o600)
    return dict(sha256=h.hexdigest(), bytes=before.st_size, source_probe_rows=n_rows,
                n_samples=len(fields) - 1, n_registered_present=len(selected),
                n_registered_absent=len(targets - seen),
                extracted_sha256=sha(output),
                structure_scope='all rows; numeric/range checks only on retained registered probes')


def load_beta(path):
    """Equivalent missing/range rules to the archived Korean paired loader."""
    with path.open(newline='') as handle:
        header = next(csv.reader(handle, delimiter='\t'))
    samples = [str(x).strip() for x in header[1:]]
    if len(samples) != len(set(samples)):
        raise ValueError('Duplicate beta sample identifier')
    raw = pd.read_csv(path, sep='\t', index_col=0, dtype=str, keep_default_na=False)
    if raw.index.duplicated().any():
        raise ValueError('Duplicate beta probe identifier')
    raw = raw.map(lambda v: np.nan if str(v).strip() in MISSING else str(v).strip())
    beta = raw.apply(pd.to_numeric, errors='raise').astype(float)
    beta.index = beta.index.astype(str)
    beta.columns = samples
    values = beta.to_numpy()
    if np.isinf(values).any() or ((values < 0) | (values > 1)).any():
        raise ValueError('Nonfinite or out-of-range beta value')
    return beta


def load_metadata(path, beta_samples, cohort):
    meta = pd.read_csv(path, sep='\t', dtype=str)
    required = {'cohort', 'sample_id', 'patient_id', 'tissue', 'pair_verified'}
    if not required <= set(meta):
        raise ValueError('Metadata missing required columns')
    for c in required:
        if meta[c].isna().any() or meta[c].str.strip().eq('').any():
            raise ValueError('Blank required metadata value')
        meta[c] = meta[c].str.strip()
    if meta.sample_id.duplicated().any() or set(meta.sample_id) != set(beta_samples):
        raise ValueError('Metadata sample identifiers not unique/exactly aligned with matrix')
    if set(meta.cohort) != {cohort} or not set(meta.tissue) <= {'T', 'N'}:
        raise ValueError('Invalid cohort or tissue labels')
    meta['pair_verified'] = meta.pair_verified.map(parse_bool)
    verified = meta[meta.pair_verified]
    if verified.duplicated(['patient_id', 'tissue']).any():
        raise ValueError('Duplicate verified patient/tissue')
    if any(set(x) != {'T', 'N'} for _, x in verified.groupby('patient_id').tissue):
        raise ValueError('Incomplete verified tumor/adjacent pair')
    return meta


def paired_delta(beta, meta, expected_pairs=None):
    verified = meta[meta.pair_verified]
    t = verified[verified.tissue.eq('T')].set_index('patient_id')
    n = verified[verified.tissue.eq('N')].set_index('patient_id')
    patients = sorted(set(t.index) & set(n.index))
    if expected_pairs is not None and len(patients) != expected_pairs:
        raise ValueError('Paired patient count differs from frozen cohort')
    delta = beta[t.loc[patients, 'sample_id']].to_numpy().T - beta[n.loc[patients, 'sample_id']].to_numpy().T
    return pd.DataFrame(delta, index=patients, columns=beta.index)


def bh_fixed(p_values, eligible, family_size=FAMILY_SIZE):
    """BH with all unavailable/ineligible hypotheses fixed at P=1."""
    p = np.asarray(p_values, float)
    eligible = np.asarray(eligible, bool)
    if len(p) != family_size or np.any(eligible & ~np.isfinite(p)):
        raise ValueError('Invalid fixed hypothesis family')
    internal = np.where(eligible, p, 1.0)
    order = np.argsort(internal, kind='stable')
    q = np.empty(family_size)
    q[order] = np.minimum(1, np.minimum.accumulate(
        (internal[order] * family_size / np.arange(1, family_size + 1))[::-1])[::-1])
    q[~eligible] = np.nan
    return q


def mean_test(delta):
    good = delta[np.isfinite(delta)]
    if len(good) < 3:
        return np.nan, 'insufficient_pairs'
    if np.all(good == 0):
        return 1.0, 'constant_zero'
    if np.all(good == good[0]):
        return 0.0, 'constant_nonzero'
    return float(stats.ttest_1samp(good, 0).pvalue), 'estimated'


def bootstrap_means(delta, seed, n_boot=N_BOOT):
    """Resample complete patient vectors, preserving cross-probe dependence."""
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, delta.shape[1]), np.nan)
    for b in range(n_boot):
        x = delta[rng.integers(0, len(delta), len(delta))]
        counts = np.isfinite(x).sum(axis=0)
        out[b] = np.divide(np.nansum(x, axis=0), counts,
                           out=np.full(delta.shape[1], np.nan), where=counts >= 3) * 100
    return out


def interval(values):
    good = np.asarray(values)[np.isfinite(values)]
    if len(good) < .95 * len(values):
        return np.nan, np.nan, len(good)
    lo, hi = np.quantile(good, [.025, .975])
    return float(lo), float(hi), len(good)


def probe_effects(delta, universe, cohort, seed):
    present = universe.probe.isin(delta.columns).to_numpy()
    x = delta.reindex(columns=universe.probe).to_numpy()
    boots = bootstrap_means(x, seed)
    rows = []
    for j, r in enumerate(universe.itertuples(index=False)):
        a = x[:, j]
        good = a[np.isfinite(a)]
        estimable = present[j] and len(good) >= 3
        eligible = estimable and not r.MASK_general and r.is_cpg
        p, method = mean_test(a)
        ci_lo, ci_hi, n_valid = interval(boots[:, j])
        status = 'absent_matrix_row' if not present[j] else 'insufficient_pairs' if not estimable else 'non_cpg_descriptive' if not r.is_cpg else 'masked_descriptive' if r.MASK_general else 'tested'
        rows.append(dict(cohort=cohort, probe=r.probe, selected=r.selected,
                         probe_type=r.probe_type, is_cpg=r.is_cpg,
                         MASK_general=r.MASK_general, pooled_location=r.pooled_location,
                         present=present[j], n_cohort_pairs=len(x), n_complete_pairs=len(good),
                         n_incomplete_pairs=len(x)-len(good), estimable=estimable,
                         inferential_eligible=eligible, status=status,
                         mean_delta_pp=float(good.mean()*100) if estimable else np.nan,
                         sd_delta_pp=float(good.std(ddof=1)*100) if estimable else np.nan,
                         ci_low_pp=ci_lo if eligible else np.nan,
                         ci_high_pp=ci_hi if eligible else np.nan,
                         bootstrap_valid=n_valid if eligible else 0, n_boot=N_BOOT, seed=seed,
                         ci_status='percentile_patient_bootstrap' if eligible else 'not_reported_ineligible',
                         n_tumor_higher=int((good>0).sum()), n_tumor_lower=int((good<0).sum()),
                         n_equal=int((good==0).sum()), paired_t_p=p if eligible else np.nan,
                         test_status=method if eligible else 'not_tested'))
    out = pd.DataFrame(rows)
    out['paired_t_q391'] = bh_fixed(out.paired_t_p, out.inferential_eligible)
    return out, boots


def summarize_groups(effects, mapping, boots, cohort):
    """Fixed-site effect distributions; no gene mean or probe-independent test."""
    idx = {p: i for i, p in enumerate(effects.probe)}
    rows, boot_rows = [], []
    cpg_effects = effects[effects.is_cpg]
    joined = mapping.merge(cpg_effects, on='probe', validate='many_to_one')
    group_specs = []
    for selection, flag in [('selected', True), ('unselected', False)]:
        sel = cpg_effects[cpg_effects.selected.eq(flag)]
        group_specs.append(('ALL', 'all_locations', selection, sel))
        for loc in ['promoter', 'body', 'other']:
            group_specs.append(('ALL', loc, selection, sel[sel.pooled_location.eq(loc)]))
        for gene in GENES:
            sub = joined[joined.gene.eq(gene) & joined.selected.eq(flag)]
            group_specs.append((gene, 'all_locations', selection, sub))
            for loc in ['promoter', 'body', 'other']:
                group_specs.append((gene, loc, selection, sub[sub.location.eq(loc)]))
    for gene, loc, selection, sub in group_specs:
        sub = sub.drop_duplicates('probe')
        for scope in ['all_measured_descriptive', 'annotation_unmasked']:
            use = sub[sub.estimable]
            if scope == 'annotation_unmasked':
                use = use[~use.MASK_general]
            e = use.mean_delta_pp.to_numpy()
            q = use.paired_t_q391.to_numpy()
            pos = [idx[p] for p in use.probe]
            if len(pos):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    med_boot = np.nanmedian(boots[:, pos], axis=1)
                lo, hi, valid = interval(med_boot)
                median = float(np.median(e))
                quartiles = np.quantile(e, [.25, .75])
            else:
                med_boot = np.full(N_BOOT, np.nan)
                lo, hi, valid = np.nan, np.nan, 0
                median, quartiles = np.nan, [np.nan, np.nan]
            if scope == 'all_measured_descriptive':
                lo, hi, valid = np.nan, np.nan, 0
            row = dict(cohort=cohort, gene=gene, location=loc, selection=selection, scope=scope,
                       n_registered=len(sub), n_present=int(sub.present.sum()),
                       n_absent=int((~sub.present).sum()), n_masked_registered=int(sub.MASK_general.sum()),
                       n_masked_present=int((sub.MASK_general & sub.present).sum()),
                       n_insufficient_present=int((sub.present & ~sub.estimable).sum()),
                       n_analyzed=len(e), min_complete_pairs=int(use.n_complete_pairs.min()) if len(e) else 0,
                       max_complete_pairs=int(use.n_complete_pairs.max()) if len(e) else 0,
                       median_delta_pp=median, q25_delta_pp=quartiles[0], q75_delta_pp=quartiles[1],
                       min_delta_pp=float(e.min()) if len(e) else np.nan,
                       max_delta_pp=float(e.max()) if len(e) else np.nan,
                       n_positive=int((e>0).sum()), n_negative=int((e<0).sum()), n_zero=int((e==0).sum()),
                       n_positive_q391_lt_005=int(((e>0)&(q<.05)).sum()),
                       n_negative_q391_lt_005=int(((e<0)&(q<.05)).sum()),
                       median_ci_low_pp=lo, median_ci_high_pp=hi, bootstrap_valid=valid,
                       summary_type='distribution of individual CpG effects; not gene-level test')
            rows.append(row)
            if scope == 'annotation_unmasked' and gene == 'ALL':
                boot_rows.extend(dict(cohort=cohort, location=loc, selection=selection,
                                      bootstrap=b+1, median_delta_pp=v)
                                 for b, v in enumerate(med_boot))
    return pd.DataFrame(rows), pd.DataFrame(boot_rows)


def coverage_by_type(effects, cohort):
    rows = []
    for selection, flag in [('selected', True), ('unselected', False)]:
        for typ in ['cg','ch']:
            e = effects[effects.selected.eq(flag)&effects.probe_type.eq(typ)]
            rows.append(dict(cohort=cohort, selection=selection, probe_type=typ,
                             n_registered=len(e), n_present=int(e.present.sum()),
                             n_absent=int((~e.present).sum()),
                             n_masked_registered=int(e.MASK_general.sum()),
                             n_annotation_unmasked_registered=int((~e.MASK_general).sum()),
                             n_masked_present=int((e.MASK_general&e.present).sum()),
                             n_estimable=int(e.estimable.sum()),
                             n_inferential_eligible=int(e.inferential_eligible.sum())))
    return pd.DataFrame(rows)


def validate_registry(registry, plan):
    manifest = json.loads((registry/'REGISTRY_MANIFEST.json').read_text())
    if sha(plan) != manifest['plan_sha256']:
        raise ValueError('Frozen plan hash mismatch')
    for name, digest in manifest['files'].items():
        if sha(registry/name) != digest:
            raise ValueError('Frozen registry hash mismatch')
    u = pd.read_csv(registry/'probe_universe.tsv', sep='\t')
    for c in ['selected', 'MASK_general']:
        u[c] = u[c].map(parse_bool)
    if len(u) != FAMILY_SIZE or u.probe.duplicated().any() or int(u.selected.sum()) != 77:
        raise ValueError('Frozen universe count/uniqueness mismatch')
    mapping = pd.read_csv(registry/'probe_gene_mapping.tsv', sep='\t')
    if mapping.duplicated(['probe', 'gene']).any() or set(mapping.probe) != set(u.probe):
        raise ValueError('Probe/gene mapping mismatch')
    tm = json.loads((registry/'ANNOTATION_TYPE_MANIFEST.json').read_text())
    if (tm['original_registry_manifest_sha256'] != sha(registry/'REGISTRY_MANIFEST.json') or
            tm['zhou_sha256'] != manifest['input_sha256']['zhou_hg19'] or
            tm['type_registry_sha256'] != sha(registry/'probe_types.tsv') or
            tm['correction_sha256'] != sha(plan.parent/'ANNOTATION_TYPE_CORRECTION.md')):
        raise ValueError('Additive annotation-type correction hash mismatch')
    types = pd.read_csv(registry/'probe_types.tsv',sep='\t')
    if len(types) != 391 or set(types.probe_type) != {'cg','ch'} or types.probe_type.eq('cg').sum() != 388:
        raise ValueError('Expected precisely 388 cg and 3 ch probes')
    if set(types.loc[types.probe_type.eq('ch'),'probe']) != EXPECTED_CH:
        raise ValueError('Unexpected non-CpG set')
    types['is_cpg'] = types.is_cpg.map(parse_bool)
    if not np.array_equal(types.is_cpg,types.probe_type.eq('cg')):
        raise ValueError('CpG flag disagrees with source type')
    u = u.merge(types,on='probe',validate='one_to_one',sort=False)
    if u.loc[~u.is_cpg,'selected'].any():
        raise ValueError('Selected registry unexpectedly contains non-CpG probe')
    return u, mapping


def run(config, registry, plan, private_dir, output_dir, cached_provenance=None):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Aggregate output must be empty')
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    private_dir.chmod(0o700)
    universe, mapping = validate_registry(registry, plan)
    if set(config['cohorts']) != set(COHORTS):
        raise ValueError('Exactly three frozen cohorts are required')
    all_effects, all_groups, all_boots, all_coverage = [], [], [], []
    cached = json.loads(cached_provenance.read_text()) if cached_provenance else None
    if cached and (cached['plan_sha256'] != sha(plan) or cached['registry_manifest_sha256'] != sha(registry/'REGISTRY_MANIFEST.json')):
        raise ValueError('Cached extraction plan/registry differs')
    provenance = dict(analysis='post-hoc annotation-defined CpG scope',
                      plan_sha256=sha(plan), registry_manifest_sha256=sha(registry/'REGISTRY_MANIFEST.json'),
                      annotation_type_manifest_sha256=sha(registry/'ANNOTATION_TYPE_MANIFEST.json'),
                      annotation_type_correction_sha256=sha(plan.parent/'ANNOTATION_TYPE_CORRECTION.md'),
                      n_boot=N_BOOT, family_size=FAMILY_SIZE,
                      environment=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                                       scipy=scipy.__version__, platform=platform.platform()), cohorts={})
    for cohort, (expected_n, seed) in COHORTS.items():
        c = config['cohorts'][cohort]
        raw, metadata, selected_beta = (Path(c[k]) for k in ['raw_matrix', 'pair_metadata', 'selected_beta'])
        input_hashes = {'pair_metadata': sha(metadata), 'selected_beta': sha(selected_beta)}
        extraction = private_dir/(cohort+'_universe_beta.tsv')
        if cached:
            previous = cached['cohorts'][cohort]
            raw_audit = previous['raw']
            if (sha(extraction) != raw_audit['extracted_sha256'] or
                    input_hashes != previous['input_sha256'] or
                    raw_audit['sha256'] != c['raw_sha256'] or raw_audit['bytes'] != c['raw_bytes']):
                raise ValueError('Cached extraction/input hashes differ from completed raw audit')
        else:
            if extraction.exists():
                raise ValueError('Private extraction already exists; choose a new private run directory')
            raw_audit = extract_registered(raw, extraction, set(universe.probe), c['raw_sha256'], c['raw_bytes'])
        beta = load_beta(extraction)
        meta = load_metadata(metadata, beta.columns, cohort)
        delta = paired_delta(beta, meta, expected_n)
        old_beta = load_beta(selected_beta)
        old_meta = load_metadata(metadata, old_beta.columns, cohort)
        old_delta = paired_delta(old_beta, old_meta, expected_n)
        sel = set(universe.loc[universe.selected, 'probe']) & set(beta.index)
        if sel != set(old_beta.index):
            raise ValueError('Measured selected-probe membership differs from archived primary data')
        a = delta.loc[old_delta.index, old_delta.columns].to_numpy()
        b = old_delta.to_numpy()
        if not np.array_equal(a, b, equal_nan=True):
            raise ValueError('Selected-probe paired values changed')
        effects, boots = probe_effects(delta, universe, cohort, seed)
        groups, group_boots = summarize_groups(effects, mapping, boots, cohort)
        all_effects.append(effects)
        all_groups.append(groups)
        all_boots.append(group_boots)
        all_coverage.append(coverage_by_type(effects,cohort))
        if input_hashes != {'pair_metadata': sha(metadata), 'selected_beta': sha(selected_beta)}:
            raise ValueError('Metadata/selected reference changed during analysis')
        provenance['cohorts'][cohort] = dict(raw=raw_audit, input_sha256=input_hashes,
                                           n_pairs=len(delta), seed=seed,
                                           selected_paired_values_exact_match=True,
                                           n_selected_measured=len(sel),
                                           n_universe_measured=len(beta),
                                           n_cpg_measured=int((effects.is_cpg&effects.present).sum()),
                                           n_non_cpg_measured=int((~effects.is_cpg&effects.present).sum()),
                                           n_tested=int(effects.inferential_eligible.sum()))
        print(json.dumps(dict(cohort=cohort, n_pairs=len(delta), n_present=len(beta),
                              n_tested=int(effects.inferential_eligible.sum()),
                              selected_paired_values_exact_match=True)), flush=True)
    pd.concat(all_effects, ignore_index=True).to_csv(output_dir/'probe_effects.tsv', sep='\t', index=False)
    group_data = pd.concat(all_groups, ignore_index=True)
    group_data.to_csv(output_dir/'group_distributions.tsv', sep='\t', index=False)
    pd.concat(all_boots, ignore_index=True).to_csv(output_dir/'pooled_group_bootstrap.tsv', sep='\t', index=False)
    pd.concat(all_coverage, ignore_index=True).to_csv(output_dir/'coverage_by_probe_type.tsv',sep='\t',index=False)
    group_data[group_data.gene.eq('ALL') & group_data.location.eq('all_locations')].to_csv(
        output_dir/'coverage_and_pooled_summary.tsv', sep='\t', index=False)
    provenance['output_sha256'] = {p.name: sha(p) for p in sorted(output_dir.glob('*.tsv'))}
    provenance['code_sha256'] = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    provenance['regeneration_mode'] = 'verified_private_extractions' if cached else 'full_raw_stream'
    if cached:
        provenance['prior_raw_audit_provenance_sha256'] = sha(cached_provenance)
    (output_dir/'run_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--registry-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cached-provenance', type=Path,
                        help='Optional completed run provenance: verify private extraction hashes and recompute summaries without rereading raw matrices')
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.registry_dir, args.plan,
        args.private_dir, args.output_dir, args.cached_provenance)


if __name__ == '__main__':
    main()
