"""Rebuild manuscript Table S11 display files from public aggregate outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ['CMCBSN', 'SNUH', 'ASAN']


def num(value):
    return f'{value:.2f}' if np.isfinite(value) else 'NA'


def build(aggregate_dir, output_dir):
    groups = pd.read_csv(aggregate_dir/'group_distributions.tsv', sep='\t')
    coverage = pd.read_csv(aggregate_dir/'coverage_by_probe_type.tsv',sep='\t')
    d = groups[groups.gene.eq('ALL') & groups.scope.eq('annotation_unmasked')]
    table_a = dict(id='S11A', title='Table S11A. Coverage of annotation-defined selected and unselected CpGs.',
                   columns=['Cohort', 'Selection', 'Annotated probes (CpGs)', 'Measured probes (CpGs)',
                            'Measured, masked CpGs', 'Analyzed CpGs', 'Pairs per analyzed CpG'], rows=[],
                   footnote=('The fixed Zhou HM450 hg19 union contains 391 unique array probes linked to the ten target genes. '
                             'These comprise 77 originally selected and 314 unselected probes. The source annotation identified 388 CpG '
                             'probes and 3 unselected non-CpG probes. The latter are retained in the coverage record but excluded '
                             'from CpG-specific inference and distributions. Parentheses give CpG-only counts. '
                             'Measured denotes row presence in the supplied '
                             'processed matrix. Analyzed CpGs are measured, have MASK_general=False, and have at least '
                             'three complete tumor–adjacent patient pairs. The 18 annotation-masked CpGs include 12 measured '
                             'in each cohort (2 selected and 10 unselected). Each matrix lacks 109 registered probes '
                             '(107 CpG and 2 non-CpG probes). Cohort-specific measured membership is retained despite identical totals. '
                             'No missing beta values occurred in paired measurements of present probes. Annotation filtering '
                             'does not establish detection-P or laboratory QC. Existing 77-probe primary results are unchanged.'))
    table_b = dict(id='S11B', title='Table S11B. Location-specific paired effects at selected and unselected CpGs.',
                   columns=['Cohort', 'Location', 'Selection', 'CpGs', 'Median Δ, pp (95% CI)',
                            'Range Δ, pp', 'Positive / negative', 'Significant positive / negative'], rows=[],
                   footnote=('Only measured CpGs with MASK_general=False and at least three complete patient pairs are included. '
                             'Each CpG effect is its patient-mean tumor-minus-adjacent beta difference multiplied by 100. '
                             'The median and range describe the distribution of these individual-CpG effects, not a gene-level '
                             'methylation score. Promoter denotes gene-matched TSS200, TSS1500, 5′UTR or 1stExon RefGene groups. '
                             'Body denotes Body without a promoter group. Other includes 3′UTR and no matching target-gene group. '
                             'Promoter takes priority for overlapping transcript groups. Positive/negative counts use mean-effect '
                             'direction. BH counts use two-sided paired t tests with Q<0.05, retaining the fixed 391-probe family '
                             'per cohort. Non-CpG, absent, masked or insufficient-pair hypotheses contribute internal P=1 and have no '
                             'reported P/Q. There are no zero mean effects among analyzed probes. Confidence intervals are '
                             'percentile intervals from 2,000 bootstrap draws of whole paired patient vectors, using seeds '
                             '2026091501/2026091502/2026091503 for CMCBSN/SNUH/ASAN. Probe means and group medians are recomputed '
                             'within each draw. Intervals condition on the frozen annotation, supplied matrices and previously '
                             'analysed cohorts. No selected-versus-unselected or gene-level hypothesis test was performed. '
                             'Full per-gene/location distributions and per-probe values are provided in Supplementary Data 1.'))
    for cohort in ORDER:
        for selection in ['selected', 'unselected']:
            r = d[d.cohort.eq(cohort)&d.selection.eq(selection)&d.location.eq('all_locations')].iloc[0]
            cov = coverage[coverage.cohort.eq(cohort)&coverage.selection.eq(selection)]
            n_pairs = str(int(r.min_complete_pairs)) if r.min_complete_pairs == r.max_complete_pairs else f'{int(r.min_complete_pairs)}–{int(r.max_complete_pairs)}'
            table_a['rows'].append([cohort, selection.capitalize(), f'{int(cov.n_registered.sum())} ({int(r.n_registered)})',
                                    f'{int(cov.n_present.sum())} ({int(r.n_present)})',
                                    int(r.n_masked_present), int(r.n_analyzed), n_pairs])
        for loc in ['promoter', 'body', 'other']:
            for selection in ['selected', 'unselected']:
                r = d[d.cohort.eq(cohort)&d.selection.eq(selection)&d.location.eq(loc)].iloc[0]
                table_b['rows'].append([cohort, loc.capitalize(), selection.capitalize(), int(r.n_analyzed),
                                        f'{num(r.median_delta_pp)} ({num(r.median_ci_low_pp)} to {num(r.median_ci_high_pp)})',
                                        f'{num(r.min_delta_pp)} to {num(r.max_delta_pp)}',
                                        f'{int(r.n_positive)} / {int(r.n_negative)}',
                                        f'{int(r.n_positive_q391_lt_005)} / {int(r.n_negative_q391_lt_005)}'])
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = [table_a, table_b]
    for table in tables:
        table['display_file'] = f'Table_{table["id"]}_display.tsv'
        pd.DataFrame(table['rows'], columns=table['columns']).to_csv(output_dir/table['display_file'], sep='\t', index=False)
    (output_dir/'table_specs.json').write_text(json.dumps(dict(schema_version=1, tables=tables), indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(tables=[t['id'] for t in tables], rows=[len(t['rows']) for t in tables])))


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aggregate-dir', type=Path, default=root/'public/aggregates')
    p.add_argument('--output-dir', type=Path, default=root/'public')
    args = p.parse_args()
    build(args.aggregate_dir, args.output_dir)


if __name__ == '__main__':
    main()
