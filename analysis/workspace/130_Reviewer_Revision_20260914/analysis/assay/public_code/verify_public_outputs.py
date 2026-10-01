"""Validate public recomputation against retained aggregates and reference bases."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from map_coordinates import bisulfite, revcomp, find_compatible
from qc_sensitivity import bh, _bca_mean_ci, N_BOOT


def main(args):
    if not (bisulfite('ACGCCGCA') == 'AYGTYGTA'):
        raise AssertionError
    if not (revcomp(revcomp('ACGTYR')) == 'ACGTYR'):
        raise AssertionError
    if not (find_compatible('R', 'Y') == []):
        raise AssertionError
    np.testing.assert_allclose(bh([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
    synthetic = np.full(5, -3.0)
    indices = np.random.default_rng(9).integers(0, 5, size=(N_BOOT, 5))
    ci = _bca_mean_ci(synthetic, indices)
    if not (ci['mean_diff_bca95_low'] == ci['mean_diff_bca95_high'] == -3):
        raise AssertionError
    expected = pd.read_csv(args.assays_dir/'representative_cpg_coordinates.tsv', sep='\t')
    actual = pd.read_csv(args.mapping_dir/'representative_cpg_coordinates.tsv', sep='\t')
    pd.testing.assert_frame_equal(expected, actual)
    if not (len(actual) == 76):
        raise AssertionError
    for row in actual.itertuples():
        ref = json.loads((args.reference_dir/f'{row.gene}_hg19.json').read_text())
        i = row.cpg_dyad_plus_c_1based-ref['start']-1
        if not (ref['dna'][i:i+2].upper() == 'CG'):
            raise AssertionError
        if not (row.cpg_dyad_plus_g_1based == row.cpg_dyad_plus_c_1based+1):
            raise AssertionError
        expected_c = row.cpg_dyad_plus_c_1based if row.bisulfite_target_strand=='+' else row.cpg_dyad_plus_g_1based
        if not (row.bisulfite_target_c_coordinate_1based == expected_c):
            raise AssertionError
    expected = pd.read_csv(args.assays_dir/'array_assay_mapping.tsv', sep='\t')
    actual = pd.read_csv(args.mapping_dir/'array_assay_mapping.tsv', sep='\t')
    # The portable script records a filename rather than an original workspace path.
    pd.testing.assert_frame_equal(expected.drop(columns='array_coordinate_source'),
                                  actual.drop(columns='array_coordinate_source'))
    if not (len(actual)==77 and actual.array_probe.is_unique):
        raise AssertionError
    if not (actual.representative_CpG_overlap.sum()==10):
        raise AssertionError
    subset = actual[actual.present_in_all_three_Korean_processed_arrays]
    if not (len(subset)==56 and subset.representative_CpG_overlap.sum()==5):
        raise AssertionError
    expected = pd.read_csv(args.assays_dir/'assay_definition.tsv', sep='\t')
    actual = pd.read_csv(args.mapping_dir/'assay_definition.tsv', sep='\t')
    pd.testing.assert_frame_equal(expected, actual)
    sensitivity_checked = False
    if args.sensitivity_dir:
        expected = pd.read_csv(args.assays_dir/'flagged_summary_sensitivity.tsv', sep='\t')
        actual = pd.read_csv(args.sensitivity_dir/'flagged_summary_sensitivity.tsv', sep='\t')
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
        if not (len(actual)==30):
            raise AssertionError
        sensitivity_checked = True
    print(json.dumps({'status':'PASS','synthetic_core_checks':True,'coordinate_rows_exact':76,
        'array_rows_exact_except_source_path':77,'assay_rows_exact':10,
        'sensitivity_rows_exact':30 if sensitivity_checked else 'not_checked_without_authorized_inputs'}, indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assays-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--mapping-dir', type=Path, required=True)
    parser.add_argument('--sensitivity-dir', type=Path)
    main(parser.parse_args())
