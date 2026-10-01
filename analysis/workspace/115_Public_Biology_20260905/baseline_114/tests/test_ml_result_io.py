"""Regression for the literal null-test label that pandas otherwise treats as NA."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from summarize_ml import load_frame
from build_ml_tables import maybe_read_frame


def test_null_permutation_label_survives_summary_and_workbook_readers(tmp_path):
    path = tmp_path / 'ridge_permutation.tsv'
    path.write_text('permutation\tauc\tkind\n-1\t0.9\tobserved\n0\t0.5\tnull\n')
    for frame in [load_frame(path, []), maybe_read_frame(path)]:
        assert frame['kind'].tolist() == ['observed', 'null']
        assert frame.loc[frame['kind'].eq('null'), 'auc'].tolist() == [0.5]


def test_nonestimable_numeric_results_remain_missing(tmp_path):
    path = tmp_path / 'metrics.csv'
    path.write_text('model,value,status\nridge,NA,nonestimable\n')
    for frame in [load_frame(path, []), maybe_read_frame(path)]:
        assert pd.isna(frame.loc[0, 'value'])
        assert frame.loc[0, 'status'] == 'nonestimable'
