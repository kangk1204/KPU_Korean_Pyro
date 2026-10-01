"""Check every ML figure source estimate against canonical numerical outputs."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'figures/source_data'


def check_ci(plot, ci, keys):
    if 'plot_metric' in plot:
        plot = plot.rename(columns={'plot_metric': 'metric'})
    merged = plot.merge(ci, on=keys, suffixes=('_plot', '_canonical'), validate='one_to_one')
    assert len(merged) == len(plot)
    np.testing.assert_allclose(merged.plot_value, merged.point, rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(merged.plot_low, merged.ci_low, rtol=1e-10, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(merged.plot_high, merged.ci_high, rtol=1e-10, atol=1e-12, equal_nan=True)
    assert plot.analysis_status.eq('final').all()


def test_f6_all_model_metrics_and_intervals():
    local = pd.read_csv(SRC / 'F6_tissue_local_metrics_source_data.tsv', sep='\t')
    assert len(local) == 13
    ci = pd.read_csv(ROOT / 'results/ml/tissue/bootstrap_ci.tsv', sep='\t')
    check_ci(local, ci, ['model', 'metric'])
    public = pd.read_csv(SRC / 'F6_tissue_public_metrics_source_data.tsv', sep='\t')
    assert set(public.cohort) == {'colonomics', 'gse119526'}
    for cohort, row in public.groupby('cohort'):
        ci = pd.read_csv(ROOT / 'results/ml/public' / cohort / 'bootstrap_ci.tsv', sep='\t')
        check_ci(row, ci, ['model', 'metric'])


def test_f7_all_primary_sensitivity_and_delta_values():
    ci = pd.read_csv(ROOT / 'results/ml/recurrence/primary_bootstrap_ci.csv').rename(columns={'estimate': 'point'})
    keys = ['block', 'model', 'horizon_days', 'metric']
    for name, count in [('primary_metrics', 9), ('brier', 3)]:
        plot = pd.read_csv(SRC / f'F7_recurrence_{name}_source_data.tsv', sep='\t')
        assert len(plot) == count
        check_ci(plot, ci, keys)
    delta = pd.read_csv(SRC / 'F7_recurrence_delta_source_data.tsv', sep='\t')
    assert len(delta) == 1
    ref = ci[ci.block.eq('combined_minus_clinical') & ci.metric.eq('delta_uno_c') & ci.horizon_days.eq(1825)]
    np.testing.assert_allclose(delta[['point', 'ci_low', 'ci_high']], ref[['point', 'ci_low', 'ci_high']])
    sens = pd.read_csv(SRC / 'F7_recurrence_sensitivity_source_data.tsv', sep='\t')
    ref = pd.read_csv(ROOT / 'results/ml/recurrence/ridge_sensitivity_metrics.csv')
    assert len(sens) == len(ref) == 14
    cols = ['sensitivity', 'horizon_days']
    a = sens.set_index(cols).sort_index()
    b = ref.set_index(cols).sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a.uno_c_mean, b.uno_c_mean, rtol=1e-10, atol=1e-12, equal_nan=True)
