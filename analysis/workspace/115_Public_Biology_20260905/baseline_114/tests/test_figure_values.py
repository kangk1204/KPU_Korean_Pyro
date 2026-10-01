"""Every plotted estimate must be the verified result, not a second fit."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'figures/source_data'


def assert_columns_match(plotted, canonical, mapping):
    for display, original in mapping.items():
        np.testing.assert_allclose(plotted[display],canonical[original],rtol=1e-9,atol=1e-11,
                                   err_msg=f'Figure estimate differs from canonical {original}')


def test_f2_paired_effects_are_canonical():
    plot=pd.read_csv(SOURCE/'F2_paired_effect_source_data.tsv',sep='\t').set_index('gene')
    ref=pd.read_csv(ROOT/'results/paired.csv').set_index('gene').loc[plot.index]
    assert_columns_match(plot,ref,{'paired_delta_mean_pctpt':'mean_difference_pp',
                                  'paired_delta_ci_low_pctpt':'mean_diff_bca95_low',
                                  'paired_delta_ci_high_pctpt':'mean_diff_bca95_high',
                                  'paired_t_p_bh10':'paired_t_BH_q'})


def test_f3_pca_is_canonical():
    plot=pd.read_csv(SOURCE/'F3_pca_scores_source_data.tsv',sep='\t').set_index('study_id')
    ref=pd.read_csv(ROOT/'results/pca_scores.csv').set_index('study_id').loc[plot.index]
    assert_columns_match(plot,ref,{'PC1':'PC1','PC2':'PC2'})
    load=pd.read_csv(SOURCE/'F3_pca_loadings_source_data.tsv',sep='\t').set_index('gene')
    load_ref=pd.read_csv(ROOT/'results/pca_loadings.csv').set_index('gene').loc[load.index]
    assert_columns_match(load,load_ref,{'PC1_loading':'PC1_loading','PC2_loading':'PC2_loading'})


def test_f4_public_effects_are_canonical():
    plot=pd.read_csv(SOURCE/'F4_external_gene_effect_source_data.tsv',sep='\t')
    ref=pd.read_csv(ROOT/'results/external/paired_public_effects.tsv',sep='\t')
    for cohort in ['colonomics','gse119526']:
        a=plot.loc[plot.cohort.eq(cohort)].set_index('gene')
        b=ref.loc[ref.cohort.eq(cohort)].set_index('gene').loc[a.index]
        assert len(a)==10
        assert_columns_match(a,b,{'paired_diff_mean':'mean_delta_beta',
                                  'paired_diff_ci_low':'bca95_low','paired_diff_ci_high':'bca95_high'})


def test_f5_cox_is_canonical_r_output():
    for label,filename in [('primary','recurrence.csv'),('all_stage','recurrence_all_stage.csv')]:
        plot=pd.read_csv(SOURCE/f'F5_cox_{label}_source_data.tsv',sep='\t').set_index('gene')
        ref=pd.read_csv(ROOT/'results'/filename).set_index('gene').loc[plot.index]
        assert_columns_match(plot,ref,{'HR_per_10_pctpt':'hr_per10','HR_ci_low':'ci_lower',
                                      'HR_ci_high':'ci_upper','cox_p':'p_value','cox_p_bh10':'p_bh','PH_p':'ph_p_value'})

