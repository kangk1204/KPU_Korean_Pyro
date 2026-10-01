#!/usr/bin/env python3
"""Discordant adjacent-mucosa sample sensitivity (Table S2): primary nested tissue classifier.

Re-runs the retained 114 ridge and inner-selected single-gene models on the 86 remaining
patients with the locked configuration (5 outer folds x 20 repeats, 4 inner folds, 2,000
patient bootstrap resamples). The excluded patient is selected by the rule in paired.py.
Requires the restricted 114 input data/ml/tissue.tsv. Writes tissue_ml_bootstrap_ci.tsv.
"""
import os, sys
from pathlib import Path
import pandas as pd
os.environ.setdefault('OMP_NUM_THREADS', '1')
HERE = Path(__file__).resolve().parent
W114 = Path(os.environ.get('KPU_WORKSPACE_114', HERE.parent / '114_ML_DataDriven_20260905'))
sys.path.insert(0, str(W114))
from ml.tissue import GENES, TissueConfig, run_cohort
cfg = TissueConfig.from_json(W114 / 'registry/ml_config.json')
df = pd.read_csv(W114 / 'data/ml/tissue.tsv', sep='\t')
normal_mean = df[df.tissue == 'N'].set_index('study_id')[GENES].mean(axis=1)
drop = normal_mean[normal_mean > 30].index.tolist()
assert len(drop) == 1, drop
df = df[~df.study_id.isin(drop)]
outdir = Path(os.environ.get('KPU_SENSITIVITY_OUT', HERE / 'private_run'))
res = run_cohort(df, outdir, models=['ridge', 'best_single_gene'], repeats=cfg.outer_repeats, config=cfg,
                 bootstrap=cfg.bootstrap_replicates, optimism=0, permutations=0, genes=GENES)
res['bootstrap_ci'].to_csv(HERE / 'tissue_ml_bootstrap_ci.tsv', sep='\t', index=False)
print(res['bootstrap_ci'].query("metric == 'auc'").to_string())
