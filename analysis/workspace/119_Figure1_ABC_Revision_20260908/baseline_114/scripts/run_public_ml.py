#!/usr/bin/env python3
"""Fit the fixed ridge procedure within each frozen public tissue cohort."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.tissue import TissueConfig, canonical_hash, dataframe_hash, prepare_tissue_frame, run_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', choices=['all', 'colonomics', 'gse119526'], default='all')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--workers', type=int, choices=[1,2,3], default=3)
    args = parser.parse_args()
    cfg_path = ROOT / 'registry/ml_config.json'
    cfg = replace(TissueConfig.from_json(cfg_path), max_workers=args.workers)
    cohorts = ['colonomics', 'gse119526'] if args.cohort == 'all' else [args.cohort]
    for cohort in cohorts:
        input_path = ROOT / f'data/ml/public_{cohort}.tsv'
        outdir = ROOT / 'results/ml/public' / cohort
        outdir.mkdir(parents=True, exist_ok=True)
        signature = canonical_hash(
            [input_path, cfg_path, ROOT/'ml/tissue.py', Path(__file__)],
            [{'model': 'ridge', 'repeats': cfg.outer_repeats,
              'bootstrap': cfg.bootstrap_replicates, 'optimism': 0, 'permutations': 0}],
        )
        data = pd.read_csv(input_path, sep='\t')
        done = outdir / 'run_manifest.json'
        required = ['metrics.tsv', 'bootstrap_ci.tsv', 'bootstrap_draws.tsv.gz',
                    'oof_predictions.tsv.gz', 'folds.tsv', 'candidate_results.tsv.gz',
                    'model_states.tsv.gz', 'manifest.json']
        if done.exists() and not args.force:
            old = json.loads(done.read_text())
            fitted = json.loads((outdir/'manifest.json').read_text()) if (outdir/'manifest.json').exists() else {}
            model_files = list((outdir/'models').glob('*.pkl.gz'))
            if (old.get('run_hash') == signature
                    and fitted.get('source_sha256') == hashlib.sha256((ROOT/'ml/tissue.py').read_bytes()).hexdigest()
                    and fitted.get('data_sha256') == dataframe_hash(prepare_tissue_frame(data))
                    and len(model_files) == cfg.outer_folds*cfg.outer_repeats
                    and all((outdir/p).is_file() and (outdir/p).stat().st_size > 0 for p in required)):
                print(json.dumps({'cohort': cohort, 'status': 'cached'}), flush=True)
                continue
        expected = {'colonomics': 92, 'gse119526': 48}[cohort]
        assert data.study_id.nunique() == expected and len(data) == expected*2
        print(json.dumps({'cohort': cohort, 'status': 'started', 'patients': expected}), flush=True)
        run_cohort(data, outdir, models=['ridge'], repeats=cfg.outer_repeats,
                   config=cfg, bootstrap=cfg.bootstrap_replicates,
                   optimism=0, permutations=0)
        end_signature = canonical_hash(
            [input_path, cfg_path, ROOT/'ml/tissue.py', Path(__file__)],
            [{'model': 'ridge', 'repeats': cfg.outer_repeats,
              'bootstrap': cfg.bootstrap_replicates, 'optimism': 0, 'permutations': 0}],
        )
        if end_signature != signature:
            raise RuntimeError('Public input, configuration, or source changed during run')
        done.write_text(json.dumps({
            'status': 'complete', 'cohort': cohort, 'run_hash': signature,
            'patients': expected, 'specimens': len(data),
            'outer_folds': cfg.outer_folds, 'outer_repeats': cfg.outer_repeats,
            'inner_folds': cfg.inner_folds, 'bootstrap_replicates': cfg.bootstrap_replicates,
            'model': 'ridge',
            'workers': args.workers,
            'role': 'Cohort-local grouped nested CV; not frozen pyrosequencing model validation',
            'interval_scope': 'Patient-cluster bootstrap conditional on fitted CV models',
        }, indent=2)+'\n')
        print(json.dumps({'cohort': cohort, 'status': 'complete'}), flush=True)


if __name__ == '__main__':
    main()
