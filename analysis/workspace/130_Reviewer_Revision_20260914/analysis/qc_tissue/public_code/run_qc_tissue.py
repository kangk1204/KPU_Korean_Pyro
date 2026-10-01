#!/usr/bin/env python3
"""Execute the frozen, post hoc primary-ridge exclusion sensitivity plan."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("KPU_PROJECT_ROOT", BASE.parents[2])).resolve()
PRIVATE = BASE / "private"
PUBLIC = BASE / "public"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = datetime.now(timezone.utc).isoformat()
    plan_file = PRIVATE / "PLAN_PRIVATE.json"
    plan = json.loads(plan_file.read_text())
    lock = json.loads((BASE / "PLAN_LOCK.json").read_text())
    if not (sha(plan_file) == lock["plan_private_sha256"]):
        raise AssertionError
    if not (sha(BASE / "PLAN.md") == lock["plan_md_sha256"]):
        raise AssertionError
    for relative, expected in plan["preflight_guarded_files"].items():
        if not (sha(PROJECT / relative) == expected):
            raise AssertionError(relative)

    source = PROJECT / plan["source_module"]
    sys.path.insert(0, str(source.parents[1]))
    from ml import tissue
    if not (Path(tissue.__file__).resolve() == source.resolve()):
        raise AssertionError
    config = tissue.TissueConfig.from_json(PROJECT / plan["config_path"])
    if not ((config.outer_folds, config.outer_repeats, config.inner_folds, config.seed) == (5, 20, 4, 20260905)):
        raise AssertionError
    if not (list(config.logistic_C) == plan["C_grid"]):
        raise AssertionError
    if not (tissue.GENES == plan["genes"]):
        raise AssertionError

    raw = pd.read_csv(PROJECT / plan["input_path"], sep="\t")
    original = tissue.prepare_tissue_frame(raw)
    if not (original.patient_id.nunique() == 87 and len(original) == 174):
        raise AssertionError
    excluded = set(plan["excluded_study_ids"])
    crosswalk = pd.read_csv(PROJECT / plan["crosswalk_path"], sep="\t")
    if not (set(crosswalk.study_id) == excluded):
        raise AssertionError
    for row in crosswalk.itertuples():
        value = original.loc[original.patient_id.eq(row.study_id) & original.y.eq(1), row.gene]
        if not (len(value) == 1 and float(value.iloc[0]) == float(row.summary)):
            raise AssertionError
    kept_raw = raw.loc[~raw.study_id.isin(excluded)].copy()
    data = tissue.prepare_tissue_frame(kept_raw)
    if not (data.patient_id.nunique() == 85 and len(data) == 170):
        raise AssertionError
    if not (not set(data.patient_id).intersection(excluded)):
        raise AssertionError
    if not ((data.groupby("patient_id").y.agg(["size", "sum"]) == [2, 1]).all().all()):
        raise AssertionError
    tissue.write_csv(kept_raw, PRIVATE / "tissue_exclusion_input_PRIVATE.tsv", sep="\t")

    folds = tissue.patient_kfolds(data.group_id.unique(), 5, 20, config.seed)
    patient_set = set(data.patient_id)
    inner_records = []
    for fold in folds:
        train, test = set(fold["train_patients"]), set(fold["test_patients"])
        if not (not train.intersection(test) and train.union(test) == patient_set):
            raise AssertionError
        if not (len(train) == 68 and len(test) == 17):
            raise AssertionError
        for phase, seed in (("tuning", config.seed), ("inherited_threshold", config.seed + 404)):
            for index, (itr, iva) in enumerate(tissue.inner_folds(train, 4, seed, fold["repeat"], fold["fold"])):
                if not (not set(itr).intersection(iva)):
                    raise AssertionError
                if not (set(itr).union(iva) == train):
                    raise AssertionError
                if not (not (set(itr) | set(iva)).intersection(test)):
                    raise AssertionError
                inner_records.append({"repeat": fold["repeat"], "fold": fold["fold"], "phase": phase,
                                      "inner_fold": index, "train_study_ids": list(itr), "valid_study_ids": list(iva)})
    (PRIVATE / "inner_folds_PRIVATE.json").write_text(json.dumps(inner_records, indent=2) + "\n")
    (PRIVATE / "execution_config_PRIVATE.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    print(json.dumps({"state": "started", "patients": 85, "specimens": 170,
                      "model": "ridge", "outer_splits": 100, "bootstrap": 2000}), flush=True)
    result = tissue.run_cohort(data, PRIVATE / "results", models=["ridge"], repeats=20,
                               config=config, bootstrap=2000, optimism=0, permutations=0, genes=tissue.GENES)

    oof = result["oof"]
    if not (set(oof.model) == {"ridge"}):
        raise AssertionError
    if not (len(oof) == 3400 and set(oof.study_id) == patient_set):
        raise AssertionError
    if not (oof.groupby(["repeat", "study_id"]).size().eq(2).all()):
        raise AssertionError
    if not (oof.groupby(["repeat", "study_id"]).fold.nunique().eq(1).all()):
        raise AssertionError
    if not (oof.groupby(["study_id", "sample_id"]).size().eq(20).all()):
        raise AssertionError
    if not (oof.groupby("repeat").size().eq(170).all()):
        raise AssertionError
    for fold in folds:
        model_path = PRIVATE / "results/models" / f"ridge_r{fold['repeat']:02d}_f{fold['fold']:02d}.pkl.gz"
        with gzip.open(model_path, "rb") as fh:
            fitted = pickle.load(fh)
        train = data.loc[data.group_id.isin(fold["train_patients"]), tissue.GENES].to_numpy(float)
        if not (fitted["model"].penalty == "l2" and fitted["model"].C in config.logistic_C):
            raise AssertionError
        np.testing.assert_allclose(fitted["scaler"].mean_, train.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(fitted["scaler"].var_, train.var(axis=0), rtol=0, atol=1e-12)
        if not (set(fitted["state"]["train_study_ids"]) == set(fold["train_patients"])):
            raise AssertionError
        if not (set(fitted["state"]["test_study_ids"]) == set(fold["test_patients"])):
            raise AssertionError

    # Independent AUC calculation and exact same patient-bootstrap resamples using pairwise ranks.
    repeat_auc = np.array([roc_auc_score(rep.y, rep.score) for _, rep in oof.groupby("repeat")])
    point = float(repeat_auc.mean())
    patients = np.asarray(sorted(patient_set))
    matrices = []
    for _, rep in oof.groupby("repeat", sort=True):
        tumour = rep.loc[rep.y.eq(1)].set_index("study_id").loc[patients, "score"].to_numpy(float)
        normal = rep.loc[rep.y.eq(0)].set_index("study_id").loc[patients, "score"].to_numpy(float)
        matrices.append((tumour[:, None] > normal[None, :]).astype(float)
                        + 0.5 * (tumour[:, None] == normal[None, :]))
    average_comparison = np.mean(matrices, axis=0)
    rng = np.random.default_rng(config.seed + 707)
    index = {p: i for i, p in enumerate(patients)}
    weights = np.array([np.bincount([index[p] for p in rng.choice(patients, len(patients), replace=True)],
                                  minlength=len(patients)) for _ in range(2000)])
    independently_bootstrapped = np.einsum("bi,ij,bj->b", weights, average_comparison, weights, optimize=True) / 85**2
    actual_draws = result["bootstrap_draws"].sort_values("bootstrap").auc.to_numpy(float)
    np.testing.assert_allclose(independently_bootstrapped, actual_draws, rtol=0, atol=1e-12)
    interval = result["bootstrap_ci"].query("model == 'ridge' and metric == 'auc'").iloc[0]
    np.testing.assert_allclose([point, *np.quantile(independently_bootstrapped, [.025, .975])],
                               [interval.point, interval.ci_low, interval.ci_high], rtol=0, atol=1e-12)
    tissue.write_csv(pd.DataFrame(weights, columns=patients), PRIVATE / "bootstrap_patient_weights_PRIVATE.tsv.gz", sep="\t")

    baseline_dir = source.parents[1] / "results/ml/tissue"
    old_oof = pd.read_csv(baseline_dir / "oof_predictions.tsv.gz", sep="\t").query("model == 'ridge'")
    if not (len(old_oof) == 3480 and old_oof.study_id.nunique() == 87):
        raise AssertionError
    old_auc = float(np.mean([roc_auc_score(rep.y, rep.score) for _, rep in old_oof.groupby("repeat")]))
    old_metrics = pd.read_csv(baseline_dir / "metrics.tsv", sep="\t")
    recorded = old_metrics.query("model == 'ridge' and scope == 'all_repeats_pooled'").iloc[0]
    if not (abs(old_auc - recorded.auc) < 1e-12):
        raise AssertionError
    old_ci = pd.read_csv(baseline_dir / "bootstrap_ci.tsv", sep="\t").query("model == 'ridge' and metric == 'auc'").iloc[0]
    if not (abs(old_auc - old_ci.point) < 1e-12):
        raise AssertionError
    for relative, expected in plan["preflight_guarded_files"].items():
        if not (sha(PROJECT / relative) == expected):
            raise AssertionError(relative)
    if not (sha(plan_file) == lock["plan_private_sha256"]):
        raise AssertionError

    aggregate = pd.DataFrame([
        {"analysis": "original_primary", "model": "ridge", "patients": 87, "specimens": 174,
         "tumour_specimens": 87, "normal_specimens": 87, "auc": old_auc,
         "ci_low": float(old_ci.ci_low), "ci_high": float(old_ci.ci_high), "bootstrap_replicates": 2000},
        {"analysis": "posthoc_exclusion_two_flagged_patient_pairs", "model": "ridge", "patients": 85,
         "specimens": 170, "tumour_specimens": 85, "normal_specimens": 85, "auc": point,
         "ci_low": float(interval.ci_low), "ci_high": float(interval.ci_high), "bootstrap_replicates": 2000},
    ])
    aggregate.to_csv(PUBLIC / "qc_tissue_aggregate.tsv", sep="\t", index=False)
    summary = {
        "status": "complete", "analysis_type": "post hoc complete-patient exclusion robustness",
        "primary_model": "ridge logistic regression", "features": "same ten genes as primary analysis",
        "outer_folds": 5, "outer_repeats": 20, "inner_folds": 4, "C_grid": list(config.logistic_C),
        "seed": config.seed, "auc_estimator": "mean of repeat-specific specimen-level outer out-of-fold AUCs",
        "ci_method": "95% percentile interval from 2000 complete-patient cluster resamples, conditional on fitted cross-validation models",
        "original": aggregate.iloc[0].to_dict(), "sensitivity": aggregate.iloc[1].to_dict(),
        "auc_difference_descriptive_only": point - old_auc,
        "contrast_limit": "The cohorts differ. No confidence interval or significance test was calculated for their AUC difference.",
        "qc_limit": "This sensitivity analysis does not reconstruct historical assay QC or establish assay acceptance.",
        "models_changed": False, "original_files_unchanged": True,
        "patient_outputs": "retained privately; public output contains only aggregate results and methods",
    }
    (PUBLIC / "qc_tissue_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    verification = {
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_plan_sha256": sha(plan_file), "script_sha256": sha(Path(__file__)),
        "original_source_module": str(source.relative_to(PROJECT)),
        "guarded_source_hashes_unchanged": plan["preflight_guarded_files"],
        "outer_folds_checked": 100, "inner_folds_checked": len(inner_records),
        "outer_scalers_match_only_training_rows": 100, "oof_rows": 3400,
        "conditional_bootstrap_draws_independently_verified": len(actual_draws),
        "max_abs_bootstrap_difference": float(np.max(np.abs(independently_bootstrapped - actual_draws))),
        "original_auc_independently_verified_from_stored_oof": old_auc,
        "excluded_study_ids": sorted(excluded), "public_summary": summary,
    }
    (PRIVATE / "verification_PRIVATE.json").write_text(json.dumps(verification, indent=2) + "\n")
    (PUBLIC / "qc_tissue_report.md").write_text(
        "# Post hoc tissue-classification exclusion sensitivity\n\n"
        f"After removing two complete patient pairs linked to representative flagged assay measurements, "
        f"the same ridge analysis gave AUC {point:.3f} (95% conditional CI {interval.ci_low:.3f}–{interval.ci_high:.3f}) "
        f"in 85 pairs (170 specimens). The original analysis in 87 pairs gave AUC {old_auc:.3f} "
        f"(95% conditional CI {old_ci.ci_low:.3f}–{old_ci.ci_high:.3f}).\n\n"
        "The analysis retained the ten genes, training-only standardisation, five patient-grouped outer folds "
        "repeated 20 times, four patient-grouped inner folds, original ridge penalty grid and random seed. "
        "AUC is the mean of the 20 repeat-specific out-of-fold AUCs. Confidence intervals used 2,000 patient-cluster "
        "bootstrap resamples and are conditional on the fitted cross-validation models.\n\n"
        "The comparison is descriptive because the cohorts differ. No test or confidence interval for their "
        "difference was calculated. This post hoc sensitivity analysis does not restore historical assay QC. "
        "All predictions, folds and patient-level details are retained privately. Original inputs and code are unchanged.\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
