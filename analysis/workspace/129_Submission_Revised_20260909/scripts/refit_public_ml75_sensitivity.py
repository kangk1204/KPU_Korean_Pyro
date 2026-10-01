#!/usr/bin/env python3
"""Refit public paired-array ML after excluding two technical-mask CpGs.

This script is portable within 124_Integrated_Revision_20260909. It imports the
124 public-ML input builder and fit code, reuses the original OOF fold assignments,
and writes aggregate sensitivity tables only. New sensitivity OOF predictions,
model states, per-fold patient identifiers, and draw-level bootstrap rows are kept
out of submission-facing results.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ML_SCRIPT = ROOT / "scripts" / "analyze_public_cpg_ml.py"
ORIGINAL_PUBLIC_ML = ROOT / "results" / "public_ml"
DEFAULT_OUTDIR = ROOT / "results" / "qc_sensitivity"
FLAGGED = ["cg20680720", "cg04481096"]
COHORTS = ["colonomics", "gse119526"]
SEED = 20260908


def load_public_ml_module():
    scripts_dir = str((ROOT / "scripts").resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import analyze_public_cpg_ml as module
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guarded_outdir(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (ROOT / "results" / "qc_sensitivity").resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"--outdir must be inside {allowed}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_original_oof(cohort: str) -> pd.DataFrame:
    path = ORIGINAL_PUBLIC_ML / cohort / "oof_predictions.tsv.gz"
    oof = pd.read_csv(path, sep="\t")
    required = {"repeat", "fold", "study_id", "sample_id", "tissue", "y", "probability"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return oof


def folds_from_original(oof: pd.DataFrame, all_patients: list[str]) -> list[dict[str, object]]:
    patients = set(map(str, all_patients))
    folds: list[dict[str, object]] = []
    for (repeat, fold), group in oof.groupby(["repeat", "fold"], sort=True):
        test = sorted(map(str, group["study_id"].unique()))
        train = sorted(patients - set(test))
        overlap = set(train) & set(test)
        if overlap:
            raise AssertionError(f"train/test overlap in original OOF fold {repeat}/{fold}: {sorted(overlap)[:3]}")
        folds.append({"repeat": int(repeat), "fold": int(fold), "train_patients": train, "test_patients": test})
    return folds


def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(score, method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_delta(original_oof: pd.DataFrame, sensitivity_oof: pd.DataFrame, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed + 1701)
    patients = np.asarray(sorted(original_oof["study_id"].unique()))
    if set(patients) != set(sensitivity_oof["study_id"].unique()):
        raise AssertionError("original and sensitivity OOF patient sets differ")
    indexed = {}
    for repeat, orig_rep in original_oof.groupby("repeat", sort=True):
        sens_rep = sensitivity_oof[sensitivity_oof["repeat"].eq(repeat)].reset_index(drop=True)
        orig_rep = orig_rep.reset_index(drop=True)
        orig_keys = orig_rep[["study_id", "sample_id", "y"]].astype(str).agg("|".join, axis=1).tolist()
        sens_keys = sens_rep[["study_id", "sample_id", "y"]].astype(str).agg("|".join, axis=1).tolist()
        if orig_keys != sens_keys:
            raise AssertionError(f"OOF row identity/order differs for repeat {repeat}")
        indexed[int(repeat)] = {
            "y": orig_rep["y"].to_numpy(int),
            "orig_probability": orig_rep["probability"].to_numpy(float),
            "sens_probability": sens_rep["probability"].to_numpy(float),
            "patient_to_index": {pid: orig_rep.index[orig_rep["study_id"].eq(pid)].to_numpy() for pid in patients},
        }
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        rep_deltas = []
        for repeat in sorted(indexed):
            info = indexed[repeat]
            idx = np.concatenate([info["patient_to_index"][pid] for pid in sampled])
            orig_auc = fast_auc(info["y"][idx], info["orig_probability"][idx])
            sens_auc = fast_auc(info["y"][idx], info["sens_probability"][idx])
            rep_deltas.append(sens_auc - orig_auc)
        deltas.append(float(np.nanmean(rep_deltas)))
    lo, hi = np.quantile(np.asarray(deltas, float), [0.025, 0.975])
    return {"delta_auc_bootstrap_low": float(lo), "delta_auc_bootstrap_high": float(hi)}


def run_cohort(cohort: str, frame: pd.DataFrame, fixed: pd.DataFrame, config, workers: int, bootstrap: int) -> dict[str, object]:
    ml = load_public_ml_module()
    expected = {"colonomics": 92, "gse119526": 48}[cohort]
    ml.validate_frame(frame, expected)
    original_oof = read_original_oof(cohort)
    folds = folds_from_original(original_oof, sorted(frame["study_id"].unique()))
    fold_audit = pd.DataFrame([
        {
            "repeat": int(f["repeat"]),
            "fold": int(f["fold"]),
            "n_train_study_ids": len(f["train_patients"]),
            "n_test_study_ids": len(f["test_patients"]),
            "patient_overlap": len(set(f["train_patients"]) & set(f["test_patients"])),
        }
        for f in folds
    ])
    retained_fixed = fixed.loc[~fixed["cpg"].isin(FLAGGED)].copy()
    features = retained_fixed["feature"].tolist()
    results = Parallel(n_jobs=workers, prefer="processes", batch_size=1)(
        delayed(ml.fit_predict)(frame, features, fold, config) for fold in folds
    )
    sensitivity_oof = pd.concat([x[0] for x in results], ignore_index=True)
    original_ci = pd.read_csv(ORIGINAL_PUBLIC_ML / cohort / "bootstrap_ci.tsv", sep="\t")
    _, sensitivity_ci = ml.patient_bootstrap(sensitivity_oof, bootstrap, config.seed)
    delta_ci = bootstrap_delta(original_oof, sensitivity_oof, bootstrap, config.seed)
    original_metrics = ml.summarize_metrics(original_oof).query("scope == 'all_repeats_pooled'").iloc[0]
    sensitivity_metrics = ml.summarize_metrics(sensitivity_oof).query("scope == 'all_repeats_pooled'").iloc[0]
    original_auc = original_ci.loc[original_ci["metric"].eq("auc")].iloc[0]
    original_brier = original_ci.loc[original_ci["metric"].eq("brier")].iloc[0]
    sensitivity_auc = sensitivity_ci.loc[sensitivity_ci["metric"].eq("auc")].iloc[0]
    sensitivity_brier = sensitivity_ci.loc[sensitivity_ci["metric"].eq("brier")].iloc[0]
    return {
        "cohort": cohort,
        "patients": int(frame["study_id"].nunique()),
        "original_features": 77,
        "sensitivity_features": int(len(features)),
        "excluded_cpgs": ";".join(FLAGGED),
        "original_auc": float(original_metrics["auc"]),
        "exclude_auc": float(sensitivity_metrics["auc"]),
        "delta_auc_exclude_minus_original": float(sensitivity_metrics["auc"] - original_metrics["auc"]),
        "original_brier": float(original_metrics["brier"]),
        "exclude_brier": float(sensitivity_metrics["brier"]),
        "delta_brier_exclude_minus_original": float(sensitivity_metrics["brier"] - original_metrics["brier"]),
        "original_bootstrap_auc_low": float(original_auc["ci_low"]),
        "original_bootstrap_auc_high": float(original_auc["ci_high"]),
        "exclude_bootstrap_auc_low": float(sensitivity_auc["ci_low"]),
        "exclude_bootstrap_auc_high": float(sensitivity_auc["ci_high"]),
        **delta_ci,
        "patient_fold_overlap_max": int(fold_audit["patient_overlap"].max()),
        "folds_reused_from_original_oof": True,
        "n_folds_reused": int(len(folds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    outdir = guarded_outdir(args.outdir)
    t0 = time.time()
    ml = load_public_ml_module()
    fixed = ml.read_fixed_probes()
    frames, _qcs, source_files = ml.build_inputs(fixed)
    config = ml.MLConfig(max_workers=args.workers, bootstrap_replicates=args.bootstrap)
    rows = []
    for cohort in COHORTS:
        row = run_cohort(cohort, frames[cohort], fixed, config, args.workers, args.bootstrap)
        row["elapsed_seconds"] = round(time.time() - t0, 3)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "public_ml75_before_after_summary.tsv", sep="\t", index=False)
    manifest = {
        "date": "2026-09-09",
        "script": str(Path(__file__).relative_to(ROOT)),
        "scope": "public paired-array ML refit after excluding cg20680720 and cg04481096; aggregate outputs only",
        "root": str(ROOT),
        "source_builder": str(PUBLIC_ML_SCRIPT.relative_to(ROOT)),
        "source_builder_sha256": sha256_file(PUBLIC_ML_SCRIPT),
        "original_public_ml_dir": str(ORIGINAL_PUBLIC_ML.relative_to(ROOT)),
        "cohorts": COHORTS,
        "bootstrap_replicates": int(args.bootstrap),
        "conditional_patient_bootstrap": True,
        "fold_source": "original 124 results/public_ml/*/oof_predictions.tsv.gz repeat/fold study_id assignments",
        "private_outputs_written": False,
        "excluded_from_submission_outputs": ["new sensitivity OOF predictions", "model state dumps", "per-fold train/test patient identifiers", "draw-level bootstrap rows"],
        "source_files": source_files,
        "summary_sha256": sha256_file(outdir / "public_ml75_before_after_summary.tsv"),
    }
    (outdir / "public_ml75_run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"summary": rows, "outdir": str(outdir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
