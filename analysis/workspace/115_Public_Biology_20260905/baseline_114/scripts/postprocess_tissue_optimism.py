#!/usr/bin/env python3
"""Summarize ridge optimism correction after the tissue ML run finishes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.tissue import GENES, TissueConfig, _as_pred, _candidate_model, _scale_fit, prepare_tissue_frame, tune_model, write_csv


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_sha256(data: pd.DataFrame) -> str:
    cols = [c for c in ["patient_id", "group_id", "sample_id", "tissue", "y"] if c in data.columns] + GENES
    return hashlib.sha256(data.loc[:, cols].to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def write_bootstrap_weights(oof: pd.DataFrame, outdir: Path, seed: int, n_boot: int) -> Path:
    patients = np.asarray(sorted(oof["study_id"].unique()))
    rng = np.random.default_rng(seed + 707)
    rows = []
    for b in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        counts = pd.Series(sampled).value_counts()
        for study_id, count in counts.sort_index().items():
            rows.append({"bootstrap": b, "study_id": study_id, "multiplicity": int(count)})
    weights = pd.DataFrame(rows)
    path = outdir / "patient_bootstrap_weights.tsv.gz"
    write_csv(weights, path, sep="\t")
    return path


def write_calibration_diagnostics(metrics: pd.DataFrame, outdir: Path) -> Path:
    repeats = metrics[metrics["scope"].eq("repeat_pooled")].copy()
    rows = []
    for model, sub in repeats.groupby("model", sort=False):
        statuses = sub["calibration_status"].fillna("missing").astype(str)
        counts = statuses.value_counts().sort_index()
        row = {
            "model": model,
            "n_repeats": int(len(sub)),
            "n_estimable": int((statuses == "estimable").sum()),
            "n_nonestimable_or_not_applicable": int((statuses != "estimable").sum()),
            "status_counts_json": json.dumps(counts.to_dict(), sort_keys=True),
            "calibration_intercept_mean_all20": float(sub["calibration_intercept"].mean(skipna=False)) if "calibration_intercept" in sub else float("nan"),
            "calibration_slope_mean_all20": float(sub["calibration_slope"].mean(skipna=False)) if "calibration_slope" in sub else float("nan"),
            "note": "NaN if any repeat is non-estimable; no finite-subset averaging for calibration.",
        }
        rows.append(row)
    path = outdir / "calibration_diagnostics.tsv"
    write_csv(pd.DataFrame(rows), path, sep="\t")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/ml/tissue.tsv", help="TSV input relative to repo root")
    parser.add_argument("--config", default="registry/ml_config.json", help="Config JSON relative to repo root")
    parser.add_argument("--outdir", default="results/ml/tissue", help="Result directory relative to repo root")
    parser.add_argument("--cohort-name", default="tissue", help="Label stored in the postprocess summary")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Patient bootstrap count used to replay weights")
    parser.add_argument("--require-optimism", action="store_true", help="Fail if ridge optimism draws are missing")
    return parser.parse_args()


def guarded_hashes(input_path: Path, config_path: Path) -> dict[str, str]:
    return {
        "source_sha256": sha256(ROOT / "ml" / "tissue.py"),
        "input_sha256": sha256(input_path),
        "config_sha256": sha256(config_path),
    }


def main() -> None:
    args = parse_args()
    input_path = (ROOT / args.input).resolve()
    config_path = (ROOT / args.config).resolve()
    outdir = (ROOT / args.outdir).resolve()
    optimism_path = outdir / "ridge_optimism_bootstrap.tsv"
    oof_path = outdir / "oof_predictions.tsv.gz"
    metrics_path = outdir / "metrics.tsv"
    if not oof_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Missing OOF predictions or metrics from tissue ML run")

    start_hashes = guarded_hashes(input_path, config_path)
    config = TissueConfig.from_json(config_path)
    data = prepare_tissue_frame(pd.read_csv(input_path, sep="\t"))
    canonical_data_sha = data_sha256(data)

    metrics = pd.read_csv(metrics_path, sep="\t")
    oof = pd.read_csv(oof_path, sep="\t")
    calibration_path = write_calibration_diagnostics(metrics, outdir)
    weights_path = write_bootstrap_weights(oof, outdir, config.seed, args.bootstrap)

    summary = {
        "cohort": args.cohort_name,
        "input": str(input_path.relative_to(ROOT)),
        "input_sha256": start_hashes["input_sha256"],
        "canonical_data_sha256": canonical_data_sha,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": start_hashes["config_sha256"],
        "source_sha256": start_hashes["source_sha256"],
        "calibration_diagnostics": str(calibration_path.relative_to(ROOT)),
        "patient_bootstrap_weights": str(weights_path.relative_to(ROOT)),
        "bootstrap_weight_replays": int(args.bootstrap),
    }

    if optimism_path.exists():
        draws = pd.read_csv(optimism_path, sep="\t")
        if len(draws) != 500 or draws["bootstrap"].nunique() != 500:
            raise ValueError("Expected exactly 500 unique ridge optimism bootstrap draws")
        required = ["apparent_auc", "test_original_auc", "optimism_auc", "apparent_brier", "test_original_brier", "optimism_brier"]
        bad = [c for c in required if c not in draws or not np.isfinite(draws[c].to_numpy(float)).all()]
        if bad:
            raise ValueError(f"Non-finite or missing optimism columns: {bad}")

        params, candidates = tune_model(data, data["group_id"].unique(), "ridge", config, 0, 999)
        model, needs_scale = _candidate_model("ridge", params, config.seed + 999)
        X = data[GENES].to_numpy(float)
        y = data["y"].to_numpy(int)
        scaler = None
        if needs_scale:
            scaler = _scale_fit(X)
            X_fit = scaler.transform(X)
        else:
            X_fit = X
        model.fit(X_fit, y)
        if not getattr(model, "n_iter_", []):
            raise RuntimeError("Full-original ridge fit did not expose convergence iterations")
        if int(max(model.n_iter_)) >= int(model.max_iter):
            raise RuntimeError(f"Full-original ridge fit reached max_iter={model.max_iter}")
        score, prob = _as_pred(model, X_fit, "ridge")
        apparent_auc = float(roc_auc_score(y, score))
        apparent_brier = float(brier_score_loss(y, prob))
        summary.update({
            "model": "ridge",
            "selection": "full original data, same inner CV tuning grid",
            "selected_params": params,
            "full_original_apparent_auc": apparent_auc,
            "mean_bootstrap_optimism_auc": float(draws["optimism_auc"].mean()),
            "optimism_corrected_auc": float(apparent_auc - draws["optimism_auc"].mean()),
            "full_original_apparent_brier": apparent_brier,
            "mean_bootstrap_optimism_brier": float(draws["optimism_brier"].mean()),
            "optimism_corrected_brier": float(apparent_brier - draws["optimism_brier"].mean()),
            "bootstrap_draws": int(len(draws)),
            "optimism_bootstrap_draws": int(len(draws)),
            "optimism_draws": str(optimism_path.relative_to(ROOT)),
            "full_original_fit_n_iter": [int(x) for x in model.n_iter_],
        })
        write_csv(candidates, outdir / "ridge_full_original_candidate_results.tsv", sep="\t")
        with gzip.GzipFile(filename="", mode="wb", fileobj=(outdir / "ridge_full_original_model.pkl.gz").open("wb"), mtime=0) as gz:
            gz.write(pickle.dumps({"model": model, "scaler": scaler, "genes": GENES, "selected_params": params}, protocol=4))
    elif args.require_optimism:
        raise FileNotFoundError(f"Missing optimism draws: {optimism_path}")
    else:
        summary["optimism_status"] = "not_available"

    end_hashes = guarded_hashes(input_path, config_path)
    if end_hashes != start_hashes:
        raise RuntimeError(f"Inputs changed during postprocess: start={start_hashes}, end={end_hashes}")
    summary.update({f"end_{key}": value for key, value in end_hashes.items()})
    (outdir / "ridge_optimism_corrected_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
