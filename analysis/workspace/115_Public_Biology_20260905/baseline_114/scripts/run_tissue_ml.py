#!/usr/bin/env python3
"""Run the nested tissue-classification ML lane."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.tissue import GENES, TissueConfig, canonical_hash, dataframe_hash, prepare_tissue_frame, run_cohort


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_artifacts(outdir: Path, bootstrap: int, optimism: int, permutations: int) -> list[Path]:
    paths = [
        outdir / "manifest.json",
        outdir / "metrics.tsv",
        outdir / "oof_predictions.tsv.gz",
        outdir / "candidate_results.tsv.gz",
        outdir / "model_states.tsv.gz",
        outdir / "folds.tsv",
    ]
    if bootstrap:
        paths.extend([outdir / "bootstrap_draws.tsv.gz", outdir / "bootstrap_ci.tsv"])
    if optimism:
        paths.append(outdir / "ridge_optimism_bootstrap.tsv")
    if permutations:
        paths.extend([outdir / "ridge_permutation.tsv", outdir / "ridge_permutation_summary.json"])
    return paths


def cached_outputs_are_current(outdir: Path, *, input_path: Path, config_path: Path, models: list[str], repeats: int, bootstrap: int, optimism: int, permutations: int, config: TissueConfig) -> tuple[bool, str]:
    missing = [str(p.relative_to(ROOT)) for p in required_artifacts(outdir, bootstrap, optimism, permutations) if not p.exists() or p.stat().st_size == 0]
    if missing:
        return False, f"missing artifacts: {missing}"
    manifest_path = outdir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        return False, f"invalid manifest.json: {exc}"
    input_df = pd.read_csv(input_path, sep="\t")
    prepared = prepare_tissue_frame(input_df)
    source_sha = file_sha256(ROOT / "ml" / "tissue.py")
    data_sha = dataframe_hash(prepared, GENES)
    cache_payload = {
        "models": models,
        "repeats": repeats,
        "genes": GENES,
        "config": asdict(config),
        "data_sha256": data_sha,
        "source_sha256": source_sha,
    }
    expected_cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    expected = {
        "source_sha256": source_sha,
        "data_sha256": data_sha,
        "cache_key": expected_cache_key,
        "rows": int(len(input_df)),
        "models": models,
        "outer_repeats": repeats,
        "outer_folds": config.outer_folds,
        "inner_folds": config.inner_folds,
        "bootstrap": bootstrap,
        "optimism": optimism,
        "permutations": permutations,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False, f"manifest mismatch {key}: {manifest.get(key)!r} != {value!r}"
    try:
        metrics = pd.read_csv(outdir / "metrics.tsv", sep="\t")
        if not set(models).issubset(set(metrics["model"].unique())):
            return False, "metrics.tsv missing requested models"
        if not metrics[metrics["scope"].eq("all_repeats_pooled")]["aggregation"].eq("mean_of_repeat_metrics").all():
            return False, "metrics.tsv pooled aggregation contract mismatch"
        oof = pd.read_csv(outdir / "oof_predictions.tsv.gz", sep="\t", nrows=1)
        if "patient_id" in oof.columns or "study_id" not in oof.columns:
            return False, "OOF privacy/schema contract mismatch"
        if bootstrap:
            boot = pd.read_csv(outdir / "bootstrap_draws.tsv.gz", sep="\t", usecols=["bootstrap"])
            if boot["bootstrap"].nunique() != bootstrap:
                return False, "bootstrap draw count mismatch"
        if optimism:
            opt = pd.read_csv(outdir / "ridge_optimism_bootstrap.tsv", sep="\t", usecols=["bootstrap"])
            if len(opt) != optimism or opt["bootstrap"].nunique() != optimism:
                return False, "optimism draw count mismatch"
        if permutations:
            perm = pd.read_csv(outdir / "ridge_permutation.tsv", sep="\t", keep_default_na=False)
            summary = json.loads((outdir / "ridge_permutation_summary.json").read_text())
            null = perm.loc[perm["kind"].eq("null")]
            observed = perm.loc[perm["kind"].eq("observed")]
            if (len(perm) != permutations + 1 or int(summary.get("permutations", -1)) != permutations
                    or len(observed) != 1 or len(null) != permutations
                    or set(null["permutation"]) != set(range(permutations))
                    or not np.isfinite(perm["auc"].to_numpy(float)).all()):
                return False, "permutation count mismatch"
    except Exception as exc:
        return False, f"artifact validation failed: {exc}"
    return True, "ok"


def write_wrapper_manifest(done_path: Path, run_manifest: dict) -> None:
    tmp_path = done_path.with_suffix(done_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True))
    tmp_path.replace(done_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None, help="Subset of tissue models to run.")
    parser.add_argument("--repeats", type=int, default=None, help="Outer CV repeats. Defaults to locked config.")
    parser.add_argument("--bootstrap", type=int, default=None, help="Patient bootstrap replicates. Defaults to locked config.")
    parser.add_argument("--optimism", type=int, default=None, help="Ridge optimism bootstrap replicates. Defaults to locked config.")
    parser.add_argument("--permutations", type=int, default=None, help="Within-patient label-swap permutations. Defaults to locked config.")
    parser.add_argument("--smoke", action="store_true", help="Fast leakage/IO smoke run with reduced resampling.")
    parser.add_argument("--force", action="store_true", help="Re-run even when manifest hash matches.")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    input_path = ROOT / "data" / "ml" / "tissue.tsv"
    config_path = ROOT / "registry" / "ml_config.json"
    guarded_paths = [input_path, config_path, ROOT / "ml/tissue.py", Path(__file__)]
    start_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in guarded_paths}
    def guard_end() -> dict:
        end_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in guarded_paths}
        if end_hashes != start_hashes:
            raise RuntimeError("Tissue input, configuration or source changed during execution")
        return end_hashes
    config = TissueConfig.from_json(config_path)
    models = args.models or json.loads(config_path.read_text())["tissue"]["models"]
    repeats = args.repeats if args.repeats is not None else config.outer_repeats
    bootstrap = args.bootstrap if args.bootstrap is not None else config.bootstrap_replicates
    optimism = args.optimism if args.optimism is not None else config.optimism_bootstrap
    permutations = args.permutations if args.permutations is not None else config.permutations
    outdir = ROOT / "results" / "ml" / "tissue"
    verify_dir = ROOT / "verification" / "ml"
    verify_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        models = ["ridge", "best_single_gene"] if args.models is None else models
        repeats = min(2, repeats)
        bootstrap = args.bootstrap if args.bootstrap is not None else 20
        optimism = args.optimism if args.optimism is not None else 5
        permutations = args.permutations if args.permutations is not None else 9
        outdir = outdir / "smoke"

    df = pd.read_csv(input_path, sep="\t")
    run_hash = canonical_hash(
        [input_path, config_path, ROOT / "ml" / "tissue.py", Path(__file__)],
        [{"models": models, "repeats": repeats, "bootstrap": bootstrap, "optimism": optimism, "permutations": permutations, "genes": GENES}],
    )
    done_path = outdir / "run_manifest.json"
    if done_path.exists() and not args.force:
        old = json.loads(done_path.read_text())
        cache_ok, cache_reason = cached_outputs_are_current(
            outdir,
            input_path=input_path,
            config_path=config_path,
            models=models,
            repeats=repeats,
            bootstrap=bootstrap,
            optimism=optimism,
            permutations=permutations,
            config=config,
        )
        if cache_ok:
            old["run_hash"] = run_hash
            old["status"] = "smoke" if args.smoke else "complete"
            old["source_hashes"] = guard_end()
            old["cache_validation"] = "strict_artifacts_current"
            old["cache_validation_reason"] = cache_reason
            write_wrapper_manifest(done_path, old)
            (verify_dir / ("tissue_smoke_summary.json" if args.smoke else "tissue_run_summary.json")).write_text(json.dumps(old, indent=2, sort_keys=True))
            print(json.dumps({"stage": "tissue_ml", "state": "cached", "run_hash": run_hash, "outdir": str(outdir), "cache_validation": cache_reason}))
            return
        if old.get("run_hash") == run_hash:
            raise RuntimeError(f"Manifest hash matches but cache validation failed: {cache_reason}")

    print(json.dumps({"stage": "tissue_ml", "state": "started", "run_hash": run_hash, "models": models, "repeats": repeats}))
    results = run_cohort(
        df,
        outdir,
        models=models,
        repeats=repeats,
        config=config,
        bootstrap=bootstrap,
        optimism=optimism,
        permutations=permutations,
        genes=GENES,
    )
    metrics = results["metrics"]
    pooled = metrics[metrics["scope"].eq("all_repeats_pooled")].copy()
    run_manifest = {
        "status": "smoke" if args.smoke else "complete",
        "source_hashes": guard_end(),
        "run_hash": run_hash,
        "input": str(input_path.relative_to(ROOT)),
        "config": str(config_path.relative_to(ROOT)),
        "rows": int(len(df)),
        "patients": int(df["study_id"].nunique()),
        "tumor_samples": int(df["y"].sum()),
        "normal_samples": int((df["y"] == 0).sum()),
        "genes": GENES,
        "models": models,
        "outer_folds": config.outer_folds,
        "outer_repeats": repeats,
        "inner_folds": config.inner_folds,
        "bootstrap_replicates": bootstrap,
        "optimism_bootstrap": optimism,
        "permutations": permutations,
        "permutation_repeats": config.permutation_repeats if permutations else 0,
        "primary_metric": "repeat-pooled out-of-fold AUC",
        "pooled_metrics": json.loads(pooled.to_json(orient="records")),
    }
    write_wrapper_manifest(done_path, run_manifest)
    (verify_dir / ("tissue_smoke_summary.json" if args.smoke else "tissue_run_summary.json")).write_text(json.dumps(run_manifest, indent=2, sort_keys=True))
    print(json.dumps({"stage": "tissue_ml", "state": "completed", "run_hash": run_hash, "outdir": str(outdir)}))


if __name__ == "__main__":
    main()
