"""Run independent ML verification checks and write compact evidence files."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "verification" / "ml"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, sep="\t", compression="gzip", keep_default_na="permutation" not in path.name)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t", keep_default_na="permutation" not in path.name)
    return pd.read_csv(path, keep_default_na="permutation" not in path.name)


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def read_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def frozen_hash(source_freeze: dict, filename: str) -> str | None:
    for item in source_freeze.get("files", []):
        if item.get("file") == filename:
            return item.get("sha256")
    return None


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "tests/test_ml_independent.py", "-q"]
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (OUTDIR / "independent_pytest.log").write_text(run.stdout + run.stderr)

    recurrence = pd.read_csv(ROOT / "data/ml/recurrence.tsv", sep="\t")
    tissue = pd.read_csv(ROOT / "data/ml/tissue.tsv", sep="\t")
    recurrence_root = ROOT / "results/ml/recurrence"
    recurrence_oof = first_existing([
        recurrence_root / "primary_oof_predictions.csv",
        recurrence_root / "_smoke/primary_oof_predictions.csv",
    ])
    tissue_oof = first_existing([
        ROOT / "results/ml/tissue/oof_predictions.tsv.gz",
        ROOT / "results/ml/tissue/final/oof_predictions.tsv.gz",
        ROOT / "results/ml/tissue/smoke/oof_predictions.tsv.gz",
    ])
    recurrence_columns = []
    tissue_columns = []
    recurrence_repeats = None
    tissue_repeats = None
    if recurrence_oof.exists():
        r_oof = read_table(recurrence_oof)
        recurrence_columns = list(r_oof.columns)
        recurrence_repeats = int(r_oof["repeat"].nunique()) if "repeat" in r_oof else None
    if tissue_oof.exists():
        t_oof = read_table(tissue_oof)
        tissue_columns = list(t_oof.columns)
        tissue_repeats = int(t_oof["repeat"].nunique()) if "repeat" in t_oof else None
    source_freeze_path = ROOT / "verification/ml/production_source_freeze.json"
    source_freeze = read_json_if_present(source_freeze_path)
    recurrence_summary_path = first_existing([
        recurrence_root / "summary.json",
        recurrence_root / "_smoke/summary.json",
    ])
    recurrence_summary = read_json_if_present(recurrence_summary_path)
    recurrence_config = recurrence_summary.get("recurrence_config", {})
    recurrence_summary_complete = (
        recurrence_summary.get("status") == "complete"
        and recurrence_summary.get("quick") is False
        and recurrence_config.get("outer_repeats") == 25
        and recurrence_config.get("bootstrap_B") == 2000
        and recurrence_config.get("optimism_B") == 500
        and recurrence_config.get("permutation_B") == 999
    )
    recurrence_primary_metrics_path = first_existing([
        recurrence_root / "primary_repeat_metrics.csv",
        recurrence_root / "_smoke/primary_repeat_metrics.csv",
    ])
    recurrence_sensitivity_summary_path = first_existing([
        recurrence_root / "ridge_sensitivity_metrics.csv",
        recurrence_root / "_smoke/ridge_sensitivity_metrics.csv",
    ])
    recurrence_sensitivity_all87_metrics_path = first_existing([
        recurrence_root / "sensitivity_all87_stage_advanced_repeat_metrics.csv",
        recurrence_root / "_smoke/sensitivity_all87_stage_advanced_repeat_metrics.csv",
    ])
    recurrence_bootstrap_ci_path = first_existing([
        recurrence_root / "primary_bootstrap_ci.csv",
        recurrence_root / "_smoke/primary_bootstrap_ci.csv",
    ])
    recurrence_optimism_path = recurrence_root / "primary_ridge_optimism_bootstrap.csv"
    recurrence_permutation_path = recurrence_root / "primary_ridge_permutation.csv"
    recurrence_bootstrap_weights_path = first_existing([
        recurrence_root / "primary_bootstrap_patient_weights.rds",
        recurrence_root / "_smoke/primary_bootstrap_patient_weights.rds",
    ])
    recurrence_counts = {}
    if recurrence_primary_metrics_path.exists():
        recurrence_counts["primary_repeat_metric_rows"] = int(len(read_table(recurrence_primary_metrics_path)))
    primary_metrics_path = recurrence_root / "primary_metrics.csv"
    if primary_metrics_path.exists():
        primary_metrics = read_table(primary_metrics_path)
        recurrence_counts["primary_metric_rows"] = int(len(primary_metrics))
        recurrence_counts["primary_metric_n_repeats_min"] = int(primary_metrics["n_repeats"].min()) if "n_repeats" in primary_metrics else None
    if recurrence_sensitivity_summary_path.exists():
        sensitivity_summary = read_table(recurrence_sensitivity_summary_path)
        recurrence_counts["sensitivity_summary_rows"] = int(len(sensitivity_summary))
        recurrence_counts["sensitivity_summary_unique"] = int(sensitivity_summary["sensitivity"].nunique()) if "sensitivity" in sensitivity_summary else None
        recurrence_counts["sensitivity_summary_n_repeats_min"] = int(sensitivity_summary["n_repeats"].min()) if "n_repeats" in sensitivity_summary else None
    if recurrence_sensitivity_all87_metrics_path.exists():
        recurrence_counts["sensitivity_all87_stage_advanced_repeat_metric_rows"] = int(len(read_table(recurrence_sensitivity_all87_metrics_path)))
    if recurrence_bootstrap_ci_path.exists():
        boot_ci = read_table(recurrence_bootstrap_ci_path)
        recurrence_counts["bootstrap_ci_rows"] = int(len(boot_ci))
        recurrence_counts["bootstrap_n_min"] = int(boot_ci["bootstrap_n"].min()) if "bootstrap_n" in boot_ci else None
    if recurrence_optimism_path.exists():
        optimism = read_table(recurrence_optimism_path)
        recurrence_counts["optimism_rows"] = int(len(optimism))
        recurrence_counts["optimism_bootstraps"] = int(optimism["bootstrap"].nunique()) if "bootstrap" in optimism else None
        opt_key_cols = [c for c in ["block", "model", "horizon_days"] if c in optimism]
        recurrence_counts["optimism_metric_keys"] = int(optimism[opt_key_cols].drop_duplicates().shape[0]) if opt_key_cols else None
        recurrence_counts["optimism_status_counts"] = optimism["status"].value_counts(dropna=False).to_dict() if "status" in optimism else {}
    if recurrence_permutation_path.exists():
        permutation = read_table(recurrence_permutation_path)
        recurrence_counts["permutation_rows"] = int(len(permutation))
        if "kind" in permutation and "permutation" in permutation:
            recurrence_counts["permutation_null_draws"] = int(permutation.loc[permutation["kind"].eq("null"), "permutation"].nunique())
        recurrence_counts["permutation_hashes"] = int(permutation["permutation_hash"].nunique()) if "permutation_hash" in permutation else None
    recurrence_final_counts_ready = (
        recurrence_counts.get("primary_metric_rows") == 18
        and recurrence_counts.get("primary_metric_n_repeats_min") == 25
        and recurrence_counts.get("primary_repeat_metric_rows") == 25 * 3 * 3 * 2
        and recurrence_counts.get("bootstrap_n_min") == 2000
        and recurrence_counts.get("bootstrap_ci_rows") == 85
        and recurrence_counts.get("optimism_rows") == 3000
        and recurrence_counts.get("optimism_bootstraps") == 500
        and recurrence_counts.get("optimism_metric_keys") == 6
        and recurrence_counts.get("permutation_null_draws") == 999
        and recurrence_counts.get("permutation_hashes") == 999
        and recurrence_counts.get("sensitivity_summary_rows") == 14
        and recurrence_counts.get("sensitivity_summary_unique") == 7
        and recurrence_counts.get("sensitivity_summary_n_repeats_min") == 25
        and recurrence_counts.get("sensitivity_all87_stage_advanced_repeat_metric_rows") == 25 * 1 * 1 * 2
        and recurrence_bootstrap_weights_path.exists()
    )
    recurrence_final_ready = (
        recurrence_oof.exists()
        and recurrence_repeats is not None
        and recurrence_repeats >= 25
        and recurrence_summary_complete
        and recurrence_final_counts_ready
    )
    tissue_artifacts_ready = tissue_oof.exists() and tissue_repeats is not None and tissue_repeats >= 20
    local_bootstrap_path = ROOT / "results/ml/tissue/bootstrap_draws.tsv.gz"
    local_bootstrap_ready = False
    local_bootstrap_draws = None
    if local_bootstrap_path.exists():
        local_bootstrap = read_table(local_bootstrap_path)
        local_bootstrap_draws = int(local_bootstrap["bootstrap"].nunique()) if "bootstrap" in local_bootstrap else None
        local_bootstrap_ready = local_bootstrap_draws is not None and local_bootstrap_draws >= 2000
    tissue_root = ROOT / "results/ml/tissue"
    tissue_run_manifest_path = tissue_root / "run_manifest.json"
    tissue_postprocess_summary_path = tissue_root / "ridge_optimism_corrected_summary.json"
    tissue_manifest_path = tissue_root / "manifest.json"
    tissue_run_manifest = read_json_if_present(tissue_run_manifest_path)
    tissue_postprocess_summary = read_json_if_present(tissue_postprocess_summary_path)
    tissue_manifest = read_json_if_present(tissue_manifest_path)
    expected_tissue_source_hash = frozen_hash(source_freeze, "ml/tissue.py")
    expected_tissue_input_hash = frozen_hash(source_freeze, "data/ml/tissue.tsv")
    expected_tissue_config_hash = frozen_hash(source_freeze, "registry/ml_config.json")
    tissue_source_hash_ok = (
        source_freeze.get("status") == "frozen_before_production"
        and tissue_manifest.get("source_sha256") in {None, expected_tissue_source_hash}
        and tissue_postprocess_summary.get("source_sha256") == expected_tissue_source_hash
        and tissue_postprocess_summary.get("input_sha256") == expected_tissue_input_hash
        and tissue_postprocess_summary.get("config_sha256") == expected_tissue_config_hash
        and tissue_postprocess_summary.get("end_source_sha256") == expected_tissue_source_hash
        and tissue_postprocess_summary.get("end_input_sha256") == expected_tissue_input_hash
        and tissue_postprocess_summary.get("end_config_sha256") == expected_tissue_config_hash
    )
    tissue_postprocess_ready = (
        tissue_postprocess_summary_path.exists()
        and tissue_postprocess_summary.get("optimism_bootstrap_draws") == 500
        and tissue_postprocess_summary.get("bootstrap_weight_replays") == 2000
        and tissue_postprocess_summary.get("optimism_draws") == "results/ml/tissue/ridge_optimism_bootstrap.tsv"
        and tissue_postprocess_summary.get("patient_bootstrap_weights") == "results/ml/tissue/patient_bootstrap_weights.tsv.gz"
    )
    tissue_manifest_ready = (
        tissue_run_manifest_path.exists()
        and tissue_run_manifest.get("status") == "complete"
        and tissue_run_manifest.get("outer_repeats") == 20
        and tissue_run_manifest.get("bootstrap_replicates") == 2000
        and tissue_run_manifest.get("optimism_bootstrap") == 500
        and tissue_run_manifest.get("permutations") == 999
    )
    tissue_final_ready = (
        tissue_artifacts_ready
        and local_bootstrap_ready
        and tissue_manifest_ready
        and tissue_postprocess_ready
        and tissue_source_hash_ok
    )

    public_tissue = {}
    for cohort, expected_patients in {"colonomics": 92, "gse119526": 48}.items():
        cohort_dir = ROOT / "results/ml/public" / cohort
        cohort_oof = cohort_dir / "oof_predictions.tsv.gz"
        cohort_metrics = cohort_dir / "metrics.tsv"
        cohort_draws = cohort_dir / "bootstrap_draws.tsv.gz"
        cohort_ci = cohort_dir / "bootstrap_ci.tsv"
        info = {
            "oof_path": str(cohort_oof.relative_to(ROOT)),
            "metrics_path": str(cohort_metrics.relative_to(ROOT)),
            "bootstrap_draws_path": str(cohort_draws.relative_to(ROOT)),
            "bootstrap_ci_path": str(cohort_ci.relative_to(ROOT)),
            "expected_patients": expected_patients,
            "present": cohort_oof.exists() and cohort_metrics.exists(),
            "bootstrap_present": cohort_draws.exists() and cohort_ci.exists(),
        }
        if cohort_oof.exists():
            cohort_df = read_table(cohort_oof)
            info.update({
                "rows": int(len(cohort_df)),
                "patients": int(cohort_df["study_id"].nunique()),
                "repeats": int(cohort_df["repeat"].nunique()),
                "models": sorted(cohort_df["model"].astype(str).unique()),
            })
        if cohort_draws.exists():
            cohort_boot = read_table(cohort_draws)
            info["bootstrap_draws"] = int(cohort_boot["bootstrap"].nunique()) if "bootstrap" in cohort_boot else None
        public_tissue[cohort] = info
    public_tissue_ready = all(
        item.get("patients") == item["expected_patients"]
        and item.get("repeats") == 20
        and item.get("models") == ["ridge"]
        and item.get("bootstrap_draws") == 2000
        for item in public_tissue.values()
    )
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": run.returncode,
        "command": command,
        "production_source_freeze": {
            "path": str(source_freeze_path.relative_to(ROOT)),
            "present": source_freeze_path.exists(),
            "status": source_freeze.get("status"),
            "files": source_freeze.get("files", []),
        },
        "source_counts": {
            "patients": int(recurrence["study_id"].nunique()),
            "specimens": int(len(tissue)),
            "primary_patients": int(recurrence["recurrence_primary"].sum()),
            "primary_events": int(recurrence.loc[recurrence["recurrence_primary"].eq(1), "event"].sum()),
            "all_events": int(recurrence["event"].sum()),
            "cea_elevated": int(recurrence["cea_binary"].sum()),
        },
        "recurrence_oof": {
            "path": str(recurrence_oof.relative_to(ROOT)),
            "present": recurrence_oof.exists(),
            "columns": recurrence_columns,
            "repeats": recurrence_repeats,
            "final_repeat_target": 25,
        },
        "recurrence_strict_ready_inputs": {
            "summary_path": str(recurrence_summary_path.relative_to(ROOT)),
            "summary_present": recurrence_summary_path.exists(),
            "summary_status": recurrence_summary.get("status"),
            "summary_quick": recurrence_summary.get("quick"),
            "config": recurrence_config,
            "summary_complete": recurrence_summary_complete,
            "counts": recurrence_counts,
            "counts_ready": recurrence_final_counts_ready,
            "bootstrap_weights_path": str(recurrence_bootstrap_weights_path.relative_to(ROOT)),
            "bootstrap_weights_present": recurrence_bootstrap_weights_path.exists(),
        },
        "tissue_strict_ready_inputs": {
            "artifacts_ready": tissue_artifacts_ready,
            "run_manifest_path": str(tissue_run_manifest_path.relative_to(ROOT)),
            "run_manifest_present": tissue_run_manifest_path.exists(),
            "run_manifest_status": tissue_run_manifest.get("status"),
            "run_manifest": tissue_run_manifest,
            "postprocess_summary_path": str(tissue_postprocess_summary_path.relative_to(ROOT)),
            "postprocess_summary_present": tissue_postprocess_summary_path.exists(),
            "postprocess_summary": tissue_postprocess_summary,
            "manifest_path": str(tissue_manifest_path.relative_to(ROOT)),
            "manifest_present": tissue_manifest_path.exists(),
            "manifest": tissue_manifest,
            "manifest_ready": tissue_manifest_ready,
            "postprocess_ready": tissue_postprocess_ready,
            "source_hash_ok": tissue_source_hash_ok,
        },
        "tissue_oof": {
            "path": str(tissue_oof.relative_to(ROOT)),
            "present": tissue_oof.exists(),
            "columns": tissue_columns,
            "repeats": tissue_repeats,
            "final_repeat_target": 20,
        },
        "tissue_local_uncertainty": {
            "path": str(local_bootstrap_path.relative_to(ROOT)),
            "present": local_bootstrap_path.exists(),
            "bootstrap_draws": local_bootstrap_draws,
            "target_bootstrap_draws": 2000,
        },
        "public_tissue": public_tissue,
        "oof_files_present": sorted(
            str(p.relative_to(ROOT))
            for p in (ROOT / "results" / "ml").rglob("*oof*")
            if p.is_file()
        ),
    }
    (OUTDIR / "independent_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    smoke_oof = ROOT / "results/ml/recurrence/_smoke/primary_oof_predictions.csv"
    smoke_metrics = ROOT / "results/ml/recurrence/_smoke/primary_repeat_metrics.csv"
    smoke = {
        "path_oof": str(smoke_oof.relative_to(ROOT)),
        "path_metrics": str(smoke_metrics.relative_to(ROOT)),
        "present": smoke_oof.exists() and smoke_metrics.exists(),
        "validated_by_pytest": run.returncode == 0,
    }
    if smoke["present"]:
        sm_oof = read_table(smoke_oof)
        sm_met = read_table(smoke_metrics)
        smoke.update(
            {
                "repeats": int(sm_oof["repeat"].nunique()),
                "metric_rows": int(len(sm_met)),
                "has_101_grid": all(f"risk_grid_{i}" in sm_oof.columns and f"G_grid_{i}" in sm_oof.columns for i in range(1, 102)),
            }
        )
    (OUTDIR / "independent_smoke_reference.json").write_text(json.dumps(smoke, indent=2) + "\n")
    if run.returncode:
        status = "fail"
    elif recurrence_final_ready and tissue_final_ready and local_bootstrap_ready and public_tissue_ready:
        status = "pass"
    else:
        status = "partial"
    audit = {
        "status": status,
        "final_ready": {
            "recurrence": recurrence_final_ready,
            "recurrence_summary_complete": recurrence_summary_complete,
            "recurrence_final_counts": recurrence_final_counts_ready,
            "tissue": tissue_final_ready,
            "tissue_artifacts": tissue_artifacts_ready,
            "tissue_manifest": tissue_manifest_ready,
            "tissue_postprocess": tissue_postprocess_ready,
            "tissue_source_hash": tissue_source_hash_ok,
            "tissue_local_uncertainty": local_bootstrap_ready,
            "public_tissue": public_tissue_ready,
        },
        "known_nonfinal_outputs": {
            "recurrence_repeats": recurrence_repeats,
            "tissue_repeats": tissue_repeats,
            "local_tissue_bootstrap_draws": local_bootstrap_draws,
            "recurrence_summary_quick": recurrence_summary.get("quick"),
            "recurrence_summary_status": recurrence_summary.get("status"),
            "recurrence_counts": recurrence_counts,
        },
        "pytest_stdout_tail": (run.stdout + run.stderr)[-4000:],
    }
    (OUTDIR / "independent_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
