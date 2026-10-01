#!/usr/bin/env python3
"""Collect completed ML runs into one auditable manuscript input without refitting."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ROOT, sha256, write_json

sys.path.insert(0, str(ROOT))
from ml.tissue import GENES, canonical_hash, dataframe_hash, prepare_tissue_frame  # noqa: E402


PRIMARY_BLOCKS = ["clinical", "tumor10", "combined"]
RECURRENCE_MODELS = ["ridge", "elastic_net", "survival_forest"]
HORIZONS = [1095, 1825]
RECURRENCE_BASE_METRICS = ["uno_c", "auc", "brier", "harrell_c"]
RECURRENCE_DELTA_METRICS = ["delta_uno_c", "delta_brier"]
SENSITIVITIES = {
    "all87_stage_advanced",
    "no_stage",
    "cea_binary",
    "delta10",
    "clinical_delta10",
    "q95_binary_train_normals",
    "historical_cea_q95_ralyl_sfmbt2",
}
PRIVATE_COLUMNS = {"patient_id", "operation_date", "endpoint_date", "left_out_patient_id"}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def one(frame: pd.DataFrame, **keys: Any) -> dict[str, Any]:
    sub = frame.copy()
    for key, value in keys.items():
        sub = sub.loc[sub[key].eq(value)]
    if len(sub) != 1:
        raise ValueError(f"Expected one completed result for {keys}; found {len(sub)}")
    return clean(sub.iloc[0].to_dict())


def require_columns(frame: pd.DataFrame, columns: list[str] | set[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def require_no_private_columns(frame: pd.DataFrame, path: Path) -> None:
    leaked = PRIVATE_COLUMNS & set(frame.columns)
    if leaked:
        raise ValueError(f"Private columns in export {path}: {sorted(leaked)}")


def row_keys(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    require_columns(frame, columns, "row key frame")
    return frame[columns].astype(str).agg("|".join, axis=1).tolist()


def expected_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def assert_exact_keys(frame: pd.DataFrame, expected: pd.DataFrame, columns: list[str], name: str) -> None:
    actual_keys = sorted(row_keys(frame, columns))
    expected_keys = sorted(row_keys(expected, columns))
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError(f"{name} contains duplicate keys")
    if actual_keys != expected_keys:
        raise ValueError(f"{name} key mismatch: expected {len(expected_keys)}, found {len(actual_keys)}")


def assert_series_all(series: pd.Series, expected: int, name: str) -> None:
    values = pd.to_numeric(series, errors="coerce")
    if not values.eq(expected).all():
        raise ValueError(f"{name} must all equal {expected}")


def numeric(value: Any) -> float:
    if value is None:
        return math.nan
    return float(value)


def assert_close(actual: Any, expected: Any, name: str, tol: float = 1e-10) -> None:
    a = numeric(actual)
    e = numeric(expected)
    if math.isnan(a) and math.isnan(e):
        return
    if not (math.isfinite(a) and math.isfinite(e) and abs(a - e) <= tol):
        raise ValueError(f"{name} mismatch: {a} != {e}")


def expected_recurrence_hashes() -> dict[str, str]:
    return {
        "input": sha256(ROOT / "data/ml/recurrence.tsv"),
        "config": sha256(ROOT / "registry/ml_config.json"),
        "runner": sha256(ROOT / "scripts/run_recurrence_ml.R"),
        "metrics": sha256(ROOT / "ml/recurrence_metrics.R"),
        "resampling": sha256(ROOT / "ml/recurrence_resampling.R"),
    }


def load_json(path: Path, paths: list[Path]) -> dict[str, Any]:
    paths.append(path)
    return json.loads(path.read_text())


def load_frame(path: Path, paths: list[Path]) -> pd.DataFrame:
    paths.append(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} or ".tsv" in path.name else ","
    frame = pd.read_csv(path, sep=sep, keep_default_na="permutation" not in path.name)
    require_no_private_columns(frame, path)
    return frame


def planned_recurrence_metric_keys() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block in PRIMARY_BLOCKS:
        for model in RECURRENCE_MODELS:
            for horizon in HORIZONS:
                for metric in RECURRENCE_BASE_METRICS:
                    rows.append({"block": block, "model": model, "horizon_days": horizon, "metric": metric})
            rows.append({"block": block, "model": model, "horizon_days": 1825, "metric": "ibs"})
    return expected_frame(rows)


def planned_recurrence_ci_keys(include_deltas: bool = True) -> pd.DataFrame:
    base = planned_recurrence_metric_keys()
    if not include_deltas:
        return base
    delta_rows = [
        {"block": "combined_minus_clinical", "model": "ridge", "horizon_days": horizon, "metric": metric}
        for horizon in HORIZONS
        for metric in RECURRENCE_DELTA_METRICS
    ]
    return pd.concat([base, expected_frame(delta_rows)], ignore_index=True)


def assert_recurrence_ci_points(metrics: pd.DataFrame, ci: pd.DataFrame) -> None:
    for row in planned_recurrence_metric_keys().itertuples(index=False):
        got = one(ci, block=row.block, model=row.model, horizon_days=row.horizon_days, metric=row.metric)
        src = one(metrics, block=row.block, model=row.model, horizon_days=row.horizon_days)
        assert_close(got["point"], src[f"{row.metric}_mean"], f"CI point {row.block}/{row.model}/{row.horizon_days}/{row.metric}")
    for horizon in HORIZONS:
        for metric in ["uno_c", "brier"]:
            got = one(ci, block="combined_minus_clinical", model="ridge", horizon_days=horizon, metric=f"delta_{metric}")
            combined = one(metrics, block="combined", model="ridge", horizon_days=horizon)
            clinical = one(metrics, block="clinical", model="ridge", horizon_days=horizon)
            expected = numeric(combined[f"{metric}_mean"]) - numeric(clinical[f"{metric}_mean"])
            assert_close(got["point"], expected, f"Delta CI point {horizon}/{metric}")


def validate_recurrence_summary(manifest: dict[str, Any], cfg: dict[str, Any]) -> None:
    expected = expected_recurrence_hashes()
    if manifest.get("status") != "complete" or manifest.get("quick"):
        raise ValueError("Recurrence summary must be a non-quick complete production run")
    if manifest.get("source_hashes") != expected:
        raise ValueError("Recurrence source_hashes do not match current input/config/source files")
    expected_counts = {
        "outer_repeats": cfg["recurrence"]["outer_repeats"],
        "bootstrap_replicates": cfg["uncertainty"]["patient_bootstrap"],
        "optimism_bootstrap": cfg["uncertainty"]["optimism_bootstrap"],
        "permutations": cfg["uncertainty"]["permutations"],
    }
    for key, value in expected_counts.items():
        if manifest.get(key) != value:
            raise ValueError(f"Recurrence {key} expected {value}, found {manifest.get(key)}")
    checks = manifest.get("completion_checks", {})
    required = ["primary", "optimism", "permutation", "bootstrap_ci", "bootstrap_draws", "all"]
    if not checks or not all(checks.get(k) is True for k in required):
        raise ValueError(f"Recurrence completion checks are not all true: {checks}")


def validate_tissue_ci_points(aggregate: pd.DataFrame, ci: pd.DataFrame) -> None:
    for r in ci.itertuples(index=False):
        model = getattr(r, "model")
        metric = getattr(r, "metric")
        point = getattr(r, "point")
        if model.endswith("_minus_best_single_gene"):
            base = model.removesuffix("_minus_best_single_gene")
            expected = numeric(one(aggregate, model=base)[metric]) - numeric(one(aggregate, model="best_single_gene")[metric])
        elif model in set(aggregate["model"]):
            expected = numeric(one(aggregate, model=model)[metric])
        else:
            continue
        assert_close(point, expected, f"Tissue CI point {model}/{metric}")


def tissue_calibration_diagnostics(folder: Path, cfg: dict[str, Any], paths: list[Path], expected_models: set[str]) -> list[dict[str, Any]] | None:
    path = folder / "calibration_diagnostics.tsv"
    if not path.exists():
        return None
    diagnostics = load_frame(path, paths)
    require_columns(
        diagnostics,
        {"model", "n_repeats", "n_estimable", "n_nonestimable_or_not_applicable", "status_counts_json"},
        f"{path.name}",
    )
    if set(diagnostics["model"]) != expected_models:
        raise ValueError(f"Tissue calibration diagnostics model set mismatch: {folder}")
    assert_series_all(diagnostics["n_repeats"], cfg["tissue"]["outer_repeats"], f"{path.name}.n_repeats")
    invalid_counts = diagnostics["n_estimable"] + diagnostics["n_nonestimable_or_not_applicable"]
    if not invalid_counts.eq(cfg["tissue"]["outer_repeats"]).all():
        raise ValueError(f"Tissue calibration diagnostics estimable/nonestimable counts do not sum to outer repeats: {folder}")
    return clean(diagnostics.to_dict("records"))


def tissue_result(folder: Path, patients: int, cfg: dict[str, Any], paths: list[Path], *, local: bool = False) -> dict[str, Any]:
    manifest = load_json(folder / "run_manifest.json", paths)
    fitted = load_json(folder / "manifest.json", paths)
    if manifest.get("status") not in {None, "complete"}:
        raise ValueError(f"Tissue manifest is not complete: {folder}")
    assert fitted["source_sha256"] == sha256(ROOT / "ml/tissue.py"), "Stale fitted tissue source"
    input_path = ROOT / ("data/ml/tissue.tsv" if local else f"data/ml/public_{folder.name}.tsv")
    expected_data_hash = dataframe_hash(prepare_tissue_frame(pd.read_csv(input_path, sep="\t")))
    assert fitted["data_sha256"] == expected_data_hash, "Stale fitted tissue input"
    procedure = (
        {
            "models": cfg["tissue"]["models"],
            "repeats": cfg["tissue"]["outer_repeats"],
            "bootstrap": cfg["uncertainty"]["patient_bootstrap"],
            "optimism": cfg["uncertainty"]["optimism_bootstrap"],
            "permutations": cfg["uncertainty"]["permutations"],
            "genes": GENES,
        }
        if local
        else {
            "model": "ridge",
            "repeats": cfg["tissue"]["outer_repeats"],
            "bootstrap": cfg["uncertainty"]["patient_bootstrap"],
            "optimism": 0,
            "permutations": 0,
        }
    )
    wrapper = ROOT / "scripts" / ("run_tissue_ml.py" if local else "run_public_ml.py")
    expected_run_hash = canonical_hash([input_path, ROOT / "registry/ml_config.json", ROOT / "ml/tissue.py", wrapper], [procedure])
    if manifest["run_hash"] != expected_run_hash:
        raise ValueError(f"Stale tissue run manifest: {folder}")
    paths.extend([input_path, ROOT / "ml/tissue.py", wrapper])
    assert manifest["patients"] == patients
    assert manifest["outer_repeats"] == cfg["tissue"]["outer_repeats"]
    assert manifest["bootstrap_replicates"] == cfg["uncertainty"]["patient_bootstrap"]

    metrics = load_frame(folder / "metrics.tsv", paths)
    ci = load_frame(folder / "bootstrap_ci.tsv", paths)
    aggregate = metrics.loc[metrics["scope"].eq("all_repeats_pooled")].copy()
    expected_models = set(cfg["tissue"]["models"]) if local else {"ridge"}
    if not aggregate["aggregation"].eq("mean_of_repeat_metrics").all():
        raise ValueError(f"Tissue aggregate rows are not mean_of_repeat_metrics: {folder}")
    if len(metrics.loc[metrics["scope"].eq("repeat_pooled")]) != len(expected_models) * cfg["tissue"]["outer_repeats"]:
        raise ValueError(f"Tissue repeat metric count mismatch: {folder}")
    if set(aggregate["model"]) != expected_models:
        raise ValueError(f"Tissue model set mismatch: {folder}")
    if not ci["bootstrap_replicates"].eq(cfg["uncertainty"]["patient_bootstrap"]).all():
        raise ValueError(f"Tissue bootstrap replicate count mismatch: {folder}")
    validate_tissue_ci_points(aggregate, ci)

    result: dict[str, Any] = {
        "manifest": manifest,
        "ridge": one(aggregate, model="ridge"),
        "ridge_auc_ci": one(ci, model="ridge", metric="auc"),
        "model_metrics": clean(aggregate.to_dict("records")),
        "bootstrap_ci": clean(ci.to_dict("records")),
    }
    calibration = tissue_calibration_diagnostics(folder, cfg, paths, expected_models)
    if calibration is not None:
        result["calibration_diagnostics"] = calibration
    if local:
        assert manifest["optimism_bootstrap"] == cfg["uncertainty"]["optimism_bootstrap"]
        assert manifest["permutations"] == cfg["uncertainty"]["permutations"]
        result["best_single_gene"] = one(aggregate, model="best_single_gene")
        result["delta_auc_ci"] = one(ci, model="ridge_minus_best_single_gene", metric="auc")
        permutation_summary = load_json(folder / "ridge_permutation_summary.json", paths)
        null = load_frame(folder / "ridge_permutation.tsv", paths)
        null_rows = null.loc[null["kind"].eq("null")]
        observed_rows = null.loc[null["kind"].eq("observed")]
        B = cfg["uncertainty"]["permutations"]
        if (len(null_rows) != B or set(null_rows["permutation"]) != set(range(B))
                or len(observed_rows) != 1 or len(null) != B + 1
                or not np.isfinite(null["auc"].to_numpy(float)).all()):
            raise ValueError("Tissue permutation null draw count/value mismatch")
        observed = float(observed_rows.iloc[0]["auc"])
        expected_p = (1 + (null_rows["auc"] >= observed).sum()) / (B + 1)
        if (permutation_summary["permutations"] != B
                or permutation_summary["repeats"] != cfg["uncertainty"]["permutation_repeats"]
                or not np.isclose(permutation_summary["observed_auc"], observed, rtol=1e-12, atol=1e-14)
                or not np.isclose(permutation_summary["p_value"], expected_p, rtol=1e-12, atol=1e-14)):
            raise ValueError("Tissue permutation summary disagrees with complete observed/null rows")
        result["permutation"] = permutation_summary
        result["permutation_null"] = {
            "B": int(len(null_rows)),
            "observed_auc": float(permutation_summary["observed_auc"]),
            "p_value": float(permutation_summary["p_value"]),
        }
        optimism_draws = load_frame(folder / "ridge_optimism_bootstrap.tsv", paths)
        if len(optimism_draws) != cfg["uncertainty"]["optimism_bootstrap"]:
            raise ValueError("Tissue optimism bootstrap draw count mismatch")
        result["optimism_draws"] = clean(optimism_draws.to_dict("records"))
        result["optimism_corrected"] = load_json(folder / "ridge_optimism_corrected_summary.json", paths)
        if result["optimism_corrected"].get("bootstrap_draws") != cfg["uncertainty"]["optimism_bootstrap"]:
            raise ValueError("Tissue optimism corrected summary count mismatch")
        for key, source in [("input_sha256", input_path), ("config_sha256", ROOT / "registry/ml_config.json"), ("source_sha256", ROOT / "ml/tissue.py")]:
            if result["optimism_corrected"].get(key) != sha256(source):
                raise ValueError(f"Stale tissue optimism corrected summary: {key}")
        paths.append(ROOT / "scripts/postprocess_tissue_optimism.py")
    return result


def recurrence_result(folder: Path, cfg: dict[str, Any], paths: list[Path]) -> dict[str, Any]:
    manifest = load_json(folder / "summary.json", paths)
    validate_recurrence_summary(manifest, cfg)

    metrics = load_frame(folder / "primary_metrics.csv", paths)
    require_columns(metrics, {"block", "model", "horizon_days", "n_repeats", "uno_c_mean", "brier_mean", "ibs_mean"}, "primary_metrics")
    expected_primary = expected_frame(
        [{"block": block, "model": model, "horizon_days": horizon} for block in PRIMARY_BLOCKS for model in RECURRENCE_MODELS for horizon in HORIZONS]
    )
    assert_exact_keys(metrics, expected_primary, ["block", "model", "horizon_days"], "primary_metrics")
    assert_series_all(metrics["n_repeats"], cfg["recurrence"]["outer_repeats"], "primary_metrics.n_repeats")

    ci = load_frame(folder / "primary_bootstrap_ci.csv", paths)
    if "estimate" in ci.columns and "point" not in ci.columns:
        ci = ci.rename(columns={"estimate": "point"})
    require_columns(ci, {"block", "model", "horizon_days", "metric", "point", "ci_low", "ci_high", "bootstrap_n", "n_estimable", "status"}, "primary_bootstrap_ci")
    assert_exact_keys(ci, planned_recurrence_ci_keys(include_deltas=True), ["block", "model", "horizon_days", "metric"], "primary_bootstrap_ci")
    assert_series_all(ci["bootstrap_n"], cfg["uncertainty"]["patient_bootstrap"], "primary_bootstrap_ci.bootstrap_n")
    assert_recurrence_ci_points(metrics, ci)

    repeat_metrics = load_frame(folder / "primary_repeat_metrics.csv", paths)
    expected_repeat_rows = cfg["recurrence"]["outer_repeats"] * len(PRIMARY_BLOCKS) * len(RECURRENCE_MODELS) * len(HORIZONS)
    if len(repeat_metrics) != expected_repeat_rows:
        raise ValueError(f"primary_repeat_metrics row count mismatch: {len(repeat_metrics)} != {expected_repeat_rows}")

    null_km = load_frame(folder / "primary_null_km_metrics.csv", paths)
    expected_null = expected_frame([{"block": "null_km", "model": "km", "horizon_days": h} for h in HORIZONS])
    assert_exact_keys(null_km, expected_null, ["block", "model", "horizon_days"], "primary_null_km_metrics")
    assert_series_all(null_km["n_repeats"], cfg["recurrence"]["outer_repeats"], "primary_null_km_metrics.n_repeats")

    draw_means = load_frame(folder / "primary_bootstrap_repeat_mean_draws.csv", paths)
    distinct_draw_keys = draw_means[["block", "model", "horizon_days", "metric"]].drop_duplicates()
    assert_exact_keys(distinct_draw_keys, planned_recurrence_metric_keys(), ["block", "model", "horizon_days", "metric"], "primary_bootstrap_repeat_mean_draws")
    counts = draw_means.groupby(["block", "model", "horizon_days", "metric"]).size()
    if not counts.eq(cfg["uncertainty"]["patient_bootstrap"]).all():
        raise ValueError("primary_bootstrap_repeat_mean_draws must contain every draw for every base metric")

    perm = load_frame(folder / "primary_ridge_permutation.csv", paths)
    require_columns(perm, {"test", "observed", "permutation", "kind", "null_value", "permutation_hash", "p_value", "input_hash", "config_hash", "runner_hash", "metrics_hash", "module_hash"}, "primary_ridge_permutation")
    if not perm["kind"].eq("null").all():
        raise ValueError("Recurrence permutation file must contain only null rows in final schema")
    if sorted(perm["permutation"].astype(int).tolist()) != list(range(1, cfg["uncertainty"]["permutations"] + 1)):
        raise ValueError("Recurrence permutation ids are not exactly 1..B")
    if perm["permutation_hash"].nunique() != cfg["uncertainty"]["permutations"]:
        raise ValueError("Recurrence permutation hashes are not unique")
    if perm["p_value"].nunique() != 1 or perm["observed"].nunique() != 1:
        raise ValueError("Recurrence permutation observed/p_value must be constant")
    if not np.isfinite(pd.to_numeric(perm["null_value"], errors="coerce")).all():
        raise ValueError("Recurrence permutation null values must be finite")
    expected_p = (1 + (perm["null_value"] >= float(perm.iloc[0]["observed"])).sum()) / (len(perm) + 1)
    assert_close(perm.iloc[0]["p_value"], expected_p, "Recurrence permutation p value")

    optimism_draws = load_frame(folder / "primary_ridge_optimism_bootstrap.csv", paths)
    expected_opt = expected_frame(
        [{"block": block, "model": "ridge", "horizon_days": horizon, "bootstrap": b} for block in PRIMARY_BLOCKS for horizon in HORIZONS for b in range(1, cfg["uncertainty"]["optimism_bootstrap"] + 1)]
    )
    assert_exact_keys(optimism_draws, expected_opt, ["block", "model", "horizon_days", "bootstrap"], "primary_ridge_optimism_bootstrap")

    optimism_corrected = load_frame(folder / "primary_ridge_optimism_corrected.csv", paths)
    require_columns(optimism_corrected, {"block", "model", "horizon_days", "metric", "original_apparent", "mean_optimism", "corrected", "B", "n_estimable", "status", "model_path"}, "primary_ridge_optimism_corrected")
    corrected_expected = expected_frame(
        [
            {"block": block, "model": "ridge", "horizon_days": horizon, "metric": metric}
            for block in PRIMARY_BLOCKS
            for horizon in HORIZONS
            for metric in ["uno_c", "harrell_c", "auc", "brier", "cal_intercept", "cal_slope", "ibs"]
        ]
    )
    assert_exact_keys(optimism_corrected, corrected_expected, ["block", "model", "horizon_days", "metric"], "primary_ridge_optimism_corrected")
    assert_series_all(optimism_corrected["B"], cfg["uncertainty"]["optimism_bootstrap"], "primary_ridge_optimism_corrected.B")
    for row in optimism_corrected.to_dict("records"):
        draws = optimism_draws.loc[
            optimism_draws["block"].eq(row["block"]) & optimism_draws["horizon_days"].eq(row["horizon_days"]),
            "optimism_" + row["metric"],
        ].to_numpy(float)
        finite = np.isfinite(draws)
        if int(finite.sum()) != row["n_estimable"]:
            raise ValueError("Recurrence optimism estimable count disagrees with saved draws")
        if row["status"] == "complete":
            if not finite.all():
                raise ValueError("Complete optimism correction cannot drop nonestimable draws")
            assert_close(row["mean_optimism"], float(draws.mean()), "Mean recurrence optimism")
            assert_close(row["corrected"], row["original_apparent"] - row["mean_optimism"], "Corrected recurrence apparent performance")
        elif pd.notna(row["corrected"]) or pd.notna(row["mean_optimism"]):
            raise ValueError("Nonestimable optimism correction must remain missing")

    sensitivity = load_frame(folder / "ridge_sensitivity_metrics.csv", paths)
    if set(sensitivity["sensitivity"]) != SENSITIVITIES or len(sensitivity) != len(SENSITIVITIES) * len(HORIZONS):
        raise ValueError("Recurrence sensitivity metrics do not contain the seven planned sensitivity analyses")
    assert_series_all(sensitivity["n_repeats"], cfg["recurrence"]["outer_repeats"], "ridge_sensitivity_metrics.n_repeats")

    selected = load_frame(folder / "primary_selected_hyperparameters.csv", paths)
    regularization = []
    for (block, model), group in selected.loc[selected["model"].ne("survival_forest")].groupby(["block", "model"]):
        regularization.append({"block": block, "model": model, "outer_fits": int(len(group)),
                               "largest_penalty_selected": int(group["selected_lambda"].eq(cfg["recurrence"]["lambda_grid"]["max"]).sum()),
                               "selected_lambda_min": float(group["selected_lambda"].min()),
                               "selected_lambda_median": float(group["selected_lambda"].median()),
                               "selected_lambda_max": float(group["selected_lambda"].max())})

    delta_uno_c_5y = one(ci, model="ridge", block="combined_minus_clinical", metric="delta_uno_c", horizon_days=1825)
    delta_brier_5y = one(ci, model="ridge", block="combined_minus_clinical", metric="delta_brier", horizon_days=1825)
    return {
        "summary": manifest,
        "input": clean(manifest["input"]),
        "model_metrics": clean(metrics.to_dict("records")),
        "repeat_metrics": {"rows": int(len(repeat_metrics))},
        "bootstrap_ci": clean(ci.to_dict("records")),
        "bootstrap_repeat_mean_draws": {"rows": int(len(draw_means)), "B": cfg["uncertainty"]["patient_bootstrap"]},
        "null_km_metrics": clean(null_km.to_dict("records")),
        "clinical": one(metrics, model="ridge", block="clinical", horizon_days=1825),
        "methylation": one(metrics, model="ridge", block="tumor10", horizon_days=1825),
        "combined": one(metrics, model="ridge", block="combined", horizon_days=1825),
        "delta": delta_uno_c_5y,
        "combined_minus_clinical": delta_uno_c_5y,
        "delta_uno_c_5y": delta_uno_c_5y,
        "delta_brier_5y": delta_brier_5y,
        "permutation": {
            "test": str(perm["test"].iloc[0]),
            "p_value": float(perm["p_value"].iloc[0]),
            "observed": float(perm["observed"].iloc[0]),
            "B": int(len(perm)),
        },
        "sensitivity_metrics": clean(sensitivity.to_dict("records")),
        "optimism_draws": clean(optimism_draws.to_dict("records")),
        "optimism_corrected": clean(optimism_corrected.to_dict("records")),
        "regularization_diagnostics": regularization,
    }


def main() -> None:
    cfg = json.loads((ROOT / "registry/ml_config.json").read_text())
    input_manifest = json.loads((ROOT / "registry/ml_input_manifest.json").read_text())
    paths = [ROOT / "registry/ml_config.json", ROOT / "registry/ml_input_manifest.json"]
    for item in input_manifest["inputs"]:
        assert sha256(ROOT / item["file"]) == item["sha256"], f"Frozen ML input changed: {item['file']}"
        paths.append(ROOT / item["file"])

    tissue = tissue_result(ROOT / "results/ml/tissue", 87, cfg, paths, local=True)
    public = {name: tissue_result(ROOT / "results/ml/public" / name, n, cfg, paths) for name, n in [("colonomics", 92), ("gse119526", 48)]}
    recurrence = recurrence_result(ROOT / "results/ml/recurrence", cfg, paths)

    raw = np.array([tissue["permutation"]["p_value"], recurrence["permutation"]["p_value"]], dtype=float)
    if not (np.isfinite(raw).all() and ((raw > 0) & (raw <= 1)).all()):
        raise ValueError("Primary global test p-values must be finite in (0, 1]")
    order = np.argsort(raw, kind="stable")
    adjusted = np.empty(2)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(raw[order] * np.array([2, 1])))
    tests = {name: {"p_value": float(raw[i]), "p_holm": float(adjusted[i])} for i, name in enumerate(["tissue", "recurrence"])}

    payload = {
        "status": "final",
        "tissue": tissue,
        "recurrence": recurrence,
        "public": public,
        "global_tests": tests,
        "optimism": {"tissue": tissue.get("optimism_corrected"), "recurrence": recurrence.get("optimism_corrected")},
        "sensitivity": {"recurrence": recurrence.get("sensitivity_metrics")},
        "aggregation": "Mean of per-repeat held-out metrics; recurrence concordance pools within-fold counts within each repeat",
        "interval_scope": "Patient bootstrap conditional on fitted CV models; full-original optimism correction is reported separately",
        "sources": [{"file": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in dict.fromkeys(paths)],
    }
    write_json(ROOT / "results/ml/summary.json", clean(payload))
    pd.DataFrame([{"task": k, **v} for k, v in tests.items()]).to_csv(ROOT / "results/ml/global_tests.tsv", sep="\t", index=False)
    print(json.dumps({"status": "final", "source_files": len(paths), "global_tests": tests}, sort_keys=True))


if __name__ == "__main__":
    main()
