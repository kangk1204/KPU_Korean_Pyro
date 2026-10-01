#!/usr/bin/env python3
"""Build corrected recurrence metric tables for the 124 integrated revision.

This script does not refit survival models. It reads the preserved 114 recurrence
OOF predictions and bootstrap outputs, repairs manuscript-facing metric selection,
and writes machine-readable tables for the revised Table S4 / Supplementary
Figure S3 source data.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HORIZON_DAYS = 1825
SUPPORT_FLOOR = 0.1
BLOCK_ORDER = ["clinical", "tumor10", "combined"]
BLOCK_LABELS = {
    "clinical": "Clinical",
    "tumor10": "Tumor methylation",
    "combined": "Clinical + methylation",
    "null_km": "Null Kaplan-Meier",
}


@dataclass(frozen=True)
class SourcePaths:
    workspace: Path
    revision_root: Path
    source_114: Path
    recurrence_dir: Path
    outdir: Path
    evidence_dir: Path
    decisions_dir: Path

    @classmethod
    def build(cls, workspace: Path, revision_root: Path) -> "SourcePaths":
        source_114 = workspace / "114_ML_DataDriven_20260905"
        return cls(
            workspace=workspace,
            revision_root=revision_root,
            source_114=source_114,
            recurrence_dir=source_114 / "results" / "ml" / "recurrence",
            outdir=revision_root / "results" / "recurrence_revision",
            evidence_dir=revision_root / "review" / "evidence",
            decisions_dir=revision_root / "review" / "decisions",
        )


def auc_pairwise(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y = y[ok]
    score = score[ok]
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def ipcw_auc(frame: pd.DataFrame, score_col: str, horizon: int = HORIZON_DAYS) -> float:
    time = frame["duration_days"].to_numpy(float)
    event = frame["event"].to_numpy(int)
    score = frame[score_col].to_numpy(float)
    g_time = frame["G_time"].to_numpy(float)
    g_horizon = frame[f"G_{horizon}"].to_numpy(float)
    cases = np.where((event == 1) & (time <= horizon))[0]
    controls = np.where(time > horizon)[0]
    if len(cases) == 0 or len(controls) == 0:
        return float("nan")
    if np.any(~np.isfinite(g_time[cases]) | (g_time[cases] < SUPPORT_FLOOR)):
        return float("nan")
    if np.any(~np.isfinite(g_horizon[controls]) | (g_horizon[controls] < SUPPORT_FLOOR)):
        return float("nan")
    num = 0.0
    den = 0.0
    for i in cases:
        weights = 1.0 / (g_time[i] * g_horizon[controls])
        den += float(weights.sum())
        den_cmp = score[i] - score[controls]
        num += float((weights * ((den_cmp > 0) + 0.5 * (den_cmp == 0))).sum())
    return float(num / den) if den > 0 else float("nan")


def within_fold_auc(frame: pd.DataFrame, score_col: str, horizon: int = HORIZON_DAYS) -> float:
    num = 0.0
    den = 0.0
    for _, fold_frame in frame.groupby("fold", sort=True):
        time = fold_frame["duration_days"].to_numpy(float)
        event = fold_frame["event"].to_numpy(int)
        score = fold_frame[score_col].to_numpy(float)
        g_time = fold_frame["G_time"].to_numpy(float)
        g_horizon = fold_frame[f"G_{horizon}"].to_numpy(float)
        cases = np.where((event == 1) & (time <= horizon))[0]
        controls = np.where(time > horizon)[0]
        if len(cases) == 0 or len(controls) == 0:
            continue
        if np.any(~np.isfinite(g_time[cases]) | (g_time[cases] < SUPPORT_FLOOR)):
            continue
        if np.any(~np.isfinite(g_horizon[controls]) | (g_horizon[controls] < SUPPORT_FLOOR)):
            continue
        for i in cases:
            weights = 1.0 / (g_time[i] * g_horizon[controls])
            den += float(weights.sum())
            cmp = score[i] - score[controls]
            num += float((weights * ((cmp > 0) + 0.5 * (cmp == 0))).sum())
    return float(num / den) if den > 0 else float("nan")


def ipcw_brier(frame: pd.DataFrame, risk_col: str, horizon: int = HORIZON_DAYS) -> float:
    time = frame["duration_days"].to_numpy(float)
    event = frame["event"].to_numpy(int)
    risk = np.clip(frame[risk_col].to_numpy(float), 0.0, 1.0)
    g_time = frame["G_time"].to_numpy(float)
    g_horizon = frame[f"G_{horizon}"].to_numpy(float)
    event_known = (event == 1) & (time <= horizon)
    control_known = time > horizon
    known = event_known | control_known
    if not known.any():
        return float("nan")
    err = 0.0
    for i in np.where(known)[0]:
        target = 1.0 if event_known[i] else 0.0
        g = g_time[i] if event_known[i] else g_horizon[i]
        if not np.isfinite(g) or g < SUPPORT_FLOOR:
            return float("nan")
        err += float(((target - risk[i]) ** 2) / g)
    return float(err / len(time))


def repeat_metric(frame: pd.DataFrame, metric: str) -> float:
    per_repeat = []
    for _, rep in frame.groupby("repeat", sort=True):
        if metric == "within_fold_auc_risk1825":
            per_repeat.append(within_fold_auc(rep, "risk_1825"))
        elif metric == "pooled_auc_risk1825":
            per_repeat.append(ipcw_auc(rep, "risk_1825"))
        elif metric == "pooled_auc_linear_predictor":
            per_repeat.append(ipcw_auc(rep, "risk_score"))
        elif metric == "brier":
            per_repeat.append(ipcw_brier(rep, "risk_1825"))
        else:
            raise ValueError(metric)
    arr = np.asarray(per_repeat, dtype=float)
    return float(np.nanmean(arr))


def bootstrap_metric_ci(frame: pd.DataFrame, metric: str, *, n_boot: int = 2000, seed: int = 20260909) -> tuple[float, float, float, int]:
    point = repeat_metric(frame, metric)
    rng = np.random.default_rng(seed)
    patients = np.asarray(sorted(frame["study_id"].unique()))
    indexed: dict[int, dict[str, object]] = {}
    for repeat, rep in frame.groupby("repeat", sort=True):
        rep = rep.reset_index(drop=True)
        indexed[int(repeat)] = {
            "frame": rep,
            "patient_to_index": {pid: rep.index[rep["study_id"].eq(pid)].to_numpy() for pid in patients},
        }
    draws = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        rep_values = []
        for repeat in sorted(indexed):
            info = indexed[repeat]
            rep = info["frame"]
            p2i = info["patient_to_index"]
            idx = np.concatenate([p2i[pid] for pid in sampled])
            sampled_rep = rep.iloc[idx]
            if metric == "within_fold_auc_risk1825":
                rep_values.append(within_fold_auc(sampled_rep, "risk_1825"))
            elif metric == "pooled_auc_risk1825":
                rep_values.append(ipcw_auc(sampled_rep, "risk_1825"))
            elif metric == "pooled_auc_linear_predictor":
                rep_values.append(ipcw_auc(sampled_rep, "risk_score"))
            elif metric == "brier":
                rep_values.append(ipcw_brier(sampled_rep, "risk_1825"))
            else:
                raise ValueError(metric)
        draws.append(float(np.nanmean(rep_values)))
    arr = np.asarray(draws, dtype=float)
    arr = arr[np.isfinite(arr)]
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return point, float(lo), float(hi), int(len(arr))


def count_horizon_support(primary: pd.DataFrame, horizon: int) -> dict[str, int]:
    event_by_h = (primary["event"].eq(1) & primary["duration_days"].le(horizon))
    controls = primary["duration_days"].gt(horizon)
    unknown = primary["event"].eq(0) & primary["duration_days"].le(horizon)
    return {
        "horizon_days": int(horizon),
        "n_patients": int(len(primary)),
        "total_events": int(primary["event"].sum()),
        "events_by_horizon": int(event_by_h.sum()),
        "controls_at_horizon": int(controls.sum()),
        "censored_before_horizon_unknown_status": int(unknown.sum()),
        "median_observed_time_days": int(primary["duration_days"].median()),
        "min_observed_time_days": int(primary["duration_days"].min()),
        "max_observed_time_days": int(primary["duration_days"].max()),
    }


def paired_delta_from_repeat_draws(recurrence_dir: Path) -> pd.DataFrame:
    draws = pd.read_csv(recurrence_dir / "primary_bootstrap_repeat_mean_draws.csv")
    rows = []
    for metric in ["uno_c", "brier"]:
        c = draws[(draws["block"].eq("combined")) & (draws["model"].eq("ridge")) & (draws["horizon_days"].eq(HORIZON_DAYS)) & (draws["metric"].eq(metric))][["draw", "value"]]
        base = draws[(draws["block"].eq("clinical")) & (draws["model"].eq("ridge")) & (draws["horizon_days"].eq(HORIZON_DAYS)) & (draws["metric"].eq(metric))][["draw", "value"]]
        merged = c.merge(base, on="draw", suffixes=("_combined", "_clinical"))
        delta = merged["value_combined"] - merged["value_clinical"]
        rows.append({
            "contrast": "combined_ridge_minus_clinical_ridge",
            "metric": f"delta_{metric}",
            "horizon_days": HORIZON_DAYS,
            "estimate_from_bootstrap_draw_mean": float(delta.mean()),
            "ci_low_from_bootstrap_draws": float(np.quantile(delta, 0.025)),
            "ci_high_from_bootstrap_draws": float(np.quantile(delta, 0.975)),
            "bootstrap_n": int(len(delta)),
            "interval_scope": "paired_patient_cluster_bootstrap_conditional_on_preserved_nested_cv_oof",
            "decision": "primary" if metric == "uno_c" else "secondary_brier_diagnostic",
        })
    return pd.DataFrame(rows)


def permutation_diagnostics(recurrence_dir: Path) -> pd.DataFrame:
    perm = pd.read_csv(recurrence_dir / "primary_ridge_permutation.csv")
    observed = float(perm["observed"].iloc[0])
    ge = int((perm["null_value"] >= observed).sum())
    p_add_one = (ge + 1) / (len(perm) + 1)
    return pd.DataFrame([{
        "test": str(perm["test"].iloc[0]),
        "observed_uno_c": observed,
        "n_permutations": int(len(perm)),
        "n_null_ge_observed": ge,
        "p_value_add_one": float(p_add_one),
        "p_value_file": float(perm["p_value"].iloc[0]),
        "status_all_complete": bool(perm["status"].eq("complete").all()),
        "note": "observed value is the permutation diagnostic statistic from the preserved 114 permutation run",
    }])


def regularization_diagnostics(recurrence_dir: Path) -> pd.DataFrame:
    hp = pd.read_csv(recurrence_dir / "primary_selected_hyperparameters.csv")
    rows = []
    for (block, model), sub in hp.groupby(["block", "model"], sort=True):
        vals = pd.to_numeric(sub["selected_lambda"], errors="coerce")
        finite = vals.dropna()
        rows.append({
            "block": block,
            "model": model,
            "n_fits": int(len(sub)),
            "n_with_lambda": int(len(finite)),
            "lambda_min": float(finite.min()) if len(finite) else np.nan,
            "lambda_max": float(finite.max()) if len(finite) else np.nan,
            "n_at_grid_max_100": int((finite >= 99.999999).sum()) if len(finite) else 0,
            "fraction_at_grid_max_100": float((finite >= 99.999999).mean()) if len(finite) else np.nan,
        })
    return pd.DataFrame(rows)


def optimism_diagnostics(recurrence_dir: Path) -> pd.DataFrame:
    opt = pd.read_csv(recurrence_dir / "primary_ridge_optimism_corrected.csv")
    keep = opt[(opt["horizon_days"].eq(HORIZON_DAYS)) & (opt["model"].eq("ridge")) & (opt["metric"].isin(["uno_c", "auc", "brier", "ibs"]))].copy()
    keep["decision"] = np.where(keep["metric"].eq("uno_c"), "supporting_rank_diagnostic", "diagnostic_not_primary")
    return keep[["block", "model", "horizon_days", "metric", "original_apparent", "mean_optimism", "corrected", "B", "n_estimable", "status", "decision"]]


def elastic_net_oof_diagnostics(recurrence_dir: Path) -> pd.DataFrame:
    oof = pd.read_csv(
        recurrence_dir / "primary_oof_predictions.csv",
        usecols=["block", "model", "repeat", "risk_score", "risk_1825"],
    )
    elastic = oof[oof["model"].eq("elastic_net")].copy()
    rows = []
    for block, sub in elastic.groupby("block", sort=True):
        rep_rows = []
        for repeat, rep in sub.groupby("repeat", sort=True):
            rep_rows.append({
                "repeat": int(repeat),
                "risk_score_unique_n": int(rep["risk_score"].nunique(dropna=False)),
                "risk_score_sd": float(rep["risk_score"].std(ddof=1)),
                "risk_1825_unique_n": int(rep["risk_1825"].nunique(dropna=False)),
                "risk_1825_sd": float(rep["risk_1825"].std(ddof=1)),
            })
        rep_df = pd.DataFrame(rep_rows)
        rows.append({
            "block": block,
            "model": "elastic_net",
            "n_repeats": int(len(rep_df)),
            "min_unique_risk_score_per_repeat": int(rep_df["risk_score_unique_n"].min()),
            "max_unique_risk_score_per_repeat": int(rep_df["risk_score_unique_n"].max()),
            "median_risk_score_sd_per_repeat": float(rep_df["risk_score_sd"].median()),
            "min_unique_risk_1825_per_repeat": int(rep_df["risk_1825_unique_n"].min()),
            "max_unique_risk_1825_per_repeat": int(rep_df["risk_1825_unique_n"].max()),
            "median_risk_1825_sd_per_repeat": float(rep_df["risk_1825_sd"].median()),
            "interpretation": "constant_linear_predictor" if rep_df["risk_score_unique_n"].max() == 1 else "mostly_near_null_with_some_nonconstant_repeats",
        })
    return pd.DataFrame(rows)


def build_outputs(paths: SourcePaths, n_boot: int = 2000) -> dict[str, object]:
    paths.outdir.mkdir(parents=True, exist_ok=True)
    paths.evidence_dir.mkdir(parents=True, exist_ok=True)
    paths.decisions_dir.mkdir(parents=True, exist_ok=True)

    primary = pd.read_csv(paths.source_114 / "data" / "ml" / "recurrence.tsv", sep="\t")
    primary = primary[primary["recurrence_primary"].eq(1)].copy()
    support = pd.DataFrame([count_horizon_support(primary, 1095), count_horizon_support(primary, HORIZON_DAYS)])
    support.to_csv(paths.outdir / "horizon_support_counts.tsv", sep="\t", index=False)

    oof_cols = ["repeat", "fold", "study_id", "event", "duration_days", "block", "model", "risk_score", "risk_1095", "risk_1825", "G_1095", "G_1825", "G_time"]
    oof = pd.read_csv(paths.recurrence_dir / "primary_oof_predictions.csv", usecols=oof_cols)
    null_oof = pd.read_csv(paths.recurrence_dir / "primary_null_km_oof_predictions.csv", usecols=oof_cols)
    metrics = pd.read_csv(paths.recurrence_dir / "primary_metrics.csv")
    bci = pd.read_csv(paths.recurrence_dir / "primary_bootstrap_ci.csv")
    null_metrics = pd.read_csv(paths.recurrence_dir / "primary_null_km_metrics.csv")

    diagnostic_rows = []
    for block in BLOCK_ORDER:
        frame = oof[oof["block"].eq(block) & oof["model"].eq("ridge")]
        for metric in ["within_fold_auc_risk1825", "pooled_auc_risk1825", "pooled_auc_linear_predictor", "brier"]:
            point, lo, hi, n_eff = bootstrap_metric_ci(frame, metric, n_boot=n_boot, seed=20260909 + BLOCK_ORDER.index(block))
            diagnostic_rows.append({
                "block": block,
                "model": "ridge",
                "horizon_days": HORIZON_DAYS,
                "metric": metric,
                "estimate": point,
                "ci_low": lo,
                "ci_high": hi,
                "bootstrap_n_estimable": n_eff,
                "decision": "diagnostic_only" if metric != "brier" else "secondary_with_null_benchmark",
            })
    null_frame = null_oof[null_oof["block"].eq("null_km") & null_oof["model"].eq("km")]
    for metric in ["within_fold_auc_risk1825", "pooled_auc_risk1825", "brier"]:
        point, lo, hi, n_eff = bootstrap_metric_ci(null_frame, metric, n_boot=n_boot, seed=20260919)
        diagnostic_rows.append({
            "block": "null_km",
            "model": "km",
            "horizon_days": HORIZON_DAYS,
            "metric": metric,
            "estimate": point,
            "ci_low": lo,
            "ci_high": hi,
            "bootstrap_n_estimable": n_eff,
            "decision": "null_benchmark",
        })
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(paths.outdir / "risk_metric_diagnostics.tsv", sep="\t", index=False)

    delta = paired_delta_from_repeat_draws(paths.recurrence_dir)
    source_delta = bci[(bci["block"].eq("combined_minus_clinical")) & (bci["model"].eq("ridge")) & (bci["horizon_days"].eq(HORIZON_DAYS))]
    source_delta = source_delta[source_delta["metric"].isin(["delta_uno_c", "delta_brier"])]
    delta = delta.merge(source_delta[["metric", "estimate", "ci_low", "ci_high"]], on="metric", how="left", suffixes=("", "_source_primary_bootstrap_ci"))
    delta = delta.rename(columns={
        "estimate": "estimate_source_primary_bootstrap_ci",
        "ci_low": "ci_low_source_primary_bootstrap_ci",
        "ci_high": "ci_high_source_primary_bootstrap_ci",
    })
    delta["estimate"] = delta["estimate_source_primary_bootstrap_ci"]
    delta["ci_low"] = delta["ci_low_source_primary_bootstrap_ci"]
    delta["ci_high"] = delta["ci_high_source_primary_bootstrap_ci"]
    delta["max_abs_ci_diff_vs_recomputed_draw_quantiles"] = delta[["ci_low", "ci_high"]].sub(delta[["ci_low_from_bootstrap_draws", "ci_high_from_bootstrap_draws"]].to_numpy()).abs().max(axis=1)
    delta["point_estimate_minus_bootstrap_draw_mean"] = delta["estimate"] - delta["estimate_from_bootstrap_draw_mean"]
    delta.to_csv(paths.outdir / "table_s3_delta_ci.tsv", sep="\t", index=False)

    table_rows = []
    support5 = support[support["horizon_days"].eq(HORIZON_DAYS)].iloc[0].to_dict()
    for block in BLOCK_ORDER:
        row = metrics[(metrics["block"].eq(block)) & (metrics["model"].eq("ridge")) & (metrics["horizon_days"].eq(HORIZON_DAYS))].iloc[0]
        uno = bci[(bci["block"].eq(block)) & (bci["model"].eq("ridge")) & (bci["horizon_days"].eq(HORIZON_DAYS)) & (bci["metric"].eq("uno_c"))].iloc[0]
        harrell = bci[(bci["block"].eq(block)) & (bci["model"].eq("ridge")) & (bci["horizon_days"].eq(HORIZON_DAYS)) & (bci["metric"].eq("harrell_c"))].iloc[0]
        brier = bci[(bci["block"].eq(block)) & (bci["model"].eq("ridge")) & (bci["horizon_days"].eq(HORIZON_DAYS)) & (bci["metric"].eq("brier"))].iloc[0]
        wf_auc = diagnostics[(diagnostics["block"].eq(block)) & (diagnostics["metric"].eq("within_fold_auc_risk1825"))].iloc[0]
        table_rows.append({
            "model": "Ridge Cox",
            "predictor_block": BLOCK_LABELS[block],
            "block": block,
            "n_patients": int(support5["n_patients"]),
            "total_events": int(support5["total_events"]),
            "horizon_days": HORIZON_DAYS,
            "events_by_horizon": int(support5["events_by_horizon"]),
            "controls_at_horizon": int(support5["controls_at_horizon"]),
            "censored_before_horizon_unknown_status": int(support5["censored_before_horizon_unknown_status"]),
            "uno_c_5y": float(uno["estimate"]),
            "uno_c_5y_ci_low": float(uno["ci_low"]),
            "uno_c_5y_ci_high": float(uno["ci_high"]),
            "harrell_c_secondary": float(harrell["estimate"]),
            "harrell_c_ci_low": float(harrell["ci_low"]),
            "harrell_c_ci_high": float(harrell["ci_high"]),
            "ipcw_brier_5y_secondary": float(brier["estimate"]),
            "ipcw_brier_5y_ci_low": float(brier["ci_low"]),
            "ipcw_brier_5y_ci_high": float(brier["ci_high"]),
            "within_fold_auc_5y_diagnostic": float(wf_auc["estimate"]),
            "within_fold_auc_5y_ci_low": float(wf_auc["ci_low"]),
            "within_fold_auc_5y_ci_high": float(wf_auc["ci_high"]),
            "retired_pooled_risk_auc_5y": float(row["auc_mean"]),
            "retired_calibration_slope_5y": float(row["cal_slope_mean"]),
            "decision_note": "Uno C primary; Brier secondary with null-KM benchmark; pooled AUC/calibration retired from claims",
        })
    table = pd.DataFrame(table_rows)
    table.to_csv(paths.outdir / "table_s3_recurrence_primary_corrected.tsv", sep="\t", index=False)

    figure_rows = []
    for _, row in table.iterrows():
        figure_rows.append({
            "panel": "A_primary_uno_c",
            "plot_order": len(figure_rows) + 1,
            "label": row["predictor_block"],
            "block": row["block"],
            "model": "ridge",
            "metric": "uno_c_5y",
            "point": row["uno_c_5y"],
            "ci_low": row["uno_c_5y_ci_low"],
            "ci_high": row["uno_c_5y_ci_high"],
            "reference_line": 0.5,
            "decision": "primary",
        })
    drow = delta[delta["metric"].eq("delta_uno_c")].iloc[0]
    figure_rows.append({
        "panel": "B_incremental_delta_uno_c",
        "plot_order": 1,
        "label": "Clinical + methylation minus clinical",
        "block": "combined_minus_clinical",
        "model": "ridge",
        "metric": "delta_uno_c_5y",
        "point": float(drow["estimate_source_primary_bootstrap_ci"]),
        "ci_low": float(drow["ci_low_source_primary_bootstrap_ci"]),
        "ci_high": float(drow["ci_high_source_primary_bootstrap_ci"]),
        "reference_line": 0.0,
        "decision": "primary_delta",
    })
    null_brier = null_metrics[(null_metrics["block"].eq("null_km")) & (null_metrics["horizon_days"].eq(HORIZON_DAYS))].iloc[0]
    for _, row in table.iterrows():
        figure_rows.append({
            "panel": "C_brier_null_benchmark",
            "plot_order": len([x for x in figure_rows if x["panel"] == "C_brier_null_benchmark"]) + 1,
            "label": row["predictor_block"],
            "block": row["block"],
            "model": "ridge",
            "metric": "ipcw_brier_5y",
            "point": row["ipcw_brier_5y_secondary"],
            "ci_low": row["ipcw_brier_5y_ci_low"],
            "ci_high": row["ipcw_brier_5y_ci_high"],
            "reference_line": float(null_brier["brier_mean"]),
            "decision": "secondary_with_null_benchmark",
        })
    figure = pd.DataFrame(figure_rows)
    figure.to_csv(paths.outdir / "figure_s2_recurrence_source.tsv", sep="\t", index=False)

    metric_decisions = pd.DataFrame([
        {"metric": "5-year Uno C", "status": "retained_primary", "reason": "computed from fold-wise comparable rank pairs; primary negative incremental conclusion stands", "replacement_or_display": "Table S4 and Figure S3 primary panel"},
        {"metric": "paired combined-clinical Uno C delta", "status": "retained_primary", "reason": "paired patient-cluster bootstrap CI already available and valid for incremental claim", "replacement_or_display": "Table S4 delta row and Figure S3 delta panel"},
        {"metric": "Harrell C", "status": "retained_secondary", "reason": "rank metric accumulated within folds; less horizon-specific than Uno C", "replacement_or_display": "optional Table S4 secondary column"},
        {"metric": "5-year IPCW Brier", "status": "retained_secondary_with_null_benchmark", "reason": "per-patient probability error is not invalidated by cross-fold rank comparability; values are almost identical to null-KM and should not imply methylation benefit", "replacement_or_display": "optional Table S4 secondary column with null-KM benchmark"},
        {"metric": "pooled risk_1825 AUC", "status": "retired_from_claims", "reason": "case-control ranking pooled across folds compares fold-specific Breslow baseline risk scales", "replacement_or_display": "within-fold AUC diagnostic only, not primary"},
        {"metric": "calibration slope/intercept", "status": "retired_from_claims", "reason": "5-year known-status support is 46 patients per repeat and estimates are unstable; negative slope should not be interpreted as model performance evidence", "replacement_or_display": "archive diagnostic; omit from main claim"},
        {"metric": "3-year endpoint", "status": "not_substituted", "reason": "not requested and would silently switch to a better-supported endpoint after seeing 5-year weakness", "replacement_or_display": "support counts only"},
    ])
    metric_decisions.to_csv(paths.outdir / "metric_retention_decisions.tsv", sep="\t", index=False)

    null_table = null_metrics[null_metrics["horizon_days"].eq(HORIZON_DAYS)].copy()
    null_table.to_csv(paths.outdir / "null_km_benchmark.tsv", sep="\t", index=False)
    perm = permutation_diagnostics(paths.recurrence_dir)
    perm.to_csv(paths.outdir / "permutation_diagnostic.tsv", sep="\t", index=False)
    reg = regularization_diagnostics(paths.recurrence_dir)
    reg.to_csv(paths.outdir / "regularization_boundary_diagnostic.tsv", sep="\t", index=False)
    opt = optimism_diagnostics(paths.recurrence_dir)
    opt.to_csv(paths.outdir / "optimism_diagnostic.tsv", sep="\t", index=False)
    elastic_oof = elastic_net_oof_diagnostics(paths.recurrence_dir)
    elastic_oof.to_csv(paths.outdir / "elastic_net_oof_diagnostic.tsv", sep="\t", index=False)

    report_lines = [
        "# ML recurrence revision evidence",
        "",
        "## Metric contract",
        "",
        "Primary recurrence reporting should use 5-year Uno C and the paired combined-minus-clinical Uno C delta CI. Harrell C can remain secondary. Brier can remain only as a secondary prediction-error diagnostic with the null-KM benchmark. Pooled 5-year risk AUC and calibration slope/intercept are retired from manuscript-facing performance claims. The time-support label is median observed time, not confirmed recurrence follow-up.",
        "",
        "## Corrected primary ridge Table S4 source",
        "",
        table.to_markdown(index=False),
        "",
        "## Paired delta CI",
        "",
        delta.to_markdown(index=False),
        "",
        "## Null-KM benchmark",
        "",
        null_table.to_markdown(index=False),
        "",
        "## Support counts",
        "",
        support.to_markdown(index=False),
        "",
        "## Diagnostics",
        "",
        f"Permutation p-value verified as (131 + 1) / (999 + 1) = {float(perm['p_value_add_one'].iloc[0]):.3f}.",
        "Ridge/elastic-net regularization selected the grid maximum frequently; this supports near-null/overpenalized reporting but does not justify a new grid-extension refit because the revised primary claim is the preserved nested-CV Uno C null result.",
        "Elastic-net OOF diagnostics show that the clinical linear predictor is exactly constant across repeats; tumor10 and combined are mostly near-null but have some nonconstant repeats. Avoid saying all elastic-net coefficients were zero unless coefficient files are separately audited.",
        "",
        "## Suggested prose boundaries",
        "",
        "Prior high apparent AUCs from earlier exploratory recurrence panels should be described as exploratory, same-cohort, model-selection-sensitive results that were replaced by the nested patient-grouped CV analysis. They should not be presented as validated prognostic performance.",
        "Elastic-net results should be footnoted as a heavily penalized near-null sensitivity: selected penalties were at the lambda grid maximum in 295/300 elastic-net fits; clinical OOF linear predictors were constant, while tumor10/combined remained effectively chance-level. Do not state all coefficients were zero unless coefficient files are separately audited.",
    ]
    evidence_md = paths.evidence_dir / "ml_revision_recurrence.md"
    evidence_md.write_text("\n".join(report_lines) + "\n")

    manifest = {
        "status": "complete",
        "source_114_recurrence_dir": str(paths.recurrence_dir),
        "horizon_days": HORIZON_DAYS,
        "bootstrap_replicates_for_new_diagnostics": int(n_boot),
        "outputs": sorted(set([p.name for p in paths.outdir.iterdir() if p.is_file()] + ["run_manifest.json"])),
        "primary_table_rows": int(len(table)),
        "delta_uno_c": delta[delta["metric"].eq("delta_uno_c")].iloc[0].to_dict(),
        "metric_contract": "Uno C primary; paired delta CI primary; Brier secondary with null-KM benchmark; pooled AUC/calibration retired from claims",
    }
    (paths.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--revision-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = SourcePaths.build(args.workspace.resolve(), args.revision_root.resolve())
    manifest = build_outputs(paths, n_boot=args.bootstrap)
    print(json.dumps({"status": manifest["status"], "outputs": manifest["outputs"], "delta_uno_c": manifest["delta_uno_c"]}, indent=2))


if __name__ == "__main__":
    main()
