"""Independent reference calculations for ML verification tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]


@dataclass(frozen=True)
class BaselineHazard:
    times: np.ndarray
    hazard: np.ndarray

    def at(self, time: float) -> float:
        mask = self.times <= time
        if not np.any(mask):
            return 0.0
        return float(self.hazard[mask][-1])


def breslow_baseline_hazard(times, events, linear_predictor, center=None):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    lp = np.asarray(linear_predictor, dtype=float)
    if center is None:
        center = float(np.mean(lp))
    event_times = np.unique(times[events == 1])
    hazards = []
    cumulative = 0.0
    risk_score = np.exp(lp - center)
    for event_time in event_times:
        deaths = int(np.sum((times == event_time) & (events == 1)))
        risk_sum = float(np.sum(risk_score[times >= event_time]))
        if risk_sum <= 0:
            raise ValueError("risk set has non-positive denominator")
        cumulative += deaths / risk_sum
        hazards.append(cumulative)
    return BaselineHazard(times=event_times, hazard=np.asarray(hazards, dtype=float))


def survival_probability(baseline, time, linear_predictor, center):
    lp = np.asarray(linear_predictor, dtype=float)
    return np.exp(-baseline.at(time) * np.exp(lp - center))


def censoring_km(times, events):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    censor_events = 1 - events
    unique_times = np.unique(times)
    survival = 1.0
    after = {}
    before = {}
    for time in unique_times:
        before[float(time)] = survival
        at_risk = int(np.sum(times >= time))
        censored = int(np.sum((times == time) & (censor_events == 1)))
        if at_risk > 0 and censored > 0:
            survival *= 1.0 - censored / at_risk
        after[float(time)] = survival
    return {"before": before, "after": after, "last": survival}


def _km_before(km, time):
    exact = [t for t in km["before"] if t == time]
    if exact:
        return float(km["before"][exact[0]])
    candidates = [t for t in km["after"] if t < time]
    if not candidates:
        return 1.0
    return float(km["after"][max(candidates)])


def _km_after(km, time):
    candidates = [t for t in km["after"] if t <= time]
    if not candidates:
        return 1.0
    return float(km["after"][max(candidates)])


def uno_c_at_tau(times, events, risk, tau, km=None):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if km is None:
        km = censoring_km(times, events)
    concordant = 0.0
    comparable = 0.0
    for i in range(len(times)):
        if events[i] != 1 or times[i] > tau:
            continue
        g = _km_before(km, float(times[i]))
        if g <= 0:
            continue
        weight = 1.0 / (g * g)
        for j in range(len(times)):
            if times[j] <= times[i]:
                continue
            comparable += weight
            if risk[i] > risk[j]:
                concordant += weight
            elif risk[i] == risk[j]:
                concordant += 0.5 * weight
    if comparable == 0:
        return np.nan, concordant, comparable
    return concordant / comparable, concordant, comparable


def weighted_auc_at_time(times, events, risk, horizon, km=None):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risk = np.asarray(risk, dtype=float)
    if km is None:
        km = censoring_km(times, events)
    cases = np.flatnonzero((events == 1) & (times <= horizon))
    controls = np.flatnonzero(times > horizon)
    if len(cases) == 0 or len(controls) == 0:
        return np.nan
    concordant = 0.0
    total = 0.0
    for i in cases:
        gi = _km_before(km, float(times[i]))
        if gi <= 0:
            continue
        wi = 1.0 / gi
        for j in controls:
            gj = _km_after(km, float(horizon))
            if gj <= 0:
                continue
            weight = wi / gj
            total += weight
            if risk[i] > risk[j]:
                concordant += weight
            elif risk[i] == risk[j]:
                concordant += 0.5 * weight
    if total == 0:
        return np.nan
    return concordant / total


def ipcw_brier_at_time(times, events, event_probability, horizon, km=None):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    probability = np.asarray(event_probability, dtype=float)
    if km is None:
        km = censoring_km(times, events)
    weighted_error = 0.0
    for i in range(len(times)):
        if events[i] == 1 and times[i] <= horizon:
            g = _km_before(km, float(times[i]))
            if g <= 0:
                continue
            weighted_error += ((1.0 - probability[i]) ** 2) / g
        elif times[i] > horizon:
            g = _km_after(km, float(horizon))
            if g <= 0:
                continue
            weighted_error += (probability[i] ** 2) / g
    if len(times) == 0:
        return np.nan
    return weighted_error / len(times)


def km_before(km, time):
    return _km_before(km, time)


def km_after(km, time):
    return _km_after(km, time)


def uno_c_from_saved_g_by_fold(frame, horizon, score_col="risk_score", g_col="G_time"):
    numerator = 0.0
    denominator = 0.0
    for _, fold_frame in frame.groupby("fold", sort=True):
        times = fold_frame["duration_days"].to_numpy(float)
        events = fold_frame["event"].to_numpy(int)
        score = fold_frame[score_col].to_numpy(float)
        if g_col in fold_frame:
            g_time = fold_frame[g_col].to_numpy(float)
        else:
            g_time = np.ones(len(fold_frame), dtype=float)
        for i in range(len(fold_frame)):
            if events[i] != 1 or times[i] > horizon:
                continue
            if not np.isfinite(g_time[i]) or g_time[i] < 0.1:
                continue
            weight = 1.0 / (g_time[i] * g_time[i])
            comparable = np.flatnonzero(times > times[i])
            denominator += weight * len(comparable)
            numerator += weight * np.sum((score[i] > score[comparable]) + 0.5 * (score[i] == score[comparable]))
    if denominator == 0:
        return np.nan, numerator, denominator
    return numerator / denominator, numerator, denominator


def weighted_auc_from_saved_g(frame, horizon, score_col, g_time_col="G_time", g_tau_col=None):
    if g_tau_col is None:
        g_tau_col = f"G_{int(horizon)}"
    cases = frame.index[(frame["event"].eq(1)) & (frame["duration_days"].le(horizon))].to_numpy()
    controls = frame.index[frame["duration_days"].gt(horizon)].to_numpy()
    if len(cases) == 0 or len(controls) == 0:
        return np.nan
    total = 0.0
    concordant = 0.0
    for case in cases:
        g_case = float(frame.loc[case, g_time_col])
        if not np.isfinite(g_case) or g_case < 0.1:
            continue
        for control in controls:
            g_control = float(frame.loc[control, g_tau_col])
            if not np.isfinite(g_control) or g_control < 0.1:
                continue
            weight = 1.0 / (g_case * g_control)
            total += weight
            a = float(frame.loc[case, score_col])
            b = float(frame.loc[control, score_col])
            concordant += weight * ((a > b) + 0.5 * (a == b))
    if total == 0:
        return np.nan
    return concordant / total


def ipcw_brier_from_saved_g(frame, horizon, risk_col, g_time_col="G_time", g_tau_col=None):
    if g_tau_col is None:
        g_tau_col = f"G_{int(horizon)}"
    if len(frame) == 0:
        return np.nan
    risk = np.clip(frame[risk_col].to_numpy(float), 0.0, 1.0)
    times = frame["duration_days"].to_numpy(float)
    events = frame["event"].to_numpy(int)
    g_time = frame[g_time_col].to_numpy(float)
    g_tau = frame[g_tau_col].to_numpy(float)
    case_mask = (events == 1) & (times <= horizon) & np.isfinite(g_time) & (g_time >= 0.1)
    control_mask = (times > horizon) & np.isfinite(g_tau) & (g_tau >= 0.1)
    error = np.sum(((1.0 - risk[case_mask]) ** 2) / g_time[case_mask])
    error += np.sum((risk[control_mask] ** 2) / g_tau[control_mask])
    return float(error / len(frame))


def ibs_from_saved_grid(frame, risk_prefix="risk_grid_", time_max=1825.0, n_grid=101):
    grid = np.linspace(0.0, time_max, n_grid)
    values = []
    for i, horizon in enumerate(grid, start=1):
        risk_col = f"{risk_prefix}{i}"
        g_col = f"G_grid_{i}"
        if risk_col not in frame or g_col not in frame:
            raise KeyError(f"missing grid columns for index {i}")
        values.append(ipcw_brier_from_saved_g(frame, horizon, risk_col, g_tau_col=g_col))
    return float(np.trapz(values, grid) / time_max)


def tissue_metric_block(frame):
    y = frame["y"].to_numpy(int)
    score = frame["score"].to_numpy(float)
    id_col = "study_id" if "study_id" in frame.columns else "patient_id"
    out = {
        "n": float(len(frame)),
        "n_patients": float(frame[id_col].nunique()),
        "auc": float(roc_auc_score(y, score)),
    }
    if "probability" in frame.columns and np.isfinite(frame["probability"].to_numpy(float)).all():
        out["brier"] = float(brier_score_loss(y, frame["probability"].to_numpy(float)))
    else:
        out["brier"] = np.nan
    for label, col in [("threshold_0_5", "pred_0_5"), ("normal_q95", "pred_normal_q95")]:
        pred = frame[col].to_numpy(int)
        if (pred < 0).any():
            out[f"{label}_sensitivity"] = np.nan
            out[f"{label}_specificity"] = np.nan
            out[f"{label}_balanced_accuracy"] = np.nan
            continue
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out[f"{label}_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else np.nan
        out[f"{label}_specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
        out[f"{label}_balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    return out


def summarize_tissue_metrics_from_oof(oof):
    rows = []
    for model, model_frame in oof.groupby("model", sort=False):
        repeat_metrics = []
        for repeat, repeat_frame in model_frame.groupby("repeat", sort=True):
            metrics = tissue_metric_block(repeat_frame)
            metrics.update({"model": model, "scope": "repeat_pooled", "repeat": float(repeat)})
            rows.append(metrics)
            repeat_metrics.append(metrics)
        repeat_table = pd.DataFrame(repeat_metrics)
        numeric_cols = [c for c in repeat_table.select_dtypes(include=[np.number]).columns if c != "repeat"]
        summary = {c: float(repeat_table[c].mean()) for c in numeric_cols}
        summary.update({"model": model, "scope": "all_repeats_pooled", "repeat": np.nan, "aggregation": "mean_of_repeat_metrics"})
        rows.append(summary)
    return pd.DataFrame(rows)


def bootstrap_cluster_rows(frame, group_col, sampled_groups):
    parts = []
    for draw_index, group in enumerate(sampled_groups):
        chunk = frame.loc[frame[group_col].eq(group)].copy()
        chunk["bootstrap_draw"] = draw_index
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def assert_group_disjoint(train_groups, test_groups):
    overlap = set(train_groups).intersection(set(test_groups))
    if overlap:
        raise AssertionError(f"group leakage: {sorted(overlap)}")


def read_optional_tsv(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")
