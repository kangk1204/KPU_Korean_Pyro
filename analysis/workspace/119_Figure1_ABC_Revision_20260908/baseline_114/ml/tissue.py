"""Leakage-controlled tissue classification models for paired CRC methylation data."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
DEFAULT_SEED = 20260905


@dataclass(frozen=True)
class TissueConfig:
    outer_folds: int = 5
    outer_repeats: int = 20
    inner_folds: int = 4
    logistic_C: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    elastic_l1_ratio: float = 0.5
    forest_trees: int = 500
    forest_depth: tuple[int | None, ...] = (3, None)
    forest_leaf: tuple[int, ...] = (5, 10)
    svm_C: tuple[float, ...] = (0.1, 1.0, 10.0)
    svm_gamma: tuple[float, ...] = (0.01, 0.1, 1.0)
    seed: int = DEFAULT_SEED
    bootstrap_replicates: int = 2000
    optimism_bootstrap: int = 500
    permutations: int = 999
    permutation_repeats: int = 5
    max_workers: int = 3

    @classmethod
    def from_json(cls, path: Path) -> "TissueConfig":
        raw = json.loads(path.read_text())
        tissue = raw.get("tissue", raw)
        uncertainty = raw.get("uncertainty", {})
        return cls(
            outer_folds=int(tissue.get("outer_folds", 5)),
            outer_repeats=int(tissue.get("outer_repeats", 20)),
            inner_folds=int(tissue.get("inner_folds", 4)),
            logistic_C=tuple(float(x) for x in tissue.get("logistic_C", [0.01, 0.1, 1, 10])),
            elastic_l1_ratio=float(tissue.get("elastic_l1_ratio", 0.5)),
            forest_trees=int(tissue.get("forest_trees", 500)),
            forest_depth=tuple(None if x is None else int(x) for x in tissue.get("forest_depth", [3, None])),
            forest_leaf=tuple(int(x) for x in tissue.get("forest_leaf", [5, 10])),
            svm_C=tuple(float(x) for x in tissue.get("svm_C", [0.1, 1, 10])),
            svm_gamma=tuple(float(x) for x in tissue.get("svm_gamma", [0.01, 0.1, 1])),
            seed=int(raw.get("seed", DEFAULT_SEED)),
            bootstrap_replicates=int(uncertainty.get("patient_bootstrap", 2000)),
            optimism_bootstrap=int(uncertainty.get("optimism_bootstrap", 500)),
            permutations=int(uncertainty.get("permutations", 999)),
            permutation_repeats=int(uncertainty.get("permutation_repeats", 5)),
            max_workers=min(3, int(raw.get("workers", 3))),
        )


def canonical_hash(paths: Iterable[Path], payloads: Iterable[Any] = ()) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    for payload in payloads:
        h.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def dataframe_hash(data: pd.DataFrame, genes: list[str] | None = None) -> str:
    genes = list(genes or GENES)
    cols = [c for c in ["patient_id", "group_id", "sample_id", "tissue", "y"] if c in data.columns] + genes
    payload = data.loc[:, cols].copy()
    return hashlib.sha256(payload.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def write_csv(df: pd.DataFrame, path: Path, sep: str = ",", gzip_output: bool | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output is None:
        gzip_output = path.suffix == ".gz"
    if gzip_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as gz:
            df.to_csv(gz, index=False, sep=sep)
    else:
        df.to_csv(path, index=False, sep=sep)


def prepare_tissue_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "patient_id" not in data.columns:
        if "study_id" not in data.columns:
            raise ValueError("tissue input needs patient_id or study_id")
        data["patient_id"] = data["study_id"].astype(str)
    if "sample_id" not in data.columns:
        data["sample_id"] = data.get("specimen_id", data.index.astype(str)).astype(str)
    if "group_id" not in data.columns:
        data["group_id"] = data["patient_id"]
    if "y" not in data.columns:
        raise ValueError("tissue input needs binary y")
    missing = [g for g in GENES if g not in data.columns]
    if missing:
        raise ValueError(f"missing gene columns: {missing}")
    data["y"] = data["y"].astype(int)
    if set(data["y"].unique()) != {0, 1}:
        raise ValueError("tissue y must contain both 0 and 1")
    pair = data.groupby("patient_id")["y"].agg(["size", "sum"])
    bad = pair[(pair["size"] != 2) | (pair["sum"] != 1)]
    if not bad.empty:
        raise ValueError("each patient must have exactly one normal and one tumor sample")
    if data[GENES].isna().any().any():
        raise ValueError("gene matrix contains missing values")
    return data.reset_index(drop=True)


def patient_kfolds(patient_ids: np.ndarray, n_splits: int, repeats: int, seed: int) -> list[dict[str, Any]]:
    patient_ids = np.asarray(sorted(map(str, patient_ids)))
    folds: list[dict[str, Any]] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + 1009 * repeat)
        perm = patient_ids.copy()
        rng.shuffle(perm)
        chunks = np.array_split(perm, n_splits)
        for fold, test_patients in enumerate(chunks):
            train_patients = np.setdiff1d(patient_ids, test_patients, assume_unique=True)
            folds.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "train_patients": train_patients.tolist(),
                    "test_patients": test_patients.tolist(),
                }
            )
    return folds


def inner_folds(train_patients: Iterable[str], n_splits: int, seed: int, repeat: int, fold: int) -> list[tuple[np.ndarray, np.ndarray]]:
    train_patients = np.asarray(sorted(map(str, train_patients)))
    rng = np.random.default_rng(seed + 7919 * (repeat + 1) + 1543 * (fold + 1))
    perm = train_patients.copy()
    rng.shuffle(perm)
    chunks = np.array_split(perm, n_splits)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for valid in chunks:
        train = np.setdiff1d(train_patients, valid, assume_unique=True)
        out.append((train, valid))
    return out


def _scale_fit(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def _as_pred(model: Any, X: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray | None]:
    if kind == "svm":
        score = model.decision_function(X)
        return np.asarray(score, float), None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        return np.asarray(prob, float), np.asarray(prob, float)
    score = model.decision_function(X)
    prob = 1.0 / (1.0 + np.exp(-score))
    return np.asarray(score, float), np.asarray(prob, float)


def _candidate_model(model_name: str, params: dict[str, Any], seed: int) -> tuple[Any, bool]:
    if model_name == "ridge":
        return LogisticRegression(penalty="l2", C=params["C"], solver="liblinear", max_iter=5000, random_state=seed), True
    if model_name == "elastic_net":
        return (
            LogisticRegression(
                penalty="elasticnet",
                C=params["C"],
                l1_ratio=params["l1_ratio"],
                solver="saga",
                max_iter=10000,
                tol=1e-4,
                random_state=seed,
            ),
            True,
        )
    if model_name == "random_forest":
        return (
            RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=seed,
                n_jobs=1,
            ),
            False,
        )
    if model_name == "rbf_svm":
        return SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], probability=False, random_state=seed), True
    if model_name == "best_single_gene":
        return LogisticRegression(penalty="l2", C=params["C"], solver="liblinear", max_iter=5000, random_state=seed), True
    raise ValueError(model_name)


def candidate_grid(model_name: str, config: TissueConfig, genes: list[str]) -> list[dict[str, Any]]:
    if model_name == "ridge":
        return [{"C": c} for c in config.logistic_C]
    if model_name == "elastic_net":
        return [{"C": c, "l1_ratio": config.elastic_l1_ratio} for c in config.logistic_C]
    if model_name == "random_forest":
        return [
            {
                "n_estimators": config.forest_trees,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "max_features": "sqrt",
            }
            for depth in config.forest_depth
            for leaf in sorted(config.forest_leaf, reverse=True)
        ]
    if model_name == "rbf_svm":
        return [{"C": c, "gamma": g} for c in config.svm_C for g in config.svm_gamma]
    if model_name == "best_single_gene":
        return [{"gene": gene, "C": c} for gene in genes for c in config.logistic_C]
    raise ValueError(model_name)


def _metric_for_selection(model_name: str, y: np.ndarray, score: np.ndarray, prob: np.ndarray | None) -> float:
    if model_name in {"random_forest", "rbf_svm"}:
        try:
            return float(roc_auc_score(y, score))
        except ValueError:
            return float("nan")
    if prob is None:
        raise ValueError("log-loss selection requires probabilities")
    return float(log_loss(y, np.clip(prob, 1e-8, 1 - 1e-8), labels=[0, 1]))


def tune_model(
    data: pd.DataFrame,
    train_patients: Iterable[str],
    model_name: str,
    config: TissueConfig,
    repeat: int,
    fold: int,
    genes: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    genes = list(genes or GENES)
    train_patients = np.asarray(list(map(str, train_patients)))
    candidates = candidate_grid(model_name, config, genes)
    records: list[dict[str, Any]] = []
    for cand_idx, params in enumerate(candidates):
        vals: list[float] = []
        for inner_idx, (inner_train, inner_valid) in enumerate(inner_folds(train_patients, config.inner_folds, config.seed, repeat, fold)):
            train_mask = data["group_id"].isin(inner_train).to_numpy()
            valid_mask = data["group_id"].isin(inner_valid).to_numpy()
            use_genes = [params["gene"]] if model_name == "best_single_gene" else genes
            Xtr = data.loc[train_mask, use_genes].to_numpy(float)
            Xva = data.loc[valid_mask, use_genes].to_numpy(float)
            ytr = data.loc[train_mask, "y"].to_numpy(int)
            yva = data.loc[valid_mask, "y"].to_numpy(int)
            model, needs_scale = _candidate_model(model_name, params, config.seed + repeat * 1000 + fold * 100 + inner_idx)
            if needs_scale:
                scaler = _scale_fit(Xtr)
                Xtr = scaler.transform(Xtr)
                Xva = scaler.transform(Xva)
            model.fit(Xtr, ytr)
            score, prob = _as_pred(model, Xva, "svm" if model_name == "rbf_svm" else model_name)
            vals.append(_metric_for_selection(model_name, yva, score, prob))
        finite = np.asarray([v for v in vals if np.isfinite(v)], float)
        mean = float(np.mean(finite)) if len(finite) else float("nan")
        se = float(np.std(finite, ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else 0.0
        records.append({"candidate_index": cand_idx, "model": model_name, "params_json": json.dumps(params, sort_keys=True), "mean": mean, "se": se})
    tab = pd.DataFrame(records)
    if model_name in {"ridge", "elastic_net", "best_single_gene"}:
        best_idx = int(tab["mean"].idxmin())
        cutoff = float(tab.loc[best_idx, "mean"] + tab.loc[best_idx, "se"])
        eligible = tab[tab["mean"].le(cutoff)].copy()
        eligible["C_value"] = eligible["params_json"].map(lambda s: json.loads(s)["C"])
        if model_name == "best_single_gene":
            order = {g: i for i, g in enumerate(genes)}
            eligible["gene_order"] = eligible["params_json"].map(lambda s: order[json.loads(s)["gene"]])
            eligible = eligible.sort_values(["C_value", "mean", "gene_order", "candidate_index"], ascending=[True, True, True, True])
        else:
            eligible = eligible.sort_values(["C_value", "mean", "candidate_index"], ascending=[True, True, True])
        chosen_row = eligible.iloc[0]
    else:
        chosen_row = tab.sort_values(["mean", "candidate_index"], ascending=[False, True]).iloc[0]
    params = json.loads(chosen_row["params_json"])
    return params, tab


def fit_predict_outer(
    data: pd.DataFrame,
    fold_def: dict[str, Any],
    model_name: str,
    config: TissueConfig,
    genes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], bytes]:
    genes = list(genes or GENES)
    repeat, fold = int(fold_def["repeat"]), int(fold_def["fold"])
    params, candidates = tune_model(data, fold_def["train_patients"], model_name, config, repeat, fold, genes)
    train_mask = data["group_id"].isin(fold_def["train_patients"]).to_numpy()
    test_mask = data["group_id"].isin(fold_def["test_patients"]).to_numpy()
    use_genes = [params["gene"]] if model_name == "best_single_gene" else genes
    Xtr = data.loc[train_mask, use_genes].to_numpy(float)
    Xte = data.loc[test_mask, use_genes].to_numpy(float)
    ytr = data.loc[train_mask, "y"].to_numpy(int)
    model, needs_scale = _candidate_model(model_name, params, config.seed + repeat * 1000 + fold)
    scaler = None
    if needs_scale:
        scaler = _scale_fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
    model.fit(Xtr, ytr)
    score, prob = _as_pred(model, Xte, "svm" if model_name == "rbf_svm" else model_name)
    threshold_q95 = _inner_oof_normal_q95(data, fold_def["train_patients"], model_name, params, config, repeat, fold, use_genes)
    pred_q95 = (score > threshold_q95).astype(int)
    pred_05 = (prob >= 0.5).astype(int) if prob is not None else np.full(len(score), -1, dtype=int)
    out = data.loc[test_mask, ["patient_id", "sample_id", "tissue", "y"]].copy()
    out = out.rename(columns={"patient_id": "study_id"})
    out.insert(0, "fold", fold)
    out.insert(0, "repeat", repeat)
    out.insert(0, "model", model_name)
    out["score"] = score
    out["probability"] = prob if prob is not None else np.nan
    out["pred_0_5"] = pred_05
    out["normal_q95_threshold"] = threshold_q95
    out["pred_normal_q95"] = pred_q95
    out["selected_params_json"] = json.dumps(params, sort_keys=True)
    state = {
        "model": model_name,
        "repeat": repeat,
        "fold": fold,
        "genes": use_genes,
        "params": params,
        "train_study_ids": list(fold_def["train_patients"]),
        "test_study_ids": list(fold_def["test_patients"]),
        "scaler_mean": None if scaler is None else scaler.mean_.tolist(),
        "scaler_scale": None if scaler is None else scaler.scale_.tolist(),
        "normal_q95_threshold": float(threshold_q95),
    }
    blob = pickle.dumps({"model": model, "scaler": scaler, "state": state}, protocol=4)
    candidates = candidates.copy()
    candidates.insert(0, "fold", fold)
    candidates.insert(0, "repeat", repeat)
    return out, candidates, state, blob


def _cached_fit_job(
    data: pd.DataFrame,
    fold_def: dict[str, Any],
    model_name: str,
    config: TissueConfig,
    genes: list[str],
    cache_dir: Path | None,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], bytes]:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{model_name}_r{fold_def['repeat']:02d}_f{fold_def['fold']:02d}.job.pkl.gz"
        if cache_path.exists() and not force:
            with gzip.GzipFile(fileobj=cache_path.open("rb"), mode="rb") as gz:
                return pickle.loads(gz.read())
    result = fit_predict_outer(data, fold_def, model_name, config, genes)
    if cache_path is not None:
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with gzip.GzipFile(filename="", mode="wb", fileobj=tmp.open("wb"), mtime=0) as gz:
            gz.write(pickle.dumps(result, protocol=4))
        tmp.replace(cache_path)
    return result


def _inner_oof_normal_q95(
    data: pd.DataFrame,
    train_patients: Iterable[str],
    model_name: str,
    params: dict[str, Any],
    config: TissueConfig,
    repeat: int,
    fold: int,
    use_genes: list[str],
) -> float:
    rows: list[pd.DataFrame] = []
    train_patients = np.asarray(list(map(str, train_patients)))
    for inner_idx, (inner_train, inner_valid) in enumerate(inner_folds(train_patients, config.inner_folds, config.seed + 404, repeat, fold)):
        tr = data["group_id"].isin(inner_train).to_numpy()
        va = data["group_id"].isin(inner_valid).to_numpy()
        Xtr = data.loc[tr, use_genes].to_numpy(float)
        Xva = data.loc[va, use_genes].to_numpy(float)
        ytr = data.loc[tr, "y"].to_numpy(int)
        model, needs_scale = _candidate_model(model_name, params, config.seed + repeat * 1000 + fold * 100 + inner_idx + 50)
        if needs_scale:
            scaler = _scale_fit(Xtr)
            Xtr = scaler.transform(Xtr)
            Xva = scaler.transform(Xva)
        model.fit(Xtr, ytr)
        score, _ = _as_pred(model, Xva, "svm" if model_name == "rbf_svm" else model_name)
        rows.append(pd.DataFrame({"y": data.loc[va, "y"].to_numpy(int), "score": score}))
    oof = pd.concat(rows, ignore_index=True)
    normal_scores = oof.loc[oof["y"].eq(0), "score"].to_numpy(float)
    return float(np.quantile(normal_scores, 0.95, method="higher"))


def _metric_block(df: pd.DataFrame, score_col: str = "score", prob_col: str = "probability") -> dict[str, float]:
    y = df["y"].to_numpy(int)
    score = df[score_col].to_numpy(float)
    id_col = "study_id" if "study_id" in df.columns else "patient_id"
    out: dict[str, float] = {"n": float(len(df)), "n_patients": float(df[id_col].nunique())}
    try:
        out["auc"] = float(roc_auc_score(y, score))
    except ValueError:
        out["auc"] = float("nan")
    prob = df[prob_col].to_numpy(float) if prob_col in df else np.full(len(df), np.nan)
    if np.isfinite(prob).all():
        out["brier"] = float(brier_score_loss(y, prob))
        p = np.clip(prob, 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        z0 = z[y == 0]
        z1 = z[y == 1]
        if np.allclose(z, z[0]):
            out["calibration_intercept"] = float("nan")
            out["calibration_slope"] = float("nan")
            out["calibration_status"] = "nonestimable_constant_score"
        elif len(z0) and len(z1) and np.max(z0) <= np.min(z1):
            out["calibration_intercept"] = float("nan")
            out["calibration_slope"] = float("nan")
            out["calibration_status"] = "nonestimable_complete_separation_increasing"
        elif len(z0) and len(z1) and np.max(z1) <= np.min(z0):
            out["calibration_intercept"] = float("nan")
            out["calibration_slope"] = float("nan")
            out["calibration_status"] = "nonestimable_complete_separation_decreasing"
        else:
            try:
                lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
                lr.fit(z.reshape(-1, 1), y)
                out["calibration_intercept"] = float(lr.intercept_[0])
                out["calibration_slope"] = float(lr.coef_[0, 0])
                out["calibration_status"] = "estimable"
            except Exception:
                out["calibration_intercept"] = float("nan")
                out["calibration_slope"] = float("nan")
                out["calibration_status"] = "nonestimable_optimizer_failure"
    else:
        out["brier"] = float("nan")
        out["calibration_intercept"] = float("nan")
        out["calibration_slope"] = float("nan")
        out["calibration_status"] = "not_applicable_no_probability"
    for label, col in [("threshold_0_5", "pred_0_5"), ("normal_q95", "pred_normal_q95")]:
        if col not in df or (df[col] < 0).any():
            out[f"{label}_sensitivity"] = float("nan")
            out[f"{label}_specificity"] = float("nan")
            out[f"{label}_balanced_accuracy"] = float("nan")
            continue
        cm = confusion_matrix(y, df[col].to_numpy(int), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out[f"{label}_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        out[f"{label}_specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        out[f"{label}_balanced_accuracy"] = float(balanced_accuracy_score(y, df[col].to_numpy(int)))
    return out


def summarize_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, sub in oof.groupby("model", sort=False):
        rep_metrics = []
        for repeat, rep in sub.groupby("repeat", sort=True):
            met = _metric_block(rep)
            met.update({"model": model, "scope": "repeat_pooled", "repeat": int(repeat)})
            rows.append(met)
            rep_metrics.append(met)
        rep_tab = pd.DataFrame(rep_metrics)
        numeric_cols = [c for c in rep_tab.select_dtypes(include=[np.number]).columns if c != "repeat"]
        pooled = {c: float(rep_tab[c].mean()) for c in numeric_cols}
        statuses = sorted(set(str(x) for x in rep_tab["calibration_status"].dropna()))
        pooled["calibration_status"] = statuses[0] if len(statuses) == 1 else "mixed:" + ",".join(statuses)
        pooled.update({"model": model, "scope": "all_repeats_pooled", "aggregation": "mean_of_repeat_metrics"})
        rows.append(pooled)
    return pd.DataFrame(rows)


def patient_bootstrap_metrics(oof: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 707)
    id_col = "study_id" if "study_id" in oof.columns else "patient_id"
    patients = np.asarray(sorted(oof[id_col].unique()))
    rows: list[dict[str, Any]] = []
    model_names = list(oof["model"].drop_duplicates())
    metric_names = [
        "auc",
        "brier",
        "threshold_0_5_sensitivity",
        "threshold_0_5_specificity",
        "threshold_0_5_balanced_accuracy",
        "normal_q95_sensitivity",
        "normal_q95_specificity",
        "normal_q95_balanced_accuracy",
    ]

    def _fast_auc(y: np.ndarray, score: np.ndarray) -> float:
        y = y.astype(int)
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(score, kind="mergesort")
        sorted_score = score[order]
        ranks = np.empty(len(score), dtype=float)
        i = 0
        while i < len(score):
            j = i + 1
            while j < len(score) and sorted_score[j] == sorted_score[i]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
            i = j
        rank_sum_pos = float(ranks[y == 1].sum())
        return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def _metrics_from_arrays(y: np.ndarray, score: np.ndarray, prob: np.ndarray, pred05: np.ndarray, predq95: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        out["auc"] = _fast_auc(y, score)
        out["brier"] = float(np.mean((prob - y) ** 2)) if np.isfinite(prob).all() else float("nan")
        for label, pred in [("threshold_0_5", pred05), ("normal_q95", predq95)]:
            if (pred < 0).any():
                out[f"{label}_sensitivity"] = float("nan")
                out[f"{label}_specificity"] = float("nan")
                out[f"{label}_balanced_accuracy"] = float("nan")
                continue
            cm = confusion_matrix(y, pred.astype(int), labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            out[f"{label}_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
            out[f"{label}_specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
            out[f"{label}_balanced_accuracy"] = float(0.5 * (out[f"{label}_sensitivity"] + out[f"{label}_specificity"]))
        return out

    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    for model in model_names:
        indexed[model] = {}
        sub = oof[oof["model"].eq(model)]
        for repeat, rep in sub.groupby("repeat", sort=True):
            rep = rep.reset_index(drop=True)
            pid_to_idx = {pid: rep.index[rep[id_col].eq(pid)].to_numpy() for pid in patients}
            indexed[model][int(repeat)] = {
                "pid_to_idx": pid_to_idx,
                "y": rep["y"].to_numpy(int),
                "score": rep["score"].to_numpy(float),
                "prob": rep["probability"].to_numpy(float),
                "pred05": rep["pred_0_5"].to_numpy(int),
                "predq95": rep["pred_normal_q95"].to_numpy(int),
            }
    for b in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        for model in model_names:
            rep_metrics = []
            for repeat in sorted(indexed[model]):
                repi = indexed[model][repeat]
                idx = np.concatenate([repi["pid_to_idx"][pid] for pid in sampled])
                rep_metrics.append(
                    _metrics_from_arrays(
                        repi["y"][idx],
                        repi["score"][idx],
                        repi["prob"][idx],
                        repi["pred05"][idx],
                        repi["predq95"][idx],
                    )
                )
            rep_tab = pd.DataFrame(rep_metrics)
            met = {c: float(rep_tab[c].mean()) for c in rep_tab.columns}
            met.update({"bootstrap": b, "model": model})
            rows.append(met)
    draws = pd.DataFrame(rows)
    ci_rows: list[dict[str, Any]] = []
    point = summarize_metrics(oof).query("scope == 'all_repeats_pooled'").set_index("model")
    for model, sub in draws.groupby("model", sort=False):
        for metric in metric_names:
            vals = sub[metric].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                low = high = float("nan")
            else:
                low, high = np.quantile(vals, [0.025, 0.975])
            ci_rows.append({"model": model, "metric": metric, "point": float(point.loc[model, metric]), "ci_low": float(low), "ci_high": float(high), "bootstrap_replicates": n_boot})
    if "best_single_gene" not in model_names:
        return draws, pd.DataFrame(ci_rows)
    for model in model_names:
        if model == "best_single_gene":
            continue
        d = draws[draws["model"].eq(model)].set_index("bootstrap")
        s = draws[draws["model"].eq("best_single_gene")].set_index("bootstrap")
        common = d.index.intersection(s.index)
        for metric in ["auc", "brier", "normal_q95_balanced_accuracy", "threshold_0_5_balanced_accuracy"]:
            vals = d.loc[common, metric].to_numpy(float) - s.loc[common, metric].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                low, high = np.quantile(vals, [0.025, 0.975])
                pdelta = float(point.loc[model, metric] - point.loc["best_single_gene", metric])
            else:
                low = high = pdelta = float("nan")
            ci_rows.append({"model": f"{model}_minus_best_single_gene", "metric": metric, "point": pdelta, "ci_low": float(low), "ci_high": float(high), "bootstrap_replicates": n_boot})
    return draws, pd.DataFrame(ci_rows)


def run_nested_cv(
    data: pd.DataFrame,
    outdir: Path,
    config: TissueConfig,
    models: list[str],
    repeats: int | None = None,
    genes: list[str] | None = None,
    write_outputs: bool = True,
    cache_key: str | None = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    data = prepare_tissue_frame(data)
    genes = list(genes or GENES)
    repeats = int(repeats if repeats is not None else config.outer_repeats)
    folds = patient_kfolds(data["group_id"].unique(), config.outer_folds, repeats, config.seed)
    jobs = [(fold_def, model_name) for fold_def in folds for model_name in models]
    cache_dir = outdir / ".cache" / cache_key if write_outputs and cache_key else None
    model_dir = outdir / "models"
    if write_outputs:
        model_dir.mkdir(parents=True, exist_ok=True)
    results = Parallel(n_jobs=config.max_workers, prefer="processes", batch_size=1, verbose=10)(
        delayed(_cached_fit_job)(data, fold_def, model_name, config, genes, cache_dir, force) for fold_def, model_name in jobs
    )
    oof_rows: list[pd.DataFrame] = []
    candidate_rows: list[pd.DataFrame] = []
    state_rows: list[dict[str, Any]] = []
    for (fold_def, model_name), (oof, cand, state, blob) in zip(jobs, results):
        oof_rows.append(oof)
        candidate_rows.append(cand)
        state_rows.append(state)
        if write_outputs:
            pkl = model_dir / f"{model_name}_r{fold_def['repeat']:02d}_f{fold_def['fold']:02d}.pkl.gz"
            with gzip.GzipFile(filename="", mode="wb", fileobj=pkl.open("wb"), mtime=0) as gz:
                gz.write(blob)
    oof_tab = pd.concat(oof_rows, ignore_index=True)
    cand_tab = pd.concat(candidate_rows, ignore_index=True)
    state_tab = pd.DataFrame(state_rows)
    fold_tab = pd.DataFrame(
        [
            {
                "repeat": f["repeat"],
                "fold": f["fold"],
                "train_study_ids": json.dumps(f["train_patients"]),
                "test_study_ids": json.dumps(f["test_patients"]),
                "n_train_study_ids": len(f["train_patients"]),
                "n_test_study_ids": len(f["test_patients"]),
            }
            for f in folds
        ]
    )
    metrics = summarize_metrics(oof_tab)
    if write_outputs:
        write_csv(oof_tab, outdir / "oof_predictions.tsv.gz", sep="\t")
        write_csv(cand_tab, outdir / "candidate_results.tsv.gz", sep="\t")
        write_csv(state_tab, outdir / "model_states.tsv.gz", sep="\t")
        write_csv(fold_tab, outdir / "folds.tsv", sep="\t")
        write_csv(metrics, outdir / "metrics.tsv", sep="\t")
    return {"oof": oof_tab, "candidate_results": cand_tab, "model_states": state_tab, "folds": fold_tab, "metrics": metrics}


def optimism_bootstrap_ridge(data: pd.DataFrame, config: TissueConfig, n_boot: int, checkpoint_dir: Path | None = None) -> pd.DataFrame:
    data = prepare_tissue_frame(data)
    patients = np.asarray(sorted(data["patient_id"].unique()))
    rows: list[dict[str, Any]] = []
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for b in range(n_boot):
        ck = checkpoint_dir / f"optimism_{b:04d}.json" if checkpoint_dir is not None else None
        if ck is not None and ck.exists():
            rows.append(json.loads(ck.read_text()))
            continue
        rng = np.random.default_rng(config.seed + 909 + b)
        sampled = rng.choice(patients, size=len(patients), replace=True)
        pieces = []
        for dup, pid in enumerate(sampled):
            block = data[data["patient_id"].eq(pid)].copy()
            block["patient_id"] = block["patient_id"] + f"__dup{dup}"
            block["group_id"] = pid
            pieces.append(block)
        boot_data = pd.concat(pieces, ignore_index=True)
        params, _ = tune_model(boot_data, boot_data["group_id"].unique(), "ridge", config, b, 0)
        model, needs_scale = _candidate_model("ridge", params, config.seed + b)
        Xb = boot_data[GENES].to_numpy(float)
        yb = boot_data["y"].to_numpy(int)
        Xorig = data[GENES].to_numpy(float)
        if needs_scale:
            scaler = _scale_fit(Xb)
            Xb = scaler.transform(Xb)
            Xorig = scaler.transform(Xorig)
        model.fit(Xb, yb)
        score_b, prob_b = _as_pred(model, Xb, "ridge")
        score_o, prob_o = _as_pred(model, Xorig, "ridge")
        row = {
            "bootstrap": b,
            "apparent_auc": float(roc_auc_score(yb, score_b)),
            "test_original_auc": float(roc_auc_score(data["y"], score_o)),
            "optimism_auc": float(roc_auc_score(yb, score_b) - roc_auc_score(data["y"], score_o)),
            "apparent_brier": float(brier_score_loss(yb, prob_b)),
            "test_original_brier": float(brier_score_loss(data["y"], prob_o)),
            "optimism_brier": float(brier_score_loss(yb, prob_b) - brier_score_loss(data["y"], prob_o)),
        }
        if ck is not None:
            tmp = ck.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(row, sort_keys=True))
            tmp.replace(ck)
        rows.append(row)
    return pd.DataFrame(rows)


def permute_within_patients(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    perm = data.copy()
    for pid in perm["patient_id"].unique():
        idx = perm.index[perm["patient_id"].eq(pid)].to_numpy()
        if rng.random() < 0.5:
            perm.loc[idx, "y"] = perm.loc[idx, "y"].to_numpy()[::-1]
    return perm


def permutation_primary_ridge(data: pd.DataFrame, config: TissueConfig, n_perm: int, repeats: int, checkpoint_dir: Path | None = None) -> tuple[pd.DataFrame, float]:
    data = prepare_tissue_frame(data)
    short_cfg = TissueConfig(
        outer_folds=config.outer_folds,
        outer_repeats=repeats,
        inner_folds=config.inner_folds,
        logistic_C=config.logistic_C,
        seed=config.seed,
        bootstrap_replicates=config.bootstrap_replicates,
        optimism_bootstrap=config.optimism_bootstrap,
        permutations=config.permutations,
        permutation_repeats=config.permutation_repeats,
    )
    obs = run_nested_cv(data, Path("/tmp/tissue_perm_observed_no_write"), short_cfg, ["ridge"], repeats=repeats, write_outputs=False)["metrics"]
    obs_auc = float(obs.query("model == 'ridge' and scope == 'all_repeats_pooled'")["auc"].iloc[0])
    rows = [{"permutation": -1, "auc": obs_auc, "kind": "observed"}]
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "observed.json").write_text(json.dumps(rows[0], sort_keys=True))

    def _one_perm(p: int) -> dict[str, Any]:
        ck = checkpoint_dir / f"perm_{p:04d}.json" if checkpoint_dir is not None else None
        if ck is not None and ck.exists():
            return json.loads(ck.read_text())
        rng = np.random.default_rng(config.seed + 1201 + p)
        perm = permute_within_patients(data, rng)
        cfg = TissueConfig(
            outer_folds=config.outer_folds,
            outer_repeats=repeats,
            inner_folds=config.inner_folds,
            logistic_C=config.logistic_C,
            seed=config.seed,
            max_workers=1,
        )
        met = run_nested_cv(perm, Path(f"/tmp/tissue_perm_{p:04d}_no_write"), cfg, ["ridge"], repeats=repeats, write_outputs=False)["metrics"]
        auc = float(met.query("model == 'ridge' and scope == 'all_repeats_pooled'")["auc"].iloc[0])
        row = {"permutation": p, "auc": auc, "kind": "null"}
        if ck is not None:
            tmp = ck.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(row, sort_keys=True))
            tmp.replace(ck)
        return row

    rows.extend(Parallel(n_jobs=config.max_workers, prefer="processes", batch_size=1, verbose=10)(delayed(_one_perm)(p) for p in range(n_perm)))
    tab = pd.DataFrame(rows)
    pvalue = float((1 + np.sum(tab.loc[tab["kind"].eq("null"), "auc"].to_numpy() >= obs_auc)) / (n_perm + 1))
    return tab, pvalue


def run_cohort(
    dataframe: pd.DataFrame,
    outdir: str | Path,
    models: list[str] | None = None,
    repeats: int = 20,
    config: TissueConfig | None = None,
    bootstrap: int = 0,
    optimism: int = 0,
    permutations: int = 0,
    genes: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run grouped nested tissue classification for local or public tumor-normal cohorts."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = config or TissueConfig()
    models = models or ["ridge"]
    data = prepare_tissue_frame(dataframe)
    data_sha256 = dataframe_hash(data, genes or GENES)
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run_payload = {
        "models": models,
        "repeats": repeats,
        "genes": genes or GENES,
        "config": asdict(config),
        "data_sha256": data_sha256,
        "source_sha256": source_sha256,
    }
    cache_key = hashlib.sha256(json.dumps(run_payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
    result = run_nested_cv(data, outdir, config, models, repeats=repeats, genes=genes, cache_key=cache_key)
    if bootstrap:
        draws, ci = patient_bootstrap_metrics(result["oof"], bootstrap, config.seed)
        write_csv(draws, outdir / "bootstrap_draws.tsv.gz", sep="\t")
        write_csv(ci, outdir / "bootstrap_ci.tsv", sep="\t")
        result["bootstrap_draws"] = draws
        result["bootstrap_ci"] = ci
    if optimism:
        opt = optimism_bootstrap_ridge(data, config, optimism, outdir / ".cache" / "ridge_optimism" / cache_key)
        write_csv(opt, outdir / "ridge_optimism_bootstrap.tsv", sep="\t")
        result["ridge_optimism"] = opt
    if permutations:
        perm, pvalue = permutation_primary_ridge(data, config, permutations, config.permutation_repeats, outdir / ".cache" / "ridge_permutation" / cache_key)
        write_csv(perm, outdir / "ridge_permutation.tsv", sep="\t")
        (outdir / "ridge_permutation_summary.json").write_text(json.dumps({"observed_auc": float(perm.loc[perm["kind"].eq("observed"), "auc"].iloc[0]), "p_value": pvalue, "permutations": permutations, "repeats": config.permutation_repeats}, indent=2))
        result["ridge_permutation"] = perm
    current_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current_source_sha256 != source_sha256:
        raise RuntimeError(
            "ml/tissue.py changed during run; refusing to write final manifest "
            f"(start={source_sha256}, current={current_source_sha256})"
        )
    manifest = {
        "rows": int(len(data)),
        "patients": int(data["patient_id"].nunique()),
        "tumor_samples": int(data["y"].sum()),
        "normal_samples": int((data["y"] == 0).sum()),
        "genes": genes or GENES,
        "models": models,
        "outer_folds": config.outer_folds,
        "outer_repeats": repeats,
        "inner_folds": config.inner_folds,
        "bootstrap": bootstrap,
        "optimism": optimism,
        "permutations": permutations,
        "seed": config.seed,
        "data_sha256": data_sha256,
        "source_sha256": source_sha256,
        "cache_key": cache_key,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return result
