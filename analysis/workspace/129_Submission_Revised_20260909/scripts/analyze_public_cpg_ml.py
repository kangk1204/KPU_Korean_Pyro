#!/usr/bin/env python3
"""Public tissue ML rerun using the frozen individual CpG features."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE_119 = ROOT.parent / "119_Figure1_ABC_Revision_20260908"
SOURCE_114 = ROOT.parent / "114_ML_DataDriven_20260905"
DEFAULT_OUTDIR = ROOT / "results" / "public_ml"
DEFAULT_CONTRACT = ROOT / "registry" / "public_cpg_ml_contract.json"
DEFAULT_FIXED_PROBES = ROOT / "registry" / "fixed_probes.json"
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
SEED = 20260908


@dataclass(frozen=True)
class MLConfig:
    outer_folds: int = 5
    outer_repeats: int = 20
    inner_folds: int = 4
    logistic_C: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    bootstrap_replicates: int = 2000
    seed: int = SEED
    max_workers: int = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_sha256(df: pd.DataFrame) -> str:
    payload = df.copy()
    payload = payload.reindex(sorted(payload.columns), axis=1)
    return hashlib.sha256(payload.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as gz:
            df.to_csv(gz, sep="\t", index=False)
    else:
        df.to_csv(path, sep="\t", index=False)


def read_fixed_probes(path: Path = DEFAULT_FIXED_PROBES) -> pd.DataFrame:
    fixed = json.loads(path.read_text())
    rows = []
    for gene in GENES:
        for order, cpg in enumerate(fixed[gene], start=1):
            rows.append({"gene": gene, "cpg": cpg, "feature": f"cpg__{cpg}", "gene_cpg_order": order})
    out = pd.DataFrame(rows)
    if len(out) != 77 or out["cpg"].duplicated().any():
        raise ValueError("fixed registry must contain 77 unique CpGs")
    return out


def _complete_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.groupby(["study_id", "tissue"]).size().unstack(fill_value=0)
    keep = counts.index[(counts.get("N", 0) == 1) & (counts.get("T", 0) == 1)]
    out = frame.loc[frame["study_id"].isin(keep)].copy()
    if len(out) != 2 * len(keep):
        raise ValueError("paired frame contains duplicate or incomplete patient rows")
    return out.sort_values(["study_id", "y", "sample_id"]).reset_index(drop=True)


def load_colonomics(fixed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    beta_path = SOURCE_119 / "data" / "derived" / "Colonomics_beta.tsv.gz"
    meta_path = SOURCE_119 / "data" / "derived" / "Colonomics_samples.tsv"
    beta = pd.read_csv(beta_path, sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce")
    meta = pd.read_csv(meta_path, sep="\t")
    meta = meta.loc[meta["tissue"].isin(["N", "T"]) & meta["pair_verified"].astype(bool)].copy()
    values = beta.reindex(fixed["cpg"]).T
    values.columns = fixed["feature"]
    values.insert(0, "sample", values.index.astype(str))
    merged = meta.merge(values, on="sample", validate="one_to_one")
    merged["study_id"] = "colonomics_" + merged["patient"].astype(str)
    merged["sample_id"] = "colonomics_" + merged["patient"].astype(str) + "_" + merged["tissue"].astype(str)
    merged["y"] = merged["tissue"].map({"N": 0, "T": 1}).astype(int)
    frame = _complete_pairs(merged[["study_id", "sample_id", "sample", "patient", "tissue", "y", *fixed["feature"]]])
    qc = feature_qc(frame, fixed, "colonomics", mask_source="curated_119_beta")
    return frame, qc, {"beta": str(beta_path), "samples": str(meta_path)}


def _gse_sample_code(description: str) -> tuple[str, str]:
    match = re.fullmatch(r"(\d+)([NT])", str(description))
    if not match:
        raise ValueError(f"cannot parse GSE119526 sample description: {description}")
    return match.group(1), match.group(2)


def load_gse119526(fixed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    signal_path = SOURCE_114 / "data" / "public" / "processed" / "gse119526_table1_signal_intensities.tsv.gz"
    meta_path = SOURCE_114 / "data" / "public" / "processed" / "gse119526_sample_metadata.tsv"
    signals = pd.read_csv(signal_path, sep="\t", index_col=0)
    meta = pd.read_csv(meta_path, sep="\t")
    parsed = meta["description"].map(_gse_sample_code)
    meta["patient_code"] = [x[0] for x in parsed]
    meta["tissue"] = [x[1] for x in parsed]
    meta["study_id"] = "gse119526_" + meta["patient_code"].astype(str)
    meta["sample_id"] = meta["study_id"] + "_" + meta["tissue"].astype(str)
    meta["sample_code"] = meta["description"].astype(str)
    meta["y"] = meta["tissue"].map({"N": 0, "T": 1}).astype(int)
    rows = []
    for rec in meta.to_dict("records"):
        row = {k: rec[k] for k in ["study_id", "sample_id", "geo_accession", "sample_code", "patient_code", "tissue", "y"]}
        for cpg, feature in zip(fixed["cpg"], fixed["feature"]):
            m_col = f"{rec['sample_code']}_methylated_signal"
            u_col = f"{rec['sample_code']}_unmethylated_signal"
            p_col = f"{rec['sample_code']}_detection_pval"
            if cpg not in signals.index or m_col not in signals or u_col not in signals or p_col not in signals:
                row[feature] = np.nan
                continue
            det_p = float(signals.loc[cpg, p_col])
            if not np.isfinite(det_p) or det_p >= 0.01:
                row[feature] = np.nan
                continue
            m = float(signals.loc[cpg, m_col])
            u = float(signals.loc[cpg, u_col])
            row[feature] = m / (m + u + 100.0)
        rows.append(row)
    frame = _complete_pairs(pd.DataFrame(rows))
    qc = feature_qc(frame, fixed, "gse119526", mask_source="M/(M+U+100)_detection_p_lt_0.01")
    return frame, qc, {"signals": str(signal_path), "samples": str(meta_path)}


def feature_qc(frame: pd.DataFrame, fixed: pd.DataFrame, cohort: str, mask_source: str) -> pd.DataFrame:
    rows = []
    for rec in fixed.to_dict("records"):
        feature = rec["feature"]
        vals = pd.to_numeric(frame[feature], errors="coerce")
        rows.append(
            {
                "cohort": cohort,
                "gene": rec["gene"],
                "cpg": rec["cpg"],
                "feature": feature,
                "mask_source": mask_source,
                "n_rows": int(len(frame)),
                "n_nonmissing": int(vals.notna().sum()),
                "all_missing": bool(vals.notna().sum() == 0),
            }
        )
    return pd.DataFrame(rows)


def validate_frame(frame: pd.DataFrame, expected_patients: int) -> None:
    if frame["study_id"].nunique() != expected_patients or len(frame) != expected_patients * 2:
        raise ValueError(f"expected {expected_patients} paired patients")
    pair = frame.groupby("study_id")["y"].agg(["size", "sum"])
    if not ((pair["size"] == 2) & (pair["sum"] == 1)).all():
        raise ValueError("each patient must contribute one normal and one tumor")


def patient_kfolds(patient_ids: Iterable[str], n_splits: int, repeats: int, seed: int) -> list[dict[str, object]]:
    patient_ids = np.asarray(sorted(map(str, patient_ids)))
    folds = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + 1009 * repeat)
        perm = patient_ids.copy()
        rng.shuffle(perm)
        for fold, test_patients in enumerate(np.array_split(perm, n_splits)):
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


def inner_kfolds(train_patients: Iterable[str], n_splits: int, seed: int, repeat: int, fold: int) -> list[tuple[np.ndarray, np.ndarray]]:
    train_patients = np.asarray(sorted(map(str, train_patients)))
    rng = np.random.default_rng(seed + 7919 * (repeat + 1) + 1543 * (fold + 1))
    perm = train_patients.copy()
    rng.shuffle(perm)
    return [(np.setdiff1d(train_patients, chunk, assume_unique=True), chunk) for chunk in np.array_split(perm, n_splits)]


def fit_preprocessor(train: pd.DataFrame, features: list[str]) -> tuple[list[str], np.ndarray, StandardScaler]:
    med = train[features].median(axis=0, skipna=True)
    usable = med.index[med.notna()].tolist()
    if not usable:
        raise ValueError("no usable CpG features in training fold")
    median = med.loc[usable].to_numpy(float)
    x = train[usable].to_numpy(float)
    x = np.where(np.isnan(x), median, x)
    scaler = StandardScaler().fit(x)
    return usable, median, scaler


def transform_with_preprocessor(frame: pd.DataFrame, usable: list[str], median: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    x = frame[usable].to_numpy(float)
    x = np.where(np.isnan(x), median, x)
    return scaler.transform(x)


def fit_predict(data: pd.DataFrame, features: list[str], fold_def: dict[str, object], config: MLConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], bytes]:
    train_ids = list(fold_def["train_patients"])
    test_ids = list(fold_def["test_patients"])
    train = data.loc[data["study_id"].isin(train_ids)].copy()
    test = data.loc[data["study_id"].isin(test_ids)].copy()
    repeat = int(fold_def["repeat"])
    fold = int(fold_def["fold"])
    candidates = []
    for c in config.logistic_C:
        losses = []
        for inner_idx, (tr_ids, va_ids) in enumerate(inner_kfolds(train_ids, config.inner_folds, config.seed, repeat, fold)):
            tr = data.loc[data["study_id"].isin(tr_ids)]
            va = data.loc[data["study_id"].isin(va_ids)]
            usable, median, scaler = fit_preprocessor(tr, features)
            x_tr = transform_with_preprocessor(tr, usable, median, scaler)
            x_va = transform_with_preprocessor(va, usable, median, scaler)
            model = LogisticRegression(
                penalty="l2",
                C=float(c),
                solver="liblinear",
                class_weight=None,
                max_iter=5000,
                random_state=config.seed + repeat * 1000 + fold * 100 + inner_idx,
            )
            model.fit(x_tr, tr["y"].to_numpy(int))
            prob = model.predict_proba(x_va)[:, 1]
            losses.append(float(log_loss(va["y"].to_numpy(int), np.clip(prob, 1e-8, 1 - 1e-8), labels=[0, 1])))
        finite = np.asarray([x for x in losses if np.isfinite(x)], float)
        candidates.append(
            {
                "repeat": repeat,
                "fold": fold,
                "model": "ridge_all_cpg",
                "C": float(c),
                "inner_mean_log_loss": float(finite.mean()),
                "inner_se_log_loss": float(finite.std(ddof=1) / math.sqrt(len(finite))) if len(finite) > 1 else 0.0,
            }
        )
    cand = pd.DataFrame(candidates)
    best_idx = int(cand["inner_mean_log_loss"].idxmin())
    cutoff = float(cand.loc[best_idx, "inner_mean_log_loss"] + cand.loc[best_idx, "inner_se_log_loss"])
    chosen_c = float(cand.loc[cand["inner_mean_log_loss"].le(cutoff)].sort_values(["C", "inner_mean_log_loss"]).iloc[0]["C"])
    usable, median, scaler = fit_preprocessor(train, features)
    x_train = transform_with_preprocessor(train, usable, median, scaler)
    x_test = transform_with_preprocessor(test, usable, median, scaler)
    model = LogisticRegression(
        penalty="l2",
        C=chosen_c,
        solver="liblinear",
        max_iter=5000,
        random_state=config.seed + repeat * 1000 + fold,
    )
    model.fit(x_train, train["y"].to_numpy(int))
    probability = model.predict_proba(x_test)[:, 1]
    pred = (probability >= 0.5).astype(int)
    oof = test[["study_id", "sample_id", "tissue", "y"]].copy()
    oof.insert(0, "fold", fold)
    oof.insert(0, "repeat", repeat)
    oof.insert(0, "model", "ridge_all_cpg")
    oof["probability"] = probability
    oof["score"] = model.decision_function(x_test)
    oof["pred_0_5"] = pred
    oof["selected_C"] = chosen_c
    oof["n_features_fit"] = len(usable)
    state = {
        "model": "ridge_all_cpg",
        "repeat": repeat,
        "fold": fold,
        "selected_C": chosen_c,
        "train_study_ids": train_ids,
        "test_study_ids": test_ids,
        "features_requested": len(features),
        "features_fit": usable,
        "n_all_missing_training_features": int(len(features) - len(usable)),
        "median_by_feature": {f: float(v) for f, v in zip(usable, median)},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    blob = pickle.dumps({"model": model, "scaler": scaler, "state": state}, protocol=4)
    return oof, cand, state, blob


def metric_block(df: pd.DataFrame) -> dict[str, float]:
    y = df["y"].to_numpy(int)
    prob = df["probability"].to_numpy(float)
    pred = df["pred_0_5"].to_numpy(int)
    out: dict[str, float] = {"n": float(len(df)), "n_patients": float(df["study_id"].nunique())}
    out["auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")
    out["brier"] = float(brier_score_loss(y, prob))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out["threshold_0_5_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    out["threshold_0_5_specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    out["threshold_0_5_balanced_accuracy"] = 0.5 * (
        out["threshold_0_5_sensitivity"] + out["threshold_0_5_specificity"]
    )
    return out


def summarize_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rep_metrics = []
    for repeat, rep in oof.groupby("repeat", sort=True):
        met = metric_block(rep)
        met.update({"model": "ridge_all_cpg", "scope": "repeat_pooled", "repeat": int(repeat)})
        rows.append(met)
        rep_metrics.append(met)
    rep_tab = pd.DataFrame(rep_metrics)
    numeric = [c for c in rep_tab.select_dtypes(include=[np.number]).columns if c != "repeat"]
    pooled = {c: float(rep_tab[c].mean()) for c in numeric}
    pooled.update({"model": "ridge_all_cpg", "scope": "all_repeats_pooled", "aggregation": "mean_of_repeat_metrics"})
    rows.append(pooled)
    return pd.DataFrame(rows)


def patient_bootstrap(oof: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 707)
    patients = np.asarray(sorted(oof["study_id"].unique()))
    indexed = {}
    for repeat, rep in oof.groupby("repeat", sort=True):
        rep = rep.reset_index(drop=True)
        indexed[int(repeat)] = {
            "patient_to_index": {pid: rep.index[rep["study_id"].eq(pid)].to_numpy() for pid in patients},
            "frame": rep,
        }
    rows = []
    for b in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        rep_rows = []
        for repeat in sorted(indexed):
            info = indexed[repeat]
            idx = np.concatenate([info["patient_to_index"][pid] for pid in sampled])
            rep_rows.append(metric_block(info["frame"].iloc[idx]))
        met = {c: float(pd.DataFrame(rep_rows)[c].mean()) for c in rep_rows[0] if c not in {"n", "n_patients"}}
        met.update({"bootstrap": b, "model": "ridge_all_cpg"})
        rows.append(met)
    draws = pd.DataFrame(rows)
    point = summarize_metrics(oof).query("scope == 'all_repeats_pooled'").iloc[0]
    ci_rows = []
    for metric in ["auc", "brier", "threshold_0_5_sensitivity", "threshold_0_5_specificity", "threshold_0_5_balanced_accuracy"]:
        vals = draws[metric].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        lo, hi = np.quantile(vals, [0.025, 0.975])
        ci_rows.append(
            {
                "model": "ridge_all_cpg",
                "metric": metric,
                "point": float(point[metric]),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "bootstrap_replicates": int(n_boot),
                "interval_scope": "patient_cluster_bootstrap_conditional_on_fitted_cv_models",
            }
        )
    return draws, pd.DataFrame(ci_rows)


def run_cohort(
    name: str,
    frame: pd.DataFrame,
    qc: pd.DataFrame,
    outdir: Path,
    config: MLConfig,
    fixed: pd.DataFrame,
    expected_patients: int | None = None,
) -> dict[str, object]:
    expected = int(expected_patients if expected_patients is not None else {"colonomics": 92, "gse119526": 48}[name])
    validate_frame(frame, expected)
    features = fixed["feature"].tolist()
    folds = patient_kfolds(frame["study_id"].unique(), config.outer_folds, config.outer_repeats, config.seed)
    jobs = [(fold, config) for fold in folds]
    results = Parallel(n_jobs=config.max_workers, prefer="processes", batch_size=1)(
        delayed(fit_predict)(frame, features, fold, cfg) for fold, cfg in jobs
    )
    oof = pd.concat([x[0] for x in results], ignore_index=True)
    candidates = pd.concat([x[1] for x in results], ignore_index=True)
    states = pd.DataFrame([x[2] for x in results])
    fold_audit = pd.DataFrame(
        [
            {
                "repeat": int(f["repeat"]),
                "fold": int(f["fold"]),
                "train_study_ids": json.dumps(f["train_patients"]),
                "test_study_ids": json.dumps(f["test_patients"]),
                "n_train_study_ids": len(f["train_patients"]),
                "n_test_study_ids": len(f["test_patients"]),
                "patient_overlap": len(set(f["train_patients"]) & set(f["test_patients"])),
            }
            for f in folds
        ]
    )
    metrics = summarize_metrics(oof)
    draws, ci = patient_bootstrap(oof, config.bootstrap_replicates, config.seed)
    cohort_out = outdir / name
    model_dir = cohort_out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for fold, result in zip(folds, results):
        model_path = model_dir / f"ridge_all_cpg_r{int(fold['repeat']):02d}_f{int(fold['fold']):02d}.pkl.gz"
        with gzip.GzipFile(filename="", mode="wb", fileobj=model_path.open("wb"), mtime=0) as gz:
            gz.write(result[3])
    write_tsv(oof, cohort_out / "oof_predictions.tsv.gz")
    write_tsv(candidates, cohort_out / "candidate_results.tsv.gz")
    write_tsv(states, cohort_out / "model_states.tsv.gz")
    write_tsv(fold_audit, cohort_out / "fold_audit.tsv")
    write_tsv(metrics, cohort_out / "metrics.tsv")
    write_tsv(draws, cohort_out / "bootstrap_draws.tsv.gz")
    write_tsv(ci, cohort_out / "bootstrap_ci.tsv")
    write_tsv(qc, cohort_out / "feature_qc.tsv")
    manifest = {
        "status": "complete",
        "cohort": name,
        "patients": expected,
        "rows": int(len(frame)),
        "features_requested": int(len(features)),
        "features_nonmissing_any": int((~frame[features].isna().all(axis=0)).sum()),
        "model": "ridge_all_cpg",
        "outer_folds": config.outer_folds,
        "outer_repeats": config.outer_repeats,
        "inner_folds": config.inner_folds,
        "bootstrap_replicates": config.bootstrap_replicates,
        "logistic_C": list(config.logistic_C),
        "seed": config.seed,
        "data_sha256": dataframe_sha256(frame[["study_id", "sample_id", "tissue", "y", *features]]),
        "script_sha256": sha256(Path(__file__)),
        "fixed_probe_sha256": sha256(DEFAULT_FIXED_PROBES),
        "patient_fold_overlap_max": int(fold_audit["patient_overlap"].max()),
        "outputs": {},
    }
    for filename in [
        "oof_predictions.tsv.gz",
        "candidate_results.tsv.gz",
        "model_states.tsv.gz",
        "fold_audit.tsv",
        "metrics.tsv",
        "bootstrap_draws.tsv.gz",
        "bootstrap_ci.tsv",
        "feature_qc.tsv",
    ]:
        manifest["outputs"][filename] = sha256(cohort_out / filename)
    write_json(cohort_out / "run_manifest.json", manifest)
    return manifest


def build_inputs(fixed: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, str]]]:
    col_frame, col_qc, col_sources = load_colonomics(fixed)
    gse_frame, gse_qc, gse_sources = load_gse119526(fixed)
    return {"colonomics": col_frame, "gse119526": gse_frame}, {"colonomics": col_qc, "gse119526": gse_qc}, {"colonomics": col_sources, "gse119526": gse_sources}


def write_contract(path: Path, config: MLConfig, source_files: dict[str, dict[str, str]]) -> None:
    contract = {
        "status": "public_array_individual_cpg_ml_contract",
        "created_utc": "2026-09-08T00:00:00Z",
        "analysis_unit": "patient-grouped tumor/normal tissue classification",
        "feature_unit": "individual fixed CpG beta values",
        "forbidden_aggregation": ["gene-level CpG mean", "panel mean", "80 percent gene coverage score"],
        "fixed_cpg_count": 77,
        "model": "ridge logistic regression",
        "logistic_C": list(config.logistic_C),
        "preprocessing": "training-fold-only median imputation followed by training-fold-only StandardScaler; all-missing training columns dropped inside the fold",
        "resampling": {
            "outer_folds": config.outer_folds,
            "outer_repeats": config.outer_repeats,
            "inner_folds": config.inner_folds,
            "grouping": "patient/study_id",
            "bootstrap": "2000 patient-cluster draws of mean repeat metrics conditional on fitted CV models",
        },
        "metrics": ["AUC", "Brier", "sensitivity_threshold_0.5", "specificity_threshold_0.5"],
        "cohorts": {
            "colonomics": {"patients": 92, "source": "curated 119 beta/sample files"},
            "gse119526": {"patients": 48, "source": "114 Table 1 signal intensities; beta=M/(M+U+100); detection_p>=0.01 masked"},
        },
        "source_files": source_files,
    }
    write_json(path, contract)


def write_report(outdir: Path, manifests: dict[str, dict[str, object]]) -> None:
    lines = [
        "# 공개 array ML 개별 CpG 재실행 요약",
        "",
        "기존 공개 array tissue ML 결과는 gene-level CpG 평균 입력을 사용했으므로 이번 결과로 대체해야 한다.",
        "이번 실행은 고정 77개 CpG를 개별 feature로 두고 ridge logistic nested CV를 다시 수행했다.",
        "",
    ]
    for cohort in ["colonomics", "gse119526"]:
        m = manifests[cohort]
        ci = pd.read_csv(outdir / cohort / "bootstrap_ci.tsv", sep="\t")
        auc = ci.loc[ci["metric"].eq("auc")].iloc[0]
        brier = ci.loc[ci["metric"].eq("brier")].iloc[0]
        sens = ci.loc[ci["metric"].eq("threshold_0_5_sensitivity")].iloc[0]
        spec = ci.loc[ci["metric"].eq("threshold_0_5_specificity")].iloc[0]
        lines.extend(
            [
                f"## {cohort}",
                "",
                f"- patients: {m['patients']}; rows: {m['rows']}; requested CpGs: {m['features_requested']}; usable-any CpGs: {m['features_nonmissing_any']}",
                f"- AUC: {auc['point']:.6f} ({auc['ci_low']:.6f}-{auc['ci_high']:.6f})",
                f"- Brier: {brier['point']:.6f} ({brier['ci_low']:.6f}-{brier['ci_high']:.6f})",
                f"- sensitivity at 0.5: {sens['point']:.6f} ({sens['ci_low']:.6f}-{sens['ci_high']:.6f})",
                f"- specificity at 0.5: {spec['point']:.6f} ({spec['ci_low']:.6f}-{spec['ci_high']:.6f})",
                f"- max train/test patient overlap across folds: {m['patient_fold_overlap_max']}",
                "",
            ]
        )
    (outdir / "public_cpg_ml_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(outdir: Path = DEFAULT_OUTDIR, contract_path: Path = DEFAULT_CONTRACT, config: MLConfig | None = None) -> dict[str, dict[str, object]]:
    config = config or MLConfig()
    fixed = read_fixed_probes()
    frames, qcs, source_files = build_inputs(fixed)
    write_contract(contract_path, config, source_files)
    outdir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for cohort in ["colonomics", "gse119526"]:
        manifests[cohort] = run_cohort(cohort, frames[cohort], qcs[cohort], outdir, config, fixed)
    overall = {
        "status": "complete",
        "contract_sha256": sha256(contract_path),
        "script_sha256": sha256(Path(__file__)),
        "cohorts": manifests,
    }
    write_json(outdir / "public_cpg_ml_manifest.json", overall)
    write_report(outdir, manifests)
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    config = MLConfig(max_workers=args.workers, bootstrap_replicates=args.bootstrap, outer_repeats=args.repeats)
    manifests = run_analysis(args.outdir, args.contract, config)
    print(json.dumps({k: {"patients": v["patients"], "features_nonmissing_any": v["features_nonmissing_any"]} for k, v in manifests.items()}, indent=2))


if __name__ == "__main__":
    main()
