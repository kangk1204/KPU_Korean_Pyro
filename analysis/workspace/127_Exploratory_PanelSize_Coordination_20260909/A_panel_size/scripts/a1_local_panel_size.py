#!/usr/bin/env python3
"""A1 - panel-size (redundancy) curve in the local pyrosequencing cohort.

Pre-specified in PLAN.md section A1. Read-only input:
  114_ML_DataDriven_20260905/data/ml/tissue.tsv  (87 patients x 2 specimens, 10 gene values)

Two products:
  (i)  descriptive sweep over all 1,023 gene subsets, patient-grouped 5-fold CV x 20 repeats;
  (ii) honest nested "best-k": the size-k subset with the highest inner-CV AUC is chosen inside
       each outer training fold and evaluated once on the held-out fold.

Runtime shortcut declared in RESULTS.md: for the 1,023-subset descriptive sweep the ridge C is
fixed once per (repeat, fold) by inner 4-fold CV on the FULL 10-gene panel and reused for every
subset. The nested best-k procedure re-tunes C inside the inner loop for every candidate subset.
"""
from __future__ import annotations

import datetime
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as common_module  # noqa: E402
from common import (  # noqa: E402
    C_GRID,
    GENES,
    INNER_FOLDS,
    N_BOOT,
    OUTER_FOLDS,
    OUTER_REPEATS,
    RESULTS,
    SEED,
    SRC_114,
    all_subsets,
    cluster_bootstrap_auc,
    fast_auc,
    patient_folds,
    sha256,
    write_json,
)

TISSUE_TSV = Path(os.environ.get("A1_TISSUE_TSV", SRC_114 / "data" / "ml" / "tissue.tsv"))
TISSUE_LABEL = os.environ.get("A1_TISSUE_LABEL", "114_ML_DataDriven_20260905/data/ml/tissue.tsv")
ALGORITHM_VERSION = "a1_inner_train_scaling_20260917"
SUBSETS = all_subsets(GENES)
KS = np.array([len(s) for s in SUBSETS])

_STATE: dict = {}


def fit_predict(xtr, ytr, xte, c):
    model = LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=2000)
    model.fit(xtr, ytr)
    return model.predict_proba(xte)[:, 1]


def load_local() -> pd.DataFrame:
    df = pd.read_csv(TISSUE_TSV, sep="\t")
    required = ["study_id", "tissue", "y", *GENES]
    absent = [c for c in required if c not in df.columns]
    if absent:
        raise ValueError(f"tissue.tsv missing required columns: {absent}")
    if df[["study_id", "tissue", "y"]].isna().any().any():
        raise ValueError("tissue.tsv contains missing study_id, tissue, or y values")
    missing = [g for g in GENES if g not in df.columns]
    if missing:
        raise ValueError(f"tissue.tsv missing genes: {missing}")
    if df[GENES].isna().any().any():
        raise ValueError("tissue.tsv contains missing gene values")
    values = df[GENES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("tissue.tsv contains non-finite gene values")
    if (values < 0).any() or (values > 100).any():
        raise ValueError("tissue.tsv gene methylation values must be within 0-100")
    tissues = set(df["tissue"].astype(str))
    if tissues != {"N", "T"}:
        raise ValueError(f"tissue.tsv tissue labels must be exactly N and T, got {sorted(tissues)}")
    y_values = set(pd.to_numeric(df["y"], errors="raise"))
    if y_values != {0, 1}:
        raise ValueError(f"tissue.tsv y labels must be exactly 0 and 1, got {sorted(y_values)}")
    if not df.loc[df["tissue"].eq("N"), "y"].eq(0).all():
        raise ValueError("all N tissues must have y=0")
    if not df.loc[df["tissue"].eq("T"), "y"].eq(1).all():
        raise ValueError("all T tissues must have y=1")
    counts = df.groupby(["study_id", "tissue"]).size().unstack(fill_value=0)
    if not ((counts.get("N", 0) == 1) & (counts.get("T", 0) == 1)).all():
        raise ValueError("every patient must contribute exactly one N and one T specimen")
    return df.sort_values(["study_id", "y"]).reset_index(drop=True)


def _init(x, y, groups):
    _STATE["x"] = x
    _STATE["y"] = y
    _STATE["groups"] = groups


def _standardize(xtr, xte):
    mu = xtr.mean(axis=0)
    sd = xtr.std(axis=0, ddof=0)
    sd[sd < 1e-12] = 1.0
    return (xtr - mu) / sd, (xte - mu) / sd


def _inner_standardize_subset(xtr_raw, itr, ite, cols):
    return _standardize(xtr_raw[np.ix_(itr, cols)], xtr_raw[np.ix_(ite, cols)])


def _checkpoint_path(checkpoint_dir: Path, spec: dict) -> Path:
    return checkpoint_dir / f"repeat{int(spec['repeat']):02d}_fold{int(spec['fold']):02d}.npz"


def _checkpoint_metadata(input_sha: str) -> dict:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "input_label": TISSUE_LABEL,
        "input_sha256": input_sha,
        "source_fingerprint": _source_fingerprint(),
        "seed": SEED,
        "genes": list(GENES),
        "c_grid": list(C_GRID),
        "outer_folds": OUTER_FOLDS,
        "outer_repeats": OUTER_REPEATS,
        "inner_folds": INNER_FOLDS,
    }


def _expected_test_idx(spec: dict, groups: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(groups, spec["test_patients"]))


def _validate_checkpoint(res: dict, spec: dict, groups: np.ndarray, metadata: dict) -> None:
    if res["metadata"] != metadata:
        raise ValueError(f"checkpoint metadata mismatch for repeat {spec['repeat']} fold {spec['fold']}")
    if (res["repeat"], res["fold"]) != (int(spec["repeat"]), int(spec["fold"])):
        raise ValueError("checkpoint fold identity mismatch")
    expected_test_idx = _expected_test_idx(spec, groups)
    if not np.array_equal(res["test_idx"], expected_test_idx):
        raise ValueError("checkpoint test index mismatch")
    expected_shape = (len(SUBSETS), len(expected_test_idx))
    if tuple(res["sweep"].shape) != expected_shape:
        raise ValueError(f"checkpoint sweep shape mismatch: {res['sweep'].shape} != {expected_shape}")
    expected_nested_shape = (len(GENES), len(expected_test_idx))
    if tuple(res["nested"].shape) != expected_nested_shape:
        raise ValueError(
            f"checkpoint nested shape mismatch: {res['nested'].shape} != {expected_nested_shape}"
        )
    for key in ("sweep", "nested"):
        arr = np.asarray(res[key])
        if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
            raise ValueError(f"checkpoint {key} contains invalid probabilities")
    if len(res["chosen"]) != len(GENES):
        raise ValueError("checkpoint chosen subset count mismatch")
    if res["c_full"] not in C_GRID:
        raise ValueError("checkpoint full-panel C is outside C_GRID")


def _save_checkpoint(checkpoint_dir: Path, res: dict, metadata: dict) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, res)
    with tempfile.NamedTemporaryFile(dir=checkpoint_dir, suffix=".npz", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        np.savez_compressed(
            tmp,
            repeat=int(res["repeat"]),
            fold=int(res["fold"]),
            test_idx=res["test_idx"],
            sweep=res["sweep"],
            nested=res["nested"],
            chosen_json=json.dumps(res["chosen"], sort_keys=True),
            c_full=float(res["c_full"]),
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_checkpoint(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "repeat": int(data["repeat"]),
            "fold": int(data["fold"]),
            "test_idx": data["test_idx"],
            "sweep": data["sweep"],
            "nested": data["nested"],
            "chosen": json.loads(data["chosen_json"].item()),
            "c_full": float(data["c_full"]),
            "metadata": json.loads(data["metadata_json"].item()),
        }


def _source_fingerprint() -> dict:
    return {
        "a1_local_panel_size.py": sha256(Path(__file__).resolve()),
        "common.py": sha256(Path(common_module.__file__).resolve()),
    }


def run_fold(spec: dict) -> dict:
    """One outer (repeat, fold): descriptive sweep predictions + nested best-k predictions."""
    x, y, groups = _STATE["x"], _STATE["y"], _STATE["groups"]
    repeat, fold = spec["repeat"], spec["fold"]
    te = np.flatnonzero(np.isin(groups, spec["test_patients"]))
    tr = np.flatnonzero(np.isin(groups, spec["train_patients"]))
    xtr_raw, xte_raw = x[tr], x[te]
    # The outer-fold transformation is used only for final outer predictions.
    # Inner-fold model selection must fit preprocessing inside each inner training split.
    xtr, xte = _standardize(xtr_raw, xte_raw)
    ytr, yte = y[tr], y[te]
    gtr = groups[tr]

    inner = patient_folds(
        np.unique(gtr), INNER_FOLDS, 1, seed=SEED + 1009 * repeat + 104729 * (fold + 1)
    )
    inner_idx = [
        (
            np.flatnonzero(np.isin(gtr, f["train_patients"])),
            np.flatnonzero(np.isin(gtr, f["test_patients"])),
        )
        for f in inner
    ]

    # ---- inner-CV out-of-fold scores for every subset x every C -------------------------
    n_sub, n_c = len(SUBSETS), len(C_GRID)
    inner_oof = np.zeros((n_sub, n_c, len(ytr)), dtype=float)
    for itr, ite in inner_idx:
        for si, sub in enumerate(SUBSETS):
            cols = list(sub)
            a, b = _inner_standardize_subset(xtr_raw, itr, ite, cols)
            for ci, c in enumerate(C_GRID):
                inner_oof[si, ci, ite] = fit_predict(a, ytr[itr], b, c)
    inner_auc = np.array(
        [[fast_auc(ytr, inner_oof[si, ci]) for ci in range(n_c)] for si in range(n_sub)]
    )
    best_c_idx = np.nanargmax(inner_auc, axis=1)
    best_inner = inner_auc[np.arange(n_sub), best_c_idx]

    # ---- C for the descriptive sweep: tuned once on the full panel, reused for all subsets
    full_si = n_sub - 1  # last subset is the full 10-gene panel
    c_full = C_GRID[int(best_c_idx[full_si])]

    sweep = np.empty((n_sub, len(te)), dtype=float)
    for si, sub in enumerate(SUBSETS):
        cols = list(sub)
        sweep[si] = fit_predict(xtr[:, cols], ytr, xte[:, cols], c_full)

    # ---- nested best-k: pick the size-k winner by inner AUC, evaluate once on the outer fold
    nested = np.empty((len(GENES), len(te)), dtype=float)
    chosen = []
    for k in range(1, len(GENES) + 1):
        cand = np.flatnonzero(KS == k)
        si = int(cand[np.nanargmax(best_inner[cand])])
        c = C_GRID[int(best_c_idx[si])]
        cols = list(SUBSETS[si])
        nested[k - 1] = fit_predict(xtr[:, cols], ytr, xte[:, cols], c)
        chosen.append(
            {
                "repeat": repeat,
                "fold": fold,
                "k": k,
                "subset": "+".join(GENES[i] for i in SUBSETS[si]),
                "inner_auc": float(best_inner[si]),
                "C": c,
            }
        )

    return {
        "repeat": repeat,
        "fold": fold,
        "test_idx": te,
        "sweep": sweep,
        "nested": nested,
        "chosen": chosen,
        "c_full": c_full,
    }


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_local()
    x = df[GENES].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    groups = df["study_id"].astype(str).to_numpy()
    n = len(df)
    input_sha = sha256(TISSUE_TSV)
    checkpoint_metadata = _checkpoint_metadata(input_sha)
    source_fingerprint_start = checkpoint_metadata["source_fingerprint"]
    print(f"[A1] {n} specimens, {len(np.unique(groups))} patients, {len(SUBSETS)} subsets", flush=True)

    folds = patient_folds(np.unique(groups), OUTER_FOLDS, OUTER_REPEATS, SEED)
    workers = int(os.environ.get("A1_PANEL_SIZE_WORKERS", "4"))
    checkpoint_env = os.environ.get("A1_CHECKPOINT_DIR")
    checkpoint_dir = Path(checkpoint_env) if checkpoint_env else None
    results = []
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        completed = {}
        for spec in folds:
            path = _checkpoint_path(checkpoint_dir, spec)
            if path.exists():
                res = _load_checkpoint(path)
                _validate_checkpoint(res, spec, groups, checkpoint_metadata)
                completed[(res["repeat"], res["fold"])] = res
        results.extend(completed.values())
        folds = [f for f in folds if (int(f["repeat"]), int(f["fold"])) not in completed]
        if completed:
            print(f"[A1] loaded {len(completed)} checkpointed outer folds", flush=True)

    with mp.Pool(processes=min(workers, mp.cpu_count()), initializer=_init, initargs=(x, y, groups)) as pool:
        start_i = len(results)
        for i, res in enumerate(pool.imap_unordered(run_fold, folds, chunksize=1), start=start_i + 1):
            results.append(res)
            if checkpoint_dir is not None:
                _save_checkpoint(checkpoint_dir, res, checkpoint_metadata)
            if i % 10 == 0:
                print(f"[A1] {i}/100 outer folds done ({time.time()-t0:.0f}s)", flush=True)

    sweep_pred = np.full((len(SUBSETS), OUTER_REPEATS, n), np.nan)
    nested_pred = np.full((len(GENES), OUTER_REPEATS, n), np.nan)
    chosen_rows, c_full_rows = [], []
    for res in results:
        r, te = res["repeat"], res["test_idx"]
        sweep_pred[:, r, te] = res["sweep"]
        nested_pred[:, r, te] = res["nested"]
        chosen_rows.extend(res["chosen"])
        c_full_rows.append({"repeat": r, "fold": res["fold"], "C_full_panel": res["c_full"]})
    if np.isnan(sweep_pred).any() or np.isnan(nested_pred).any():
        raise RuntimeError("incomplete out-of-fold coverage")
    source_fingerprint_end = _source_fingerprint()
    if source_fingerprint_end != source_fingerprint_start:
        raise RuntimeError("source files changed during A1 run; refusing to write results")

    # ---- descriptive sweep: mean repeat-level AUC per subset ---------------------------
    sweep_auc = np.array(
        [[fast_auc(y, sweep_pred[si, r]) for r in range(OUTER_REPEATS)] for si in range(len(SUBSETS))]
    )
    subset_tab = pd.DataFrame(
        {
            "k": KS,
            "subset": ["+".join(GENES[i] for i in s) for s in SUBSETS],
            "mean_repeat_auc": sweep_auc.mean(axis=1),
            "sd_repeat_auc": sweep_auc.std(axis=1, ddof=1),
            "min_repeat_auc": sweep_auc.min(axis=1),
            "max_repeat_auc": sweep_auc.max(axis=1),
        }
    )
    subset_tab.to_csv(RESULTS / "a1_subset_auc.tsv", sep="\t", index=False, float_format="%.6f")

    per_k = (
        subset_tab.groupby("k")["mean_repeat_auc"]
        .agg(
            n_subsets="size",
            median_auc="median",
            q1_auc=lambda s: s.quantile(0.25),
            q3_auc=lambda s: s.quantile(0.75),
            min_auc="min",
            max_auc="max",
        )
        .reset_index()
    )

    # ---- nested best-k with patient-cluster bootstrap CI --------------------------------
    nested_rows = []
    for k in range(1, len(GENES) + 1):
        per_repeat = np.array([fast_auc(y, nested_pred[k - 1, r]) for r in range(OUTER_REPEATS)])
        mean_scores = nested_pred[k - 1].mean(axis=0)
        _, lo, hi = _bootstrap_mean_repeat_auc(y, nested_pred[k - 1], groups)
        nested_rows.append(
            {
                "k": k,
                "nested_mean_auc": per_repeat.mean(),
                "nested_sd_auc": per_repeat.std(ddof=1),
                "ci_lo": lo,
                "ci_hi": hi,
                "auc_repeat_averaged_scores": fast_auc(y, mean_scores),
            }
        )
    nested_tab = pd.DataFrame(nested_rows)
    per_k = per_k.merge(nested_tab, on="k")
    per_k.to_csv(RESULTS / "a1_per_k_summary.tsv", sep="\t", index=False, float_format="%.6f")
    chosen_df = pd.DataFrame(chosen_rows).sort_values(["repeat", "fold", "k"]).reset_index(drop=True)
    chosen_df.to_csv(
        RESULTS / "a1_nested_selected_subsets.tsv", sep="\t", index=False, float_format="%.6f"
    )
    pd.DataFrame(c_full_rows).sort_values(["repeat", "fold"]).to_csv(
        RESULTS / "a1_full_panel_C.tsv", sep="\t", index=False
    )

    # ---- pre-specified saturation reading -----------------------------------------------
    k10_median = float(per_k.loc[per_k.k == 10, "median_auc"].iloc[0])
    sat = None
    for _, row in per_k.iterrows():
        if row["ci_lo"] > 0.95 and abs(row["median_auc"] - k10_median) <= 0.01:
            sat = int(row["k"])
            break
    reading = {
        "rule": "smallest k with nested best-k AUC lower CI bound > 0.95 AND median random-subset AUC within 0.01 of k=10",
        "k10_median_subset_auc": k10_median,
        "saturation_k": sat,
    }

    manifest = {
        "analysis": "A1 local pyrosequencing panel-size curve",
        "seed": SEED,
        "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_seconds": round(time.time() - t0, 1),
        "design": {
            "outer_folds": OUTER_FOLDS,
            "outer_repeats": OUTER_REPEATS,
            "inner_folds": INNER_FOLDS,
            "C_grid": list(C_GRID),
            "bootstrap_draws": N_BOOT,
            "sweep_C_shortcut": "C fixed per (repeat, fold) by inner CV on the full 10-gene panel and reused for all 1,023 subsets",
            "nested_C": "re-tuned inside the inner loop for every candidate subset",
        },
        "inputs": {TISSUE_LABEL: input_sha},
        "source_fingerprint": source_fingerprint_start,
        "n_specimens": int(n),
        "n_patients": int(len(np.unique(groups))),
        "reading": reading,
        "outputs": {
            p.name: sha256(p)
            for p in [
                RESULTS / "a1_subset_auc.tsv",
                RESULTS / "a1_per_k_summary.tsv",
                RESULTS / "a1_nested_selected_subsets.tsv",
                RESULTS / "a1_full_panel_C.tsv",
            ]
        },
    }
    write_json(RESULTS / "a1_manifest.json", manifest)
    print(per_k.to_string(index=False))
    print(json.dumps(reading, indent=2))
    print(f"[A1] done in {time.time()-t0:.0f}s", flush=True)


def _bootstrap_mean_repeat_auc(y, pred_by_repeat, groups):
    """Patient-cluster bootstrap of the mean-across-repeats out-of-fold AUC."""
    uniq = np.unique(groups)
    idx_by_cluster = [np.flatnonzero(groups == c) for c in uniq]
    rng = np.random.default_rng(SEED)
    point = float(np.mean([fast_auc(y, pred_by_repeat[r]) for r in range(pred_by_repeat.shape[0])]))
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_cluster[p] for p in pick])
        yy = y[idx]
        draws[b] = np.mean([fast_auc(yy, pred_by_repeat[r][idx]) for r in range(pred_by_repeat.shape[0])])
    lo, hi = np.percentile(draws[np.isfinite(draws)], [2.5, 97.5])
    return point, float(lo), float(hi)


if __name__ == "__main__":
    main()
