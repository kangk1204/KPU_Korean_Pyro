"""Shared helpers for the exploratory panel-size analyses (Part A of PLAN.md).

Read-only inputs; every output is written under A_panel_size/.
"""
from __future__ import annotations
import os

import hashlib
import itertools
import json
import pathlib
from typing import Iterable, Sequence

import numpy as np

SEED = 20260909
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
C_GRID = (0.01, 0.1, 1.0, 10.0)
OUTER_FOLDS = 5
OUTER_REPEATS = 20
INNER_FOLDS = 4
N_BOOT = 2000

WORKSPACE_ROOT = pathlib.Path(os.environ.get("KPU_WORKSPACE_ROOT", os.environ.get("KPU_PROJECT_ROOT", pathlib.Path(__file__).resolve().parents[3])))
SOURCE_ROOT = pathlib.Path(os.environ.get("KPU_SOURCE_ROOT", os.environ.get("KPU_PROJECT_ROOT", WORKSPACE_ROOT)))
BASE = WORKSPACE_ROOT / "127_Exploratory_PanelSize_Coordination_20260909"
OUTDIR = BASE / "A_panel_size"
RESULTS = OUTDIR / "results"
FIGURES = OUTDIR / "figures"
LOGS = BASE / "logs"

SRC_114 = SOURCE_ROOT / "114_ML_DataDriven_20260905"
SRC_119 = SOURCE_ROOT / "119_Figure1_ABC_Revision_20260908"
SRC_124 = SOURCE_ROOT / "124_Integrated_Revision_20260909"
SRC_126 = SOURCE_ROOT / "126_Figure_Redesign_20260909"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def all_subsets(genes: Sequence[str] = GENES) -> list[tuple[int, ...]]:
    """All 1,023 non-empty gene index subsets, ordered by k then lexicographically."""
    out: list[tuple[int, ...]] = []
    for k in range(1, len(genes) + 1):
        out.extend(itertools.combinations(range(len(genes)), k))
    return out


def fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC with mid-ranks for ties. Returns nan if a class is absent."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = np.asarray(score)[order]
    ranks = np.empty(s.size, dtype=float)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos_ranks = ranks[np.asarray(y)[order] == 1]
    return float((pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def patient_folds(patients: Iterable[str], n_splits: int, repeats: int, seed: int) -> list[dict]:
    """Patient-grouped folds with a fresh shuffle per repeat (both specimens stay together)."""
    ids = np.asarray(sorted(map(str, patients)))
    folds = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + 1009 * repeat)
        perm = ids.copy()
        rng.shuffle(perm)
        for fold, test_ids in enumerate(np.array_split(perm, n_splits)):
            folds.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "test_patients": np.asarray(test_ids),
                    "train_patients": np.setdiff1d(ids, test_ids, assume_unique=True),
                }
            )
    return folds


def single_split_folds(patients: Iterable[str], n_splits: int, seed: int) -> list[dict]:
    return patient_folds(patients, n_splits, 1, seed)


def cluster_bootstrap_auc(
    y: np.ndarray,
    score: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Patient-cluster bootstrap percentile CI for a specimen-level AUC."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    idx_by_cluster = [np.flatnonzero(clusters == c) for c in uniq]
    rng = np.random.default_rng(seed)
    point = fast_auc(y, score)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_cluster[p] for p in pick])
        draws[b] = fast_auc(y[idx], score[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def write_json(path, obj) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
