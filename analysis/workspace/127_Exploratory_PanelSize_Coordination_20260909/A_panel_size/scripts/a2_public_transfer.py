#!/usr/bin/env python3
"""A2 - gene-level panel-size curve with external transfer on the public/Korean arrays.

Pre-specified in PLAN.md section A2. All inputs are read-only.

Training cohort: Colonomics (92 source-defined pairs, 184 specimens, 450K).
Transfer feature set: the 56 fixed CpGs measured in all three Korean EPIC matrices, restricted
further per target when a CpG is absent there. Beta values are used as-is (no cross-cohort
renormalisation); median imputation and scaling are fitted on the training cohort only.

Products
  1. internal Colonomics repeated-CV curve over all 1,023 gene subsets;
  2. external AUC for source-fitted variants of all 1,023 subsets in every target cohort;
  3. full-model and single-best-gene transfer AUC per cohort with 2,000-draw patient-cluster CIs;
  4. cross-cohort matrix: train in each row cohort on the candidate CpGs
     available in the target cohort, then apply that fitted model to the target.

Terminology corrected on 2026-09-14. The cross-cohort matrix does not train
on the pooled remaining cohorts and is not leave-one-cohort-out validation.
The retained output filename is unchanged for compatibility. No model fitting
or numerical calculation was changed by this documentation correction.
"""
from __future__ import annotations

import datetime
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    SRC_119,
    SRC_124,
    all_subsets,
    cluster_bootstrap_auc,
    fast_auc,
    patient_folds,
    sha256,
    write_json,
)

FIXED_PROBES = SRC_124 / "registry" / "fixed_probes.json"
KOREAN = ["CMCBSN", "SNUH", "ASAN"]
PUBLIC_119 = ["GSE193535", "GSE77718", "GSE42752"]
SUBSETS = all_subsets(GENES)
KS = np.array([len(s) for s in SUBSETS])
INPUTS: dict[str, str] = {}


def _register(path: Path) -> Path:
    INPUTS[str(path)] = sha256(path)
    return path


def _complete_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.groupby(["study_id", "tissue"]).size().unstack(fill_value=0)
    keep = counts.index[(counts.get("N", 0) == 1) & (counts.get("T", 0) == 1)]
    out = frame.loc[frame["study_id"].isin(keep)].copy()
    return out.sort_values(["study_id", "y"]).reset_index(drop=True)


def load_fixed() -> dict[str, list[str]]:
    return json.loads(_register(FIXED_PROBES).read_text())


def _assemble(beta: pd.DataFrame, meta: pd.DataFrame, cohort: str, cpgs: list[str]) -> pd.DataFrame:
    """meta must carry sample / study_id / tissue; beta is probes x samples."""
    values = beta.reindex(cpgs).T
    values.index = values.index.astype(str)
    frame = meta.copy()
    frame["y"] = frame["tissue"].map({"N": 0, "T": 1}).astype(int)
    frame = frame.loc[frame["sample"].astype(str).isin(values.index)].copy()
    mat = values.loc[frame["sample"].astype(str)].to_numpy(dtype=float)
    out = frame[["study_id", "sample", "tissue", "y"]].reset_index(drop=True)
    out.insert(0, "cohort", cohort)
    for j, cpg in enumerate(cpgs):
        out[cpg] = mat[:, j]
    return _complete_pairs(out)


def load_colonomics(cpgs: list[str]) -> pd.DataFrame:
    beta = pd.read_csv(_register(SRC_119 / "data/derived/Colonomics_beta.tsv.gz"), sep="\t", index_col=0)
    meta = pd.read_csv(_register(SRC_119 / "data/derived/Colonomics_samples.tsv"), sep="\t")
    meta = meta.loc[
        meta["tissue"].isin(["N", "T"])
        & meta["pair_verified"].astype(bool)
        & (~meta["excluded"].astype(bool))
    ].copy()
    meta["study_id"] = "Colonomics_" + meta["patient"].astype(str)
    return _assemble(beta.apply(pd.to_numeric, errors="coerce"), meta, "Colonomics", cpgs)


def load_korean(name: str, cpgs: list[str]) -> pd.DataFrame:
    beta = pd.read_csv(_register(SRC_124 / f"data/derived/{name}_beta.tsv"), sep="\t", index_col=0)
    meta = pd.read_csv(_register(SRC_124 / f"data/derived/{name}_samples.tsv"), sep="\t")
    meta = meta.loc[meta["tissue"].isin(["N", "T"])].copy()
    meta["sample"] = meta["sample_id"].astype(str)
    meta["study_id"] = name + "_" + meta["patient_id"].astype(str)
    return _assemble(beta.apply(pd.to_numeric, errors="coerce"), meta, name, cpgs)


def load_gse119(name: str, cpgs: list[str]) -> pd.DataFrame:
    beta = pd.read_csv(_register(SRC_119 / f"data/derived/{name}_beta.tsv.gz"), sep="\t", index_col=0)
    meta = pd.read_csv(_register(SRC_119 / f"data/derived/{name}_samples.tsv"), sep="\t")
    meta = meta.loc[meta["tissue"].isin(["N", "T"]) & meta["pair_verified"].astype(bool)].copy()
    meta["study_id"] = name + "_" + meta["patient"].astype(str)
    return _assemble(beta.apply(pd.to_numeric, errors="coerce"), meta, name, cpgs)


def load_gse119526(cpgs: list[str]) -> pd.DataFrame:
    """beta = M/(M+U+100); detection P >= 0.01 masked (parsing follows 124 analyze_public_cpg_ml.py)."""
    signals = pd.read_csv(
        _register(SRC_114 / "data/public/processed/gse119526_table1_signal_intensities.tsv.gz"),
        sep="\t",
        index_col=0,
    )
    meta = pd.read_csv(
        _register(SRC_114 / "data/public/processed/gse119526_sample_metadata.tsv"), sep="\t"
    )
    parsed = meta["description"].astype(str).map(lambda d: re.fullmatch(r"(\d+)([NT])", d))
    if parsed.isna().any():
        raise ValueError("unparsable GSE119526 sample description")
    meta["code"] = meta["description"].astype(str)
    meta["tissue"] = [m.group(2) for m in parsed]
    meta["study_id"] = "GSE119526_" + pd.Series([m.group(1) for m in parsed], index=meta.index)
    meta["sample"] = meta["code"]
    rows = []
    for rec in meta.to_dict("records"):
        row = {
            "cohort": "GSE119526",
            "study_id": rec["study_id"],
            "sample": rec["sample"],
            "tissue": rec["tissue"],
            "y": 1 if rec["tissue"] == "T" else 0,
        }
        m_col, u_col, p_col = (f"{rec['code']}_{s}" for s in ("methylated_signal", "unmethylated_signal", "detection_pval"))
        for cpg in cpgs:
            if cpg not in signals.index or m_col not in signals or u_col not in signals:
                row[cpg] = np.nan
                continue
            det = float(signals.loc[cpg, p_col]) if p_col in signals else np.nan
            if not np.isfinite(det) or det >= 0.01:
                row[cpg] = np.nan
                continue
            m, u = float(signals.loc[cpg, m_col]), float(signals.loc[cpg, u_col])
            row[cpg] = m / (m + u + 100.0)
        rows.append(row)
    return _complete_pairs(pd.DataFrame(rows))


# ---------------------------------------------------------------------------------------
class FrozenModel:
    """Median imputation + standardisation + ridge logistic, all fitted on one cohort."""

    def __init__(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float):
        self.median = np.nanmedian(x, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        xi = self._impute(x)
        self.mu = xi.mean(axis=0)
        sd = xi.std(axis=0, ddof=0)
        self.sd = np.where(sd < 1e-12, 1.0, sd)
        self.c = c
        self.model = LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=5000)
        self.model.fit(self._scale(xi), y)

    def _impute(self, x):
        out = np.array(x, dtype=float, copy=True)
        idx = np.where(np.isnan(out))
        out[idx] = np.take(self.median, idx[1])
        return out

    def _scale(self, xi):
        return (xi - self.mu) / self.sd

    def score(self, x):
        return self.model.predict_proba(self._scale(self._impute(x)))[:, 1]


def tune_c(x, y, groups, seed=SEED, folds=INNER_FOLDS) -> float:
    """Ridge C by patient-grouped CV inside the training cohort."""
    splits = patient_folds(np.unique(groups), folds, 1, seed)
    best_c, best_auc = C_GRID[0], -np.inf
    for c in C_GRID:
        oof = np.zeros(len(y))
        for f in splits:
            tr = np.flatnonzero(np.isin(groups, f["train_patients"]))
            te = np.flatnonzero(np.isin(groups, f["test_patients"]))
            oof[te] = FrozenModel(x[tr], y[tr], groups[tr], c).score(x[te])
        auc = fast_auc(y, oof)
        if auc > best_auc:
            best_auc, best_c = auc, c
    return best_c


def transfer(train_frame, target_frame, cpgs) -> tuple[np.ndarray, float]:
    xtr = train_frame[cpgs].to_numpy(float)
    ytr = train_frame["y"].to_numpy(int)
    gtr = train_frame["study_id"].to_numpy(str)
    c = tune_c(xtr, ytr, gtr)
    model = FrozenModel(xtr, ytr, gtr, c)
    return model.score(target_frame[cpgs].to_numpy(float)), c


# ---- internal Colonomics repeated CV over all subsets ----------------------------------
_ST: dict = {}


def _init_internal(x, y, groups, cols_by_subset):
    _ST.update(x=x, y=y, groups=groups, cols=cols_by_subset)


def _internal_fold(spec):
    x, y, groups, cols = _ST["x"], _ST["y"], _ST["groups"], _ST["cols"]
    tr = np.flatnonzero(np.isin(groups, spec["train_patients"]))
    te = np.flatnonzero(np.isin(groups, spec["test_patients"]))
    c_full = tune_c(x[tr], y[tr], groups[tr], seed=SEED + 1009 * spec["repeat"] + 104729 * (spec["fold"] + 1))
    out = np.empty((len(cols), len(te)))
    for si, cc in enumerate(cols):
        out[si] = FrozenModel(x[np.ix_(tr, cc)], y[tr], groups[tr], c_full).score(x[np.ix_(te, cc)])
    return spec["repeat"], te, out, c_full


def main() -> None:
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    fixed = load_fixed()

    # ---- transfer feature set: 56 CpGs measured in all three Korean EPIC matrices --------
    korean_probe_sets = []
    for name in KOREAN:
        b = pd.read_csv(SRC_124 / f"data/derived/{name}_beta.tsv", sep="\t", index_col=0, usecols=[0])
        korean_probe_sets.append(set(b.index.astype(str)))
    fixed_order = [c for g in GENES for c in fixed[g]]
    korean56 = [c for c in fixed_order if all(c in s for s in korean_probe_sets)]
    print(f"[A2] Korean-measured CpGs: {len(korean56)} of {len(fixed_order)}", flush=True)

    gene_of = {c: g for g in GENES for c in fixed[g]}

    cohorts = {"Colonomics": load_colonomics(korean56)}
    for name in KOREAN:
        cohorts[name] = load_korean(name, korean56)
    cohorts["GSE119526"] = load_gse119526(korean56)
    for name in PUBLIC_119:
        cohorts[name] = load_gse119(name, korean56)
    for name, fr in cohorts.items():
        print(f"[A2] {name}: {fr['study_id'].nunique()} pairs, {len(fr)} specimens", flush=True)

    # per-cohort available CpGs (present in the matrix and not entirely missing)
    avail = {
        name: [c for c in korean56 if fr[c].notna().any()] for name, fr in cohorts.items()
    }
    cov_rows = [
        {
            "cohort": name,
            "n_pairs": int(fr["study_id"].nunique()),
            "n_specimens": int(len(fr)),
            "n_cpgs_available": len(avail[name]),
            "pct_missing_values": round(100 * float(fr[avail[name]].isna().mean().mean()), 3),
            **{f"n_cpg_{g}": sum(gene_of[c] == g for c in avail[name]) for g in GENES},
        }
        for name, fr in cohorts.items()
    ]
    pd.DataFrame(cov_rows).to_csv(RESULTS / "a2_cohort_coverage.tsv", sep="\t", index=False)

    targets = [n for n in cohorts if n != "Colonomics"]

    # ---- 1. internal Colonomics repeated CV over the 1,023 subsets ----------------------
    col = cohorts["Colonomics"]
    cols56 = avail["Colonomics"]
    x = col[cols56].to_numpy(float)
    y = col["y"].to_numpy(int)
    groups = col["study_id"].to_numpy(str)
    idx_of = {c: i for i, c in enumerate(cols56)}
    cols_by_subset = [
        [idx_of[c] for gi in sub for c in fixed[GENES[gi]] if c in idx_of] for sub in SUBSETS
    ]
    folds = patient_folds(np.unique(groups), OUTER_FOLDS, OUTER_REPEATS, SEED)
    oof = np.full((len(SUBSETS), OUTER_REPEATS, len(y)), np.nan)
    with mp.Pool(min(8, mp.cpu_count()), initializer=_init_internal, initargs=(x, y, groups, cols_by_subset)) as pool:
        for i, (r, te, out, _c) in enumerate(pool.imap_unordered(_internal_fold, folds, chunksize=1), 1):
            oof[:, r, te] = out
            if i % 20 == 0:
                print(f"[A2] internal CV {i}/{len(folds)} ({time.time()-t0:.0f}s)", flush=True)
    internal_auc = np.array(
        [[fast_auc(y, oof[si, r]) for r in range(OUTER_REPEATS)] for si in range(len(SUBSETS))]
    ).mean(axis=1)

    # ---- 2. frozen-model transfer for every subset x every target -----------------------
    rows = []
    model_cache: dict[tuple[str, ...], FrozenModel] = {}
    for name in targets:
        tgt = cohorts[name]
        tgt_avail = set(avail[name])
        yt = tgt["y"].to_numpy(int)
        for si, sub in enumerate(SUBSETS):
            cpgs = tuple(c for gi in sub for c in fixed[GENES[gi]] if c in tgt_avail and c in idx_of)
            if not cpgs:
                continue
            if cpgs not in model_cache:
                cl = list(cpgs)
                xtr = col[cl].to_numpy(float)
                c = tune_c(xtr, y, groups)
                model_cache[cpgs] = FrozenModel(xtr, y, groups, c)
            m = model_cache[cpgs]
            rows.append(
                {
                    "target": name,
                    "k": int(KS[si]),
                    "subset": "+".join(GENES[i] for i in sub),
                    "n_cpgs": len(cpgs),
                    "colonomics_internal_auc": internal_auc[si],
                    "external_auc": fast_auc(yt, m.score(tgt[list(cpgs)].to_numpy(float))),
                }
            )
        print(f"[A2] transfer sweep done for {name} ({time.time()-t0:.0f}s)", flush=True)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(RESULTS / "a2_subset_transfer_auc.tsv", sep="\t", index=False, float_format="%.6f")

    per_k = (
        sweep.groupby(["target", "k"])["external_auc"]
        .agg(median_auc="median", q1_auc=lambda s: s.quantile(0.25), q3_auc=lambda s: s.quantile(0.75), min_auc="min", max_auc="max", n="size")
        .reset_index()
    )
    per_k.to_csv(RESULTS / "a2_per_k_by_target.tsv", sep="\t", index=False, float_format="%.6f")
    pooled = (
        sweep.groupby("k")["external_auc"]
        .agg(median_auc="median", q1_auc=lambda s: s.quantile(0.25), q3_auc=lambda s: s.quantile(0.75), min_auc="min", max_auc="max", n="size")
        .reset_index()
    )
    pooled.to_csv(RESULTS / "a2_per_k_pooled.tsv", sep="\t", index=False, float_format="%.6f")

    internal_tab = pd.DataFrame(
        {"k": KS, "subset": ["+".join(GENES[i] for i in s) for s in SUBSETS], "colonomics_internal_auc": internal_auc}
    )
    internal_tab.to_csv(RESULTS / "a2_colonomics_internal_auc.tsv", sep="\t", index=False, float_format="%.6f")

    # ---- 3. full model and single best gene (chosen on Colonomics internal CV only) -----
    single_idx = [i for i in range(len(SUBSETS)) if KS[i] == 1]
    best_single = int(single_idx[int(np.argmax(internal_auc[single_idx]))])
    best_gene = GENES[SUBSETS[best_single][0]]
    print(f"[A2] single best gene by Colonomics internal CV: {best_gene}", flush=True)

    frozen_rows = []
    for label, cpg_pool in [("full_10gene", set(korean56)), (f"single_gene_{best_gene}", set(fixed[best_gene]))]:
        for name in targets:
            tgt = cohorts[name]
            cpgs = [c for c in korean56 if c in cpg_pool and c in set(avail[name])]
            s, c = transfer(col, tgt, cpgs)
            point, lo, hi = cluster_bootstrap_auc(
                tgt["y"].to_numpy(int), s, tgt["study_id"].to_numpy(str), N_BOOT, SEED
            )
            frozen_rows.append(
                {
                    "model": label,
                    "target": name,
                    "n_pairs": int(tgt["study_id"].nunique()),
                    "n_specimens": int(len(tgt)),
                    "n_cpgs": len(cpgs),
                    "C": c,
                    "auc": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "transfer_success": bool(lo > 0.90),
                }
            )
    frozen = pd.DataFrame(frozen_rows)
    frozen.to_csv(RESULTS / "a2_frozen_transfer.tsv", sep="\t", index=False, float_format="%.6f")

    # ---- 4. leave-one-cohort-out for the full model ------------------------------------
    loco_rows = []
    for trn in ["Colonomics"] + KOREAN:
        tr_frame = cohorts[trn]
        tr_avail = set(avail[trn])
        for name in cohorts:
            if name == trn:
                continue
            tgt = cohorts[name]
            cpgs = [c for c in korean56 if c in tr_avail and c in set(avail[name])]
            s, c = transfer(tr_frame, tgt, cpgs)
            point, lo, hi = cluster_bootstrap_auc(
                tgt["y"].to_numpy(int), s, tgt["study_id"].to_numpy(str), N_BOOT, SEED
            )
            loco_rows.append(
                {"train": trn, "target": name, "n_cpgs": len(cpgs), "C": c, "auc": point, "ci_lo": lo, "ci_hi": hi}
            )
        print(f"[A2] LOCO train={trn} done ({time.time()-t0:.0f}s)", flush=True)
    loco = pd.DataFrame(loco_rows)
    loco.to_csv(RESULTS / "a2_leave_one_cohort_out.tsv", sep="\t", index=False, float_format="%.6f")
    loco.pivot(index="train", columns="target", values="auc").to_csv(
        RESULTS / "a2_loco_matrix.tsv", sep="\t", float_format="%.4f"
    )

    # ---- pre-specified readings ---------------------------------------------------------
    full = frozen.loc[frozen["model"] == "full_10gene"]
    k10 = pooled.loc[pooled["k"] == 10, "median_auc"].iloc[0]
    reading = {
        "transfer_rule": "successful when the frozen full-model AUC lower CI bound exceeds 0.90",
        "transfer_success": {r["target"]: bool(r["ci_lo"] > 0.90) for _, r in full.iterrows()},
        "single_best_gene_by_colonomics_internal_cv": best_gene,
        "pooled_k10_median_external_auc": float(k10),
        "genes_needed_pooled": {
            "rule": "smallest k whose pooled median external AUC is within 0.01 of the k=10 pooled median",
            "k": int(
                min(
                    (int(r["k"]) for _, r in pooled.iterrows() if abs(r["median_auc"] - k10) <= 0.01),
                    default=-1,
                )
            ),
        },
        "genes_needed_per_target": {
            t: int(
                min(
                    (
                        int(r["k"])
                        for _, r in per_k.loc[per_k["target"] == t].iterrows()
                        if abs(r["median_auc"] - per_k.loc[(per_k["target"] == t) & (per_k["k"] == 10), "median_auc"].iloc[0]) <= 0.01
                    ),
                    default=-1,
                )
            )
            for t in targets
        },
    }

    write_json(
        RESULTS / "a2_manifest.json",
        {
            "analysis": "A2 public-array panel-size curve and external transfer",
            "seed": SEED,
            "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runtime_seconds": round(time.time() - t0, 1),
            "design": {
                "training_cohort": "Colonomics",
                "transfer_feature_set": f"{len(korean56)} CpGs measured in CMCBSN+SNUH+ASAN, restricted per target",
                "C_grid": list(C_GRID),
                "C_selection": "4-fold patient-grouped CV on the training cohort",
                "internal_cv": f"{OUTER_FOLDS}-fold patient-grouped x {OUTER_REPEATS} repeats; C fixed per fold on the full feature set and reused for subsets",
                "bootstrap_draws": N_BOOT,
                "normalisation": "beta used as-is; median imputation and scaling fitted on the training cohort only",
            },
            "inputs": INPUTS,
            "cohorts": {n: {"pairs": int(f['study_id'].nunique()), "specimens": int(len(f)), "cpgs": len(avail[n])} for n, f in cohorts.items()},
            "reading": reading,
            "outputs": {
                p.name: sha256(p)
                for p in sorted(RESULTS.glob("a2_*.tsv"))
            },
        },
    )
    print(frozen.to_string(index=False))
    print(pooled.to_string(index=False))
    print(json.dumps(reading, indent=2))
    print(f"[A2] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
