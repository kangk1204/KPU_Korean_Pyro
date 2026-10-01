"""Shared helpers for Part B (coordination-axis biology), 127 exploratory analyses.

Pure functions only: axis construction, effect sizes with bootstrap CIs, and the
pre-specified test wrappers. No I/O side effects beyond hashing.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260909
N_BOOT = 2000
MIN_GROUP = 5  # groups smaller than this are described, never tested


# ---------------------------------------------------------------- provenance
def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- axes
def zscore_columns(mat: np.ndarray) -> np.ndarray:
    """z per feature (column). Zero-variance columns are dropped by the caller."""
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=1)
    return (mat - mu) / sd


def pc1(mat_z: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """PC1 scores, variance explained fraction, loading vector.

    mat_z: samples x features, already standardised per feature.
    """
    centred = mat_z - mat_z.mean(axis=0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    scores = u[:, 0] * s[0]
    var = s ** 2
    return scores, float(var[0] / var.sum()), vt[0]


def build_axis(beta_sub: pd.DataFrame, orient_ref: np.ndarray) -> dict:
    """beta_sub: samples (rows) x CpGs (cols), complete cases only.

    Returns scores oriented so that higher = more methylation (positive
    Spearman correlation with `orient_ref`, the mean tumor beta).
    """
    mat = beta_sub.to_numpy(dtype=float)
    keep = mat.std(axis=0, ddof=1) > 0
    mat = mat[:, keep]
    cpgs = list(np.asarray(beta_sub.columns)[keep])
    z = zscore_columns(mat)
    scores, ve, load = pc1(z)
    rho = stats.spearmanr(scores, orient_ref).statistic
    flipped = bool(rho < 0)
    if flipped:
        scores = -scores
        load = -load
    return {
        "scores": scores,
        "var_explained": ve,
        "loadings": pd.Series(load, index=cpgs),
        "n_cpg": len(cpgs),
        "n_sample": mat.shape[0],
        "flipped": flipped,
        "orient_rho": float(stats.spearmanr(scores, orient_ref).statistic),
    }


# ----------------------------------------------------------- effect sizes/CI
def _rng(tag: str) -> np.random.Generator:
    """Deterministic, test-specific stream derived from the fixed project seed."""
    mix = int.from_bytes(hashlib.sha256(f"{SEED}:{tag}".encode()).digest()[:8], "big")
    return np.random.default_rng(mix)


def boot_median_diff(a: np.ndarray, b: np.ndarray, tag: str, n_boot: int = N_BOOT):
    """median(a) - median(b) with a stratified percentile bootstrap CI."""
    rng = _rng(tag)
    point = float(np.median(a) - np.median(b))
    draws = np.empty(n_boot)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        draws[i] = np.median(rng.choice(a, na, replace=True)) - np.median(
            rng.choice(b, nb, replace=True)
        )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def boot_spearman(x: np.ndarray, y: np.ndarray, tag: str, n_boot: int = N_BOOT):
    rng = _rng(tag)
    point = float(stats.spearmanr(x, y).statistic)
    n = len(x)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xi, yi = x[idx], y[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            draws[i] = np.nan
        else:
            draws[i] = stats.spearmanr(xi, yi).statistic
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def epsilon_squared(groups: list[np.ndarray]) -> float:
    """Kruskal-Wallis epsilon^2 = (H - k + 1) / (n - k)."""
    h = stats.kruskal(*groups).statistic
    n = sum(len(g) for g in groups)
    k = len(groups)
    if n - k <= 0:
        return float("nan")
    return float((h - k + 1) / (n - k))


def boot_epsilon_squared(groups: list[np.ndarray], tag: str, n_boot: int = N_BOOT):
    rng = _rng(tag)
    point = epsilon_squared(groups)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        res = [rng.choice(g, len(g), replace=True) for g in groups]
        try:
            draws[i] = epsilon_squared(res)
        except ValueError:
            draws[i] = np.nan
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


# ------------------------------------------------------------- test wrappers
def rec(**kw) -> dict:
    base = dict(
        cohort="", axis="", label="", comparison="", test="", n_total=np.nan,
        group_sizes="", group_medians="", statistic=np.nan, p_value=np.nan,
        effect_name="", effect=np.nan, ci_lo=np.nan, ci_hi=np.nan,
        effect_r=np.nan, tested=False, note="", in_family=True,
    )
    base.update(kw)
    return base


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation from the Mann-Whitney U of a vs b, in [-1, 1].

    Used only to put two-group, four-group and continuous effects on one
    comparable |r| scale for the pre-specified "stronger than" reading.
    """
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def test_two_group(values, labels, pos, neg, *, cohort, axis, label, comparison,
                   in_family=True, note=""):
    """Wilcoxon rank-sum (Mann-Whitney U), two-sided; median(pos) - median(neg)."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    a = values[labels == pos]
    b = values[labels == neg]
    sizes = f"{pos}={len(a)}; {neg}={len(b)}"
    meds = (f"{pos}={np.median(a):.3g}; {neg}={np.median(b):.3g}"
            if len(a) and len(b) else "")
    if len(a) < MIN_GROUP or len(b) < MIN_GROUP:
        return rec(cohort=cohort, axis=axis, label=label, comparison=comparison,
                   test="Wilcoxon rank-sum (not run)", n_total=len(a) + len(b),
                   group_sizes=sizes, group_medians=meds, tested=False, in_family=in_family,
                   effect_name="median difference",
                   effect=float(np.median(a) - np.median(b)) if len(a) and len(b) else np.nan,
                   note=(note + "; " if note else "") + f"group < {MIN_GROUP} patients: descriptive only")
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    tag = f"{cohort}|{axis}|{comparison}"
    eff, lo, hi = boot_median_diff(a, b, tag)
    return rec(cohort=cohort, axis=axis, label=label, comparison=comparison,
               test="Wilcoxon rank-sum", n_total=len(a) + len(b), group_sizes=sizes,
               group_medians=meds, statistic=float(u.statistic), p_value=float(u.pvalue),
               effect_name="median difference", effect=eff, ci_lo=lo, ci_hi=hi,
               effect_r=rank_biserial(a, b), tested=True, in_family=in_family, note=note)


def test_kruskal(values, labels, levels, *, cohort, axis, label, comparison,
                 in_family=True, note=""):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    groups, sizes, kept, meds = [], [], [], []
    for lv in levels:
        g = values[labels == lv]
        sizes.append(f"{lv}={len(g)}")
        if len(g):
            meds.append(f"{lv}={np.median(g):.3g}")
        if len(g) >= MIN_GROUP:
            groups.append(g)
            kept.append(lv)
    sizes_s = "; ".join(sizes)
    meds_s = "; ".join(meds)
    dropped = [lv for lv in levels if lv not in kept]
    note_full = note
    if dropped:
        note_full = (note + "; " if note else "") + \
            f"levels with < {MIN_GROUP} patients excluded from the test: {','.join(dropped)}"
    if len(groups) < 2:
        return rec(cohort=cohort, axis=axis, label=label, comparison=comparison,
                   test="Kruskal-Wallis (not run)", group_sizes=sizes_s,
                   group_medians=meds_s, tested=False,
                   in_family=in_family, note=note_full or "fewer than two testable groups")
    kw = stats.kruskal(*groups)
    tag = f"{cohort}|{axis}|{comparison}"
    eff, lo, hi = boot_epsilon_squared(groups, tag)
    return rec(cohort=cohort, axis=axis, label=label,
               comparison=comparison + " [" + ",".join(kept) + "]",
               test="Kruskal-Wallis", n_total=int(sum(len(g) for g in groups)),
               group_sizes=sizes_s, group_medians=meds_s,
               statistic=float(kw.statistic), p_value=float(kw.pvalue),
               effect_name="epsilon^2", effect=eff, ci_lo=lo, ci_hi=hi,
               effect_r=float(np.sqrt(max(eff, 0.0))), tested=True,
               in_family=in_family, note=note_full)


def test_spearman(values, covar, *, cohort, axis, label, comparison,
                  in_family=True, note=""):
    values = np.asarray(values, dtype=float)
    covar = np.asarray(covar, dtype=float)
    ok = np.isfinite(values) & np.isfinite(covar)
    x, y = values[ok], covar[ok]
    if len(x) < MIN_GROUP:
        return rec(cohort=cohort, axis=axis, label=label, comparison=comparison,
                   test="Spearman (not run)", n_total=len(x), tested=False,
                   in_family=in_family, note=(note + "; " if note else "") + "n < 5")
    sp = stats.spearmanr(x, y)
    tag = f"{cohort}|{axis}|{comparison}"
    eff, lo, hi = boot_spearman(x, y, tag)
    return rec(cohort=cohort, axis=axis, label=label, comparison=comparison,
               test="Spearman correlation", n_total=len(x), group_sizes=f"n={len(x)}",
               statistic=float(sp.statistic), p_value=float(sp.pvalue),
               effect_name="rho", effect=eff, ci_lo=lo, ci_hi=hi,
               effect_r=abs(eff), tested=True, in_family=in_family, note=note)


def bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values over the finite entries of `pvals`."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    fin = np.flatnonzero(np.isfinite(p))
    if fin.size == 0:
        return q
    order = fin[np.argsort(p[fin])]
    m = fin.size
    raw = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum(1.0, np.minimum.accumulate(raw[::-1])[::-1])
    return q
