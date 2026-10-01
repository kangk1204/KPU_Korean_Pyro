#!/usr/bin/env python3
"""Summary figure for Part B: coordination axis vs the available labels.

Per cohort the coordination axis is the delta-PC1 where verified pairs exist and
the tumor-PC1 otherwise. Panels are strip + box, with the test statistic printed.
Style tokens come from 126_Figure_Redesign_20260909/figstyle.py (imported, not modified).
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KPU_PROJECT_ROOT", Path(__file__).resolve().parents[3]))  # project root; override with KPU_PROJECT_ROOT
sys.path.insert(0, str(ROOT / "126_Figure_Redesign_20260909"))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from coordination_axis_lib import sha256  # noqa: E402

OUTDIR = HERE.parent
RESULTS = OUTDIR / "results"
FIGDIR = OUTDIR / "figures"
SEED = 20260909

COORD = {  # cohort -> (score column of the coordination axis, short axis label)
    "Colonomics": ("delta_axis", "Colonomics\ndelta-PC1"),
    "GSE164811 (MATCH)": ("tumor_axis", "MATCH\ntumor-PC1"),
    "CMCBSN": ("delta_axis", "CMCBSN\ndelta-PC1"),
    "GSE48684": ("tumor_axis", "GSE48684\ntumor-PC1"),
    "GSE77718": ("delta_axis", "GSE77718\ndelta-PC1"),
    "TCGA-COADREAD": ("tumor_axis", "TCGA\ntumor-PC1"),
}

# (cohort, score column, label column, ordered levels, panel title, test key)
PANELS = [
    ("Colonomics", "delta_axis", "BRAF_V600E", ["No", "Yes"], "BRAF V600E",
     ("Colonomics", "delta_axis", "BRAF V600E mutant vs wild-type")),
    ("Colonomics", "delta_axis", "cms", ["CMS1", "CMS2", "CMS3", "CMS4"], "CMS",
     ("Colonomics", "delta_axis", "CMS four groups [CMS1,CMS2,CMS3,CMS4]")),
    ("Colonomics", "delta_axis", "site2", ["Left", "Right"], "Site",
     ("Colonomics", "delta_axis", "right- vs left-sided")),
    ("Colonomics", "delta_axis", "sex", ["F", "M"], "Sex",
     ("Colonomics", "delta_axis", "male vs female")),
    ("Colonomics", "delta_axis", "stromal_score", None, "Stromal score",
     ("Colonomics", "delta_axis", "ESTIMATE-style stromal score")),

    ("GSE164811 (MATCH)", "tumor_axis", "cms", ["CMS2", "CMS3"], "CMS",
     ("GSE164811", "tumor_axis", "CMS3 vs CMS2")),
    ("GSE164811 (MATCH)", "tumor_axis", "site2", ["Left", "Right"], "Site",
     ("GSE164811", "tumor_axis", "right- vs left-sided")),
    ("GSE164811 (MATCH)", "tumor_axis", "sex", ["F", "M"], "Sex",
     ("GSE164811", "tumor_axis", "male vs female")),
    ("CMCBSN", "delta_axis", "cms", ["CMS1", "CMS2", "CMS3", "CMS4"], "CMS",
     ("CMCBSN", "delta_axis", "CMS four groups [CMS1,CMS2,CMS3,CMS4]")),
    ("CMCBSN", "delta_axis", "msi2", ["MSS/MSI-L", "MSI-H"], "MSI",
     ("CMCBSN", "delta_axis", "MSI-H vs MSS/MSI-L")),

    ("CMCBSN", "delta_axis", "site2", ["Left", "Right"], "Site",
     ("CMCBSN", "delta_axis", "right- vs left-sided")),
    ("GSE48684", "tumor_axis", "site2", ["Left", "Right"], "Site",
     ("GSE48684", "tumor_axis", "right- vs left-sided")),
    ("GSE77718", "delta_axis", "site2", ["Left", "Right"], "Site",
     ("GSE77718", "delta_axis", "right- vs left-sided")),
    ("TCGA-COADREAD", "tumor_axis", "msi", ["MSI-negative", "MSI-positive"], "MSI",
     ("TCGA-COADREAD", "tumor_axis", "MSI-positive vs MSI-negative")),
    ("TCGA-COADREAD", "tumor_axis", "braf", ["BRAF normal", "BRAF abnormal"], "BRAF",
     ("TCGA-COADREAD", "tumor_axis", "BRAF abnormal vs normal")),

    ("TCGA-COADREAD", "tumor_axis", "site2", ["Left", "Right"], "Site",
     ("TCGA-COADREAD", "tumor_axis", "right- vs left-sided")),
    ("TCGA-COADREAD", "tumor_axis", "age", None, "Age (years)",
     ("TCGA-COADREAD", "tumor_axis", "age (years)")),
]

SHORT = {"Colonomics": "Colonomics", "GSE164811 (MATCH)": "MATCH", "CMCBSN": "CMCBSN",
         "GSE48684": "GSE48684", "GSE77718": "GSE77718", "TCGA-COADREAD": "TCGA"}

NCOL, NROW = 5, 4
LEFT0, TOP0, PW, PH, GX, GY = 13.0, 12.0, 27.0, 26.0, 7.0, 19.0
GROUP_FILL = [fs.C["box_fill"][i % len(fs.C["box_fill"])] for i in range(6)]
POINT = dict(s=2.2, linewidths=0, alpha=0.75, zorder=3)


def fmt_p(v: float) -> str:
    if not np.isfinite(v):
        return "n.s."
    if v < 1e-4:
        return f"P={v:.0e}".replace("e-0", "e-")
    return f"P={v:.3g}"


def stat_line(row: pd.Series | None) -> str:
    if row is None:
        return "not tested"
    if not bool(row["tested"]):
        return "n<5, descriptive"
    q = row["q_value"]
    qs = ("q<1e-4" if q < 1e-4 else f"q={q:.3g}") if np.isfinite(q) else "q=NA"
    if row["effect_name"] == "rho":
        return f"rho={row['effect']:+.2f}  {fmt_p(row['p_value'])}  {qs}"
    if row["effect_name"] == "epsilon^2":
        return f"H={row['statistic']:.1f}  {fmt_p(row['p_value'])}  {qs}"
    return f"{fmt_p(row['p_value'])}  {qs}"


def strip_box(ax, groups: list[np.ndarray], names: list[str]):
    rng = np.random.default_rng(SEED)
    for i, (g, nm) in enumerate(zip(groups, names)):
        x = i + 1
        if len(g):
            bp = ax.boxplot([g], positions=[x], widths=0.55, showfliers=False,
                            patch_artist=True, zorder=2)
            bp["boxes"][0].set(facecolor=GROUP_FILL[i], edgecolor=fs.C["box_edge"], lw=0.5)
            for key in ("whiskers", "caps"):
                for a in bp[key]:
                    a.set(color=fs.C["box_edge"], lw=0.5)
            bp["medians"][0].set(color=fs.C["ink"], lw=0.9)
            jitter = (rng.random(len(g)) - 0.5) * 0.32
            ax.scatter(x + jitter, g, color=fs.C["ink2"], **POINT)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([f"{n}\nn={len(g)}" for n, g in zip(names, groups)],
                       fontsize=fs.FS["tiny"])
    ax.set_xlim(0.4, len(names) + 0.6)


def main():
    scores = pd.read_csv(RESULTS / "B_axis_scores.tsv", sep="\t")
    tests = pd.read_csv(RESULTS / "B_axis_tests.tsv", sep="\t")
    tkey = tests.set_index(["cohort", "axis", "comparison"])

    height = TOP0 + NROW * PH + (NROW - 1) * GY + 26
    fig = fs.figure(fs.W_DOUBLE, height)

    letters = "abcdefghijklmnopqrstuvwxyz"
    for idx, (cohort, col, lab, levels, title, key) in enumerate(PANELS):
        r, c = divmod(idx, NCOL)
        left = LEFT0 + c * (PW + GX)
        top = TOP0 + r * (PH + GY)
        ax = fs.ax_mm(fig, left, top, PW, PH)
        fs.despine(ax)
        ax.tick_params(labelsize=fs.FS["tiny"], pad=1)

        sub = scores[(scores["cohort"] == cohort) &
                     (scores["axis_role"].str.startswith(col))]
        sub = sub[sub[col].notna()]
        row = tkey.loc[key] if key in tkey.index else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        if levels is None:  # continuous covariate -> scatter
            ok = sub[lab].notna()
            ax.scatter(sub.loc[ok, lab], sub.loc[ok, col], color=fs.C["ink2"], **POINT)
            ax.set_xlabel(f"{title}  (n={int(ok.sum())})", fontsize=fs.FS["tiny"], labelpad=1)
        else:
            groups = [sub.loc[sub[lab] == lv, col].to_numpy(float) for lv in levels]
            shown = [lv.replace("MSI-negative", "MSI−").replace("MSI-positive", "MSI+")
                     .replace("BRAF normal", "wt").replace("BRAF abnormal", "abn")
                     .replace("MSS/MSI-L", "MSS/L") for lv in levels]
            strip_box(ax, groups, shown)
            ax.set_xlabel(title, fontsize=fs.FS["tiny"], labelpad=1)

        if c == 0:
            ax.set_ylabel(COORD[cohort][1], fontsize=fs.FS["tiny"], labelpad=2)
        ax.set_title(f"{SHORT[cohort]} · {title}", fontsize=fs.FS["small"], pad=8,
                     loc="left", color=fs.C["ink"])
        ax.text(0.0, 1.02, stat_line(row), transform=ax.transAxes,
                fontsize=fs.FS["tiny"], color=fs.C["ink2"], ha="left", va="bottom")
        fs.panel_label(fig, left - 5.5, top - 7.5, letters[idx])

    caption = (
        "Coordination axis = PC1 of gene-standardised tumour-minus-normal deltas where verified pairs exist\n"
        "(Colonomics, CMCBSN, GSE77718), otherwise PC1 of z-scored tumour-only beta (MATCH/GSE164811, GSE48684,\n"
        "TCGA-COADREAD). Every axis is oriented so that higher = more methylation. P from two-sided Wilcoxon rank-sum,\n"
        "Kruskal-Wallis or Spearman; q = Benjamini-Hochberg within cohort. Groups with fewer than five patients are\n"
        "shown but not tested. Exploratory analysis; the 77 candidate CpGs were selected on tumour-vs-normal contrasts."
    )
    fig.text(LEFT0 * fs.MM / fig.get_size_inches()[0],
             1 - (height - 3.0) * fs.MM / fig.get_size_inches()[1],
             caption, fontsize=fs.FS["tiny"], color=fs.C["ink2"], ha="left", va="bottom",
             linespacing=1.6)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGDIR / f"B_coordination_axis_labels.{ext}", dpi=600,
                    bbox_inches=None, pad_inches=0)
    plt.close(fig)

    man = {
        "figure": "B_coordination_axis_labels",
        "width_mm": fs.W_DOUBLE, "height_mm": round(height, 1), "seed": SEED,
        "style": str(ROOT / "126_Figure_Redesign_20260909" / "figstyle.py"),
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in
                    [RESULTS / "B_axis_scores.tsv", RESULTS / "B_axis_tests.tsv"]],
        "outputs": {ext: sha256(FIGDIR / f"B_coordination_axis_labels.{ext}")
                    for ext in ("pdf", "png", "svg")},
    }
    (FIGDIR / "B_coordination_axis_labels_manifest.json").write_text(
        json.dumps(man, indent=2) + "\n")
    print("wrote", FIGDIR / "B_coordination_axis_labels.pdf")


if __name__ == "__main__":
    main()
