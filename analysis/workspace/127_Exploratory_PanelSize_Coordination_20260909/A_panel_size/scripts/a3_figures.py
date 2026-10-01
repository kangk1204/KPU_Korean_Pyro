#!/usr/bin/env python3
"""Two summary figures for Part A, drawn with the 126 figure-style tokens.

figstyle.py is imported read-only (its own save() targets 126/output, so this script writes
through a local save() into A_panel_size/figures/).
"""
from __future__ import annotations
import os
import pathlib

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT = pathlib.Path(os.environ.get("KPU_PROJECT_ROOT", pathlib.Path(__file__).resolve().parents[3]))
_FIGSTYLE = os.environ.get("KPU_FIGSTYLE_DIR") or next((str(_PROJECT / d) for d in ("126_Figure_Redesign_20260909", "129_Submission_Revised_20260909/figures_redesign") if (_PROJECT / d / "figstyle.py").is_file()), str(_PROJECT / "126_Figure_Redesign_20260909"))
sys.path.insert(0, _FIGSTYLE)  # figstyle.py location; override with KPU_FIGSTYLE_DIR
sys.path.insert(0, str(Path(__file__).resolve().parent))

import figstyle as FS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES, RESULTS, sha256, write_json  # noqa: E402

MANUSCRIPT_SINGLE = 0.943
MANUSCRIPT_FULL = 0.968

TARGET_ORDER = ["CMCBSN", "SNUH", "ASAN", "GSE119526", "GSE193535", "GSE77718", "GSE42752"]
TARGET_COLOR = {
    "CMCBSN": FS.C["cohort"]["CMCBSN"],
    "SNUH": FS.C["cohort"]["SNUH"],
    "ASAN": FS.C["cohort"]["ASAN"],
    "GSE119526": "#CC79A7",
    "GSE193535": "#56B4E9",
    "GSE77718": "#D55E00",
    "GSE42752": "#444444",
}


def save(fig, name: str, sources: list[Path], notes: str = "") -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=600, pad_inches=0)
    w, h = fig.get_size_inches()
    write_json(
        FIGURES / f"{name}_manifest.json",
        {
            "figure": name,
            "width_mm": round(w / FS.MM, 1),
            "height_mm": round(h / FS.MM, 1),
            "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "style_module": "126_Figure_Redesign_20260909/figstyle.py",
            "sources": [{"path": str(p), "sha256": sha256(p)} for p in sources],
            "notes": notes,
            "outputs": {e: sha256(FIGURES / f"{name}.{e}") for e in ("pdf", "png")},
        },
    )
    plt.close(fig)


def figure_a1() -> None:
    src = RESULTS / "a1_per_k_summary.tsv"
    d = pd.read_csv(src, sep="\t")
    fig = FS.figure(88, 72)
    ax = FS.ax_mm(fig, 14, 7, 70, 54)
    k = d["k"].to_numpy()

    band = ax.fill_between(k, d["min_auc"], d["max_auc"], color=FS.C["local"], alpha=0.16,
                           linewidth=0, zorder=1)
    (line,) = ax.plot(k, d["median_auc"], color=FS.C["local"], lw=1.0, zorder=3)
    bars = ax.errorbar(
        k, d["nested_mean_auc"],
        yerr=np.vstack([d["nested_mean_auc"] - d["ci_lo"], d["ci_hi"] - d["nested_mean_auc"]]),
        fmt="o", ms=3, mfc="white", mec=FS.C["ink"], ecolor=FS.C["ink2"], elinewidth=0.6,
        capsize=1.4, capthick=0.6, lw=0, zorder=4,
    )
    for val, txt, ls, xy, va, ha in [
        (MANUSCRIPT_FULL, "manuscript full panel 0.968", (0, (4, 2)), 0.75, "bottom", "left"),
        (MANUSCRIPT_SINGLE, "manuscript single gene 0.943", (0, (1.2, 1.6)), 10.3, "top", "right"),
    ]:
        ax.axhline(val, color=FS.C["rule"], lw=0.5, ls=ls, zorder=2)
        off = 0.0025 if va == "bottom" else -0.0025
        ax.text(xy, val + off, txt, fontsize=FS.FS["tiny"], color=FS.C["muted"], va=va, ha=ha)

    ax.set_xlim(0.6, 10.4)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.80, 1.005)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Genes in panel ($k$)")
    ax.set_ylabel("Tissue-discrimination AUC")
    FS.despine(ax)
    FS.light_grid(ax, axis="y")
    ax.legend([(band, line), bars],
              ["median across the $\\mathregular{C}$(10,$k$) subsets (band: min\u2013max)",
               "nested best-$k$ (95% patient-cluster CI)"],
              loc="lower right", fontsize=FS.FS["tiny"], borderpad=0.2, labelspacing=0.35,
              handlelength=1.8)
    FS.panel_label(fig, 2, 3, "a")
    save(fig, "figA1_local_panel_size", [src],
         "Local pyrosequencing cohort (87 patients, 174 specimens); patient-grouped 5-fold CV x 20 repeats.")


def figure_a2() -> None:
    s_per_k = RESULTS / "a2_per_k_by_target.tsv"
    s_frozen = RESULTS / "a2_frozen_transfer.tsv"
    s_loco = RESULTS / "a2_leave_one_cohort_out.tsv"
    per_k = pd.read_csv(s_per_k, sep="\t")
    frozen = pd.read_csv(s_frozen, sep="\t")
    frozen = frozen.loc[frozen["model"] == "full_10gene"].set_index("target")
    loco = pd.read_csv(s_loco, sep="\t")

    fig = FS.figure(180, 96)
    ax = FS.ax_mm(fig, 14, 8, 72, 66)
    handles = []
    for t in TARGET_ORDER:
        sub = per_k.loc[per_k["target"] == t].sort_values("k")
        (ln,) = ax.plot(sub["k"], sub["median_auc"], color=TARGET_COLOR[t], lw=0.9, zorder=3)
        handles.append(ln)
        row = frozen.loc[t]
        ax.errorbar(
            10.4, row["auc"], yerr=[[row["auc"] - row["ci_lo"]], [row["ci_hi"] - row["auc"]]],
            fmt="D", ms=2.6, mfc="white", mec=TARGET_COLOR[t], ecolor=TARGET_COLOR[t],
            elinewidth=0.6, capsize=1.2, capthick=0.5, lw=0, zorder=4,
        )
    ax.axhline(0.90, color=FS.C["rule"], lw=0.5, ls=(0, (4, 2)), zorder=2)
    ax.text(10.55, 0.90, "0.90", fontsize=FS.FS["tiny"], color=FS.C["muted"], va="center", ha="left")
    ax.set_xlim(0.6, 10.9)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.74, 1.008)
    ax.set_xlabel("Genes in panel ($k$)")
    ax.set_ylabel("External AUC (frozen Colonomics model)")
    FS.despine(ax)
    FS.light_grid(ax, axis="y")
    leg = ax.legend(handles, TARGET_ORDER, loc="lower right", ncol=2, fontsize=FS.FS["tiny"],
                    borderpad=0.2, labelspacing=0.3, columnspacing=0.9, handlelength=1.4)
    leg.set_zorder(6)
    ax.text(0.015, 0.02,
            "lines: median across gene subsets\ndiamonds: full model (95% CI)\n"
            "dashed: transfer threshold 0.90",
            transform=ax.transAxes, fontsize=FS.FS["tiny"], color=FS.C["ink2"], va="bottom")
    FS.panel_label(fig, 2, 4, "a")

    trains = ["Colonomics", "CMCBSN", "SNUH", "ASAN"]
    cols = ["Colonomics"] + TARGET_ORDER
    mat = np.full((len(trains), len(cols)), np.nan)
    for i, tr in enumerate(trains):
        for j, tg in enumerate(cols):
            v = loco.loc[(loco["train"] == tr) & (loco["target"] == tg), "auc"]
            if len(v):
                mat[i, j] = v.iloc[0]

    axh = FS.ax_mm(fig, 118, 12, 56, 26)
    norm = plt.Normalize(vmin=0.85, vmax=1.0)
    axh.imshow(np.ma.masked_invalid(mat), cmap=FS.CMAP_SEQ, norm=norm, aspect="auto")
    for i in range(len(trains)):
        for j in range(len(cols)):
            if np.isnan(mat[i, j]):
                FS.na_cell(axh, j, i)
            else:
                axh.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                         fontsize=FS.FS["tiny"],
                         color="white" if mat[i, j] > 0.955 else FS.C["ink"])
    axh.set_xticks(range(len(cols)))
    axh.set_xticklabels(cols, rotation=45, ha="right", fontsize=FS.FS["tiny"])
    axh.set_yticks(range(len(trains)))
    axh.set_yticklabels(trains, fontsize=FS.FS["tiny"])
    axh.set_ylabel("Training cohort", fontsize=FS.FS["small"], labelpad=2)
    axh.set_xlabel("Target cohort", fontsize=FS.FS["small"], labelpad=2)
    axh.xaxis.set_label_position("top")
    axh.tick_params(length=0)
    for sp in ("top", "right", "left", "bottom"):
        axh.spines[sp].set_visible(False)
    FS.colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=FS.CMAP_SEQ), 118, 62, 56, 2.6,
                "Specimen-level AUC", ticks=[0.85, 0.9, 0.95, 1.0])
    w, h = fig.get_size_inches()
    fig.text(118 * FS.MM / w, 1 - 74 * FS.MM / h,
             "Leave-one-cohort-out, full model with the Korean-measured CpGs\n"
             "restricted to those present in the target; self cells (\u00d7) not estimated.",
             fontsize=FS.FS["tiny"], color=FS.C["ink2"], va="top")
    FS.panel_label(fig, 96, 4, "b")
    save(fig, "figA2_external_transfer", [s_per_k, s_frozen, s_loco],
         "Frozen models trained on the named cohort with the Korean-measured CpGs restricted per target.")


if __name__ == "__main__":
    figure_a1()
    figure_a2()
    print("figures written to", FIGURES)
