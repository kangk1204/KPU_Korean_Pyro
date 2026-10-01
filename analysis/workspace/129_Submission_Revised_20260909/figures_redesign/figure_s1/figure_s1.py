#!/usr/bin/env python3
"""Figure S1 - panel-size and external-transfer summary (three panels, 180 x 150 mm).

Reads the supplied Part-A result tables and preserves the shared design system.
KPU_S1_RESULTS, KPU_FIGSTYLE_DIR and KPU_S1_OUT can override the input, style and
output directories. A local save() writes Figure_S1.pdf, Figure_S1.svg,
Figure_S1.png (600 dpi), and input hashes.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
PROJECT = Path(os.environ.get("KPU_PROJECT_ROOT", Path(__file__).resolve().parents[2]))  # folder containing the numbered analysis folders
FIGSTYLE_DIR = Path(os.environ.get("KPU_FIGSTYLE_DIR", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(FIGSTYLE_DIR))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Verify the unchanged shared design system before creating any output.
_REQUIRED_STYLE = {"font.family": ["sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42}
for _setting, _value in _REQUIRED_STYLE.items():
    if plt.rcParams[_setting] != _value:
        raise ValueError(f"Shared figure style must set {_setting} to {_value!r}")

SRC = Path(os.environ.get("KPU_127_DIR", PROJECT / "127_Exploratory_PanelSize_Coordination_20260909"))
A = Path(os.environ.get("KPU_S1_RESULTS", SRC / "A_panel_size" / "results"))
OUT = Path(os.environ.get("KPU_S1_OUT", Path(__file__).resolve().parents[1] / "figures"))

MANUSCRIPT_SINGLE = 0.943
MANUSCRIPT_FULL = 0.968
TRANSFER_REF = 0.90

TARGET_ORDER = ["CMCBSN", "SNUH", "ASAN", "GSE119526", "GSE193535", "GSE77718", "GSE42752"]
TRAIN_ORDER = ["Colonomics", "CMCBSN", "SNUH", "ASAN"]
COL_ORDER = ["Colonomics"] + TARGET_ORDER

# Korean arrays keep their fixed entity colours; the four public GSE cohorts use the
# neutral grey tokens of the design system, separated additionally by dash pattern.
TARGET_STYLE = {
    "CMCBSN":    (fs.C["cohort"]["CMCBSN"], "solid"),
    "SNUH":      (fs.C["cohort"]["SNUH"],   "solid"),
    "ASAN":      (fs.C["cohort"]["ASAN"],   "solid"),
    "GSE119526": (fs.C["ink"],     "solid"),
    "GSE193535": (fs.C["ink2"],    (0, (3.5, 1.5))),
    "GSE77718":  (fs.C["muted"],   "solid"),
    "GSE42752":  (fs.C["na_mark"], (0, (1.3, 1.3))),
}

SMALL = fs.FS["small"]   # 6 pt annotations
BODY = fs.FS["body"]     # 7 pt text



def _rel(p: Path) -> str:
    """Record source paths relative to PROJECT so the manifest carries no machine-specific prefix."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(PROJECT.resolve()))
    except ValueError:
        return p.name

def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, name: str, sources: list[Path], notes: str = "") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, pad_inches=0,
                    metadata=None if ext == "png" else {"Title": name})
    w, h = fig.get_size_inches()
    man = {
        "figure": name,
        "width_mm": round(w / fs.MM, 1),
        "height_mm": round(h / fs.MM, 1),
        "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "style_module": _rel(Path(fs.__file__)),
        "style_sha256": sha256(Path(fs.__file__)),
        "plotting_script_sha256": sha256(Path(__file__)),
        "sources": [{"path": _rel(p), "sha256": sha256(p)} for p in sources],
        "notes": notes,
        "outputs": {e: sha256(OUT / f"{name}.{e}") for e in ("pdf", "svg", "png")},
    }
    (OUT / f"{name}_manifest.json").write_text(json.dumps(man, indent=1))
    plt.close(fig)
    return man


def build():
    """Draw the figure and return (fig, sources, registry of drawn values) for QA."""
    s_perk = A / "a1_per_k_summary.tsv"
    s_bytarget = A / "a2_per_k_by_target.tsv"
    s_frozen = A / "a2_frozen_transfer.tsv"
    s_loco = A / "a2_loco_matrix.tsv"
    s_locolong = A / "a2_leave_one_cohort_out.tsv"
    s_cov = A / "a2_cohort_coverage.tsv"

    d = pd.read_csv(s_perk, sep="\t")
    per_k = pd.read_csv(s_bytarget, sep="\t")
    frozen = pd.read_csv(s_frozen, sep="\t")
    frozen = frozen.loc[frozen["model"] == "full_10gene"].set_index("target")
    loco = pd.read_csv(s_loco, sep="\t").set_index("train")

    fig = fs.figure(180, 150)

    # ----------------------------------------------------------------- panel a
    ax = fs.ax_mm(fig, 15, 11, 67, 54)
    k = d["k"].to_numpy()
    band = ax.fill_between(k, d["min_auc"], d["max_auc"], color=fs.C["local"], alpha=0.16,
                           linewidth=0, zorder=1)
    (line,) = ax.plot(k, d["median_auc"], color=fs.C["local"], lw=1.0, zorder=3)
    bars = ax.errorbar(
        k, d["nested_mean_auc"],
        yerr=np.vstack([d["nested_mean_auc"] - d["ci_lo"], d["ci_hi"] - d["nested_mean_auc"]]),
        fmt="o", ms=2.8, mfc="white", mec=fs.C["ink"], ecolor=fs.C["ink2"], elinewidth=0.6,
        capsize=1.3, capthick=0.6, lw=0, zorder=4,
    )
    ref_full = ax.axhline(MANUSCRIPT_FULL, color=fs.C["rule"], lw=0.6, ls=(0, (4, 2)), zorder=2)
    ref_single = ax.axhline(MANUSCRIPT_SINGLE, color=fs.C["rule"], lw=0.6, ls=(0, (1.4, 1.6)), zorder=2)
    ax.set_xlim(0.55, 10.45)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.80, 1.002)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Genes in panel ($k$)", fontsize=BODY)
    ax.set_ylabel("Tissue-discrimination AUC", fontsize=BODY)
    fs.despine(ax)
    fs.light_grid(ax, axis="y")
    leg = ax.legend([(band, line), bars, ref_full, ref_single],
                    ["median across C(10,$k$) subsets (min–max band)",
                     "nested best-$k$ (95% patient-cluster CI)",
                     "manuscript full panel (0.968)",
                     "manuscript single gene (0.943)"],
                    loc="lower right", fontsize=SMALL, borderpad=0.2, labelspacing=0.35,
                    handlelength=1.7, borderaxespad=0.25)
    leg.set_zorder(6)
    fs.panel_label(fig, 2.5, 3.5, "A")

    # ----------------------------------------------------------------- panel b
    axb = fs.ax_mm(fig, 104, 11, 67, 54)
    handles, labels, b_markers = [], [], {}
    for i, t in enumerate(TARGET_ORDER):
        col, ls = TARGET_STYLE[t]
        lw = 1.05 if t in ("GSE42752", "GSE77718") else 0.9
        sub = per_k.loc[per_k["target"] == t].sort_values("k")
        (ln,) = axb.plot(sub["k"], sub["median_auc"], color=col, lw=lw, ls=ls, zorder=3)
        handles.append(ln)
        labels.append(t)
        row = frozen.loc[t]
        # frozen full-panel variant estimates all refer to k = 10; the x positions are staggered
        # only so that the seven markers and their CIs remain individually readable.
        xm = 10.34 + 0.10 * i
        axb.errorbar(xm, row["auc"],
                     yerr=[[row["auc"] - row["ci_lo"]], [row["ci_hi"] - row["auc"]]],
                     fmt="D", ms=2.4, mfc="white", mec=col, ecolor=col,
                     elinewidth=0.6, capsize=1.1, capthick=0.5, lw=0, zorder=4)
        b_markers[t] = {"x": xm, "auc": float(row["auc"]),
                        "ci": [float(row["ci_lo"]), float(row["ci_hi"])]}
    axb.axhline(TRANSFER_REF, color=fs.C["rule"], lw=0.5, ls=(0, (4, 2)), zorder=2)
    axb.text(0.70, TRANSFER_REF - 0.004, "0.90", fontsize=SMALL, color=fs.C["muted"],
             va="top", ha="left")
    axb.set_xlim(0.55, 11.15)
    axb.set_xticks(range(1, 11))
    axb.set_ylim(0.75, 1.003)
    axb.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    axb.set_xlabel("Genes in panel ($k$)", fontsize=BODY)
    axb.set_ylabel("External AUC (models fitted in Colonomics)", fontsize=BODY)
    fs.despine(axb)
    fs.light_grid(axb, axis="y")
    legb = axb.legend(handles, labels, loc="lower right", ncol=1, fontsize=SMALL,
                      borderpad=0.2, labelspacing=0.28, handlelength=1.7, borderaxespad=0.25)
    legb.set_zorder(6)
    axb.text(0.012, 0.015,
             "lines: median across C(10,$k$) subsets\n"
             "diamonds: 10-gene models at $k$=10 (95% CI)\n"
             "dashed: transfer reference 0.90",
             transform=axb.transAxes, fontsize=SMALL, color=fs.C["ink2"], va="bottom",
             ha="left", linespacing=1.35)
    fs.panel_label(fig, 91, 3.5, "B")

    # ----------------------------------------------------------------- panel c
    mat = np.full((len(TRAIN_ORDER), len(COL_ORDER)), np.nan)
    for i, tr in enumerate(TRAIN_ORDER):
        for j, tg in enumerate(COL_ORDER):
            v = loco.loc[tr, tg]
            mat[i, j] = float(v) if pd.notna(v) else np.nan

    c_text = []
    axh = fs.ax_mm(fig, 27, 86, 117, 32)
    norm = plt.Normalize(vmin=0.85, vmax=1.00)
    axh.imshow(np.ma.masked_invalid(mat), cmap=fs.CMAP_SEQ, norm=norm, aspect="auto")
    for i in range(len(TRAIN_ORDER)):
        for j in range(len(COL_ORDER)):
            if np.isnan(mat[i, j]):
                fs.na_cell(axh, j, i)
                axh.texts[-1].set_fontsize(SMALL)
                c_text.append((i, j, axh.texts[-1].get_text()))
            else:
                t_ = axh.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=SMALL,
                              color="white" if mat[i, j] > 0.955 else fs.C["ink"])
                c_text.append((i, j, t_.get_text()))
    axh.set_xticks(range(len(COL_ORDER)))
    axh.set_xticklabels(COL_ORDER, rotation=45, ha="right", fontsize=SMALL)
    axh.set_yticks(range(len(TRAIN_ORDER)))
    axh.set_yticklabels(TRAIN_ORDER, fontsize=SMALL)
    axh.set_ylabel("Training cohort", fontsize=BODY, labelpad=2)
    axh.set_xlabel("Target cohort", fontsize=BODY, labelpad=3)
    axh.xaxis.set_label_position("top")
    axh.tick_params(length=0, pad=1.5)
    for sp in ("top", "right", "left", "bottom"):
        axh.spines[sp].set_visible(False)
    fs.colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=fs.CMAP_SEQ), 27, 138, 52, 2.6,
                "Specimen-level AUC", ticks=[0.85, 0.90, 0.95, 1.00])
    fs.panel_label(fig, 2.5, 78, "C")
    w, h = fig.get_size_inches()
    fig.text(92 * fs.MM / w, 1 - 137 * fs.MM / h,
             "Cross-cohort evaluation: models were fitted in each training cohort using the\n"
             "candidate CpGs measured in the target cohort, then applied without refitting.\n"
             "Grey × = self cell (not estimated).",
             fontsize=SMALL, color=fs.C["ink2"], va="top", ha="left", linespacing=1.35)

    reg = {
        "a_line": {"x": list(line.get_xdata()), "y": list(line.get_ydata())},
        "a_band_lo": list(d["min_auc"]), "a_band_hi": list(d["max_auc"]),
        "a_points": {"x": list(bars[0].get_xdata()), "y": list(bars[0].get_ydata())},
        "a_ci": [[float(v) for v in seg[:, 1]] for seg in bars[2][0].get_segments()],
        "a_ref": [MANUSCRIPT_SINGLE, MANUSCRIPT_FULL],
        "a_ylim": list(ax.get_ylim()),
        "b_lines": {t: list(h.get_ydata()) for t, h in zip(labels, handles)},
        "b_lines_x": {t: list(h.get_xdata()) for t, h in zip(labels, handles)},
        "b_markers": b_markers,
        "b_ref": TRANSFER_REF, "b_ylim": list(axb.get_ylim()),
        "c_matrix": mat.tolist(), "c_rows": TRAIN_ORDER, "c_cols": COL_ORDER,
        "c_text": {f"{i},{j}": t for (i, j, t) in c_text},
        "c_norm": [norm.vmin, norm.vmax],
    }
    sources = [s_perk, s_bytarget, s_frozen, s_loco, s_locolong, s_cov]
    return fig, sources, reg


def main() -> None:
    fig, sources, _ = build()
    man = save(fig, "Figure_S1", sources,
               "a: local pyrosequencing cohort, patient-grouped 5-fold CV x 20 repeats, all C(10,k) "
               "gene subsets. b: models fitted in Colonomics using the candidate CpGs measured in each "
               "target, then applied without refitting. c: cross-cohort evaluation, with models "
               "fitted in each row cohort using the candidate CpGs measured in each target. "
               "2,000-resample patient-cluster "
               "bootstrap CIs throughout.")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
