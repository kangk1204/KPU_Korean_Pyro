#!/usr/bin/env python3
"""Supplementary Figure S4 - stage-definition sensitivity of recurrence concordance (180 x 60 mm).

A  5-year Uno concordance for clinical and combined ridge Cox models under three stage definitions.
B  paired combined-minus-clinical difference, on the same three rows as A.
"""
from __future__ import annotations

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

import figstyle as fs

SRC = fs.F124 / "Recurrence_Revision" / "source_data"
F_SUM = SRC / "S4_stage_plot_ready_summary.tsv"
F_DEL = SRC / "S4_stage_plot_ready_deltas.tsv"
F_AUDIT = SRC / "S4_stage_patient_audit.tsv"

W, H = 180.0, 60.0
AX_TOP, AX_H = 11.0, 27.0
A_AX_L, A_AX_W, A_TXT_R = 31.0, 53.0, 108.0
B_AX_L, B_AX_W, B_TXT_R = 114.0, 36.0, 178.0

STAGES = [("provider", "Recorded stage"), ("tnm", "TNM-derived stage"), ("nostage", "Stage omitted")]
MODELS = [("clinical", "Clinical", fs.C["clinical"], "s", -0.20),
          ("combined", "Clinical + methylation", fs.C["combined"], "o", +0.20)]
YLIM = (2.6, -0.6)


def ci_text(est, lo, hi, signed=False):
    if signed:
        return f"{est:.3f} ({lo:.3f} to {hi:.3f})".replace("-", "−")
    return f"{est:.3f} ({lo:.3f}–{hi:.3f})"


def make_axes(fig, left, width):
    ax = fs.ax_mm(fig, left, AX_TOP, width, AX_H)
    fs.light_grid(ax, "x")
    ax.set_ylim(*YLIM)
    for y in range(3):
        ax.axhline(y, color=fs.C["grid"], lw=0.3, zorder=0)
    ax.tick_params(axis="x", pad=1.5)
    fs.despine(ax, left=False, bottom=True)
    return ax


def text_column(fig, left, width, header):
    tax = fs.ax_mm(fig, left, AX_TOP, width, AX_H)
    tax.set_axis_off()
    tax.set_xlim(0, 1)
    tax.set_ylim(*YLIM)
    tax.text(1.0, YLIM[1], header, ha="right", va="bottom",
             fontsize=fs.FS["small"], color=fs.C["ink2"])
    return tax


def main():
    summ = pd.read_csv(F_SUM, sep="\t").set_index(["stage_definition", "model_block"])
    dels = pd.read_csv(F_DEL, sep="\t")
    dels = dels[dels.metric == "delta_uno_c"].set_index("stage_definition")
    audit = pd.read_csv(F_AUDIT, sep="\t")
    n_pat = int(audit.analysis_included.sum())
    n_evt = int(audit.loc[audit.analysis_included, "event"].sum())

    fig = fs.figure(W, H)

    # ---- panel A ---------------------------------------------------------
    fs.panel_label(fig, 2.0, 3.2, "A")
    fig.text(6.5 * fs.MM / (W * fs.MM), 1 - 4.4 * fs.MM / (H * fs.MM),
             f"{n_pat} patients with interpretable stage records, {n_evt} events",
             fontsize=fs.FS["small"], color=fs.C["ink2"], ha="left", va="top")

    ax = make_axes(fig, A_AX_L, A_AX_W)
    ax.set_xlim(0.35, 0.85)
    ax.xaxis.set_major_locator(FixedLocator([0.4, 0.5, 0.6, 0.7, 0.8]))
    ax.set_yticks(range(3))
    ax.set_yticklabels([s[1] for s in STAGES])
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.axvline(0.5, color=fs.C["rule"], lw=0.6, ls=(0, (2.5, 1.8)), zorder=1)
    ax.set_xlabel("Uno concordance through 5 years")

    tax = text_column(fig, A_AX_L + A_AX_W + 1.0, A_TXT_R - (A_AX_L + A_AX_W + 1.0),
                      "Uno C (95% CI)")
    for i, (stage, _lab) in enumerate(STAGES):
        for block, _mlab, col, mk, off in MODELS:
            r = summ.loc[(stage, block)]
            y = i + off
            ax.plot([r["ci_low.uno_c"], r["ci_high.uno_c"]], [y, y], color=col, lw=0.9,
                    solid_capstyle="butt", zorder=3)
            ax.plot([r["estimate.uno_c"]], [y], marker=mk, ms=3.0, mfc=col, mec="white",
                    mew=0.3, linestyle="none", zorder=4)
            tax.text(1.0, y, ci_text(r["estimate.uno_c"], r["ci_low.uno_c"], r["ci_high.uno_c"]),
                     ha="right", va="center", fontsize=fs.FS["small"], color=col)

    handles = [Line2D([], [], color=col, marker=mk, ms=3.0, mfc=col, mec="white", mew=0.3,
                      lw=0.9, label=mlab) for _b, mlab, col, mk, _o in MODELS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.30), ncol=2,
              fontsize=fs.FS["small"], borderaxespad=0, columnspacing=1.6, handletextpad=0.4)

    # ---- panel B ---------------------------------------------------------
    fs.panel_label(fig, 108.0, 3.2, "B")
    bx = make_axes(fig, B_AX_L, B_AX_W)
    bx.set_xlim(-0.25, 0.25)
    bx.xaxis.set_major_locator(FixedLocator([-0.2, -0.1, 0.0, 0.1, 0.2]))
    bx.set_yticks([])
    bx.axvline(0.0, color=fs.C["rule"], lw=0.6, ls=(0, (2.5, 1.8)), zorder=1)
    bx.set_xlabel("Combined − clinical difference in Uno C")

    btax = text_column(fig, B_AX_L + B_AX_W + 1.0, B_TXT_R - (B_AX_L + B_AX_W + 1.0),
                       "Difference (95% CI)")
    col = fs.C["combined"]
    for i, (stage, _lab) in enumerate(STAGES):
        r = dels.loc[stage]
        bx.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=0.9, solid_capstyle="butt", zorder=3)
        bx.plot([r.estimate], [i], marker="o", ms=3.0, mfc=col, mec="white", mew=0.3,
                linestyle="none", zorder=4)
        btax.text(1.0, i, ci_text(r.estimate, r.ci_low, r.ci_high, signed=True),
                  ha="right", va="center", fontsize=fs.FS["small"], color=fs.C["ink"])

    fig.text(2.0 * fs.MM / (W * fs.MM), 1 - (H - 2.4) * fs.MM / (H * fs.MM),
             "Points, nested-CV estimates from 25 repeats of fourfold cross-validation; bars, 95% "
             "patient bootstrap intervals; dashed lines, Uno C 0.5 (A) and zero (B).",
             fontsize=fs.FS["small"], color=fs.C["ink2"], ha="left", va="bottom")

    man = fs.save(fig, "Figure_S4", sources=[F_SUM, F_DEL, F_AUDIT],
                  notes="Stage-definition sensitivity: recorded (provider), TNM-derived (tnm) and "
                        "omitted (nostage) stage; clinical vs combined ridge Cox; 79 patients, "
                        "14 events (counted from S4_stage_patient_audit.tsv).")
    print(man["width_mm"], man["height_mm"])


if __name__ == "__main__":
    main()
