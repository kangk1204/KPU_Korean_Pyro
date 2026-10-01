#!/usr/bin/env python3
"""Supplementary Figure S3 - recurrence-prediction concordance (180 x 50 mm).

A  5-year Uno concordance for the clinical, tumor-methylation and combined ridge Cox blocks.
B  paired combined-minus-clinical difference, drawn on the same row grid as A.
"""
from __future__ import annotations

import pandas as pd
from matplotlib.ticker import FixedLocator

import figstyle as fs

SRC = fs.F124 / "Recurrence_Revision" / "source_data"
F_MAIN = SRC / "S3_table_s3_recurrence_primary_corrected.tsv"
F_DELTA = SRC / "S3_table_s3_delta_ci.tsv"
F_SUPPORT = SRC / "S3_horizon_support_counts.tsv"

W, H = 180.0, 50.0
AX_TOP, AX_H = 9.5, 21.0
ROWS = [("clinical", "Clinical", fs.C["clinical"]),
        ("tumor10", "Tumor methylation", fs.C["methylation"]),
        ("combined", "Clinical + methylation", fs.C["combined"])]

A_AX_L, A_AX_W, A_TXT_R = 29.0, 55.0, 106.0
B_AX_L, B_AX_W, B_TXT_R = 113.0, 38.0, 178.0


def ci_text(est, lo, hi, dp=3, signed=False):
    if signed:
        return f"{est:.3f} ({lo:.3f} to {hi:.3f})".replace("-", "−")
    return f"{est:.{dp}f} ({lo:.{dp}f}–{hi:.{dp}f})"


def text_column(fig, left, width, header):
    tax = fs.ax_mm(fig, left, AX_TOP, width, AX_H)
    tax.set_axis_off()
    tax.set_xlim(0, 1)
    tax.set_ylim(2.6, -0.6)
    tax.text(1.0, -0.6, header, ha="right", va="bottom",
             fontsize=fs.FS["small"], color=fs.C["ink2"])
    return tax


def main():
    main_df = pd.read_csv(F_MAIN, sep="\t").set_index("block")
    delta = pd.read_csv(F_DELTA, sep="\t")
    delta = delta[delta.metric == "delta_uno_c"].iloc[0]
    sup = pd.read_csv(F_SUPPORT, sep="\t")
    sup = sup[sup.horizon_days == 1825].iloc[0]

    fig = fs.figure(W, H)

    # ---- panel A ---------------------------------------------------------
    fs.panel_label(fig, 2.0, 3.0, "A")
    ax = fs.ax_mm(fig, A_AX_L, AX_TOP, A_AX_W, AX_H)
    fs.light_grid(ax, "x")
    ax.set_xlim(0.35, 0.85)
    ax.xaxis.set_major_locator(FixedLocator([0.4, 0.5, 0.6, 0.7, 0.8]))
    ax.set_ylim(2.6, -0.6)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r[1] for r in ROWS])
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", pad=1.5)
    fs.despine(ax, left=False, bottom=True)
    for y in range(3):
        ax.axhline(y, color=fs.C["grid"], lw=0.3, zorder=0)
    ax.axvline(0.5, color=fs.C["rule"], lw=0.6, ls=(0, (2.5, 1.8)), zorder=1)
    ax.set_xlabel("Uno concordance through 5 years")

    tax = text_column(fig, A_AX_L + A_AX_W + 1.0, A_TXT_R - (A_AX_L + A_AX_W + 1.0),
                      "Uno C (95% CI)")
    for i, (block, _lab, col) in enumerate(ROWS):
        r = main_df.loc[block]
        ax.plot([r.uno_c_5y_ci_low, r.uno_c_5y_ci_high], [i, i], color=col, lw=0.9,
                solid_capstyle="butt", zorder=3)
        ax.plot([r.uno_c_5y], [i], marker="o", ms=3.2, mfc=col, mec="white", mew=0.3,
                linestyle="none", zorder=4)
        tax.text(1.0, i, ci_text(r.uno_c_5y, r.uno_c_5y_ci_low, r.uno_c_5y_ci_high),
                 ha="right", va="center", fontsize=fs.FS["small"], color=fs.C["ink"])

    # ---- panel B ---------------------------------------------------------
    fs.panel_label(fig, 108.0, 3.0, "B")
    bx = fs.ax_mm(fig, B_AX_L, AX_TOP, B_AX_W, AX_H)
    fs.light_grid(bx, "x")
    bx.set_xlim(-0.25, 0.25)
    bx.xaxis.set_major_locator(FixedLocator([-0.2, -0.1, 0.0, 0.1, 0.2]))
    bx.set_ylim(2.6, -0.6)
    bx.set_yticks([])
    bx.tick_params(axis="x", pad=1.5)
    fs.despine(bx, left=False, bottom=True)
    for y in range(3):
        bx.axhline(y, color=fs.C["grid"], lw=0.3, zorder=0)
    bx.axvline(0.0, color=fs.C["rule"], lw=0.6, ls=(0, (2.5, 1.8)), zorder=1)
    bx.set_xlabel("Combined − clinical difference in Uno C")

    col = fs.C["combined"]
    bx.plot([delta.ci_low, delta.ci_high], [2, 2], color=col, lw=0.9,
            solid_capstyle="butt", zorder=3)
    bx.plot([delta.estimate], [2], marker="o", ms=3.2, mfc=col, mec="white", mew=0.3,
            linestyle="none", zorder=4)
    btax = text_column(fig, B_AX_L + B_AX_W + 1.0, B_TXT_R - (B_AX_L + B_AX_W + 1.0),
                       "Difference (95% CI)")
    btax.text(1.0, 2, ci_text(delta.estimate, delta.ci_low, delta.ci_high, signed=True),
              ha="right", va="center", fontsize=fs.FS["small"], color=fs.C["ink"])
    note1 = (f"{int(main_df.loc['clinical'].n_patients)} patients, "
             f"{int(main_df.loc['clinical'].total_events)} events; "
             f"{int(sup.events_by_horizon)} events by 5 years, "
             f"{int(sup.controls_at_horizon)} observed beyond, "
             f"{int(sup.censored_before_horizon_unknown_status)} censored earlier.")
    note2 = ("Points, nested-CV estimates; bars, 95% bootstrap intervals; dashed lines, "
             "Uno C 0.5 (A) and zero (B); B is drawn on the combined row of A.")
    for dy, txt in ((6.4, note1), (2.4, note2)):
        fig.text(2.0 * fs.MM / (W * fs.MM), 1 - (H - dy) * fs.MM / (H * fs.MM), txt,
                 fontsize=fs.FS["small"], color=fs.C["ink2"], ha="left", va="bottom")

    man = fs.save(fig, "Figure_S3", sources=[F_MAIN, F_DELTA, F_SUPPORT],
                  notes="Uno C through 5 years for clinical / tumor methylation / combined "
                        "ridge Cox blocks and the paired combined-minus-clinical difference; "
                        "82 patients, 14 events.")
    print(man["width_mm"], man["height_mm"])


if __name__ == "__main__":
    main()
