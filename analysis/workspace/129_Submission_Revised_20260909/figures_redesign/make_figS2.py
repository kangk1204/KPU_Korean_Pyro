#!/usr/bin/env python3
"""Supplementary Figure S2 - single-gene recurrence Cox associations (180 x 70 mm).

Two forest panels sharing the gene axis:
  A  primary stage I-III non-palliative set (82 patients, 14 events)
  B  all-stage recurrence-or-progression sensitivity set (87 patients, 17 events)
"""
from __future__ import annotations

import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter

import figstyle as fs

SRC = fs.F124 / "Recurrence_Revision" / "source_data"
F_PRIMARY = SRC / "S2_cox_primary_source_data.tsv"
F_ALL = SRC / "S2_cox_all_stage_source_data.tsv"

W, H = 180.0, 70.0
XLIM = (0.4, 1.7)
XTICKS = [0.5, 0.75, 1.0, 1.5]
XTICKLABELS = ["0.5", "0.75", "1", "1.5"]

# panel geometry in mm (left edge, axes left, axes width, estimate-column right edge)
PANELS = [
    dict(letter="A", label_x=15.0, ax_left=16.0, ax_w=52.0, txt_right=89.0,
         file=F_PRIMARY, color=fs.C["methylation"],
         header="Primary set: 82 patients, 14 events"),
    dict(letter="B", label_x=104.0, ax_left=105.0, ax_w=52.0, txt_right=178.0,
         file=F_ALL, color=fs.C["methylation"],
         header="All-stage sensitivity set: 87 patients, 17 events"),
]
AX_TOP, AX_H = 9.5, 44.0


def fmt(hr, lo, hi):
    return f"{hr:.2f} ({lo:.2f}–{hi:.2f})"


def load(path):
    df = pd.read_csv(path, sep="\t")
    df = df.set_index("gene").loc[fs.GENES].reset_index()
    return df


def draw_panel(fig, cfg):
    df = load(cfg["file"])
    ax = fs.ax_mm(fig, cfg["ax_left"], AX_TOP, cfg["ax_w"], AX_H)
    fs.light_grid(ax, "x")
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_major_formatter(FixedFormatter(XTICKLABELS))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(9.5, -0.5)
    ax.set_yticks(range(10))
    ax.set_yticklabels(fs.GENES, fontstyle="italic")
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", pad=1.5)
    fs.despine(ax, left=False, bottom=True)
    ax.axvline(1.0, color=fs.C["rule"], lw=0.6, ls=(0, (2.5, 1.8)), zorder=1)

    for i, r in df.iterrows():
        ax.plot([r.HR_ci_low, r.HR_ci_high], [i, i], color=cfg["color"], lw=0.9,
                solid_capstyle="butt", zorder=3)
        ax.plot([r.HR_per_10_pctpt], [i], marker="o", ms=3.0, mfc=cfg["color"],
                mec="white", mew=0.3, linestyle="none", zorder=4)
    ax.set_xlabel("Hazard ratio per 10 percentage points (log scale)")

    # dedicated right-hand estimate column
    tx_left = cfg["ax_left"] + cfg["ax_w"] + 1.0
    tax = fs.ax_mm(fig, tx_left, AX_TOP, cfg["txt_right"] - tx_left, AX_H)
    tax.set_axis_off()
    tax.set_xlim(0, 1)
    tax.set_ylim(9.5, -0.5)
    tax.text(1.0, -0.5, "HR (95% CI)", ha="right", va="bottom",
             fontsize=fs.FS["small"], color=fs.C["ink2"])
    for i, r in df.iterrows():
        tax.text(1.0, i, fmt(r.HR_per_10_pctpt, r.HR_ci_low, r.HR_ci_high),
                 ha="right", va="center", fontsize=fs.FS["small"], color=fs.C["ink"])
    return df


def main():
    fig = fs.figure(W, H)
    for cfg in PANELS:
        fs.panel_label(fig, cfg["label_x"] - 13.0, 3.0, cfg["letter"])
        fig.text((cfg["label_x"] - 13.0 + 4.5) * fs.MM / (W * fs.MM),
                 1 - 4.2 * fs.MM / (H * fs.MM), cfg["header"],
                 fontsize=fs.FS["small"], color=fs.C["ink2"], ha="left", va="top")
        draw_panel(fig, cfg)

    fig.text(2.0 * fs.MM / (W * fs.MM), 1 - (H - 2.0) * fs.MM / (H * fs.MM),
             "Points, hazard ratio per 10 percentage-point increase in tumor methylation; "
             "bars, Wald 95% confidence intervals; dashed line, no association.",
             fontsize=fs.FS["small"], color=fs.C["ink2"], ha="left", va="bottom")

    man = fs.save(fig, "Figure_S2", sources=[F_PRIMARY, F_ALL],
                  notes="Single-gene Cox forest plots; HR per 10 percentage points, log x axis "
                        "0.4-1.7; gene order fs.GENES; panel A primary set (n=82, 14 events), "
                        "panel B all-stage sensitivity set (n=87, 17 events).")
    print(man["width_mm"], man["height_mm"])


if __name__ == "__main__":
    main()
