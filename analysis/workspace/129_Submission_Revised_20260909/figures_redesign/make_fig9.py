#!/usr/bin/env python3
"""Figure 9 - patient-grouped tissue-classification AUC forest (88 x 45 mm).

Four cohort-local nested-cross-validation models: two local pyrosequencing models and two
public-array ridge models. Single panel, so no panel letter is drawn.
"""
from __future__ import annotations

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

import figstyle as fs

F_AUC = fs.F124 / "Tissue_ML" / "tissue_ml_auc_forest_source.tsv"
F_COL_MAN = fs.R124 / "public_ml" / "colonomics" / "run_manifest.json"
F_G19_MAN = fs.R124 / "public_ml" / "gse119526" / "run_manifest.json"

W, H = 88.0, 45.0
AX_L, AX_W, TXT_R = 34.0, 30.0, 86.0
AX_TOP, AX_H = 8.5, 27.0

# (cohort, model) -> (row label line 1, row label line 2, color group)
ROWS = [
    ("Local PSQ, 87 patients", "ridge",
     "Local PSQ, ridge", "10 genes, 87 patients", "local"),
    ("Local PSQ, 87 patients", "inner-selected single gene",
     "Local PSQ, single gene", "inner-selected, 87 patients", "local"),
    ("Colonomics, 92 patients", "ridge",
     "Colonomics, ridge", "77 CpGs, 92 patients", "public"),
    ("GSE119526, 48 patients", "ridge",
     "GSE119526, ridge", "71 CpGs, 48 patients", "public"),
]


def main():
    df = pd.read_csv(F_AUC, sep="\t")
    fig = fs.figure(W, H)

    ax = fs.ax_mm(fig, AX_L, AX_TOP, AX_W, AX_H)
    fs.light_grid(ax, "x")
    ax.set_xlim(0.85, 1.0)
    ax.xaxis.set_major_locator(FixedLocator([0.85, 0.90, 0.95, 1.00]))
    ax.set_xticklabels(["0.85", "0.90", "0.95", "1.00"])
    ax.set_ylim(3.6, -0.6)
    ax.set_yticks(range(4))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", pad=1.5)
    fs.despine(ax, left=False, bottom=True)
    for y in range(4):
        ax.axhline(y, color=fs.C["grid"], lw=0.3, zorder=0)
    ax.set_xlabel("Area under the ROC curve (AUC)")

    tax = fs.ax_mm(fig, AX_L + AX_W + 1.0, AX_TOP, TXT_R - (AX_L + AX_W + 1.0), AX_H)
    tax.set_axis_off()
    tax.set_xlim(0, 1)
    tax.set_ylim(3.6, -0.6)
    tax.text(1.0, -0.6, "AUC (95% CI)", ha="right", va="bottom",
             fontsize=fs.FS["small"], color=fs.C["ink2"])

    lax = fs.ax_mm(fig, 1.0, AX_TOP, AX_L - 2.0, AX_H)
    lax.set_axis_off()
    lax.set_xlim(0, 1)
    lax.set_ylim(3.6, -0.6)

    for i, (cohort, model, lab1, lab2, group) in enumerate(ROWS):
        r = df[(df.cohort == cohort) & (df.model == model)].iloc[0]
        col = fs.C[group]
        ax.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=0.9, solid_capstyle="butt", zorder=3)
        ax.plot([r.auc], [i], marker="o", ms=3.5, mfc=col, mec="white", mew=0.35,
                linestyle="none", zorder=4)
        lax.text(1.0, i - 0.20, lab1, ha="right", va="center",
                 fontsize=fs.FS["body"], color=fs.C["ink"])
        lax.text(1.0, i + 0.20, lab2, ha="right", va="center",
                 fontsize=fs.FS["small"], color=fs.C["ink2"])
        tax.text(1.0, i, f"{r.auc:.3f} ({r.ci_low:.3f}–{r.ci_high:.3f})", ha="right",
                 va="center", fontsize=fs.FS["small"], color=fs.C["ink"])

    handles = [Line2D([], [], color=fs.C["local"], marker="o", ms=3.5, mfc=fs.C["local"],
                      mec="white", mew=0.35, lw=0.9, label="Local pyrosequencing (PSQ)"),
               Line2D([], [], color=fs.C["public"], marker="o", ms=3.5, mfc=fs.C["public"],
                      mec="white", mew=0.35, lw=0.9, label="Public arrays")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-1.07, 1.06), ncol=2,
              fontsize=fs.FS["small"], borderaxespad=0, columnspacing=1.4, handletextpad=0.4)

    man = fs.save(fig, "Figure_9", sources=[F_AUC, F_COL_MAN, F_G19_MAN],
                  notes="Tissue-classification AUC forest; CpG counts (77 Colonomics, 71 GSE119526) "
                        "read from features_nonmissing_any in the public_ml run manifests.")
    print(man["width_mm"], man["height_mm"])


if __name__ == "__main__":
    main()
