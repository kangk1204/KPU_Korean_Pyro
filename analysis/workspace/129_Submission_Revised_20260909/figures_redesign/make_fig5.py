"""Figure 5 - Individual CpG contrasts in public colorectal tissue cohorts.

77 fixed CpGs (rows, gene blocks, ascending hg19 position within gene) x 12
public tissue contrasts (columns, three blocks).  Color = mean difference in
percentage points, symmetric +/-60.  Dot = BH q < 0.05 within the 462-test
family.  Grey x = non-estimable.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

import figstyle as fs

MM = fs.MM
SRC = fs.F124 / "Public_CpG_contrasts"
CONTRASTS = SRC / "Public_CpG_contrasts_source_data.tsv"
PROBECTX = SRC / "input_probe_context.tsv"

# ---------------------------------------------------------------- row layout
def row_order() -> pd.DataFrame:
    """77 CpGs: fs.GENES order, ascending hg19 pos within gene."""
    p = pd.read_csv(PROBECTX, sep="\t")
    p["_g"] = p["fixed_gene"].map({g: i for i, g in enumerate(fs.GENES)})
    assert p["_g"].notna().all(), "unknown gene in probe context"
    p = p.sort_values(["_g", "pos"], kind="mergesort").reset_index(drop=True)
    p["row"] = np.arange(len(p))
    return p[["row", "fixed_gene", "probe", "chr", "pos", "promoter_match"]]


# ---------------------------------------------------------------- columns
BLOCKS = [
    ("Adjacent vs healthy (N–H)",
     [("Colonomics", "N-H"), ("GSE48684", "N-H"), ("GSE42752", "N-H")]),
    ("Tumor vs healthy (T–H)",
     [("Colonomics", "T-H"), ("GSE48684", "T-H"), ("GSE42752", "T-H")]),
    ("Tumor vs adjacent (T–N)",
     [("Colonomics", "T-N"), ("GSE42752", "T-N"), ("GSE193535", "T-N"),
      ("GSE77718", "T-N"), ("GSE77954", "T-N"), ("GSE48684", "T-N")]),
]

# ---------------------------------------------------------------- geometry (mm)
W, H = 180.0, 225.0
LEFT, RIGHT_PAD = 36.0, 4.0
TOP = 21.0                       # heatmap top edge
HM_W = W - LEFT - RIGHT_PAD      # 140 mm
HM_H = 192.0                     # 77 rows -> 2.494 mm pitch
GAP_MM = 3.0
N_COL = sum(len(b[1]) for b in BLOCKS)
COL_W = (HM_W - GAP_MM * (len(BLOCKS) - 1)) / N_COL
GAP_U = GAP_MM / COL_W
ROW_H = HM_H / 77.0
LIM = 60.0


def mm2y(mm_from_fig_top: float) -> float:
    """Figure-top mm -> heatmap data y (0 at heatmap top row edge)."""
    return (mm_from_fig_top - TOP) / ROW_H


def mm2x(mm_from_fig_left: float) -> float:
    return (mm_from_fig_left - LEFT) / COL_W


def fig_text(fig, x, y, s, **kw):
    w, h = fig.get_size_inches()
    return fig.text(x * MM / w, 1 - y * MM / h, s, **kw)


def main():
    d = pd.read_csv(CONTRASTS, sep="\t")
    rows = row_order()
    rmap = dict(zip(rows["probe"], rows["row"]))

    # ---- data checks recorded for QA -----------------------------------
    report = {}
    report["n_rows"] = len(d)
    report["max_abs_effect"] = float(d["effect_pp"].abs().max())
    report["status_counts"] = d["status"].value_counts().to_dict()
    report["sig_rule"] = sorted(d["significance_rule"].dropna().unique())

    # independent BH recomputation over p_for_bh within each family
    def bh(p):
        p = np.asarray(p, float)
        n = p.size
        o = np.argsort(p, kind="mergesort")
        q = np.empty(n)
        q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
        return np.minimum(q, 1.0)

    d["q_recomputed"] = np.nan
    for fam, idx in d.groupby("family").groups.items():
        d.loc[idx, "q_recomputed"] = bh(d.loc[idx, "p_for_bh"].values)
        report[f"family_size_{fam}"] = len(idx)
    report["max_q_delta"] = float((d["q_recomputed"] - d["q_BH_family"]).abs().max())
    d["sig"] = d["q_recomputed"] < 0.05
    report["sig_disagreement_vs_file"] = int((d["sig"] != d["significant_bh_family"]).sum())

    # ---- figure ---------------------------------------------------------
    fig = fs.figure(W, H)
    ax = fs.ax_mm(fig, LEFT, TOP, HM_W, HM_H)
    ax.set_xlim(0, N_COL + GAP_U * (len(BLOCKS) - 1))
    ax.set_ylim(77, 0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-LIM, vmax=LIM)
    cmap = fs.CMAP_DIV

    dot_counts, col_centres, col_x0 = {}, [], []
    ci = 0
    for bi, (btitle, cols) in enumerate(BLOCKS):
        for cohort, contrast in cols:
            x0 = ci + GAP_U * bi
            xc = x0 + 0.5
            col_centres.append(xc); col_x0.append(x0)
            sub = d[(d["cohort"] == cohort) & (d["contrast"] == contrast)]
            assert len(sub) == 77, (cohort, contrast, len(sub))
            n_sig = 0
            for _, r in sub.iterrows():
                y = rmap[r["probe"]] + 0.5
                if r["status"] != "estimated" or not np.isfinite(r["effect_pp"]):
                    fs.na_cell(ax, xc, y, w=1.0, h=1.0)
                    continue
                ax.add_patch(plt.Rectangle((x0, y - 0.5), 1.0, 1.0,
                                           facecolor=cmap(norm(r["effect_pp"])),
                                           edgecolor="none", zorder=1))
                if bool(r["sig"]):
                    fs.sig_dot(ax, xc, y, size=1.5)
                    n_sig += 1
            dot_counts[f"{cohort} {contrast}"] = n_sig

            # column header: cohort (7 pt) + n (6 pt grey), rotated 90 deg
            f = sub.iloc[0]
            nlab = (f"{int(f['n_pairs'])} pairs" if bool(f["paired"])
                    else f"n {int(f['n_high'])}/{int(f['n_low'])}")
            ax.text(xc - 0.20, mm2y(TOP - 1.0), cohort, rotation=90, ha="center",
                    va="bottom", fontsize=fs.FS["body"], color=fs.C["ink"], clip_on=False)
            ax.text(xc + 0.24, mm2y(TOP - 1.0), nlab, rotation=90, ha="center",
                    va="bottom", fontsize=fs.FS["small"], color=fs.C["ink2"], clip_on=False)
            ci += 1

        # block header + rule
        first, last = col_x0[-len(cols)], col_x0[-1] + 1.0
        mid = (first + last) / 2
        ax.text(mid, mm2y(3.3), btitle, ha="center", va="center",
                fontsize=fs.FS["body"], fontweight="bold", color=fs.C["ink"], clip_on=False)
        ax.plot([first, last], [mm2y(5.4)] * 2, color=fs.C["ink2"], lw=0.6,
                clip_on=False, solid_capstyle="butt")

    # ---- row labels, gene brackets, block separators ---------------------
    bracket_x = mm2x(17.0)
    cpg_x = mm2x(34.6)
    lab_dx = 4.0 / COL_W                      # 4 mm gap, italic label left of bracket
    for gene, g in rows.groupby("fixed_gene", sort=False):
        y0, y1 = g["row"].min(), g["row"].max() + 1.0
        # same geometry as fs.gene_bracket, but a true italic face (mathtext keeps
        # digits upright, which reads badly for ZNF568 / ADHFE1 / HOXA2)
        ax.plot([bracket_x, bracket_x], [y0 + 0.12, y1 - 0.12], color=fs.C["ink2"],
                lw=0.6, clip_on=False, solid_capstyle="butt")
        ax.text(bracket_x - lab_dx, (y0 + y1) / 2, gene, ha="right", va="center",
                fontsize=fs.FS["body"], color=fs.C["ink"], style="italic", clip_on=False)
        if y0 > 0:   # thin separator between gene blocks
            ax.plot([0, N_COL + GAP_U * (len(BLOCKS) - 1)], [y0, y0],
                    color="white", lw=0.6, zorder=4, solid_capstyle="butt")
    for _, r in rows.iterrows():
        ax.text(cpg_x, r["row"] + 0.5, r["probe"], ha="right", va="center",
                fontsize=fs.FS["tiny"], color=fs.C["ink2"], clip_on=False)

    # frame around each block
    for bi, (_, cols) in enumerate(BLOCKS):
        start = sum(len(b[1]) for b in BLOCKS[:bi]) + GAP_U * bi
        ax.add_patch(plt.Rectangle((start, 0), len(cols), 77, fill=False,
                                   edgecolor=fs.C["rule"], lw=0.4, zorder=5))

    # ---- color bar ------------------------------------------------------
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fs.colorbar(fig, sm, LEFT, 217.0, 40.0, 2.5,
                "Mean difference (percentage points)",
                ticks=[-60, -30, 0, 30, 60])

    # ---- marker legend ---------------------------------------------------
    lax = fs.ax_mm(fig, 86.0, 215.5, 26.0, 6.0)
    lax.set_xlim(0, 26); lax.set_ylim(0, 6); lax.axis("off")
    lax.plot(1.2, 4.3, marker="o", ms=1.5, mfc=fs.C["sig"], mec="none", ls="none")
    lax.text(3.0, 4.3, "BH q < 0.05", ha="left", va="center", fontsize=fs.FS["small"])
    lax.add_patch(plt.Rectangle((0.3, 0.6), 1.8, 1.8, facecolor=fs.C["na"], edgecolor="none"))
    lax.text(1.2, 1.5, "×", ha="center", va="center", fontsize=fs.FS["tiny"],
             color=fs.C["na_mark"])
    lax.text(3.0, 1.5, "not estimable", ha="left", va="center", fontsize=fs.FS["small"])

    # ---- footnote --------------------------------------------------------
    note = ("Dots mark BH q < 0.05 within the pre-specified 462-test\n"
            "healthy-reference or tumor–normal family. × marks CpGs absent\n"
            "from a beta matrix (19 cells). Color scale symmetric at ±60 pp.")
    fig_text(fig, 118.0, 215.8, note, fontsize=fs.FS["small"], color=fs.C["ink2"],
             ha="left", va="top", linespacing=1.45)

    man = fs.save(fig, "Figure_5", sources=[CONTRASTS, PROBECTX],
                  notes=("77 CpGs x 12 public tissue contrasts; effect_pp on symmetric "
                         "+/-60 diverging scale; dots = BH q<0.05 within 462-test family; "
                         "grey x = probe absent."))
    report["dot_counts"] = dot_counts
    report["manifest"] = man
    return report


if __name__ == "__main__":
    import json
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "manifest"}, indent=1, default=str))
    print("sources:", [(s["path"].split("/")[-1], s["sha256"]) for s in r["manifest"]["sources"]])
    print("size mm:", r["manifest"]["width_mm"], r["manifest"]["height_mm"])
