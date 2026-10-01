"""Figure 2 - Paired pyrosequencing methylation by gene (180 x 90 mm).

2 rows x 5 gene panels, shared y (0-90 %), paired grey lines, N/T points.
Sources: F2_paired_raw_source_data.tsv, F2_paired_effect_source_data.tsv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figstyle as fs

SRC = fs.F124 / "Paired_PSQ"
RAW = SRC / "F2_paired_raw_source_data.tsv"
EFF = SRC / "F2_paired_effect_source_data.tsv"

# ---- data ---------------------------------------------------------------
raw = pd.read_csv(RAW, sep="\t")
eff = pd.read_csv(EFF, sep="\t").set_index("gene")
wide = raw.pivot(index=["study_id", "gene"], columns="tissue",
                 values="methylation_pct").reset_index()

N_PAIRS = int(eff["n_pairs"].iloc[0])
assert (eff["n_pairs"] == N_PAIRS).all() and N_PAIRS == 87

# integrity: recompute the effect columns from the raw file
for g in fs.GENES:
    d = wide[wide.gene == g]
    assert len(d) == N_PAIRS, g
    diff = d["Tumor"].to_numpy() - d["Normal"].to_numpy()
    assert np.isclose(diff.mean(), eff.loc[g, "mean_difference_pp"], atol=1e-8), g
    assert int((diff > 0).sum()) == int(eff.loc[g, "tumor_greater_own_normal_n"]), g

# ---- layout (mm) --------------------------------------------------------
W, H = 180.0, 90.0
L0, RMARG, GUT = 13.0, 3.0, 5.4
PW = (W - L0 - RMARG - 4 * GUT) / 5.0          # 28.48 mm
LEFTS = [L0 + i * (PW + GUT) for i in range(5)]
AX_TOP = [16.2, 57.2]
AX_H = 27.0
TITLE_Y = [7.4, 48.4]        # gene symbol (va = top)
DELTA_Y = [10.8, 51.8]       # delta line
COUNT_Y = [13.8, 54.8]       # count line

YLIM = (0.0, 90.0)
YTICKS = [0, 20, 40, 60, 80]
JIT = 0.06

rng = np.random.default_rng(20260909)
fig = fs.figure(W, H)

for k, gene in enumerate(fs.GENES):
    r, c = divmod(k, 5)
    ax = fs.ax_mm(fig, LEFTS[c], AX_TOP[r], PW, AX_H)
    d = wide[wide.gene == gene]
    nvals = d["Normal"].to_numpy(float)
    tvals = d["Tumor"].to_numpy(float)
    jn = rng.uniform(-JIT, JIT, size=nvals.size)
    jt = rng.uniform(-JIT, JIT, size=tvals.size)

    fs.light_grid(ax, axis="y")
    # paired lines
    ax.plot(np.vstack([jn, 1 + jt]), np.vstack([nvals, tvals]),
            color=fs.C["rule"], lw=0.4, alpha=0.6, zorder=1, solid_capstyle="round")
    ax.plot(jn, nvals, ls="none", marker="o", ms=2.2, mfc=fs.C["normal"],
            mec="none", alpha=0.85, zorder=2)
    ax.plot(1 + jt, tvals, ls="none", marker="o", ms=2.2, mfc=fs.C["tumor"],
            mec="none", alpha=0.85, zorder=2)

    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(*YLIM)
    ax.set_xticks([0, 1], ["N", "T"])
    ax.set_yticks(YTICKS)
    fs.despine(ax)
    if c == 0:
        ax.set_ylabel("Methylation (%)", labelpad=2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=2 if c == 0 else 0)

    # header block: gene symbol, delta with BCa limits, count of tumor-higher pairs
    md = eff.loc[gene, "mean_difference_pp"]
    lo = eff.loc[gene, "mean_diff_bca95_low"]
    hi = eff.loc[gene, "mean_diff_bca95_high"]
    ng = int(eff.loc[gene, "tumor_greater_own_normal_n"])
    cx = LEFTS[c] + PW / 2
    fig.text(cx * fs.MM / (W * fs.MM), 1 - TITLE_Y[r] / H, gene,
             ha="center", va="top", fontsize=fs.FS["body"], color=fs.C["ink"],
             fontstyle="italic")
    fig.text(cx * fs.MM / (W * fs.MM), 1 - DELTA_Y[r] / H,
             f"Δ {md:.1f} pp ({lo:.1f} to {hi:.1f})",
             ha="center", va="top", fontsize=fs.FS["small"], color=fs.C["ink2"])
    fig.text(cx * fs.MM / (W * fs.MM), 1 - COUNT_Y[r] / H,
             f"{ng}/{N_PAIRS} higher",
             ha="center", va="top", fontsize=fs.FS["small"], color=fs.C["ink2"])

# ---- single legend strip + n statement ---------------------------------
lax = fs.ax_mm(fig, L0, 1.4, 70, 4.2)
lax.set_axis_off()
handles = [
    Line2D([], [], ls="none", marker="o", ms=2.8, mfc=fs.C["normal"], mec="none",
           label="N, adjacent mucosa"),
    Line2D([], [], ls="none", marker="o", ms=2.8, mfc=fs.C["tumor"], mec="none",
           label="T, tumor"),
    Line2D([], [], color=fs.C["rule"], lw=0.6, alpha=0.9, label="same patient"),
]
lax.legend(handles=handles, loc="center left", ncol=3, frameon=False,
           fontsize=fs.FS["small"], handlelength=1.2, handletextpad=0.4,
           columnspacing=1.1, borderpad=0, borderaxespad=0)
fig.text(1 - RMARG * fs.MM / (W * fs.MM), 1 - 3.4 / H, f"n = {N_PAIRS} pairs",
         ha="right", va="center", fontsize=fs.FS["small"], color=fs.C["ink2"])

man = fs.save(fig, "Figure_2", sources=[RAW, EFF],
              notes=("Paired pyrosequencing, 87 patient pairs, 10 genes. Points are "
                     "individual specimens with +-0.06 horizontal jitter; grey lines join "
                     "the two specimens of one patient. Delta = mean paired tumor-minus-normal "
                     "difference in percentage points with 95% BCa bootstrap limits "
                     "(5,000 patient-pair resamples) taken from the effect source file; "
                     "counts are pairs with tumor > own normal. Shared y axis 0-90 %. "
                     "No panel letters (single panel type)."))
print(man["width_mm"], man["height_mm"])
