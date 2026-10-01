"""Figure 4 - Korean cohorts, individual CpGs (180 x 225 mm).

A  dot-and-interval plot of the mean paired tumor-minus-adjacent-mucosa
   difference (percentage points, BCa 95% CI) at the 77 fixed CpGs in the three
   Korean EPIC cohorts.
B-D between-CpG Spearman correlation matrices of the paired differences for the
   56 measured CpGs, one per cohort.
E  distribution of the 1,381 between-gene correlations per cohort.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import figstyle as fs
import rows77

K = fs.F124 / "Korean_CpG"
EFFECTS = K / "effects_source.tsv"
CORRELATIONS = K / "correlations_source.tsv"
COORDINATION = K / "coordination_source.tsv"
ANNOTATION = K / "probe_annotation_source.tsv"
MATRICES = {c: K / f"{c}_correlation_matrix.tsv" for c in fs.COHORTS_KR}
PC1 = fs.R124 / "coordination_sensitivity" / "korean_delta_pc1_fraction.tsv"

X_MAX = 50.0
X_MIN = -2.0
OFFSETS = {"CMCBSN": -0.26, "SNUH": 0.0, "ASAN": 0.26}

# ---------------------------------------------------------------- data ----
rows = rows77.load_rows(ANNOTATION)
effects = pd.read_csv(EFFECTS, sep="\t")
estimated = effects[effects["status"] == "estimated"].copy()
coord = pd.read_csv(COORDINATION, sep="\t").set_index("cohort")
corr = pd.read_csv(CORRELATIONS, sep="\t")
pc1 = pd.read_csv(PC1, sep="\t").set_index("cohort")

n_pairs = {c: int(estimated.loc[estimated["cohort"] == c, "n_pairs"].iloc[0]) for c in fs.COHORTS_KR}
measured = sorted(set(estimated["probe"]))
missing = sorted(set(rows["probe"]) - set(measured))
assert len(measured) == 56 and len(missing) == 21, (len(measured), len(missing))

between = {c: corr[(corr["cohort"] == c) & (corr["family"] == "between_gene") & corr["rho"].notna()]["rho"].to_numpy()
           for c in fs.COHORTS_KR}
for c, v in between.items():
    assert len(v) == 1381 and (v > 0).all(), (c, len(v), v.min())

mrows = rows[rows["probe"].isin(measured)].reset_index(drop=True)
mats = {}
for c, path in MATRICES.items():
    m = pd.read_csv(path, sep="\t", index_col=0)
    assert m.shape == (56, 56), (c, m.shape)
    mats[c] = m.reindex(index=mrows["probe"], columns=mrows["probe"])
    assert mats[c].notna().all().all()

y = rows77.y_of(rows)

# ---------------------------------------------------------------- figure ----
fig = fs.figure(180, 225)

A_GUT_L, A_GUT_W = 0.5, 26.5          # gene brackets + CpG identifiers
A_L, A_W = 27.5, 64.0
A_TOP, A_H = 18.0, 196.0
R_LAB_L = 93.0                        # right column gene-label gutter
R_L, R_W = 104.0, 63.0                # matrices
CB_L, CB_W = 169.0, 2.6

fs.panel_label(fig, 1.0, 4.0, "A")

# ---- legend strip for panel A -------------------------------------------
# Three cohort entries laid out left to right, each advanced by the *measured*
# width of its own label so a handle can never overlap the preceding text.
LEG_Y, CAP1_Y, CAP2_Y = 0.86, 0.44, 0.06     # handle row and the two caption rows
HANDLE_W, HANDLE_PAD, ENTRY_GAP = 0.036, 0.011, 0.050

leg = fs.ax_mm(fig, A_GUT_L, 6.0, A_L + A_W - A_GUT_L, 9.0)
leg.set_xlim(0, 1); leg.set_ylim(0, 1); leg.axis("off")
fig.canvas.draw()                              # renderer needed for text metrics
_rend = fig.canvas.get_renderer()
_inv = leg.transAxes.inverted()

xm = 0.005
for c in fs.COHORTS_KR:
    leg.plot([xm, xm + HANDLE_W], [LEG_Y, LEG_Y], color=fs.C["cohort"][c], lw=0.6,
             solid_capstyle="butt", clip_on=False)
    leg.plot([xm + HANDLE_W / 2], [LEG_Y], marker="o", ms=2.4, mfc=fs.C["cohort"][c],
             mec="white", mew=0.25, linestyle="none", clip_on=False)
    t = leg.text(xm + HANDLE_W + HANDLE_PAD, LEG_Y, f"{c}, {n_pairs[c]} pairs",
                 fontsize=fs.FS["small"], color=fs.C["ink"], ha="left", va="center")
    bb = t.get_window_extent(renderer=_rend)
    w_ax = _inv.transform((bb.x1, bb.y0))[0] - _inv.transform((bb.x0, bb.y0))[0]
    xm += HANDLE_W + HANDLE_PAD + w_ax + ENTRY_GAP
assert xm < 1.0, f"panel A cohort legend overflows its strip ({xm:.3f})"

leg.text(0.005, CAP1_Y, "Mean paired difference, BCa 95% confidence interval.",
         fontsize=fs.FS["small"], color=fs.C["ink2"], ha="left", va="center")
leg.text(0.005, CAP2_Y, "21 of the 77 CpGs are absent from all three processed matrices.",
         fontsize=fs.FS["small"], color=fs.C["ink2"], ha="left", va="center")

# ---- A: dot-and-interval -------------------------------------------------
axA = fs.ax_mm(fig, A_L, A_TOP, A_W, A_H)
rows77.apply_row_axis(axA, rows)
axA.set_xlim(X_MIN, X_MAX)
rows77.block_bands(axA, rows, X_MIN, X_MAX)
axA.set_xticks(np.arange(0, X_MAX + 1, 10))
fs.light_grid(axA, axis="x")
fs.despine(axA, left=False, bottom=True)
axA.tick_params(axis="y", length=0)
axA.axvline(0.0, color=fs.C["rule"], lw=0.6, ls=(0, (3, 2)), zorder=1)
axA.set_xlabel("Tumor − adjacent mucosa (percentage points)")

for cohort in fs.COHORTS_KR:
    col = fs.C["cohort"][cohort]
    sub = estimated[estimated["cohort"] == cohort]
    dy = OFFSETS[cohort]
    for probe, mean, lo, hi in zip(sub["probe"], sub["mean_delta_pp"], sub["ci_low_pp"], sub["ci_high_pp"]):
        yy = y[probe] + dy
        axA.plot([lo, hi], [yy, yy], color=col, lw=0.6, solid_capstyle="butt", zorder=2)
        axA.plot([mean], [yy], marker="o", ms=2.4, mfc=col, mec="white", mew=0.25,
                 linestyle="none", zorder=3)

for probe in missing:
    axA.text(1.0, y[probe], "not measured", fontsize=fs.FS["tiny"], color=fs.C["muted"],
             ha="left", va="center", zorder=2)

rows77.draw_row_gutter(fig, A_GUT_L, A_TOP, A_GUT_W, A_H, rows, bracket_frac=0.595)

# ---- B-D: correlation matrices ------------------------------------------
norm_corr = Normalize(vmin=0.0, vmax=1.0)
mat_tops = {"CMCBSN": 16.0, "SNUH": 76.0, "ASAN": 136.0}
MAT_H = 52.0
letters = {"CMCBSN": "B", "SNUH": "C", "ASAN": "D"}
sub_y = np.arange(56, dtype=float)
gene_first = {}
for gene, chunk in mrows.groupby("gene", sort=False):
    gene_first[gene] = (float(chunk.index.min()), float(chunk.index.max()))

for cohort in fs.COHORTS_KR:
    top = mat_tops[cohort]
    fs.panel_label(fig, R_LAB_L - 1.5, top - 8.5, letters[cohort])
    med = coord.loc[cohort, "median_between_gene_rho"]
    lo, hi = coord.loc[cohort, "ci_low"], coord.loc[cohort, "ci_high"]
    fig.text(R_LAB_L * fs.MM / fig.get_size_inches()[0],
             1 - (top - 3.2) * fs.MM / fig.get_size_inches()[1],
             f"{cohort}  ·  median between-gene ρ {med:.2f} (95% CI {lo:.2f}–{hi:.2f})",
             fontsize=fs.FS["small"], color=fs.C["ink"], ha="left", va="baseline")

    ax = fs.ax_mm(fig, R_L, top, R_W, MAT_H)
    im = ax.imshow(mats[cohort].to_numpy(), cmap=fs.CMAP_CORR, norm=norm_corr,
                   origin="upper", extent=(-0.5, 55.5, 55.5, -0.5), aspect="auto",
                   interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_edgecolor(fs.C["ink2"])
    for gene, (i0, i1) in gene_first.items():
        if i1 < 55:
            ax.plot([-0.5, 55.5], [i1 + 0.5, i1 + 0.5], color="white", lw=0.5, zorder=3)
            ax.plot([i1 + 0.5, i1 + 0.5], [-0.5, 55.5], color="white", lw=0.5, zorder=3)
        ax.text(-1.2, (i0 + i1) / 2, gene, fontsize=fs.FS["small"], style="italic",
                color=fs.C["ink"], ha="right", va="center")

cb = fs.colorbar(fig, ScalarMappable(norm=norm_corr, cmap=fs.CMAP_CORR),
                 CB_L, mat_tops["SNUH"], CB_W, MAT_H, "Spearman ρ",
                 ticks=[0, 0.5, 1.0], orientation="vertical")
cb.ax.yaxis.set_label_position("right")

# ---- E: distribution of between-gene correlations ------------------------
E_TOP, E_H, E_L, E_W = 196.0, 17.0, 118.0, 50.0
fs.panel_label(fig, R_LAB_L - 1.5, E_TOP - 6.0, "E")
axE = fs.ax_mm(fig, E_L, E_TOP, E_W, E_H)
axE.set_xlim(0, 1); axE.set_ylim(3.0, 0.0)
positions = {c: i + 0.5 for i, c in enumerate(fs.COHORTS_KR)}
parts = axE.violinplot([between[c] for c in fs.COHORTS_KR],
                       positions=[positions[c] for c in fs.COHORTS_KR],
                       vert=False, widths=0.78, showextrema=False, showmedians=False)
for body, cohort in zip(parts["bodies"], fs.COHORTS_KR):
    body.set_facecolor(fs.C["cohort"][cohort]); body.set_alpha(0.35)
    body.set_edgecolor(fs.C["cohort"][cohort]); body.set_linewidth(0.4)
for cohort in fs.COHORTS_KR:
    p = positions[cohort]
    med = coord.loc[cohort, "median_between_gene_rho"]
    lo, hi = coord.loc[cohort, "ci_low"], coord.loc[cohort, "ci_high"]
    axE.plot([lo, hi], [p, p], color=fs.C["ink"], lw=0.7, solid_capstyle="butt", zorder=4)
    axE.plot([med], [p], marker="o", ms=2.6, mfc=fs.C["cohort"][cohort], mec=fs.C["ink"],
             mew=0.35, linestyle="none", zorder=5)
axE.set_yticks([positions[c] for c in fs.COHORTS_KR])
axE.set_yticklabels([f"{c}\nPC1 {100 * pc1.loc[c, 'pc1_explained_variance_fraction']:.1f}%"
                     for c in fs.COHORTS_KR], fontsize=fs.FS["small"], color=fs.C["ink"])
axE.tick_params(axis="y", length=0, pad=1)
axE.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
fs.despine(axE, left=False, bottom=True)
fs.light_grid(axE, axis="x")
axE.set_xlabel("Between-gene Spearman ρ")
fig.text(R_LAB_L * fs.MM / fig.get_size_inches()[0],
         1 - (E_TOP - 2.2) * fs.MM / fig.get_size_inches()[1],
         "1,381 between-gene CpG pairs per cohort; point and bar = median (95% CI).",
         fontsize=fs.FS["small"], color=fs.C["ink2"], ha="left", va="baseline")

notes = (f"A: 77 CpGs, {n_pairs['CMCBSN']}/{n_pairs['SNUH']}/{n_pairs['ASAN']} pairs, BCa 95% CI; "
         "21 CpGs absent from every processed matrix. B-D: 56 measured CpGs, Spearman rho 0-1 "
         "(all between-CpG correlations are positive). E: 1,381 between-gene correlations per cohort.")
fs.save(fig, "Figure_4",
        sources=[EFFECTS, CORRELATIONS, COORDINATION, ANNOTATION, *MATRICES.values(), PC1],
        notes=notes)

print("measured", len(measured), "missing", len(missing))
for c in fs.COHORTS_KR:
    s = estimated[estimated["cohort"] == c]
    print(c, "mean range %.4f - %.4f" % (s["mean_delta_pp"].min(), s["mean_delta_pp"].max()),
          "median rho %.4f" % np.median(between[c]))
print("missing probes:", missing)
