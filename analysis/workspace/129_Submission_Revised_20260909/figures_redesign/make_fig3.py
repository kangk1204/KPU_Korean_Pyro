"""Figure 3 - Patient-level methylation-difference patterns (180 x 95 mm).

A  gene-standardized paired differences, patients ordered by PC1, with an event strip
B  lower-triangle Spearman correlation matrix of the paired differences
C  PC1 vs PC2 scores colored by event status

Sources: F3_delta_matrix_source_data.tsv, F3_spearman_correlation_source_data.tsv,
         F3_pca_scores_source_data.tsv, F3_pca_loadings_source_data.tsv, pattern_summary.json
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import figstyle as fs

SRC = fs.F124 / "Delta_Patterns"
F_DELTA = SRC / "F3_delta_matrix_source_data.tsv"
F_CORR = SRC / "F3_spearman_correlation_source_data.tsv"
F_SCORES = SRC / "F3_pca_scores_source_data.tsv"
F_LOAD = SRC / "F3_pca_loadings_source_data.tsv"
F_SUM = SRC / "pattern_summary.json"

G = fs.GENES
ZLIM = 3.0

# ---- data ---------------------------------------------------------------
delta = pd.read_csv(F_DELTA, sep="\t")
scores = pd.read_csv(F_SCORES, sep="\t")
corr = pd.read_csv(F_CORR, sep="\t").set_index("gene").loc[G, G]
load = pd.read_csv(F_LOAD, sep="\t").set_index("gene").loc[G]
summ = json.loads(F_SUM.read_text())

N = len(delta)
assert N == len(scores) == 87
assert (delta["study_id"].to_numpy() == scores["study_id"].to_numpy()).all()
assert (delta["event"].to_numpy() == scores["event"].to_numpy()).all()
N_EVENT = int(delta["event"].sum())

X = delta[G].to_numpy(float)
Z = (X - X.mean(axis=0)) / X.std(axis=0)            # StandardScaler, ddof = 0

# reproduce the PCA (SVD of the standardized matrix / sqrt(n-1) = correlation-matrix PCA)
U, sv, Vt = np.linalg.svd(Z / np.sqrt(N - 1), full_matrices=False)
evr = sv ** 2 / (sv ** 2).sum()
PC1_PCT = summ["pca_explained_variance_ratio"]["PC1"] * 100
PC2_PCT = summ["pca_explained_variance_ratio"]["PC2"] * 100
assert np.isclose(evr[0] * 100, PC1_PCT, atol=1e-6) and np.isclose(evr[1] * 100, PC2_PCT, atol=1e-6)
assert round(PC1_PCT, 1) == 52.2 and round(PC2_PCT, 1) == 10.9

# sign convention: the published file flips each component so the largest |loading| is
# positive; our raw SVD comes out sign-flipped, so use the file's scores throughout.
own = Z @ Vt.T
for k in (0, 1):
    assert np.allclose(own[:, k], -scores[f"PC{k+1}"].to_numpy(), atol=1e-6)
assert np.allclose(Vt[0], -load["PC1_loading"].to_numpy(), atol=1e-6)

pc1 = scores["PC1"].to_numpy()
pc2 = scores["PC2"].to_numpy()
event = delta["event"].to_numpy(int)
order = np.argsort(pc1, kind="stable")
Zo = Z[order].T                                     # genes x patients
event_o = event[order]

W, H = 180.0, 95.0
fig = fs.figure(W, H)

# ======================= A : heatmap + event strip =======================
A_L, A_W = 19.0, 128.0
A_TOP, A_H = 6.0, 24.0
STRIP_TOP, STRIP_H = 31.2, 2.4

norm = TwoSlopeNorm(vcenter=0.0, vmin=-ZLIM, vmax=ZLIM)
axA = fs.ax_mm(fig, A_L, A_TOP, A_W, A_H)
im = axA.imshow(np.clip(Zo, -ZLIM, ZLIM), cmap=fs.CMAP_DIV, norm=norm,
                aspect="auto", interpolation="nearest",
                extent=(0, N, len(G), 0))
axA.set_yticks(np.arange(len(G)) + 0.5)
axA.set_yticklabels(G, fontsize=fs.FS["small"], fontstyle="italic")
axA.set_xticks([])
axA.tick_params(axis="y", length=0, pad=2)
for s in axA.spines.values():
    s.set_visible(True); s.set_linewidth(0.4); s.set_color(fs.C["ink2"])

axS = fs.ax_mm(fig, A_L, STRIP_TOP, A_W, STRIP_H)
for i, e in enumerate(event_o):
    axS.add_patch(Rectangle((i, 0), 1, 1, facecolor=fs.C["event"] if e else fs.C["noevent"],
                            edgecolor="none"))
axS.set_xlim(0, N); axS.set_ylim(0, 1)
axS.set_xticks([]); axS.set_yticks([])
for s in axS.spines.values():
    s.set_visible(True); s.set_linewidth(0.4); s.set_color(fs.C["ink2"])
fig.text((A_L - 1.5) / W, 1 - (STRIP_TOP + STRIP_H / 2) / H, "Event",
         ha="right", va="center", fontsize=fs.FS["small"], color=fs.C["ink"])

fig.text((A_L + A_W / 2) / W, 1 - 36.0 / H,
         f"{N} patients ordered by PC1 score  →", ha="center", va="top",
         fontsize=fs.FS["small"], color=fs.C["ink2"])

cb = fs.colorbar(fig, ScalarMappable(norm=norm, cmap=fs.CMAP_DIV),
                 151.0, 11.0, 25.0, 2.0, "z score", ticks=[-3, -1.5, 0, 1.5, 3])

# event key under the colorbar
for i, (lab, col) in enumerate([(f"Event ({N_EVENT})", fs.C["event"]),
                                (f"No event ({N - N_EVENT})", fs.C["noevent"])]):
    y = 21.0 + i * 3.4
    fig.patches.append(Rectangle((151.0 * fs.MM / (W * fs.MM), 1 - (y + 1.6) / H),
                                 2.2 * fs.MM / (W * fs.MM), 1.6 * fs.MM / (H * fs.MM),
                                 transform=fig.transFigure, facecolor=col,
                                 edgecolor="none", figure=fig))
    fig.text(154.4 / W, 1 - (y + 0.8) / H, lab, ha="left", va="center",
             fontsize=fs.FS["small"], color=fs.C["ink"])

fig.text(151.0 / W, 1 - 29.5 / H,
         "z scores clipped at ±3\n(2 of 870 cells)", ha="left", va="top",
         fontsize=fs.FS["small"], color=fs.C["ink2"], linespacing=1.35)

fs.panel_label(fig, 2.0, 3.2, "A")

# ======================= B : lower-triangle Spearman =====================
B_L, B_TOP = 15.0, 41.0
CW, CH = 7.5, 4.8
NB = len(G) - 1                                        # 9 rows x 9 cols
B_W, B_H = NB * CW, NB * CH
axB = fs.ax_mm(fig, B_L, B_TOP, B_W, B_H)
R = corr.to_numpy()
tri = np.full((NB, NB), np.nan)
for i in range(1, len(G)):
    for j in range(i):
        tri[i - 1, j] = R[i, j]
axB.imshow(np.ma.masked_invalid(tri), cmap=fs.CMAP_CORR, vmin=0.0, vmax=1.0,
           aspect="auto", interpolation="nearest", extent=(0, NB, NB, 0))
for i in range(1, len(G)):
    for j in range(i):
        v = R[i, j]
        axB.text(j + 0.5, i - 0.5, f"{v:.2f}", ha="center", va="center",
                 fontsize=fs.FS["small"], color="white" if v > 0.6 else fs.C["ink"])
axB.set_xlim(0, NB); axB.set_ylim(NB, 0)
axB.set_yticks(np.arange(NB) + 0.5)
axB.set_yticklabels(G[1:], fontsize=fs.FS["small"], fontstyle="italic")
axB.set_xticks(np.arange(NB) + 0.5)
axB.set_xticklabels(G[:-1], fontsize=fs.FS["small"], fontstyle="italic",
                    rotation=45, ha="right", rotation_mode="anchor")
axB.tick_params(length=0, pad=1.5)
for s in axB.spines.values():
    s.set_visible(False)

# color bar + note inside the free upper-right triangle
offdiag = R[np.triu_indices(len(G), 1)]
fs.colorbar(fig, ScalarMappable(norm=Normalize(0, 1), cmap=fs.CMAP_CORR),
            B_L + 3.4 * CW, B_TOP + 1.6, 24.0, 2.0, "Spearman rho",
            ticks=[0, 0.25, 0.5, 0.75, 1.0])
fig.text((B_L + 3.4 * CW) / W, 1 - (B_TOP + 9.6) / H,
         f"45 gene pairs, rho {offdiag.min():.2f} to {offdiag.max():.2f}",
         ha="left", va="top", fontsize=fs.FS["small"], color=fs.C["ink2"])

fs.panel_label(fig, 2.0, 38.4, "B")

# ======================= C : PC1 vs PC2 ==================================
C_L, C_TOP, C_W, C_H = 95.5, 41.0, 77.13, 44.5
axC = fs.ax_mm(fig, C_L, C_TOP, C_W, C_H)
axC.axhline(0, color=fs.C["rule"], lw=0.6, ls=(0, (3, 2)), zorder=0)
axC.axvline(0, color=fs.C["rule"], lw=0.6, ls=(0, (3, 2)), zorder=0)
m = event == 1
axC.plot(pc1[~m], pc2[~m], ls="none", marker="o", ms=3.2, mfc=fs.C["noevent"],
         mec="white", mew=0.3, zorder=2)
axC.plot(pc1[m], pc2[m], ls="none", marker="o", ms=3.2, mfc=fs.C["event"],
         mec="white", mew=0.3, zorder=3)
axC.set_xlim(-5.2, 5.2); axC.set_ylim(-3.0, 3.0)     # equal score-unit aspect
axC.set_xticks([-4, -2, 0, 2, 4]); axC.set_yticks([-2, 0, 2])
axC.set_xlabel(f"PC1 ({PC1_PCT:.1f} % of variance)", labelpad=2)
axC.set_ylabel(f"PC2 ({PC2_PCT:.1f} % of variance)", labelpad=2)
fs.despine(axC)
handles = [Line2D([], [], ls="none", marker="o", ms=3.2, mfc=fs.C["noevent"],
                  mec="white", mew=0.3, label=f"No event (n = {N - N_EVENT})"),
           Line2D([], [], ls="none", marker="o", ms=3.2, mfc=fs.C["event"],
                  mec="white", mew=0.3, label=f"Event (n = {N_EVENT})")]
axC.legend(handles=handles, loc="upper left", frameon=False, fontsize=fs.FS["small"],
           handlelength=1.0, handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)

fs.panel_label(fig, 86.5, 38.4, "C")

man = fs.save(fig, "Figure_3", sources=[F_DELTA, F_CORR, F_SCORES, F_LOAD, F_SUM],
              notes=("A: per-gene z scores of the paired tumor-minus-normal differences "
                     "(mean 0, SD 1 within gene, population SD), clipped to +-3 for display "
                     "(2 of 870 cells clipped); symmetric TwoSlopeNorm centered at 0. Columns "
                     "are the 87 patients sorted by increasing PC1 score from "
                     "F3_pca_scores_source_data.tsv. Event strip uses the all-stage event "
                     "column (17 events). B: published Spearman matrix, lower triangle, "
                     "diagonal omitted, scale 0-1. C: PC1/PC2 scores from the source file "
                     "(sign convention: each component flipped so the largest absolute loading "
                     "is positive); variance percentages from pattern_summary.json. "
                     "Equal score-unit aspect (x +-5.2, y +-3)."))
print(man["width_mm"], man["height_mm"], "events", N_EVENT,
      "clipped", int((np.abs(Z) > ZLIM).sum()), "of", Z.size,
      "rho range", round(offdiag.min(), 4), round(offdiag.max(), 4))
