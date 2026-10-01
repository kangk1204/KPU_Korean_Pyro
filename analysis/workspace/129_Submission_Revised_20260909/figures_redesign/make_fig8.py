"""Figure 8 - CMCBSN expression and CMS context (180 x 225 mm).

A  CMS3 - CMS2 mean methylation difference (percentage points) with bootstrap
   95% CI at the 77 fixed CpGs (37 CMS3 versus 39 CMS2 tumors).
B  Spearman correlation between CpG methylation and expression of the mapped
   gene in 165 tumors, unadjusted and adjusted for age, sex and site.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figstyle as fs
import rows77

CTX = fs.R124 / "context"
CMS = CTX / "cms_cpg_contrasts.tsv"
EXPR = CTX / "expression_cpg_associations.tsv"
HEATMAP = fs.F124 / "Korean_Context" / "cpg_context_heatmap_source.tsv"
CONTEXT = fs.F124 / "Public_CpG_contrasts" / "input_probe_context.tsv"

UNADJ_COLOR = fs.C["normal"]        # blue, open circle
ADJ_COLOR = fs.C["tumor"]           # vermilion, filled circle

# ---------------------------------------------------------------- data ----
rows = rows77.load_rows(CONTEXT)
cms = pd.read_csv(CMS, sep="\t").set_index("cpg")
expr = pd.read_csv(EXPR, sep="\t").set_index("cpg")
heat = pd.read_csv(HEATMAP, sep="\t").set_index("cpg")
assert len(cms) == len(expr) == len(heat) == 77

cms_ok = cms[cms["status"] == "estimated"]
n_cms2, n_cms3 = int(cms_ok["n_cms2"].iloc[0]), int(cms_ok["n_cms3"].iloc[0])
assert (n_cms2, n_cms3) == (39, 37), (n_cms2, n_cms3)
assert int((cms_ok["welch_bh_q"] < 0.05).sum()) == 0

expr_ok = expr[expr["status"] == "estimated"]
n_expr = int(expr_ok["n"].iloc[0])
sig_unadj = set(expr_ok.index[expr_ok["bh_q"] < 0.05])
sig_adj = set(expr_ok.index[expr_ok["partial_bh_q_age_sex_site"] < 0.05])
assert len(expr_ok) == 54 and len(sig_unadj) == 28 and len(sig_adj) == 25 and sig_adj <= sig_unadj
missing_cpg = set(expr.index[expr["status"] == "missing_cpg"])
missing_rna = set(expr.index[expr["status"] == "missing_rna_gene"])
assert len(missing_cpg) == 21 and len(missing_rna) == 2
assert set(cms.index[cms["status"] == "missing_cpg"]) == missing_cpg

# cross-check against the published heat-map source table
chk = heat.loc[expr_ok.index]
assert np.nanmax(np.abs(chk["rho"] - expr_ok["rho"])) < 1e-12
assert np.nanmax(np.abs(chk["partial_rho_age_sex_site"] - expr_ok["partial_rho_age_sex_site"])) < 1e-12
assert np.nanmax(np.abs(heat.loc[cms_ok.index, "cms3_minus_cms2_pp"]
                        - cms_ok["mean_cms3_minus_cms2_beta"] * 100)) < 1e-10
assert int(heat["expression_bh77_significant"].sum()) == 28
assert int(heat["partial_expression_bh77_significant"].sum()) == 25
assert int(heat["cms_bh77_significant"].sum()) == 0

y = rows77.y_of(rows)

# ---------------------------------------------------------------- figure ----
fig = fs.figure(180, 225)

GUT_L, GUT_W = 0.5, 26.5
A_L, A_W = 27.5, 64.0
B_L, B_W = 98.0, 74.0
DOT_L, DOT_W = 173.0, 5.0
TOP, HEIGHT = 20.0, 196.0
A_MIN, A_MAX = -14.0, 24.0
B_MIN, B_MAX = -0.8, 0.4

fs.panel_label(fig, 0.5, 1.0, "A")
fs.panel_label(fig, 94.5, 1.0, "B")

rows77.draw_row_gutter(fig, GUT_L, TOP, GUT_W, HEIGHT, rows, bracket_frac=0.595, bands=True)

# ---- A: CMS3 - CMS2 ------------------------------------------------------
axA = fs.ax_mm(fig, A_L, TOP, A_W, HEIGHT)
rows77.apply_row_axis(axA, rows)
axA.set_xlim(A_MIN, A_MAX)
rows77.block_bands(axA, rows, A_MIN, A_MAX)
axA.set_xticks([-10, 0, 10, 20])
fs.light_grid(axA, axis="x")
fs.despine(axA, left=False, bottom=True)
axA.tick_params(axis="y", length=0)
axA.axvline(0.0, color=fs.C["rule"], lw=0.6, ls=(0, (3, 2)), zorder=1)
axA.set_xlabel(f"CMS3 − CMS2 (percentage points), {n_cms3} vs {n_cms2} tumors")

for probe, yy in y.items():
    rec = cms.loc[probe]
    if rec["status"] != "estimated":
        axA.text(A_MIN + 1.0, yy, "not measured", fontsize=fs.FS["tiny"],
                 color=fs.C["muted"], ha="left", va="center")
        continue
    lo, hi, mid = rec["ci_low"] * 100, rec["ci_high"] * 100, rec["mean_cms3_minus_cms2_beta"] * 100
    axA.plot([lo, hi], [yy, yy], color=fs.C["ink2"], lw=0.6, solid_capstyle="butt", zorder=2)
    axA.plot([mid], [yy], marker="o", ms=2.4, mfc=fs.C["ink2"], mec="white", mew=0.25,
             linestyle="none", zorder=3)

# ---- B: expression correlations -----------------------------------------
axB = fs.ax_mm(fig, B_L, TOP, B_W, HEIGHT)
rows77.apply_row_axis(axB, rows)
axB.set_xlim(B_MIN, B_MAX)
rows77.block_bands(axB, rows, B_MIN, B_MAX)
axB.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
fs.light_grid(axB, axis="x")
fs.despine(axB, left=False, bottom=True)
axB.tick_params(axis="y", length=0)
axB.axvline(0.0, color=fs.C["rule"], lw=0.6, ls=(0, (3, 2)), zorder=1)
axB.set_xlabel(f"Spearman ρ / partial ρ with gene expression (n = {n_expr} tumors)")

for probe, yy in y.items():
    rec = expr.loc[probe]
    if rec["status"] == "missing_cpg":
        axB.text(B_MIN + 0.02, yy, "not measured", fontsize=fs.FS["tiny"],
                 color=fs.C["muted"], ha="left", va="center")
        continue
    if rec["status"] == "missing_rna_gene":
        axB.text(B_MIN + 0.02, yy, "no RNA", fontsize=fs.FS["tiny"],
                 color=fs.C["muted"], ha="left", va="center")
        continue
    axB.plot([rec["rho_ci_low"], rec["rho_ci_high"]], [yy - 0.2, yy - 0.2],
             color=UNADJ_COLOR, lw=0.6, solid_capstyle="butt", zorder=2)
    axB.plot([rec["rho"]], [yy - 0.2], marker="o", ms=2.4, mfc="white", mec=UNADJ_COLOR,
             mew=0.6, linestyle="none", zorder=3)
    axB.plot([rec["partial_rho_ci_low"], rec["partial_rho_ci_high"]], [yy + 0.2, yy + 0.2],
             color=ADJ_COLOR, lw=0.6, solid_capstyle="butt", zorder=2)
    axB.plot([rec["partial_rho_age_sex_site"]], [yy + 0.2], marker="o", ms=2.4,
             mfc=ADJ_COLOR, mec="white", mew=0.25, linestyle="none", zorder=3)

# ---- significance strip --------------------------------------------------
axS = rows77.gutter_axes(fig, DOT_L, TOP, DOT_W, HEIGHT, rows)
for probe in sig_unadj:
    if probe in sig_adj:
        axS.plot([0.35], [y[probe]], marker="o", ms=1.9, mfc=fs.C["sig"], mec="none",
                 linestyle="none", zorder=3)
    else:
        axS.plot([0.35], [y[probe]], marker="o", ms=1.3, mfc=fs.C["muted"], mec="none",
                 linestyle="none", zorder=3)

# ---- legend strip --------------------------------------------------------
LEG_L, LEG_TOP, LEG_H = GUT_L, 5.0, 13.0
LEG_W = DOT_L + DOT_W - GUT_L
leg = fs.ax_mm(fig, LEG_L, LEG_TOP, LEG_W, LEG_H)
leg.set_xlim(0, 1); leg.set_ylim(0, 1); leg.axis("off")


def key(x, yy, label, mfc, mec, mew, line_color):
    leg.plot([x, x + 0.026], [yy, yy], color=line_color, lw=0.6,
             solid_capstyle="butt", clip_on=False)
    leg.plot([x + 0.013], [yy], marker="o", ms=2.4, mfc=mfc, mec=mec, mew=mew,
             linestyle="none", clip_on=False)
    leg.text(x + 0.034, yy, label, fontsize=fs.FS["small"], color=fs.C["ink"],
             ha="left", va="center")


def dot_key(x, yy, label, ms, color):
    leg.plot([x], [yy], marker="o", ms=ms, mfc=color, mec="none",
             linestyle="none", clip_on=False)
    leg.text(x + 0.011, yy, label, fontsize=fs.FS["small"], color=fs.C["ink"],
             ha="left", va="center")


key(0.005, 0.87, "A: mean CMS3 − CMS2 difference, bootstrap 95% CI",
    fs.C["ink2"], "white", 0.25, fs.C["ink2"])
dot_key(0.620, 0.87, "adjusted BH q < 0.05 (right margin)", 1.9, fs.C["sig"])
dot_key(0.855, 0.87, "unadjusted only", 1.3, fs.C["muted"])
key(0.005, 0.62, "B: Spearman ρ, 95% CI", "white", UNADJ_COLOR, 0.6, UNADJ_COLOR)
key(0.230, 0.62, "B: partial ρ adjusted for age, sex and site, 95% CI",
    ADJ_COLOR, "white", 0.25, ADJ_COLOR)
leg.text(0.005, 0.37, f"No CpG reaches BH q < 0.05 for the CMS contrast "
                      f"({len(cms_ok)} estimable CpGs).",
         fontsize=fs.FS["small"], color=fs.C["ink2"], ha="left", va="center")
def text_runs(x, yy, runs, **kw):
    """Draw consecutive (text, italic) runs left to right in axes coordinates.

    Used instead of mathtext so that the gene symbol keeps slanted digits while
    the surrounding sentence stays upright.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = leg.transAxes.inverted()
    cx = x
    for txt, ital in runs:
        t = leg.text(cx, yy, txt, style="italic" if ital else "normal",
                     ha="left", va="center", **kw)
        bb = t.get_window_extent(renderer=rend)
        cx += inv.transform((bb.x1, bb.y0))[0] - inv.transform((bb.x0, bb.y0))[0]
    return cx


text_runs(0.005, 0.12, [
    (f"{len(sig_unadj)} CpGs significant unadjusted and {len(sig_adj)} adjusted; "
     f"{len(missing_cpg)} CpGs absent from the matrix (\"not measured\") and "
     f"{len(missing_rna)} ", False),
    ("RALYL", True),
    (f" CpGs without RNA (\"no RNA\"), leaving {len(expr_ok)} estimable associations.", False),
], fontsize=fs.FS["small"], color=fs.C["ink2"])

fs.save(fig, "Figure_8", sources=[CMS, EXPR, HEATMAP, CONTEXT],
        notes=(f"A: cms_cpg_contrasts.tsv, beta x 100, {n_cms3} CMS3 vs {n_cms2} CMS2, "
               f"0 CpGs with welch_bh_q < 0.05. B: expression_cpg_associations.tsv, n = {n_expr}, "
               f"{len(sig_unadj)} unadjusted and {len(sig_adj)} adjusted BH q < 0.05, "
               f"{len(expr_ok)} estimable, {len(missing_cpg)} missing CpGs, "
               f"{len(missing_rna)} RALYL CpGs without RNA. Cross-checked against "
               "cpg_context_heatmap_source.tsv."))

print("CMS pp range %.3f .. %.3f" % (cms_ok["mean_cms3_minus_cms2_beta"].min() * 100,
                                     cms_ok["mean_cms3_minus_cms2_beta"].max() * 100))
print("CMS CI span %.3f .. %.3f" % (cms_ok["ci_low"].min() * 100, cms_ok["ci_high"].max() * 100))
print("rho %.3f..%.3f  CI %.3f..%.3f" % (expr_ok["rho"].min(), expr_ok["rho"].max(),
                                         expr_ok["rho_ci_low"].min(), expr_ok["rho_ci_high"].max()))
print("partial %.3f..%.3f  CI %.3f..%.3f" % (
    expr_ok["partial_rho_age_sex_site"].min(), expr_ok["partial_rho_age_sex_site"].max(),
    expr_ok["partial_rho_ci_low"].min(), expr_ok["partial_rho_ci_high"].max()))
print("sig unadj", len(sig_unadj), "sig adj", len(sig_adj), "only unadj", len(sig_unadj - sig_adj))
print("no RNA:", sorted(missing_rna))
