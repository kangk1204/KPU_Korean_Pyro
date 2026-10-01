"""Figure 6 - Individual CpG methylation in adenomas and carcinomas (88 x 225 mm).

Three columns: GSE48684 adenoma - healthy (A-H), GSE48684 carcinoma - adenoma
(T-A) and GSE77954 carcinoma - adenoma (T-A) at the 77 fixed CpGs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

import figstyle as fs
import rows77

L = fs.F124 / "Lesion_CpG"
SOURCE = L / "Lesion_CpG_source_data.tsv"
CONTEXT = L / "input_probe_context.tsv"

LIMIT = 60.0                     # symmetric color limit, percentage points
COLUMNS = [("GSE48684", "A-H", "A − H"), ("GSE48684", "T-A", "T − A"), ("GSE77954", "T-A", "T − A")]

# ---------------------------------------------------------------- data ----
rows = rows77.load_rows(CONTEXT)
src = pd.read_csv(SOURCE, sep="\t")
assert len(src) == 231, len(src)

Q_COL = "q_BH_family"             # BH q over p_for_bh across all 231 lesion tests
assert Q_COL in src.columns
# independent re-computation of the BH step-up over the 231 p_for_bh values
order = np.argsort(src["p_for_bh"].to_numpy())
p_sorted = src["p_for_bh"].to_numpy()[order]
n = len(p_sorted)
q_sorted = np.minimum(np.minimum.accumulate((p_sorted * n / np.arange(1, n + 1))[::-1])[::-1], 1.0)
q_check = np.empty(n)
q_check[order] = q_sorted
assert np.nanmax(np.abs(q_check - src[Q_COL].to_numpy())) < 1e-12

y = rows77.y_of(rows)
cells = {}
for cohort, contrast, _ in COLUMNS:
    sub = src[(src["cohort"] == cohort) & (src["contrast"] == contrast)].set_index("probe")
    assert len(sub) == 77, (cohort, contrast, len(sub))
    cells[(cohort, contrast)] = sub

counts = {k: (int(v["n_high"].iloc[0]), int(v["n_low"].iloc[0])) for k, v in cells.items()}
n_sig = {k: int((v[Q_COL] < 0.05).sum()) for k, v in cells.items()}
assert list(n_sig.values()) == [77, 1, 9], n_sig
for k, v in cells.items():
    if k[1] == "T-A":
        assert not ((v[Q_COL] < 0.05) & (v["effect_pp"] > 0)).any(), k

# ---------------------------------------------------------------- figure ----
fig = fs.figure(88, 225)

GUT_L, GUT_W = 0.5, 26.5
HM_L, HM_W = 27.5, 58.5
TOP, HEIGHT = 19.0, 180.0
COL_W = HM_W / 3

norm = TwoSlopeNorm(vcenter=0.0, vmin=-LIMIT, vmax=LIMIT)
ax = fs.ax_mm(fig, HM_L, TOP, HM_W, HEIGHT)
ax.set_xlim(-0.5, 2.5)
rows77.apply_row_axis(ax, rows, pad=0.55)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

for j, (cohort, contrast, _) in enumerate(COLUMNS):
    sub = cells[(cohort, contrast)]
    for probe, yy in y.items():
        rec = sub.loc[probe]
        if rec["status"] != "estimated" or pd.isna(rec["effect_pp"]):
            fs.na_cell(ax, j, yy)
            continue
        ax.add_patch(Rectangle((j - 0.5, yy - 0.5), 1.0, 1.0,
                               facecolor=fs.CMAP_DIV(norm(rec["effect_pp"])),
                               edgecolor="none", zorder=1))
        if rec[Q_COL] < 0.05:
            fs.sig_dot(ax, j, yy, size=1.3)

for j in range(1, 3):                             # thin white column separators
    ax.plot([j - 0.5, j - 0.5], [rows["y"].min() - 0.55, rows["y"].max() + 0.55],
            color="white", lw=0.8, zorder=4, clip_on=False)

rows77.draw_row_gutter(fig, GUT_L, TOP, GUT_W, HEIGHT, rows,
                       bracket_frac=0.595, pad=0.55, bands=True)

# ---- column headers ------------------------------------------------------
head = fs.ax_mm(fig, HM_L, 4.5, HM_W, 13.0)
head.set_xlim(-0.5, 2.5); head.set_ylim(0, 1); head.axis("off")
for j, (cohort, contrast, pretty) in enumerate(COLUMNS):
    hi, lo = counts[(cohort, contrast)]
    head.text(j, 0.80, cohort, ha="center", va="center", fontsize=fs.FS["body"], color=fs.C["ink"])
    head.text(j, 0.46, pretty, ha="center", va="center", fontsize=fs.FS["body"], color=fs.C["ink"])
    head.text(j, 0.13, f"{hi} vs {lo}", ha="center", va="center",
              fontsize=fs.FS["small"], color=fs.C["ink2"])

# ---- colorbar and footnote ----------------------------------------------
cb = fs.colorbar(fig, ScalarMappable(norm=norm, cmap=fs.CMAP_DIV),
                 HM_L + (HM_W - 44.0) / 2, 203.0, 44.0, 2.5,
                 "Mean difference (percentage points)", ticks=[-60, -30, 0, 30, 60])

foot = fs.ax_mm(fig, GUT_L, 212.0, 87.0 - GUT_L, 11.0)
foot.set_xlim(0, 1); foot.set_ylim(0, 1); foot.axis("off")
foot.plot([0.010], [0.88], marker="o", ms=1.3, mfc=fs.C["sig"], mec="none",
          linestyle="none", clip_on=False)
lines = [
    (0.88, 0.030, "BH q < 0.05 within the 231-test lesion family."),
    (0.63, 0.0, "A − H, adenoma minus healthy mucosa; T − A, carcinoma minus adenoma."),
    (0.38, 0.0, "Counts under each column header are the two group sizes."),
    (0.13, 0.0, f"Color scale symmetric about zero at ±{LIMIT:.0f} pp. "
                f"Significant CpGs: {n_sig[('GSE48684', 'A-H')]} / "
                f"{n_sig[('GSE48684', 'T-A')]} / {n_sig[('GSE77954', 'T-A')]}."),
]
for yy, xx, txt in lines:
    foot.text(xx, yy, txt, fontsize=fs.FS["small"], color=fs.C["ink2"], ha="left", va="center")

fs.save(fig, "Figure_6", sources=[SOURCE, CONTEXT],
        notes=("77 fixed CpGs x 3 lesion contrasts; color = effect_pp, diverging, symmetric +/-60 pp; "
               "dots = q_BH_family < 0.05 (BH over p_for_bh across all 231 rows, re-derived and verified). "
               f"Significant: {n_sig}."))

print("dots per column:", n_sig)
print("group sizes:", counts)
for k, v in cells.items():
    print(k, "effect range %.2f .. %.2f" % (v["effect_pp"].min(), v["effect_pp"].max()))
