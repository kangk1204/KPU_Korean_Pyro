"""Figure 7 - Individual CpG associations with expression and molecular context.

Dot matrix: 77 fixed CpGs as columns (gene blocks, ascending hg19 position),
split into two stacked halves with identical column pitch; 7 analysis rows in
three encoding groups (rho, Kruskal-Wallis H, percentage points).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable

import figstyle as fs

MM = fs.MM
CTX = fs.F124 / "Public_Context" / "Public_Context_source_data.tsv"
PROBECTX = fs.F124 / "Public_CpG_contrasts" / "input_probe_context.tsv"

MM2PT2 = (1 / MM) ** 2 / 72.0 ** 2 * 72.0 ** 2  # placeholder, see mm2_to_pt2
def mm2_to_pt2(a_mm2: float) -> float:
    return a_mm2 * (72.0 / 25.4) ** 2


# ------------------------------------------------------------------ rows/cols
def probe_order() -> pd.DataFrame:
    p = pd.read_csv(PROBECTX, sep="\t")
    p["_g"] = p["fixed_gene"].map({g: i for i, g in enumerate(fs.GENES)})
    p = p.sort_values(["_g", "pos"], kind="mergesort").reset_index(drop=True)
    return p[["fixed_gene", "probe", "chr", "pos", "promoter_match"]]


ROWS = [
    # (label, cohort, model, analysis, group, value column)
    ("Expression rho, Colonomics (n 94)", "Colonomics", "unadjusted",
     "methylation_expression_spearman", 0, "effect"),
    ("Expression rho, adjusted (n 94)", "Colonomics", "rank_residual_age_sex_site_stroma",
     "methylation_expression_spearman", 0, "effect"),
    ("Expression rho, ColoCare (n 77)", "ColoCare_GSE101764_GSE106582", "unadjusted",
     "methylation_expression_spearman", 0, "effect"),
    ("Stromal rho, Colonomics (n 94)", "Colonomics", "unadjusted",
     "stromal_spearman", 0, "effect"),
    ("CMS Kruskal–Wallis H (n 83)", "Colonomics", "source_cms_labels",
     "CMS_global_Kruskal", 1, "effect"),
    ("MATCH CMS3 − CMS2 (22 vs 124)", "GSE164811_MATCH", "Welch_CMS3_minus_CMS2",
     "CMS3-CMS2", 2, "effect_percentage_points"),
    ("MATCH CMS3 − CMS2, adjusted", "GSE164811_MATCH",
     "sex_site_adjusted_OLS_CMS3_minus_CMS2", "CMS3-CMS2_adjusted", 2,
     "effect_percentage_points"),
]

HALVES = [fs.GENES[:5], fs.GENES[5:]]

# ------------------------------------------------------------------ geometry
W = 180.0
LEFT, RIGHT_PAD = 37.0, 2.0
GENE_GAP_MM = 1.4
N_TOP = 49
PITCH = (W - LEFT - RIGHT_PAD - GENE_GAP_MM * 4) / N_TOP     # mm per column
ROW_PITCH = 4.4
GRP_GAP_MM = 1.5
GRID_H = 7 * ROW_PITCH + 2 * GRP_GAP_MM                      # 31.0 mm

TOP_A, TOP_B = 6.6, 61.5
CPGID_MM = 11.5
H = 111.0                                                    # figure height

RHO_LIM, PP_LIM = 0.7, 25.0
SIZE_CAP = 6.0
AREA_MIN, AREA_MAX = 2.0, 5.0        # mm^2 at -log10 q = 0 and >= 6
RING_PAD_MM = 0.30


def area_mm2(mlq: float) -> float:
    v = 0.0 if not np.isfinite(mlq) else min(max(mlq, 0.0), SIZE_CAP)
    return AREA_MIN + (AREA_MAX - AREA_MIN) * v / SIZE_CAP


def ring_area_mm2(a: float) -> float:
    d = 2 * np.sqrt(a / np.pi)
    return np.pi * (d / 2 + RING_PAD_MM) ** 2


def fig_text(fig, x, y, s, **kw):
    w, h = fig.get_size_inches()
    return fig.text(x * MM / w, 1 - y * MM / h, s, **kw)


def row_y_units():
    """Row centres in row-pitch units, including 1.5 mm group gaps."""
    gap_u = GRP_GAP_MM / ROW_PITCH
    ys, off = [], 0.0
    for i, r in enumerate(ROWS):
        if i > 0 and r[4] != ROWS[i - 1][4]:
            off += gap_u
        ys.append(i + 0.5 + off)
    return ys, ROWS[-1][4] and (len(ROWS) + 2 * gap_u)


def build_half(genes, order):
    """Return (col_centres_units, gene_spans_units, probes) for a half."""
    gap_u = GENE_GAP_MM / PITCH
    centres, spans, probes = [], [], []
    off, k = 0.0, 0
    for gi, g in enumerate(genes):
        sub = order[order["fixed_gene"] == g]
        if gi > 0:
            off += gap_u
        x0 = k + off
        for _, r in sub.iterrows():
            centres.append(k + off + 0.5)
            probes.append(r)
            k += 1
        spans.append((g, x0, k + off))
    return np.array(centres), spans, pd.DataFrame(probes).reset_index(drop=True)


def gene_bracket_h(ax, x0, x1, y, label):
    ax.plot([x0 + 0.06, x1 - 0.06], [y, y], color=fs.C["ink2"], lw=0.6,
            clip_on=False, solid_capstyle="butt")
    ax.text((x0 + x1) / 2, y - 0.22, label, ha="center", va="bottom",
            fontsize=fs.FS["body"], color=fs.C["ink"], style="italic", clip_on=False)


def main():
    c = pd.read_csv(CTX, sep="\t")
    order = probe_order()

    report = {"n_rows": len(c), "n_probes": order.shape[0],
              "promoter_cpgs": int(order["promoter_match"].sum()),
              "status_counts": c["status"].value_counts().to_dict()}

    # per-row slices keyed by probe
    row_data = []
    for lab, coh, mod, ana, grp, vcol in ROWS:
        sub = c[(c["cohort"] == coh) & (c["model"] == mod) & (c["analysis"] == ana)]
        assert len(sub) == 77, (lab, len(sub))
        row_data.append(sub.set_index("cpg"))
    report["ring_counts"] = {ROWS[i][0]: int((s["q_BH77"] < 0.05).sum())
                             for i, s in enumerate(row_data)}
    report["non_estimable"] = {ROWS[i][0]: sorted(s.index[s["status"] != "estimated"])
                               for i, s in enumerate(row_data)
                               if (s["status"] != "estimated").any()}
    report["value_ranges"] = {ROWS[i][0]: (float(np.nanmin(s[ROWS[i][5]])),
                                           float(np.nanmax(s[ROWS[i][5]])))
                              for i, s in enumerate(row_data)}

    norm_rho = TwoSlopeNorm(vcenter=0.0, vmin=-RHO_LIM, vmax=RHO_LIM)
    hmax = float(np.nanmax(row_data[4]["effect"]))
    norm_h = Normalize(vmin=0.0, vmax=hmax)
    norm_pp = TwoSlopeNorm(vcenter=0.0, vmin=-PP_LIM, vmax=PP_LIM)
    NORMS = [norm_rho, norm_rho, norm_rho, norm_rho, norm_h, norm_pp, norm_pp]
    CMAPS = [fs.CMAP_DIV] * 4 + [fs.CMAP_SEQ] + [fs.CMAP_DIV] * 2
    report["H_max"] = hmax

    ys, total_u = row_y_units()
    gap_u = GRP_GAP_MM / ROW_PITCH
    ylim = len(ROWS) + 2 * gap_u

    fig = fs.figure(W, H)

    for half_i, genes in enumerate(HALVES):
        centres, spans, probes = build_half(genes, order)
        n_units = centres[-1] + 0.5
        width_mm = n_units * PITCH
        top = TOP_A if half_i == 0 else TOP_B
        ax = fs.ax_mm(fig, LEFT, top, width_mm, GRID_H)
        ax.set_xlim(0, n_units); ax.set_ylim(ylim, 0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])

        # very light row bands to guide the eye
        for ri, y in enumerate(ys):
            if ri % 2 == 0:
                ax.add_patch(plt.Rectangle((0, y - 0.5), n_units, 1.0,
                                           facecolor="#F7F7F7", edgecolor="none", zorder=0))
        # gene block separators
        for _, x0, x1 in spans[1:]:
            ax.plot([x0 - GENE_GAP_MM / PITCH / 2] * 2, [0, ylim],
                    color=fs.C["grid"], lw=0.4, zorder=0)

        for ri, (lab, *_rest) in enumerate(ROWS):
            sub = row_data[ri]
            y = ys[ri]
            xs, areas, faces, edges, elws = [], [], [], [], []
            rx, ra = [], []
            for k, pr in probes.iterrows():
                probe = pr["probe"]
                r = sub.loc[probe]
                x = centres[k]
                val = r[ROWS[ri][5]]
                if r["status"] != "estimated" or not np.isfinite(val):
                    ax.text(x, y, "×", ha="center", va="center",
                            fontsize=fs.FS["tiny"], color=fs.C["na_mark"], zorder=3)
                    continue
                a = area_mm2(r["minus_log10_q"])
                xs.append(x); areas.append(mm2_to_pt2(a))
                faces.append(CMAPS[ri](NORMS[ri](val)))
                if bool(pr["promoter_match"]):
                    edges.append(fs.C["ink"]); elws.append(0.4)
                else:
                    edges.append(fs.C["muted"]); elws.append(0.3)
                if bool(r["q_lt_0_05"]):
                    rx.append(x); ra.append(mm2_to_pt2(ring_area_mm2(a)))
            if rx:
                ax.scatter(rx, [y] * len(rx), s=ra, facecolors="none",
                           edgecolors=fs.C["ink2"], linewidths=0.6, zorder=2)
            ax.scatter(xs, [y] * len(xs), s=areas, facecolors=faces,
                       edgecolors=edges, linewidths=elws, zorder=3)

        # gene brackets above the half
        for g, x0, x1 in spans:
            gene_bracket_h(ax, x0, x1, -0.55, g)
        # CpG identifiers under the half
        for k, pr in probes.iterrows():
            ax.text(centres[k], ylim + 0.18, pr["probe"], rotation=90, ha="center",
                    va="top", fontsize=fs.FS["tiny"], color=fs.C["ink2"], clip_on=False)
        # row labels (left of the top half only would leave the lower half
        # unlabeled, so both halves carry them)
        for ri, (lab, *_rest) in enumerate(ROWS):
            ax.text(-0.35, ys[ri], lab, ha="right", va="center",
                    fontsize=fs.FS["small"], color=fs.C["ink"], clip_on=False)

    # ------------------------------------------------------------ legends
    lx = 124.0                                # right of the bottom half
    cb_w = 36.0
    sm_rho = ScalarMappable(norm=norm_rho, cmap=fs.CMAP_DIV); sm_rho.set_array([])
    sm_h = ScalarMappable(norm=norm_h, cmap=fs.CMAP_SEQ); sm_h.set_array([])
    sm_pp = ScalarMappable(norm=norm_pp, cmap=fs.CMAP_DIV); sm_pp.set_array([])
    fs.colorbar(fig, sm_rho, lx, 62.5, cb_w, 2.2,
                "Spearman / rank-residual rho", ticks=[-0.7, 0, 0.7])
    fs.colorbar(fig, sm_h, lx, 71.0, cb_w, 2.2,
                "Kruskal–Wallis H (CMS)", ticks=[0, 10, 20])
    fs.colorbar(fig, sm_pp, lx, 79.5, cb_w, 2.2,
                "CMS3 − CMS2 (percentage points)", ticks=[-25, 0, 25])

    # size key
    KW = 54.0
    kax = fs.ax_mm(fig, lx, 88.0, KW, 8.0)
    kax.set_xlim(0, KW); kax.set_ylim(0, 8); kax.axis("off")
    kax.text(0, 7.8, "Marker size, −log$_{10}$ $q$ (capped at 6)",
             fontsize=fs.FS["small"], ha="left", va="top", color=fs.C["ink"])
    for i, (mlq, lab) in enumerate([(-np.log10(0.05), "0.05"), (3, "1e−3"), (6, "1e−6")]):
        x = 5.0 + i * 15.0
        a = area_mm2(mlq)
        kax.scatter([x], [4.0], s=mm2_to_pt2(a), facecolors="#F3F3F1",
                    edgecolors=fs.C["ink"], linewidths=0.4)
        kax.text(x + 2.6, 4.0, f"$q$ = {lab}", ha="left", va="center",
                 fontsize=fs.FS["small"], color=fs.C["ink2"])

    # marker/encoding key
    max_ = fs.ax_mm(fig, lx, 96.5, KW, 11.0)
    max_.set_xlim(0, KW); max_.set_ylim(0, 11); max_.axis("off")
    a0 = area_mm2(3)
    entries = [
        ("ring", "$q$ < 0.05 within the 77-CpG family"),
        ("black", "promoter CpG (black edge)"),
        ("grey", "non-promoter CpG (grey edge)"),
        ("x", "not estimable"),
    ]
    for i, (kind, lab) in enumerate(entries):
        y = 9.6 - i * 2.7
        if kind == "ring":
            max_.scatter([2.0], [y], s=mm2_to_pt2(ring_area_mm2(a0)), facecolors="none",
                         edgecolors=fs.C["ink2"], linewidths=0.6)
            max_.scatter([2.0], [y], s=mm2_to_pt2(a0), facecolors="#E48B5C",
                         edgecolors=fs.C["ink"], linewidths=0.4)
        elif kind == "black":
            max_.scatter([2.0], [y], s=mm2_to_pt2(a0), facecolors="#E48B5C",
                         edgecolors=fs.C["ink"], linewidths=0.4)
        elif kind == "grey":
            max_.scatter([2.0], [y], s=mm2_to_pt2(a0), facecolors="#E48B5C",
                         edgecolors=fs.C["muted"], linewidths=0.3)
        else:
            max_.text(2.0, y, "×", ha="center", va="center", fontsize=fs.FS["tiny"],
                      color=fs.C["na_mark"])
        max_.text(4.4, y, lab, ha="left", va="center", fontsize=fs.FS["small"],
                  color=fs.C["ink"])

    man = fs.save(fig, "Figure_7", sources=[CTX, PROBECTX],
                  notes=("77 CpGs (columns, two stacked halves, identical pitch) x 7 "
                         "context analyses; rho +/-0.7, H 0-max, pp +/-25; ring q_BH77<0.05; "
                         "size -log10 q capped at 6; black edge = promoter CpG."))
    report["manifest"] = man
    report["pitch_mm"] = PITCH
    report["fig_mm"] = (man["width_mm"], man["height_mm"])
    return report


if __name__ == "__main__":
    import json
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k != "manifest"}, indent=1, default=str))
    print("sources:", [(s["path"].split("/")[-1], s["sha256"]) for s in r["manifest"]["sources"]])
