#!/usr/bin/env python3
"""Figure 1 - study overview (180 x 105 mm).

Three aligned columns drawn with matplotlib patches on an exact mm grid:
  A  historical candidate selection from four public HumanMethylation450 datasets
  B  Korean pyrosequencing cohort and the analyses fitted in it
  C  public individual-CpG analyses of the 77-CpG candidate subset

Every count is read from the frozen source files; box heights are measured with the
renderer so that text can never overflow its box.
"""
from __future__ import annotations

import json

import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import figstyle as fs

SRC = fs.F124 / "Study_Flow"
F_SEL = SRC / "selection_source.json"
F_LOCAL = SRC / "local_flow_source.json"
F_KOREAN = SRC / "korean_cohort_flow_source.tsv"

W, H = 180.0, 105.0
MARGIN = 2.0
GUTTER = 6.0
COL_W = (W - 2 * MARGIN - 2 * GUTTER) / 3.0          # 54.67 mm
COL_X = [MARGIN + i * (COL_W + GUTTER) for i in range(3)]

BOX_PAD = 0.4          # FancyBboxPatch pad (mm)
ROUND = 1.2            # corner radius (mm)
INNER_X = 1.5          # text inset from the box edge (mm)
INNER_TOP = 1.3
INNER_BOT = 1.4
HEAD_GAP = 0.7         # headline -> detail gap (mm)
LINE_H = 2.45          # detail line pitch (mm)
HEAD_H = 2.75          # headline line pitch (mm)
MIN_GAP = 4.5          # minimum vertical gap between boxes (mm)

BAND_TOP, BAND_BOT = 11.8, 102.6
LETTER_Y, HEADER_Y = 2.4, 6.2

EDGE = fs.C["box_edge"]
ARROW = fs.C["muted"]


# --------------------------------------------------------------------------- utils
def y(v: float) -> float:
    """mm from the top of the figure -> data y on an axes with ylim (0, H)."""
    return H - v


class Measurer:
    """Renderer-based text measurement in mm on the invisible full-figure axes."""

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        fig.canvas.draw()
        self.renderer = fig.canvas.get_renderer()
        self.px_per_mm = fig.get_size_inches()[0] * fig.dpi / W
        self._cache = {}

    def width(self, text: str, size: float, weight: str = "normal", style: str = "normal") -> float:
        key = (text, size, weight, style)
        if key not in self._cache:
            t = self.ax.text(0, 0, text, fontsize=size, fontweight=weight, fontstyle=style)
            bb = t.get_window_extent(self.renderer)
            t.remove()
            self._cache[key] = bb.width / self.px_per_mm
        return self._cache[key]

    def wrap(self, text: str, size: float, max_mm: float, weight: str = "normal") -> list:
        words, lines, cur = text.split(), [], ""
        for wd in words:
            trial = f"{cur} {wd}".strip()
            if cur and self.width(trial, size, weight) > max_mm:
                lines.append(cur)
                cur = wd
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines


# --------------------------------------------------------------------------- items
class Box:
    def __init__(self, head, details, tint=0, muted_last=False, genes=None, x_frac=(0.0, 1.0)):
        self.head, self.details, self.tint = head, details, tint
        self.muted_last, self.genes, self.x_frac = muted_last, genes, x_frac
        self.head_lines, self.detail_lines = [], []
        self.height = 0.0

    def measure(self, m: Measurer, col_w: float):
        w = col_w * (self.x_frac[1] - self.x_frac[0])
        inner = w - 2 * (BOX_PAD + INNER_X)
        self.head_lines = m.wrap(self.head, fs.FS["body"], inner, weight="bold")
        self.detail_lines = [m.wrap(d, fs.FS["small"], inner) for d in self.details]
        n_det = sum(len(d) for d in self.detail_lines)
        h = INNER_TOP + HEAD_H * len(self.head_lines)
        if n_det:
            h += HEAD_GAP + LINE_H * n_det
        if self.genes is not None:
            h += HEAD_GAP + LINE_H * ((len(self.genes) + 1) // 2)
        self.height = h + INNER_BOT
        return self.height

    def draw(self, ax, x0, top, col_w):
        w = col_w * (self.x_frac[1] - self.x_frac[0])
        xa = x0 + col_w * self.x_frac[0]
        ax.add_patch(FancyBboxPatch(
            (xa + BOX_PAD, y(top + self.height) + BOX_PAD),
            w - 2 * BOX_PAD, self.height - 2 * BOX_PAD,
            boxstyle=f"round,pad={BOX_PAD},rounding_size={ROUND}",
            linewidth=0.5, edgecolor=EDGE, facecolor=fs.C["box_fill"][self.tint], zorder=2))
        tx = xa + BOX_PAD + INNER_X
        cur = top + INNER_TOP
        for line in self.head_lines:
            ax.text(tx, y(cur), line, fontsize=fs.FS["body"], fontweight="bold",
                    color=fs.C["ink"], ha="left", va="top", zorder=3)
            cur += HEAD_H
        if self.detail_lines or self.genes is not None:
            cur += HEAD_GAP
        for j, block in enumerate(self.detail_lines):
            col = fs.C["muted"] if (self.muted_last and j == len(self.detail_lines) - 1) \
                else fs.C["ink2"]
            for line in block:
                ax.text(tx, y(cur), line, fontsize=fs.FS["small"], color=col,
                        ha="left", va="top", zorder=3)
                cur += LINE_H
        if self.genes is not None:
            inner = w - 2 * (BOX_PAD + INNER_X)
            half = (len(self.genes) + 1) // 2
            for k, g in enumerate(self.genes):
                gx = tx + (inner / 2.0) * (k // half)
                gy = cur + LINE_H * (k % half)
                ax.text(gx, y(gy), g, fontsize=fs.FS["small"], fontstyle="italic",
                        color=fs.C["ink"], ha="left", va="top", zorder=3)


class Pair:
    """Two boxes side by side occupying one row."""

    def __init__(self, left: Box, right: Box):
        self.left, self.right = left, right
        self.height = 0.0

    def measure(self, m, col_w):
        self.height = max(self.left.measure(m, col_w), self.right.measure(m, col_w))
        self.left.height = self.right.height = self.height
        return self.height

    def draw(self, ax, x0, top, col_w):
        self.left.draw(ax, x0, top, col_w)
        self.right.draw(ax, x0, top, col_w)


class Note:
    def __init__(self, text):
        self.text, self.lines, self.height = text, [], 0.0

    def measure(self, m, col_w):
        self.lines = m.wrap(self.text, fs.FS["small"], col_w - 0.5)
        self.height = LINE_H * len(self.lines)
        return self.height

    def draw(self, ax, x0, top, col_w):
        for i, line in enumerate(self.lines):
            ax.text(x0, y(top + i * LINE_H), line, fontsize=fs.FS["small"],
                    color=fs.C["muted"], ha="left", va="top", zorder=3)


# --------------------------------------------------------------------------- arrows
def varrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y(y0)), (x, y(y1)), arrowstyle="-|>",
                                 mutation_scale=4.0, lw=0.6, color=ARROW,
                                 shrinkA=0, shrinkB=0, zorder=4))


def line(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y(y0), y(y1)], color=ARROW, lw=0.6, solid_capstyle="round", zorder=4)


def split_arrows(ax, cx, y_from, y_to, xl, xr):
    mid = (y_from + y_to) / 2.0
    line(ax, cx, y_from, cx, mid)
    line(ax, xl, mid, xr, mid)
    varrow(ax, xl, mid, y_to)
    varrow(ax, xr, mid, y_to)


def merge_arrow(ax, x_from, cx, y_from, y_to):
    mid = (y_from + y_to) / 2.0
    line(ax, x_from, y_from, x_from, mid)
    line(ax, x_from, mid, cx, mid)
    varrow(ax, cx, mid, y_to)


# --------------------------------------------------------------------------- content
def build_content():
    sel = json.loads(F_SEL.read_text())["discovery"]
    loc = json.loads(F_LOCAL.read_text())
    kor = pd.read_csv(F_KOREAN, sep="\t")

    kr = {r.cohort: int(r.source_verified_pairs) for r in kor.itertuples()}
    kr_total = sum(kr.values())
    hyper, hypo = sel["hypermethylated_cpgs"], sel["hypomethylated_cpgs"]
    genes390 = sel["genes_annotated_to_hypermethylated_cpgs"]
    ds = sel["datasets"]
    q, d_hi = sel["q_cutoff_exclusive"], sel["hypermethylation_delta_beta_exclusive"]
    n_pat, n_spec = loc["korean"]["patients"], loc["korean"]["specimens"]
    rec, tis = loc["local_analyses"]["recurrence_cv"], loc["local_analyses"]["tissue_cv"]
    n_rec, n_evt = loc["local_analyses"]["recurrence_patients"], loc["local_analyses"]["recurrence_events"]
    pub = loc["public"]

    col_a = [
        Box("Public methylome screen",
            ["Four Illumina HumanMethylation450 datasets",
             "Adenoma and tumor versus non-cancerous mucosa",
             f"minfi; q < {q}; beta difference beyond ±{d_hi}"], tint=0),
        Pair(Box(f"{hyper:,} CpGs", ["hypermethylated", f"{genes390} annotated genes"],
                 tint=0, x_frac=(0.0, 0.52)),
             Box(f"{hypo} CpGs", ["hypomethylated", "not carried forward"],
                 tint=0, muted_last=True, x_frac=(0.55, 1.0))),
        Box("Literature review",
            ["Prioritization for colorectal cancer relevance; recurrence outcomes were not "
             "used for selection"], tint=0),
        Box("Ten candidate genes", [], tint=3, genes=list(sel["selected_genes"])),
        Note(f"{ds[0]} and {ds[1]}, adenoma sets; {ds[2]} and {ds[3]}, tumor sets. The screen "
             f"counts are historical source records and were not recomputed here."),
    ]

    col_b = [
        Box(f"{n_pat} Korean patients, {n_spec} paired specimens",
            ["Pyrosequencing of the ten genes in tumor and adjacent mucosa"], tint=1),
        Box("Paired tumor − mucosa differences",
            ["Per-gene mean differences with BCa 95% confidence intervals from 5,000 "
             "patient-pair resamples"], tint=1),
        Box("Patient-level patterns",
            ["Between-gene Spearman correlations and principal component analysis of "
             "gene-standardized differences"], tint=1),
        Box("Tissue classification",
            [f"Ridge logistic regression; {tis['outer_folds']} outer folds × "
             f"{tis['outer_repeats']} repeats, 4 inner folds; single-gene comparator"], tint=1),
        Box("Recurrence models",
            [f"{n_rec} stage I–III non-palliative patients, {n_evt} events",
             f"Clinical, methylation and combined ridge Cox; {rec['outer_folds']} outer folds × "
             f"{rec['outer_repeats']} repeats"], tint=1),
    ]

    col_c = [
        Box("77 source-derived candidate CpGs",
            ["Fixed list recorded for the ten genes in the original study; each CpG analyzed "
             "separately"], tint=2),
        Box("Three Korean MethylationEPIC cohorts",
            [f"CMCBSN {kr['CMCBSN']}, SNUH {kr['SNUH']} and ASAN {kr['ASAN']} source-defined "
             f"pairs ({kr_total} in total)",
             "56 of the 77 CpGs present in the processed matrices"], tint=2),
        Box("Six public tissue and lesion cohorts",
            ["Healthy mucosa, adjacent mucosa, adenoma and carcinoma contrasts",
             "Source-defined tumor − adjacent pairs"], tint=2),
        Box("Expression, stromal and CMS context",
            ["Colonomics and ColoCare expression; ESTIMATE stromal score",
             "Consensus molecular subtype (CMS) contrasts"], tint=2),
        Box("Public-array tissue classifiers",
            [f"Colonomics {pub['colonomics_pairs']} patients; GSE119526 "
             f"{pub['gse119526_pairs']} patients"], tint=2),
    ]
    return col_a, col_b, col_c


# --------------------------------------------------------------------------- layout
def layout(items, m, col_w):
    heights = [it.measure(m, col_w) for it in items]
    total = sum(heights)
    n_gap = len(items) - 1
    gap = max(MIN_GAP, (BAND_BOT - BAND_TOP - total) / n_gap)
    tops, cur = [], BAND_TOP
    for h in heights:
        tops.append(cur)
        cur += h + gap
    return tops, heights, gap


def main():
    fig = fs.figure(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    m = Measurer(fig, ax)
    columns = build_content()
    headers = ["Historical candidate selection",
               "Korean pyrosequencing cohort",
               "Public individual-CpG analyses"]

    for ci, (items, header) in enumerate(zip(columns, headers)):
        x0 = COL_X[ci]
        cx = x0 + COL_W / 2.0
        fs.panel_label(fig, x0, LETTER_Y, "ABC"[ci])
        ax.text(x0, y(HEADER_Y), header, fontsize=fs.FS["body"], fontweight="bold",
                color=fs.C["ink"], ha="left", va="top")
        ax.plot([x0, x0 + COL_W], [y(HEADER_Y + 3.4)] * 2, color=fs.C["rule"], lw=0.5,
                solid_capstyle="butt", zorder=1)

        tops, heights, _gap = layout(items, m, COL_W)
        print("ABC"[ci], "gap %.2f" % _gap, "bottom %.2f" % (tops[-1] + heights[-1]),
              ["%.1f" % h for h in heights])
        for it, top, h in zip(items, tops, heights):
            it.draw(ax, x0, top, COL_W)

        # connectors
        for k in range(len(items) - 1):
            a, b = items[k], items[k + 1]
            y_from, y_to = tops[k] + heights[k], tops[k + 1]
            if isinstance(b, Note):
                continue
            if isinstance(b, Pair):
                xl = x0 + COL_W * (b.left.x_frac[0] + b.left.x_frac[1]) / 2
                xr = x0 + COL_W * (b.right.x_frac[0] + b.right.x_frac[1]) / 2
                split_arrows(ax, cx, y_from, y_to, xl, xr)
            elif isinstance(a, Pair):
                xl = x0 + COL_W * (a.left.x_frac[0] + a.left.x_frac[1]) / 2
                merge_arrow(ax, xl, cx, y_from, y_to)
            else:
                varrow(ax, cx, y_from, y_to)

    man = fs.save(fig, "Figure_1", sources=[F_SEL, F_LOCAL, F_KOREAN],
                  notes="Study overview. Counts read from selection_source.json, "
                        "local_flow_source.json and korean_cohort_flow_source.tsv; the "
                        "56-of-77 CpG coverage statement follows the manuscript Methods.")
    print(man["width_mm"], man["height_mm"])


if __name__ == "__main__":
    main()
