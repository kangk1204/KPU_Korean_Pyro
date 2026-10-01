#!/usr/bin/env python3
"""Render final-size study design from documented historical source counts."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "119_Figure1_ABC_Revision_20260908"
EXPORT_SUFFIXES = (".png", ".pdf", ".svg")


def strip_svg(path: Path) -> None:
    text = path.read_text()
    svg_start = text.find("<svg")
    if svg_start >= 0:
        text = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n' + text[svg_start:]
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text, count=1, flags=re.S)
    path.write_text(text)


def box(ax, xy, wh, title, body, fc="#f7f7f7", ec="#555555", title_fs=6.0, body_fs=5.0):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.02", facecolor=fc, edgecolor=ec, linewidth=0.65)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h - 0.040, title, ha="center", va="top", fontsize=title_fs, fontweight="bold")
    ax.text(x + w/2, y + h*0.23, body, ha="center", va="center", fontsize=body_fs, linespacing=1.12)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, linewidth=0.7, color="#333333"))


def main():
    out = ROOT / "figures" / "Study_Flow"
    out.mkdir(parents=True, exist_ok=True)
    registry = json.loads((OLD / "registry/original_gene_selection.json").read_text())
    shutil.copy2(OLD / "registry/original_gene_selection.json", out / "selection_source.json")
    shutil.copy2(ROOT / "registry/sample_flow.tsv", out / "korean_cohort_flow_source.tsv")
    shutil.copy2(OLD / "submission/figure_source_data/F1_original_design_counts.json", out / "local_flow_source.json")
    genes = registry["discovery"]["selected_genes"]

    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype":"none", "pdf.fonttype":42, "font.size":6.2})
    fig = plt.figure(figsize=(7.05, 5.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.08, 1.0], hspace=0.30, wspace=0.24)
    ax_a = fig.add_subplot(gs[:, 0]); ax_b = fig.add_subplot(gs[0, 1]); ax_c = fig.add_subplot(gs[1, 1])
    for ax in [ax_a, ax_b, ax_c]:
        ax.set(xlim=(0,1), ylim=(0,1)); ax.axis("off")
    ax_a.text(0, 1.02, "A  Historical candidate selection", fontsize=7.6, fontweight="bold", va="bottom")
    ax_b.text(0, 1.02, "B  Korean pyrosequencing cohort", fontsize=7.6, fontweight="bold", va="bottom")
    ax_c.text(0, 1.02, "C  Public individual-CpG analyses", fontsize=7.6, fontweight="bold", va="bottom")

    box(ax_a, (0.05,0.78), (0.90,0.16), "Original public 450K screen", "Four datasets\nNormal, adenoma and/or CRC groups", fc="#eef3fb")
    box(ax_a, (0.05,0.56), (0.42,0.15), "Hypermethylated", "1,088 CpGs\n390 annotated genes", fc="#fbe9e7")
    box(ax_a, (0.53,0.56), (0.42,0.15), "Hypomethylated", "345 CpGs\ntracked, not selected", fc="#e8f0fe")
    box(ax_a, (0.05,0.34), (0.90,0.14), "Literature-prioritized panel", "10 genes; recurrence not used for selection", fc="#fff4de")
    ax_a.text(0.50, 0.235, "Selected genes", ha="center", va="center", fontsize=6.2, fontweight="bold")
    for i, gene in enumerate(genes):
        x = 0.12 + (i % 2) * 0.40
        y = 0.18 - (i // 2) * 0.032
        ax_a.text(x, y, gene, fontsize=5.9, fontstyle="italic", ha="left", va="center")
    ax_a.text(0.05, 0.025, "Counts are historical source records, not freshly recomputed in this revision.", fontsize=5.0, color="#555555")
    for a,b in [((0.50,0.78),(0.29,0.71)), ((0.50,0.78),(0.71,0.71)), ((0.29,0.56),(0.50,0.48))]: arrow(ax_a,a,b)

    box(ax_b, (0.05,0.66), (0.90,0.23), "Local paired specimens", "87 patients / 174 specimens\n10 supplied gene-level assays", fc="#eef3fb")
    box(ax_b, (0.05,0.36), (0.90,0.20), "Tissue methylation", "Paired differences, correlations\nand tissue classification", fc="#f7f7f7")
    box(ax_b, (0.05,0.08), (0.90,0.18), "Recurrence models", "82 eligible patients / 14 events\nClinical vs methylation blocks", fc="#fff4de")
    arrow(ax_b,(0.5,0.66),(0.5,0.56)); arrow(ax_b,(0.5,0.36),(0.5,0.26))

    box(ax_c, (0.05,0.68), (0.90,0.20), "Fixed 77 candidate CpGs", "No CpG averaging; each site analyzed separately", fc="#eef3fb")
    box(ax_c, (0.05,0.40), (0.90,0.18), "Korean public arrays", "CMCBSN 103; SNUH 142; ASAN 128 pairs\n56 measured CpGs per matrix", fc="#f7f7f7")
    box(ax_c, (0.05,0.11), (0.90,0.19), "External tissue/context analyses", "Public tissue and lesion contrasts\nExpression, stromal score, CMS, ML", fc="#f7f7f7")
    ax_c.text(0.05, 0.025, "Dataset roles: GSE139404/GSE129364 adenoma sets; GSE101764/GSE68838 tumor-context sets.", fontsize=5.0, color="#555555")
    arrow(ax_c,(0.5,0.68),(0.5,0.58)); arrow(ax_c,(0.5,0.40),(0.5,0.30))

    for suffix in EXPORT_SUFFIXES:
        path = out / f"Study_Flow{suffix}"
        fig.savefig(path, dpi=600 if suffix == ".png" else None, bbox_inches="tight", pad_inches=0.035)
        if suffix == ".svg": strip_svg(path)
    plt.close(fig)
    (out / "legend.md").write_text("Figure 1. Candidate selection and study design. A, Historical public 450K screen counts from the original study records, including 1,088 hypermethylated and 345 hypomethylated CpGs; literature review prioritized ten genes from 390 genes annotated to hypermethylated CpGs. The historical counts and complete selection rules were not independently reconstructed. B, Korean paired pyrosequencing and recurrence-analysis cohorts. C, Public individual-CpG reanalysis, with CpGs analyzed separately and no cross-CpG methylation averaging.\n")
    shutil.copy2(Path(__file__), out / "plot_study_flow.py")

if __name__ == "__main__":
    main()
