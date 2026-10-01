#!/usr/bin/env python3
"""Regenerate editable local PSQ Figure 2 and Figure 3 from frozen source data."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent / "119_Figure1_ABC_Revision_20260908" / "figures" / "source_data"
SOURCE_SUBMISSION = ROOT.parent / "119_Figure1_ABC_Revision_20260908" / "submission" / "figure_source_data"
F2_DIR = ROOT / "figures" / "Paired_PSQ"
F3_DIR = ROOT / "figures" / "Delta_Patterns"
GENE_ORDER = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
EXPORT_SUFFIXES = (".png", ".pdf", ".svg")
COL_T = "#0072B2"
COL_N = "#D55E00"
COL_EVENT = "#009E73"


def strip_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    svg_start = text.find("<svg")
    if svg_start >= 0:
        text = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n' + text[svg_start:]
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text, count=1, flags=re.S)
    path.write_text(text, encoding="utf-8")


def copy_source(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def save(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in EXPORT_SUFFIXES:
        target = outdir / f"{stem}{suffix}"
        fig.savefig(target, dpi=600 if suffix == ".png" else None, facecolor="white")
        if suffix == ".svg":
            strip_svg(target)
    plt.close(fig)


def configure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def figure2() -> None:
    outdir = F2_DIR
    raw_path = copy_source(SOURCE_ROOT / "F2_paired_raw_source_data.tsv", outdir)
    effect_path = copy_source(SOURCE_SUBMISSION / "F2_paired_effect_source_data.tsv", outdir)
    raw = pd.read_csv(raw_path, sep="\t")
    effects = pd.read_csv(effect_path, sep="\t").set_index("gene")
    fig, axes = plt.subplots(2, 5, figsize=(7.05, 5.25), sharey=True)
    for ax, gene in zip(axes.flat, GENE_ORDER):
        g = raw.loc[raw["gene"].eq(gene)].pivot(index="study_id", columns="tissue", values="methylation_pct").dropna()
        n = g["Normal"].to_numpy(float)
        t = g["Tumor"].to_numpy(float)
        offsets = np.linspace(-0.022, 0.022, len(g))
        x_n = offsets
        x_t = 1.0 - offsets
        for i in range(len(g)):
            ax.plot([x_n[i], x_t[i]], [n[i], t[i]], color="#c9c9c9", linewidth=0.35, alpha=0.55, zorder=1)
        ax.scatter(x_n, n, s=6, color=COL_N, alpha=0.85, linewidths=0, zorder=2)
        ax.scatter(x_t, t, s=6, color=COL_T, alpha=0.85, linewidths=0, zorder=2)
        st = effects.loc[gene]
        mean = float(st["paired_delta_mean_pctpt"])
        lo = float(st["paired_delta_ci_low_pctpt"])
        hi = float(st["paired_delta_ci_high_pctpt"])
        ax.errorbar(1.55, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color="black", markersize=3.0, capsize=2.2, linewidth=0.8)
        ax.axhline(0, color="#dddddd", linewidth=0.6)
        ax.set_title(gene, fontsize=6.6, fontstyle="italic", pad=3)
        ax.set_xlim(-0.25, 1.85)
        ax.set_ylim(0, 90)
        ax.set_xticks([0, 1, 1.55], ["N", "T", "Delta"], fontsize=5.4)
        ax.tick_params(axis="y", labelsize=5.4, length=2)
    axes[0, 0].set_ylabel("Methylation (%)\nand delta (pp)", fontsize=5.8)
    axes[1, 0].set_ylabel("Methylation (%)\nand delta (pp)", fontsize=5.8)
    fig.suptitle("Figure 2 | Paired pyrosequencing methylation by gene", fontsize=8.0, fontweight="bold", y=0.982)
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.075, top=0.90, wspace=0.25, hspace=0.36)
    save(fig, outdir, "Paired_PSQ")
    (outdir / "legend.md").write_text("Figure 2. Paired pyrosequencing methylation by gene. Lines connect adjacent normal and tumor methylation percentages from the same patient; black points and intervals show mean paired tumor-minus-normal differences and 95% BCa CIs.\n", encoding="utf-8")
    shutil.copy2(Path(__file__), outdir / "plot_local_psq_patterns.py")


def figure3() -> None:
    outdir = F3_DIR
    delta_path = copy_source(SOURCE_ROOT / "F3_delta_matrix_source_data.tsv", outdir)
    pca_path = copy_source(SOURCE_ROOT / "F3_pca_scores_source_data.tsv", outdir)
    load_path = copy_source(SOURCE_ROOT / "F3_pca_loadings_source_data.tsv", outdir)
    corr_path = copy_source(SOURCE_SUBMISSION / "F3_spearman_correlation_source_data.tsv", outdir)
    pattern_path = copy_source(ROOT.parent / "115_Public_Biology_20260905" / "baseline_114" / "results" / "pattern_summary.json", outdir)
    delta = pd.read_csv(delta_path, sep="\t")
    scores = pd.read_csv(pca_path, sep="\t")
    corr = pd.read_csv(corr_path, sep="\t").set_index("gene").loc[GENE_ORDER, GENE_ORDER]
    pattern = json.loads(pattern_path.read_text())
    mat = delta[GENE_ORDER].to_numpy(float)
    z = (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0)
    order = np.argsort(scores["PC1"].to_numpy(float))
    fig = plt.figure(figsize=(7.05, 5.80))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0], wspace=0.31, hspace=0.40)
    ax1 = fig.add_subplot(gs[:, 0])
    im = ax1.imshow(z[order, :], aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax1.set_xticks(range(len(GENE_ORDER)), GENE_ORDER, rotation=90, fontsize=5.6, rotation_mode="anchor", ha="right")
    for label in ax1.get_xticklabels():
        label.set_fontstyle("italic")
    ax1.set_yticks([])
    ax1.set_title("A. Standardized paired differences", fontsize=6.8, loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax1, fraction=0.034, pad=0.018)
    cb.set_label("z score", fontsize=5.6)
    cb.ax.tick_params(labelsize=5.2)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax2.set_xticks(range(len(GENE_ORDER)), GENE_ORDER, rotation=90, fontsize=5.0, rotation_mode="anchor", ha="right")
    ax2.set_yticks(range(len(GENE_ORDER)), GENE_ORDER, fontsize=5.0)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontstyle("italic")
    ax2.set_title("B. Spearman correlation of deltas", fontsize=6.8, loc="left", fontweight="bold")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.038, pad=0.020)
    cb2.ax.tick_params(labelsize=5.2)
    ax3 = fig.add_subplot(gs[1, 1])
    xy = scores[["PC1", "PC2"]].to_numpy(float)
    event = scores["event"].to_numpy(int)
    ax3.scatter(xy[event == 0, 0], xy[event == 0, 1], s=14, color="#999999", label="No event", alpha=0.9)
    ax3.scatter(xy[event == 1, 0], xy[event == 1, 1], s=16, color=COL_EVENT, label="Event", alpha=0.95)
    ax3.axhline(0, color="#dddddd", linewidth=0.6)
    ax3.axvline(0, color="#dddddd", linewidth=0.6)
    evr = pattern["pca_explained_variance_ratio"]
    ax3.set_xlabel(f"PC1 ({evr['PC1']*100:.1f}%)", fontsize=5.8)
    ax3.set_ylabel(f"PC2 ({evr['PC2']*100:.1f}%)", fontsize=5.8)
    ax3.set_title("C. Outcome-blind PCA", fontsize=6.8, loc="left", fontweight="bold")
    ax3.tick_params(labelsize=5.2)
    ax3.legend(frameon=False, fontsize=5.4, loc="best")
    fig.suptitle("Figure 3 | Patient-level methylation-difference patterns", fontsize=8.0, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.070, right=0.950, bottom=0.115, top=0.91)
    save(fig, outdir, "Delta_Patterns")
    (outdir / "legend.md").write_text("Figure 3. Patient-level methylation-difference patterns. Panel A shows standardized paired tumor-minus-normal differences ordered by PC1; panel B shows Spearman correlations among gene deltas; panel C shows outcome-blind PCA scores colored by recorded event status.\n", encoding="utf-8")
    shutil.copy2(Path(__file__), outdir / "plot_local_psq_patterns.py")


def main() -> None:
    configure()
    figure2()
    figure3()
    print(F2_DIR.resolve())
    print(F3_DIR.resolve())


if __name__ == "__main__":
    main()
