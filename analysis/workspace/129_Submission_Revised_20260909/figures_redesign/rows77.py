"""Shared 77-CpG row engine for Figures 4, 6 and 8.

The three figures share one conceptual row axis: the 77 fixed candidate CpGs
ordered by ``figstyle.GENES`` and, inside every gene, by ascending hg19
position. Gene blocks are separated by a 0.6-row gap so the brackets read as
blocks rather than as one continuous list.

Row coordinates increase downwards (row 0 at the top). Every consumer axes is
built with ``set_ylim(y_max, y_min)`` through :func:`apply_row_axis`, so the
first CpG of *EYA4* sits at the top of every figure.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

import figstyle as fs

BLOCK_GAP = 0.6          # rows of empty space between two gene blocks
BAND_COLOR = "#F7F7F7"   # alternating gene-block band
ID_FONTSIZE = fs.FS["tiny"]

ANNOTATION_SOURCES = (
    fs.F124 / "Korean_CpG" / "probe_annotation_source.tsv",
    fs.F124 / "Public_CpG_contrasts" / "input_probe_context.tsv",
    fs.F124 / "Lesion_CpG" / "input_probe_context.tsv",
)


def annotation_path(preferred: str | pathlib.Path | None = None) -> pathlib.Path:
    """Return the probe annotation file to use (first existing candidate)."""
    candidates = ([pathlib.Path(preferred)] if preferred else []) + list(ANNOTATION_SOURCES)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no probe annotation found among {candidates}")


def load_rows(preferred: str | pathlib.Path | None = None) -> pd.DataFrame:
    """Return the 77 CpGs with row order and y positions.

    Columns: ``probe``, ``gene``, ``chr``, ``pos``, ``strand``, ``promoter_match``,
    ``block`` (0-9 gene index), ``row`` (0-76 running index) and ``y``
    (plot coordinate including the between-block gaps).
    """
    path = annotation_path(preferred)
    raw = pd.read_csv(path, sep="\t")
    gene_col = "fixed_gene" if "fixed_gene" in raw.columns else "gene"
    keep = [gene_col, "probe", "chr", "pos", "strand", "promoter_match"]
    keep = [c for c in keep if c in raw.columns]
    tidy = raw[keep].rename(columns={gene_col: "gene"}).copy()

    missing_genes = sorted(set(tidy["gene"]) - set(fs.GENES))
    if missing_genes:
        raise ValueError(f"unexpected gene symbols in {path}: {missing_genes}")

    tidy["block"] = pd.Categorical(tidy["gene"], categories=fs.GENES, ordered=True).codes
    tidy = tidy.sort_values(["block", "pos"], kind="mergesort").reset_index(drop=True)
    if len(tidy) != 77:
        raise ValueError(f"expected 77 CpGs, found {len(tidy)} in {path}")
    if tidy["probe"].duplicated().any():
        raise ValueError("duplicated probe identifiers in the annotation source")

    tidy["row"] = np.arange(len(tidy))
    tidy["y"] = tidy["row"].to_numpy(float) + BLOCK_GAP * tidy["block"].to_numpy(float)
    tidy["source_file"] = str(path)
    return tidy


def y_of(rows: pd.DataFrame) -> dict:
    """probe -> y coordinate."""
    return dict(zip(rows["probe"], rows["y"]))


def blocks(rows: pd.DataFrame):
    """Yield ``(gene, y_first, y_last, block_index)`` for each gene block."""
    for block, chunk in rows.groupby("block", sort=True):
        yield chunk["gene"].iloc[0], float(chunk["y"].min()), float(chunk["y"].max()), int(block)


def y_limits(rows: pd.DataFrame, pad: float = 0.75) -> tuple:
    """(bottom, top) suitable for ``ax.set_ylim`` — inverted so row 0 is on top."""
    return float(rows["y"].max()) + pad, float(rows["y"].min()) - pad


def apply_row_axis(ax, rows: pd.DataFrame, pad: float = 0.75) -> None:
    """Set the inverted row axis and remove y ticks."""
    ax.set_ylim(*y_limits(rows, pad))
    ax.set_yticks([])


def block_bands(ax, rows: pd.DataFrame, x0: float, x1: float, start: int = 0,
                color: str = BAND_COLOR, zorder: float = 0.2) -> None:
    """Alternating very light bands, one per gene block, spanning x0..x1."""
    for _, y_first, y_last, block in blocks(rows):
        if (block % 2) != (start % 2):
            continue
        ax.add_patch(
            fs.plt.Rectangle(
                (x0, y_first - 0.5 - BLOCK_GAP / 2), x1 - x0,
                (y_last - y_first) + 1 + BLOCK_GAP,
                facecolor=color, edgecolor="none", zorder=zorder, clip_on=True,
            )
        )


def block_separators(ax, rows: pd.DataFrame, x0: float, x1: float,
                     color: str = "white", lw: float = 0.5, zorder: float = 3) -> None:
    """White separator lines between gene blocks (for heat maps)."""
    ys = [y_last + 0.5 + BLOCK_GAP / 2 for _, _, y_last, block in blocks(rows) if block < len(fs.GENES) - 1]
    for y in ys:
        ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=zorder, clip_on=True)


def gutter_axes(fig, left_mm: float, top_mm: float, width_mm: float, height_mm: float,
                rows: pd.DataFrame, pad: float = 0.75):
    """Invisible axes spanning the row range; x runs 0..1 across its width."""
    ax = fs.ax_mm(fig, left_mm, top_mm, width_mm, height_mm)
    ax.set_xlim(0, 1)
    apply_row_axis(ax, rows, pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_alpha(0)
    return ax


LABEL_DX = 0.35          # x-units between the bracket and the gene symbol


def draw_gene_brackets(ax, rows: pd.DataFrame, x: float = 1.0, tick: float = 0.03) -> None:
    """Thin vertical bracket per gene block with an italic gene label.

    The label is drawn locally (not via :func:`figstyle.gene_bracket`) so that a
    real italic font face is used: mathtext keeps digits upright, which reads
    badly for ZNF568 / ADHFE1 / HOXA2 / BEND5. The bracket axes spans 0..1 in x
    and the bracket sits at its right edge, leaving 65 % of the gutter for the
    gene symbol.
    """
    for gene, y_first, y_last, _ in blocks(rows):
        y0, y1 = y_first - 0.42, y_last + 0.42
        ax.plot([x, x], [y0, y1], color=fs.C["ink2"], lw=0.6, clip_on=False,
                solid_capstyle="butt")
        ax.text(x - LABEL_DX, (y0 + y1) / 2, gene, ha="right", va="center",
                fontsize=fs.FS["body"], color=fs.C["ink"], style="italic",
                clip_on=False)
        for y in (y0, y1):                                    # small serifs
            ax.plot([x, x + tick], [y, y], color=fs.C["ink2"], lw=0.6, clip_on=False)


def draw_cpg_labels(ax, rows: pd.DataFrame, x: float = 0.0,
                    fontsize: float = ID_FONTSIZE, color: str | None = None) -> None:
    """CpG identifiers, 5.5 pt, ink2, left aligned at ``x``."""
    color = fs.C["ink2"] if color is None else color
    for probe, y in zip(rows["probe"], rows["y"]):
        ax.text(x, y, probe, ha="left", va="center", fontsize=fontsize,
                color=color, clip_on=False)


def draw_row_gutter(fig, left_mm, top_mm, width_mm, height_mm, rows,
                    bracket_frac: float = 0.595, pad: float = 0.75, bands: bool = False):
    """Row gutter = gene-bracket axes (left part) + CpG identifier axes (right).

    Two separate axes are used so that the italic gene symbol and the 5.5 pt
    CpG identifier each get a fixed, predictable amount of millimetres and
    never collide with the bracket.
    """
    w_br = width_mm * bracket_frac
    ax_br = gutter_axes(fig, left_mm, top_mm, w_br, height_mm, rows, pad)
    draw_gene_brackets(ax_br, rows, x=1.0)
    ax_id = gutter_axes(fig, left_mm + w_br, top_mm, width_mm - w_br, height_mm, rows, pad)
    if bands:
        block_bands(ax_id, rows, 0.0, 1.0)
    draw_cpg_labels(ax_id, rows, x=0.04)
    return ax_br, ax_id


def measured_rows(rows: pd.DataFrame, probes) -> pd.DataFrame:
    """Subset of ``rows`` restricted to ``probes``, keeping the row order."""
    keep = rows[rows["probe"].isin(set(probes))].copy()
    keep["sub_row"] = np.arange(len(keep))
    keep["sub_y"] = keep["sub_row"].to_numpy(float) + BLOCK_GAP * pd.Categorical(
        keep["gene"], categories=[g for g in fs.GENES if g in set(keep["gene"])], ordered=True
    ).codes.astype(float)
    return keep
