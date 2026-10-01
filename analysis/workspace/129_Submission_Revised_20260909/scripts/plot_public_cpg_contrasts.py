"""Render final-size CpG-level public contrast heatmaps."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRASTS = ROOT / "results" / "public_contrasts" / "public_probe_contrasts.tsv"
DEFAULT_ANNOTATION = ROOT.parent / "119_Figure1_ABC_Revision_20260908" / "registry" / "reviewer" / "probe_context.tsv"
PUBLIC_FIG_DIR = ROOT / "figures" / "Public_CpG_contrasts"
LESION_FIG_DIR = ROOT / "figures" / "Lesion_CpG"
GENE_ORDER = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
EFFECT_LIMIT = 60.0
CMAP = LinearSegmentedColormap.from_list("cpg_delta", ["#2166ac", "#f7f7f7", "#b2182b"], N=256)
CMAP.set_bad("#d9d9d9")
NORM = TwoSlopeNorm(vmin=-EFFECT_LIMIT, vcenter=0.0, vmax=EFFECT_LIMIT)
EXPORT_SUFFIXES = (".png", ".pdf", ".svg")
FINAL_WIDTH_MM = 163.8


def strip_svg_dtd(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    svg_start = text.find("<svg")
    if svg_start >= 0:
        text = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n' + text[svg_start:]
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text, count=1, flags=re.S)
    path.write_text(text, encoding="utf-8")


def read_order(annotation_path: Path) -> pd.DataFrame:
    annot = pd.read_csv(annotation_path, sep="\t").rename(columns={"fixed_gene": "gene", "cpg": "probe"})
    annot["gene"] = pd.Categorical(annot["gene"], categories=GENE_ORDER, ordered=True)
    annot["pos"] = pd.to_numeric(annot["pos"], errors="coerce")
    annot = annot.sort_values(["gene", "chr", "pos", "probe"]).reset_index(drop=True)
    annot["cpg_order"] = np.arange(1, len(annot) + 1)
    return annot[["gene", "probe", "chr", "pos", "strand", "promoter_groups", "Relation_to_Island", "cpg_order"]]


def ordered_source(contrasts_path: Path, annotation_path: Path, figure_kind: str) -> pd.DataFrame:
    contrasts = pd.read_csv(contrasts_path, sep="\t")
    annot = read_order(annotation_path)
    if figure_kind == "public":
        keep = contrasts["contrast"].isin(["N-H", "T-H", "T-N"])
        order = [
            ("Colonomics", "N-H"), ("GSE48684", "N-H"), ("GSE42752", "N-H"),
            ("Colonomics", "T-H"), ("GSE48684", "T-H"), ("GSE42752", "T-H"),
            ("Colonomics", "T-N"), ("GSE48684", "T-N"), ("GSE42752", "T-N"),
            ("GSE193535", "T-N"), ("GSE77718", "T-N"), ("GSE77954", "T-N"),
        ]
    elif figure_kind == "lesion":
        keep = contrasts["family"].eq("lesion")
        order = [("GSE48684", "A-H"), ("GSE48684", "T-A"), ("GSE77954", "T-A")]
    else:
        raise ValueError(f"unknown figure kind: {figure_kind}")
    source = contrasts.loc[keep].merge(annot, on=["gene", "probe"], how="left", validate="many_to_one")
    source["column_key"] = source["cohort"] + " " + source["contrast"]
    source["column_order"] = source.apply(lambda r: order.index((r["cohort"], r["contrast"])) + 1 if (r["cohort"], r["contrast"]) in order else np.nan, axis=1)
    before_filter = len(source)
    source = source[source["column_order"].notna()].copy()
    if len(source) == 0 or len(source) > before_filter:
        raise ValueError("invalid public contrast row filter")
    source["column_order"] = source["column_order"].astype(int)
    source["gene"] = pd.Categorical(source["gene"], categories=GENE_ORDER, ordered=True)
    source = source.sort_values(["cpg_order", "column_order"]).reset_index(drop=True)
    family_sizes = contrasts.groupby("family")["probe"].size().to_dict()
    source["bh_family_size"] = source["family"].map(family_sizes).astype(int)
    source["significant_bh_family"] = source["q_BH_family"] < 0.05
    source["significance_rule"] = "q_BH_family < 0.05 within " + source["bh_family_size"].astype(str) + "-row contrast family"
    source["plotted_effect_pp"] = source["effect_pp"].where(source["status"].eq("estimated"))
    return source


def add_gene_block_labels(ax, rows: pd.DataFrame, n_columns: int) -> None:
    grouped = rows.reset_index(drop=True).groupby("gene", observed=False, sort=False)
    label_x = n_columns + 0.06
    for gene, block in grouped:
        y_mid = (block.index.min() + block.index.max()) / 2
        ax.text(label_x, y_mid, str(gene), ha="left", va="center", fontsize=6.4, fontstyle="italic", clip_on=False)
    for _, block in grouped:
        last = block.index.max()
        if last < len(rows) - 1:
            ax.axhline(last + 0.5, color="white", linewidth=0.75)


def draw_heatmap(source: pd.DataFrame, outdir: Path, stem: str, title: str, footer_text: str, contrasts_path: Path, annotation_path: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    source_path = outdir / f"{stem}_source_data.tsv"
    source.to_csv(source_path, sep="\t", index=False)
    rows = source[["cpg_order", "gene", "probe", "chr", "pos"]].drop_duplicates().sort_values("cpg_order").reset_index(drop=True)
    columns = source[["column_order", "column_key"]].drop_duplicates().sort_values("column_order").reset_index(drop=True)
    matrix = source.pivot(index="cpg_order", columns="column_order", values="plotted_effect_pp").reindex(index=rows["cpg_order"], columns=columns["column_order"])
    sig = source.pivot(index="cpg_order", columns="column_order", values="significant_bh_family").reindex(index=rows["cpg_order"], columns=columns["column_order"]).fillna(False)
    missing = matrix.isna()

    width_in = 6.45 if stem == "Public_CpG_contrasts" else 6.80
    height_in = 9.45
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.2,
    })
    fig, ax = plt.subplots(figsize=(width_in, height_in), constrained_layout=False)
    im = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), aspect="auto", cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_title(title, fontsize=8.5, fontweight="bold", loc="left", pad=7)
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns["column_key"].str.replace(" ", "\n", regex=False), fontsize=5.6, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows["probe"], fontsize=5.05)
    ax.tick_params(length=0, pad=1.2)
    for y, x in np.argwhere(sig.to_numpy(dtype=bool)):
        ax.plot(x, y, marker="o", color="black", markersize=1.9, linestyle="none")
    for y, x in np.argwhere(missing.to_numpy(dtype=bool)):
        ax.text(x, y, "x", ha="center", va="center", fontsize=5.0, color="#333333")
    add_gene_block_labels(ax, rows, len(columns))
    ax.set_ylabel("CpGs grouped by gene; ordered by hg19 position", fontsize=6.2, labelpad=26)
    ax.set_xlim(-0.5, len(columns) + 1.34)
    cbar = fig.colorbar(im, ax=ax, shrink=0.58, pad=0.045, aspect=22)
    cbar.set_label("Effect (percentage points; -60 to +60)", fontsize=6.2)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    ax.text(0, -0.070, footer_text, transform=ax.transAxes, fontsize=5.6, va="top")
    fig.subplots_adjust(left=0.215, right=0.865, bottom=0.105, top=0.965)
    for suffix in EXPORT_SUFFIXES:
        fig.savefig(outdir / f"{stem}{suffix}", dpi=600 if suffix == ".png" else None)
    strip_svg_dtd(outdir / f"{stem}.svg")
    plt.close(fig)

    has_missing = bool(missing.to_numpy().any())
    legend_lines = [
        f"# {stem}", "",
        "CpG-level public contrasts for the frozen 77-CpG panel.",
        "Rows follow the fixed gene order and, within each gene, ascending hg19 genomic position. Gene symbols are separate italic block labels; CpG identifiers are regular text.",
        "Color encodes methylation beta difference in percentage points on a symmetric -60 to +60 scale centered at zero. Black dots mark q_BH_family < 0.05 within the pre-specified family.",
    ]
    if has_missing:
        legend_lines.append("Gray cells marked x indicate absent or non-estimable CpGs.")
    legend_lines.extend(["", f"Generated: {datetime.now(timezone.utc).isoformat()}"])
    (outdir / "legend.md").write_text("\n".join(legend_lines) + "\n", encoding="utf-8")
    manifest = {
        "figure_id": stem,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "width_mm_target_max": 180,
        "height_mm_target_max": 247,
        "font_floor_pt_target": 5,
        "effect_scale_pp": [-EFFECT_LIMIT, EFFECT_LIMIT],
        "row_order": "fixed gene order; within gene ascending hg19 position",
        "source_rows": int(len(source)),
        "missing_cells": int(missing.to_numpy().sum()),
        "significant_cells": int(sig.to_numpy(dtype=bool).sum()),
    }
    (outdir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(Path(__file__), outdir / "plot_public_cpg_contrasts.py")
    shutil.copy2(contrasts_path, outdir / "input_public_probe_contrasts.tsv")
    shutil.copy2(annotation_path, outdir / "input_probe_context.tsv")


def render_all(contrasts_path: Path, annotation_path: Path) -> None:
    public = ordered_source(contrasts_path, annotation_path, "public")
    lesion = ordered_source(contrasts_path, annotation_path, "lesion")
    draw_heatmap(
        public, PUBLIC_FIG_DIR, "Public_CpG_contrasts", "Figure 5 | Public tissue CpG contrasts",
        "Black dots: q<0.05 within 462-test families; x: absent/non-estimable. GSE77954 T-N: 4 paired samples in active source.",
        contrasts_path, annotation_path,
    )
    draw_heatmap(
        lesion, LESION_FIG_DIR, "Lesion_CpG", "Figure 6 | Adenoma and carcinoma CpG contrasts",
        "Black dots: q<0.05 within the 231-test lesion family.",
        contrasts_path, annotation_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrasts", type=Path, default=DEFAULT_CONTRASTS)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    args = parser.parse_args()
    render_all(args.contrasts, args.annotation)


if __name__ == "__main__":
    main()
