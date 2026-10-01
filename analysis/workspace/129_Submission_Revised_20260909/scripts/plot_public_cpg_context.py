#!/usr/bin/env python3
"""Render final-size public CpG context as gene-block small multiples."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "public_context"
FIGDIR = ROOT / "figures" / "Public_Context"
STEM = "Public_Context"
PROBE_CONTEXT = ROOT / "figures" / "Public_CpG_contrasts" / "input_probe_context.tsv"
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
EXPORT_SUFFIXES = (".png", ".pdf", ".svg")
TRACKS = [
    ("Expr: Colonomics", "Colonomics", "unadjusted", "methylation_expression_spearman", "rho"),
    ("Expr: adjusted", "Colonomics", "rank_residual_age_sex_site_stroma", "methylation_expression_spearman", "rho"),
    ("Expr: ColoCare", "ColoCare_GSE101764_GSE106582", "unadjusted", "methylation_expression_spearman", "rho"),
    ("Stromal rho", "Colonomics", "unadjusted", "stromal_spearman", "rho"),
    ("CMS KW H", "Colonomics", "source_cms_labels", "CMS_global_Kruskal", "kw"),
    ("MATCH Welch", "GSE164811_MATCH", "Welch_CMS3_minus_CMS2", "CMS3-CMS2", "pp"),
    ("MATCH adjusted", "GSE164811_MATCH", "sex_site_adjusted_OLS_CMS3_minus_CMS2", "CMS3-CMS2_adjusted", "pp"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, media_type: str) -> dict[str, object]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "media_type": media_type}


def strip_svg(path: Path) -> None:
    text = path.read_text()
    svg_start = text.find("<svg")
    if svg_start >= 0:
        text = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n' + text[svg_start:]
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text, count=1, flags=re.S)
    path.write_text(text)


def ordered_source() -> pd.DataFrame:
    all_rows = pd.read_csv(RESULTS / "all_public_cpg_context.tsv", sep="\t")
    cpgs = pd.read_csv(RESULTS / "fixed77_cpg_annotation.tsv", sep="\t")
    probe_context = pd.read_csv(PROBE_CONTEXT, sep="\t")[["probe", "chr", "pos"]].rename(columns={"probe": "cpg"})
    order = cpgs[["gene", "cpg", "promoter_match", "promoter_groups"]].copy()
    order = order.merge(probe_context, on="cpg", how="left", validate="one_to_one")
    order["gene"] = pd.Categorical(order["gene"], categories=GENES, ordered=True)
    order = order.sort_values(["gene", "chr", "pos", "cpg"]).reset_index(drop=True)
    order["gene_order"] = order["gene"].map({gene: i for i, gene in enumerate(GENES)}).astype(int)
    order["x"] = order.groupby("gene", observed=False).cumcount()
    source = all_rows.merge(order[["cpg", "gene_order", "x", "chr", "pos"]], on="cpg", how="left", validate="many_to_one")
    source["minus_log10_q"] = -np.log10(source["q_BH77"].clip(lower=np.nextafter(0, 1)))
    source["q_lt_0_05"] = source["q_BH77"] < 0.05
    for col in ["effect", "ci_low", "ci_high"]:
        source[f"{col}_percentage_points"] = np.where(source["unit"].eq("beta_difference"), source[col] * 100, np.nan)
    return source.sort_values(["gene_order", "x", "cohort", "model", "analysis"])


def value_and_style(row: pd.Series | None, mode: str, h_max: float) -> tuple[float, object, Normalize, str]:
    if row is None or row.empty or row.get("status") != "estimated":
        return np.nan, "#d9d9d9", Normalize(0, 1), "missing"
    if mode == "rho":
        return float(row["effect"]), plt.get_cmap("coolwarm"), Normalize(-1, 1), "rho"
    if mode == "kw":
        return float(row["effect"]), plt.get_cmap("viridis"), Normalize(0, h_max), "kw"
    return float(row["effect_percentage_points"]), plt.get_cmap("RdBu_r"), TwoSlopeNorm(vmin=-35, vcenter=0, vmax=35), "pp"


def render() -> dict[str, object]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    source = ordered_source()
    source_path = FIGDIR / f"{STEM}_source_data.tsv"
    source.to_csv(source_path, sep="\t", index=False)
    h_max = float(source.loc[source["analysis"].eq("CMS_global_Kruskal"), "effect"].max())
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(5, 2, figsize=(7.05, 9.35), sharey=True)
    axes = axes.ravel()
    for ax, gene in zip(axes, GENES):
        g = source.loc[source["gene"].eq(gene)]
        cpgs = g[["cpg", "x", "promoter_match"]].drop_duplicates().sort_values("x")
        for _, cpg_row in cpgs.iterrows():
            x = int(cpg_row["x"])
            for y, (_, cohort, model, analysis, mode) in enumerate(TRACKS):
                hit = g.loc[g["cpg"].eq(cpg_row["cpg"]) & g["cohort"].eq(cohort) & g["model"].eq(model) & g["analysis"].eq(analysis)]
                row = hit.iloc[0] if len(hit) else None
                val, cmap, norm, kind = value_and_style(row, mode, h_max)
                if kind == "missing" or not np.isfinite(val):
                    ax.text(x, y, "x", ha="center", va="center", fontsize=5.0, color="#444444")
                    continue
                size = 18 + 24 * min(float(row["minus_log10_q"]), 6) / 6
                ax.scatter(x, y, c=[val], cmap=cmap, norm=norm, s=size, edgecolors="black" if bool(cpg_row["promoter_match"]) else "#888888", linewidths=0.35)
                if bool(row["q_lt_0_05"]):
                    ax.scatter(x, y, facecolors="none", edgecolors="black", s=size + 24, linewidths=0.75)
        ax.set_title(str(gene), fontsize=7.0, fontstyle="italic", pad=2.5)
        ax.set_xlim(-0.6, max(cpgs["x"]) + 0.6)
        ax.set_ylim(len(TRACKS) - 0.6, -0.6)
        ax.set_xticks(cpgs["x"])
        ax.set_xticklabels(cpgs["cpg"], rotation=90, rotation_mode="anchor", ha="right", fontsize=5.0)
        ax.tick_params(length=0, pad=1)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.35)
    for ax in axes[::2]:
        ax.set_yticks(range(len(TRACKS)))
        ax.set_yticklabels([t[0] for t in TRACKS], fontsize=5.6)
    for ax in axes[1::2]:
        ax.tick_params(labelleft=False)
    fig.suptitle("Figure 7 | CpG context in public tumor cohorts", fontsize=8.5, fontweight="bold", x=0.02, y=0.995, ha="left")
    cax1 = fig.add_axes([0.19, 0.018, 0.18, 0.010])
    cb1 = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap="coolwarm"), cax=cax1, orientation="horizontal")
    cb1.set_label("rho", fontsize=5.6); cb1.ax.tick_params(labelsize=5.0, length=1.5)
    cax2 = fig.add_axes([0.46, 0.018, 0.18, 0.010])
    cb2 = fig.colorbar(ScalarMappable(norm=Normalize(0, h_max), cmap="viridis"), cax=cax2, orientation="horizontal")
    cb2.set_label("Kruskal-Wallis H", fontsize=5.6); cb2.ax.tick_params(labelsize=5.0, length=1.5)
    cax3 = fig.add_axes([0.73, 0.018, 0.18, 0.010])
    cb3 = fig.colorbar(ScalarMappable(norm=TwoSlopeNorm(vmin=-35, vcenter=0, vmax=35), cmap="RdBu_r"), cax=cax3, orientation="horizontal")
    cb3.set_label("CMS3-CMS2 pp", fontsize=5.6); cb3.ax.tick_params(labelsize=5.0, length=1.5)
    fig.text(0.02, 0.035, "Outer ring: BH77 q<0.05; black edge: promoter CpG; x: unavailable/not estimable.", fontsize=5.6)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.955, bottom=0.085, hspace=0.62, wspace=0.18)
    outputs = {}
    for suffix in EXPORT_SUFFIXES:
        path = FIGDIR / f"{STEM}{suffix}"
        fig.savefig(path, dpi=600 if suffix == ".png" else None, bbox_inches="tight", pad_inches=0.025)
        if suffix == ".svg": strip_svg(path)
        outputs[suffix.lstrip(".")] = path
    plt.close(fig)
    (FIGDIR / f"{STEM}_layout.json").write_text(json.dumps({
        "layout": "10 gene-block small multiples, 5 rows x 2 columns",
        "width_in": 7.05,
        "height_in": 9.35,
        "row_order": "fixed gene order; within each gene sorted by hg19 chromosome position",
        "tracks": [t[0] for t in TRACKS],
        "encodings": {"outer_ring":"BH77 q<0.05", "black_edge":"promoter CpG", "x":"unavailable/not estimable"},
    }, indent=2) + "\n")
    legend = FIGDIR / f"{STEM}_legend.md"
    legend.write_text("\n".join([
        "# Public Context Individual-CpG Figure Legend", "",
        "Gene-block small multiples show all 77 fixed CpGs in the same fixed gene order and within-gene CpG order used by the other CpG figures.",
        "Rows encode Colonomics/ColoCare methylation-expression correlations, Colonomics stromal and CMS context, and MATCH CMS3-minus-CMS2 contrasts.",
        "Color scales are separate by row family: rho (-1 to +1), Kruskal-Wallis H, and CMS3-CMS2 percentage-point difference (-35 to +35).",
        "Outer rings mark BH77 q<0.05, black marker edges identify promoter CpGs, and x marks unavailable or non-estimable results.",
    ]) + "\n")
    manifest = {
        "schema_version":"1.1", "figure_id":STEM, "script": os.path.relpath(ROOT/"scripts"/"plot_public_cpg_context.py", FIGDIR),
        "width_mm_target_max":180, "height_mm_target_max":247, "font_floor_pt_target":5,
        "source_rows": int(len(source)), "outputs": [file_record(outputs[e], {"png":"image/png","pdf":"application/pdf","svg":"image/svg+xml"}[e]) for e in ["png","pdf","svg"]],
        "source_data":[file_record(source_path,"text/tab-separated-values")],
    }
    (FIGDIR / f"{STEM}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    render(); return 0

if __name__ == "__main__":
    raise SystemExit(main())
