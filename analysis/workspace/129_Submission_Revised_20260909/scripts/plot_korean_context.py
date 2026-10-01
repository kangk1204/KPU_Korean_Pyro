"""Plot per-CpG CMCBSN context results without methylation averaging."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT = ROOT / "results" / "context"
DEFAULT_FIGDIR = ROOT / "figures" / "Korean_Context"
PROBE_CONTEXT = ROOT / "figures" / "Public_CpG_contrasts" / "input_probe_context.tsv"
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
EXPORT_SUFFIXES = (".png", ".pdf", ".svg")


def build_source(context_dir: Path = DEFAULT_CONTEXT) -> pd.DataFrame:
    cms = pd.read_csv(context_dir / "cms_cpg_contrasts.tsv", sep="\t")
    expr = pd.read_csv(context_dir / "expression_cpg_associations.tsv", sep="\t")
    source = cms[
        [
            "target_order",
            "gene",
            "cpg",
            "present_in_beta",
            "n_cms2",
            "n_cms3",
            "mean_cms3_minus_cms2_beta",
            "welch_p",
            "welch_bh_q",
            "status",
            "reason",
        ]
    ].rename(columns={"status": "cms_status", "reason": "cms_reason"})
    source = source.merge(
        expr[
            [
                "target_order",
                "cpg",
                "n",
                "rho",
                "p",
                "bh_q",
                "partial_rho_age_sex_site",
                "partial_p_age_sex_site",
                "partial_bh_q_age_sex_site",
                "status",
                "reason",
            ]
        ].rename(
            columns={
                "n": "expression_n",
                "p": "expression_p",
                "bh_q": "expression_bh_q",
                "status": "expression_status",
                "reason": "expression_reason",
            }
        ),
        on=["target_order", "cpg"],
        validate="one_to_one",
    )
    source["cms3_minus_cms2_pp"] = source["mean_cms3_minus_cms2_beta"] * 100.0
    source["cms_bh77_significant"] = source["welch_bh_q"] < 0.05
    source["expression_bh77_significant"] = source["expression_bh_q"] < 0.05
    source["partial_expression_bh77_significant"] = source["partial_bh_q_age_sex_site"] < 0.05
    probe_context = pd.read_csv(PROBE_CONTEXT, sep="\t")[["probe", "chr", "pos"]].rename(columns={"probe": "cpg"})
    source = source.merge(probe_context, on="cpg", how="left", validate="one_to_one")
    source["gene_order"] = pd.Categorical(source["gene"], categories=GENES, ordered=True).codes
    source = source.sort_values(["gene_order", "chr", "pos", "cpg"]).reset_index(drop=True)
    source["display_order"] = np.arange(len(source))
    source = source[
        [
            "display_order",
            "target_order",
            "gene_order",
            "gene",
            "cpg",
            "chr",
            "pos",
            "present_in_beta",
            "cms3_minus_cms2_pp",
            "welch_p",
            "welch_bh_q",
            "cms_bh77_significant",
            "expression_n",
            "rho",
            "expression_p",
            "expression_bh_q",
            "expression_bh77_significant",
            "partial_rho_age_sex_site",
            "partial_p_age_sex_site",
            "partial_bh_q_age_sex_site",
            "partial_expression_bh77_significant",
            "cms_status",
            "cms_reason",
            "expression_status",
            "expression_reason",
        ]
    ]
    return source


def plot_heatmap(source: pd.DataFrame, figdir: Path = DEFAULT_FIGDIR) -> dict[str, Path]:
    figdir.mkdir(parents=True, exist_ok=True)
    source_path = figdir / "cpg_context_heatmap_source.tsv"
    source.to_csv(source_path, sep="\t", index=False)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.2,
    })
    fig = plt.figure(figsize=(7.05, 9.35))
    grid = fig.add_gridspec(1, 4, width_ratios=[0.92, 0.07, 1.44, 0.07], wspace=0.30)
    ax_cms = fig.add_subplot(grid[0, 0])
    cax_cms = fig.add_subplot(grid[0, 1])
    ax_rna = fig.add_subplot(grid[0, 2], sharey=ax_cms)
    cax_rna = fig.add_subplot(grid[0, 3])

    cms_cmap = plt.get_cmap("RdBu_r").copy()
    rna_cmap = plt.get_cmap("coolwarm").copy()
    missing_color = "#c7c7c7"
    cms_cmap.set_bad(missing_color)
    rna_cmap.set_bad(missing_color)

    cms_values = np.ma.masked_invalid(source[["cms3_minus_cms2_pp"]].to_numpy(dtype=float))
    rna_values = np.ma.masked_invalid(source[["rho", "partial_rho_age_sex_site"]].to_numpy(dtype=float))
    cms_image = ax_cms.imshow(cms_values, aspect="auto", cmap=cms_cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-15, vmax=15))
    rna_image = ax_rna.imshow(rna_values, aspect="auto", cmap=rna_cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-0.7, vmax=0.7))

    labels = [row.cpg for row in source.itertuples(index=False)]
    ax_cms.set_yticks(np.arange(len(source)))
    ax_cms.set_yticklabels(labels, fontsize=5.05)
    ax_rna.tick_params(axis="y", left=False, labelleft=False)
    ax_cms.tick_params(axis="y", length=0)
    ax_cms.set_xticks([0])
    ax_cms.set_xticklabels(["CMS3-CMS2\npp"])
    ax_rna.set_xticks([0, 1])
    ax_rna.set_xticklabels(["RNA rho", "Adjusted\nRNA rho"])
    ax_cms.set_title("A. CMS3-CMS2", loc="left", fontsize=7.2, fontweight="bold")
    ax_rna.set_title("B. Methylation-expression", loc="left", fontsize=7.2, fontweight="bold")

    for ax in [ax_cms, ax_rna]:
        ax.tick_params(axis="x", labelsize=6.0)
        for boundary in source.groupby("gene", sort=False).tail(1).index[:-1]:
            ax.axhline(boundary + 0.5, color="black", linewidth=0.25, alpha=0.45)
        ax.set_ylim(len(source) - 0.5, -0.5)

    for row_i, row in enumerate(source.itertuples(index=False)):
        if not np.isfinite(row.cms3_minus_cms2_pp):
            ax_cms.text(0, row_i, "x", ha="center", va="center", color="black", fontsize=5.0)
        if row.cms_bh77_significant:
            ax_cms.plot(0, row_i, "o", color="black", markersize=2.2)
        for col_i, value in enumerate([row.rho, row.partial_rho_age_sex_site]):
            if not np.isfinite(value):
                ax_rna.text(col_i, row_i, "x", ha="center", va="center", color="black", fontsize=5.0)
        if row.expression_bh77_significant:
            ax_rna.plot(0, row_i, "o", color="black", markersize=2.2)
        if row.partial_expression_bh77_significant:
            ax_rna.plot(1, row_i, "o", color="black", markersize=2.2)

    cbar_cms = fig.colorbar(cms_image, cax=cax_cms, extend="both")
    cbar_cms.ax.set_title("pp", fontsize=6.0, pad=4)
    cbar_cms.ax.tick_params(labelsize=5.2, length=2)
    cbar_rna = fig.colorbar(rna_image, cax=cax_rna, extend="both")
    cbar_rna.ax.set_title("rho", fontsize=6.0, pad=4)
    cbar_rna.ax.tick_params(labelsize=5.2, length=2)
    for gene, block in source.reset_index(drop=True).groupby("gene", sort=False):
        y_mid = (block.index.min() + block.index.max()) / 2
        ax_cms.text(-1.18, y_mid, str(gene), ha="right", va="center", fontsize=6.4, fontstyle="italic", clip_on=False)
    ax_cms.set_ylabel("")
    fig.subplots_adjust(left=0.245, right=0.93, top=0.97, bottom=0.035, wspace=0.30)

    outputs = {"source": source_path}
    for suffix in EXPORT_SUFFIXES:
        path = figdir / f"cpg_context_heatmap{suffix}"
        fig.savefig(path, dpi=600 if suffix == ".png" else None, bbox_inches="tight", pad_inches=0.035)
        if suffix == ".svg":
            _strip_svg_doctype(path)
        outputs[suffix.lstrip(".")] = path
    plt.close(fig)

    legend_text = (
        "\n".join(
            [
                "# CMCBSN Per-CpG Context Heatmap Legend",
                "",
                "Rows are the 77 fixed CpGs in fixed gene order and within-gene hg19 genomic position; CpG identifiers are regular text and separate block labels show full italic gene symbols.",
                "Panel A shows source-CMS3 minus source-CMS2 mean methylation difference in percentage points, with a separate +/-15 pp color scale.",
                "Panel B shows tumor-only CpG methylation versus mapped-gene RNA expression correlations, with unadjusted and raw-age/sex/site-adjusted columns on the same blue-white-red rho scale used for Figure 7.",
                "Gray cells with x marks indicate missing CpG or missing RNA-gene results. Black dots mark BH77 q<0.05 cells.",
                "No CpG values are averaged across genes or panels in this figure.",
            ]
        )
        + "\n"
    )
    legend = figdir / "cpg_context_heatmap_legend.md"
    legend.write_text(legend_text)
    summary_legend = figdir / "Korean_Context_legend.md"
    summary_legend.write_text(legend_text)
    outputs["legend"] = legend
    outputs["summary_legend"] = summary_legend
    return outputs


def _strip_svg_doctype(path: Path) -> None:
    text = path.read_text()
    lines = []
    skip = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<!DOCTYPE"):
            skip = not stripped.endswith(">")
            continue
        if skip:
            if stripped.endswith(">"):
                skip = False
            continue
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dir", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_heatmap(build_source(args.context_dir), args.figdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
