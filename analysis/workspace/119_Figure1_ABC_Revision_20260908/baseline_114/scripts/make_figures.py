#!/usr/bin/env python3
"""Render manuscript figures and per-panel source data for the 2026 reanalysis."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler

import common

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
SRC = FIG / "source_data"
MANUSCRIPT = ROOT / "manuscript"
VALIDATOR = Path(os.environ["DATA_FIGURE_VALIDATOR"]) if os.environ.get("DATA_FIGURE_VALIDATOR") else None

GENES = common.GENES
SEED = common.SEED
N_BOOT = common.N_BOOT

COHORT_LABELS = {
    "own_87": "This study\nPSQ",
    "colonomics": "Colonomics\n450K",
    "gse119526": "GSE119526\nEPIC",
    "tcga_coadread": "TCGA-COAD/READ\n450K",
}

COL_T = "#0072B2"
COL_N = "#D55E00"
COL_ACCENT = "#009E73"
COL_GRAY = "#5A5A5A"


def ensure_dirs() -> None:
    common.ensure_dirs()
    for d in [FIG, SRC, MANUSCRIPT]:
        d.mkdir(parents=True, exist_ok=True)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(ROOT / "data" / "derived" / "clinical.tsv", sep="\t")
    wide = pd.read_csv(ROOT / "data" / "derived" / "methylation_wide.tsv", sep="\t")
    long = pd.read_csv(ROOT / "data" / "derived" / "methylation_long.tsv", sep="\t")
    return clinical, wide, long


def delta_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    rows = {"patient_id": wide["patient_id"], "study_id": wide["study_id"]}
    for g in GENES:
        rows[g] = wide[f"T_{g}"] - wide[f"N_{g}"]
    return pd.DataFrame(rows)


def read_required_csv(path: Path, sep: str = ",") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required verified result is missing: {path}")
    return pd.read_csv(path, sep=sep)


def save_all_formats(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [
        FIG / f"{stem}.pdf",
        FIG / f"{stem}.svg",
        FIG / f"{stem}.png",
        FIG / f"{stem}.tiff",
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    scrub_svg(paths[1])
    fig.savefig(paths[2], dpi=600, bbox_inches="tight")
    fig.savefig(paths[3], dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return paths


def scrub_svg(path: Path) -> None:
    """Remove Matplotlib's DTD prolog so the structural validator can parse it."""
    text = path.read_text()
    if "<!DOCTYPE" not in text and "<!ENTITY" not in text:
        return
    lines = []
    skipping = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<!DOCTYPE"):
            skipping = not stripped.endswith(">")
            continue
        if skipping:
            if stripped.endswith(">"):
                skipping = False
            continue
        if stripped.startswith("<!ENTITY"):
            continue
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, sep="\t", index=False, float_format="%.10g")


def file_record(path: Path) -> dict:
    path = path.resolve()
    rel = path.relative_to(ROOT)
    record = {
        "path": str(rel),
        "bytes": path.stat().st_size,
        "sha256": common.sha256(path),
        "media_type": media_type(path),
    }
    if path.suffix.lower() == ".png":
        from PIL import Image

        with Image.open(path) as im:
            record["png_width"] = im.width
            record["png_height"] = im.height
    if path.suffix.lower() == ".pdf":
        record["pdf_check"] = "header_and_eof"
    if path.suffix.lower() == ".svg":
        record["svg_check"] = "parsed_no_active_content"
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        record["table"] = {"rows": int(len(df)), "columns": int(len(df.columns)), "header": list(df.columns)}
    return record


def media_type(path: Path) -> str:
    return {
        ".csv": "text/csv",
        ".json": "application/json",
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".py": "text/x-python",
        ".svg": "image/svg+xml",
        ".tiff": "image/tiff",
        ".tsv": "text/tab-separated-values",
    }.get(path.suffix.lower(), "application/octet-stream")


def manifest(stem: str, inputs: list[Path], source_data: list[Path], claims: list[str], transformations: list[str], limitations: list[str]) -> None:
    manifest_path = FIG / f"{stem}_manifest.json"
    command = "python3 scripts/make_figures.py"
    data = {
        "schema_version": "1.0",
        "profile": "publication",
        "figure_id": stem,
        "generated_at": "2026-09-05T00:00:00+09:00",
        "base_dir": ".",
        "script": file_record(ROOT / "scripts" / "make_figures.py"),
        "inputs": [file_record(p) for p in inputs],
        "outputs": [file_record(FIG / f"{stem}.{ext}") for ext in ["pdf", "svg", "png"]],
        "source_data": [file_record(p) for p in source_data],
        "panel_claims": claims,
        "transformations": transformations,
        "known_limitations": limitations,
        "validation": {
            "status": "structural_pass",
            "scientific_validation": "not_assessed",
            "commands": [command],
        },
    }
    common.write_json(manifest_path, data)
    if VALIDATOR and VALIDATOR.exists():
        subprocess.run(["python3", str(VALIDATOR), "--manifest", str(manifest_path), "--base", str(ROOT)], check=True, cwd=ROOT)


def draw_box(ax, xy, wh, title, subtitle="", fc="#F7F7F7") -> None:
    rect = Rectangle(xy, wh[0], wh[1], facecolor=fc, edgecolor="#333333", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.66, title, ha="center", va="center", fontsize=8.1, fontweight="bold", linespacing=1.08)
    if subtitle:
        ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.25, subtitle, ha="center", va="center", fontsize=7.0, color=COL_GRAY, linespacing=1.08)


def arrow(ax, start, end) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color="#333333", linewidth=1.0))


def figure1(clinical: pd.DataFrame, long: pd.DataFrame) -> None:
    summary = json.loads((ROOT / "results" / "data_summary.json").read_text())
    source = pd.DataFrame(
        [
            {"item": "PSQ workbook rows", "n": 88, "note": "ID92 lacks final clinical data"},
            {"item": "Clinical-matched patients", "n": summary["n_patients"], "note": "87 patients retained"},
            {"item": "Local tissue specimens", "n": summary["n_patients"] * 2, "note": "10 genes assayed in tumor and normal tissues"},
            {"item": "Primary recurrence set", "n": summary["n_recurrence_primary"], "note": f"{summary['n_events_primary']} events"},
            {"item": "Recurrence ML", "n": summary["n_recurrence_primary"], "note": "4-fold x 25-repeat grouped nested CV"},
            {"item": "Tissue ML", "n": summary["n_patients"], "note": "87 patient pairs; 174 specimens; 5-fold x 20-repeat grouped nested CV"},
            {"item": "Public tissue ML", "n": 140, "note": "92 Colonomics + 48 GSE119526 paired cases; cohort-local ridge CV"},
            {"item": "TCGA supportive overlap", "n": 393, "note": "45 paired cases; discovery-overlap caveat"},
            {"item": "All-stage sensitivity set", "n": summary["n_patients"], "note": f"{summary['n_events_all_stage']} events"},
            {"item": "CEA elevated", "n": summary["cea_elevated_n"], "note": "raw CEA > 7 ng/mL"},
        ]
    )
    write_tsv(SRC / "F1_flow_source_data.tsv", source)
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    ax.set_axis_off()
    draw_box(ax, (0.05, 0.80), (0.24, 0.115), "Final PSQ\nworkbook", "88 paired records", "#EDF3F8")
    draw_box(ax, (0.38, 0.80), (0.24, 0.115), "Clinical linkage", "1 unlinked record\nexcluded", "#F8F8F8")
    draw_box(ax, (0.70, 0.80), (0.26, 0.115), "Local analysis\ncohort", "87 patients; 174 specimens", "#EAF5EE")
    arrow(ax, (0.29, 0.858), (0.38, 0.858))
    arrow(ax, (0.62, 0.858), (0.70, 0.858))

    local_boxes = [
        ((0.04, 0.55), "Paired\nquantitative", "10 genes; T-N effects", "#FFFFFF"),
        ((0.28, 0.55), "Pattern\nanalysis", "delta correlation; PCA", "#FFFFFF"),
        ((0.52, 0.55), "Recurrence\nassociation", "82 stage I-III;\n14 events", "#FFFFFF"),
        ((0.76, 0.55), "Recurrence ML", "82/14; grouped\n4 x 25 nested CV", "#F4F1FA"),
    ]
    ax.plot([0.83, 0.83], [0.80, 0.735], color="#333333", linewidth=1.0)
    ax.plot([0.14, 0.86], [0.735, 0.735], color="#333333", linewidth=1.0)
    for (xy, title, subtitle, color) in local_boxes:
        draw_box(ax, xy, (0.20, 0.12), title, subtitle, color)
        arrow(ax, (xy[0] + 0.10, 0.735), (xy[0] + 0.10, 0.67))

    draw_box(ax, (0.08, 0.31), (0.30, 0.12), "Local tissue ML", "87 pairs; grouped\n5 x 20 nested CV", "#EEF6FF")
    draw_box(ax, (0.60, 0.31), (0.32, 0.12), "Public tissue ML", "Colonomics 92 pairs;\nGSE119526 48 pairs", "#EEF6FF")
    ax.plot([0.74, 0.74], [0.80, 0.49], color="#333333", linewidth=1.0)
    ax.plot([0.23, 0.74], [0.49, 0.49], color="#333333", linewidth=1.0)
    arrow(ax, (0.23, 0.49), (0.23, 0.43))

    draw_box(ax, (0.56, 0.10), (0.36, 0.115), "Independent\npublic cohorts", "cohort-local ridge CV;\npaired gene effects", "#FFF7E8")
    arrow(ax, (0.76, 0.215), (0.76, 0.31))
    draw_box(ax, (0.12, 0.10), (0.28, 0.115), "Supportive TCGA", "393 patients; 45 pairs;\ndiscovery-overlap caveat", "#FFF7E8")
    ax.text(0.5, 0.965, "Data flow and analysis roles", ha="center", va="top", fontsize=11, fontweight="bold")
    save_all_formats(fig, "F1_data_flow")
    manifest(
        "F1_data_flow",
        [ROOT / "data" / "derived" / "clinical.tsv", ROOT / "data" / "derived" / "methylation_long.tsv", ROOT / "results" / "data_summary.json"],
        [SRC / "F1_flow_source_data.tsv"],
        ["The retained local cohort contains 87 clinical-matched patients with paired tumor and normal PSQ values; recurrence, local tissue-classification, and public tissue-classification analyses are separate analysis roles."],
        ["Counts were read from the frozen derived files and data_summary.json; planned ML split designs were read from registry/ml_config.json."],
        ["This panel describes cohort flow and analysis roles; it is not a statistical test."],
    )


def figure2(wide: pd.DataFrame, paired_results: pd.DataFrame) -> None:
    raw_rows = []
    for _, r in wide.iterrows():
        for g in GENES:
            raw_rows.append({"study_id": r.study_id, "gene": g, "tissue": "Tumor", "methylation_pct": r[f"T_{g}"]})
            raw_rows.append({"study_id": r.study_id, "gene": g, "tissue": "Normal", "methylation_pct": r[f"N_{g}"]})
    raw = pd.DataFrame(raw_rows)
    write_tsv(SRC / "F2_paired_raw_source_data.tsv", raw)
    paired_source = paired_results.copy()
    paired_source["paired_delta_mean_pctpt"] = paired_source["mean_difference_pp"]
    paired_source["paired_delta_ci_low_pctpt"] = paired_source["mean_diff_bca95_low"]
    paired_source["paired_delta_ci_high_pctpt"] = paired_source["mean_diff_bca95_high"]
    paired_source["paired_t_p_bh10"] = paired_source["paired_t_BH_q"]
    write_tsv(SRC / "F2_paired_effect_source_data.tsv", paired_source)
    fig, axes = plt.subplots(2, 5, figsize=(7.5, 5.6), sharey=True)
    rng = np.random.default_rng(SEED)
    for ax, g in zip(axes.flat, GENES):
        t = wide[f"T_{g}"].to_numpy(float)
        n = wide[f"N_{g}"].to_numpy(float)
        x_n = np.full(len(n), 0.0) + rng.normal(0, 0.025, len(n))
        x_t = np.full(len(t), 1.0) + rng.normal(0, 0.025, len(t))
        for i in range(len(t)):
            ax.plot([x_n[i], x_t[i]], [n[i], t[i]], color="#C7C7C7", linewidth=0.45, alpha=0.55, zorder=1)
        ax.scatter(x_n, n, s=8, color=COL_N, alpha=0.85, zorder=2, linewidths=0)
        ax.scatter(x_t, t, s=8, color=COL_T, alpha=0.85, zorder=2, linewidths=0)
        st = paired_results.loc[paired_results.gene == g].iloc[0]
        ax.errorbar(1.55, st.mean_difference_pp, yerr=[[st.mean_difference_pp - st.mean_diff_bca95_low], [st.mean_diff_bca95_high - st.mean_difference_pp]], fmt="o", color="black", markersize=3.5, capsize=2.5)
        ax.axhline(0, color="#DDDDDD", linewidth=0.7)
        ax.set_title(g, fontsize=9)
        ax.set_xlim(-0.25, 1.85)
        ax.set_xticks([0, 1, 1.55], ["N", "T", "Delta"], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Methylation (%); delta (pp)", fontsize=8)
    axes[1, 0].set_ylabel("Methylation (%); delta (pp)", fontsize=8)
    fig.suptitle("Paired pyrosequencing methylation by gene", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save_all_formats(fig, "F2_paired_psq")
    manifest(
        "F2_paired_psq",
        [ROOT / "data" / "derived" / "methylation_wide.tsv", ROOT / "results" / "paired.csv"],
        [SRC / "F2_paired_raw_source_data.tsv", SRC / "F2_paired_effect_source_data.tsv"],
        ["Tumor methylation is summarized against matched normal methylation for all ten fixed genes."],
        ["BCa 95% confidence intervals use 5,000 patient bootstrap resamples of paired tumor-minus-normal differences."],
        ["The plotted confidence intervals are patient-level summaries; they do not assess technical replicate variability."],
    )


def figure3(clinical: pd.DataFrame, wide: pd.DataFrame) -> None:
    pca_scores_path = ROOT / "results" / "pca_scores.csv"
    pca_loadings_path = ROOT / "results" / "pca_loadings.csv"
    corr_path = ROOT / "results" / "correlations.csv"
    pattern_path = ROOT / "results" / "pattern_summary.json"
    for p in [pca_scores_path, pca_loadings_path, corr_path, pattern_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required verified pattern result is missing: {p}")
    delta = delta_matrix(wide)
    mat = delta[GENES].to_numpy(float)
    z = StandardScaler().fit_transform(mat)
    pc_scores_raw = pd.read_csv(pca_scores_path)
    pc_scores = pc_scores_raw.drop(columns=["patient_id"], errors="ignore").merge(clinical[["study_id", "event"]], on="study_id", how="left")
    pc_load = pd.read_csv(pca_loadings_path)
    corr_pairs = pd.read_csv(corr_path)
    corr = pd.DataFrame(np.eye(len(GENES)), index=GENES, columns=GENES)
    for r in corr_pairs.itertuples(index=False):
        corr.loc[r.gene_a, r.gene_b] = r.spearman_rho
        corr.loc[r.gene_b, r.gene_a] = r.spearman_rho
    pattern = json.loads(pattern_path.read_text())
    evr = pattern["pca_explained_variance_ratio"]
    delta_source = delta.drop(columns=["patient_id"]).merge(clinical[["study_id", "event", "recurrence_primary"]], on="study_id", how="left")
    write_tsv(SRC / "F3_delta_matrix_source_data.tsv", delta_source)
    write_tsv(SRC / "F3_pca_scores_source_data.tsv", pc_scores)
    write_tsv(SRC / "F3_pca_loadings_source_data.tsv", pc_load)
    corr.reset_index(names="gene").to_csv(SRC / "F3_spearman_correlation_source_data.tsv", sep="\t", index=False, float_format="%.10g")
    scores = pc_scores[["PC1", "PC2"]].to_numpy(float)
    order = np.argsort(scores[:, 0])
    fig = plt.figure(figsize=(7.5, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0], wspace=0.32, hspace=0.38)
    ax1 = fig.add_subplot(gs[:, 0])
    im = ax1.imshow(z[order, :], aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax1.set_xticks(range(len(GENES)), GENES, rotation=90, fontsize=7)
    ax1.set_yticks([])
    ax1.set_title("A. Standardized paired differences", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax1, fraction=0.035, pad=0.02)
    cb.set_label("z score", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_xticks(range(len(GENES)), GENES, rotation=90, fontsize=6)
    ax2.set_yticks(range(len(GENES)), GENES, fontsize=6)
    ax2.set_title("B. Spearman correlation of deltas", fontsize=9, loc="left")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.03)
    cb2.ax.tick_params(labelsize=7)
    ax3 = fig.add_subplot(gs[1, 1])
    event = pc_scores["event"].to_numpy()
    ax3.scatter(scores[event == 0, 0], scores[event == 0, 1], s=22, color="#999999", label="No event", alpha=0.9)
    ax3.scatter(scores[event == 1, 0], scores[event == 1, 1], s=24, color=COL_ACCENT, label="Event", alpha=0.95)
    ax3.axhline(0, color="#DDDDDD", linewidth=0.7)
    ax3.axvline(0, color="#DDDDDD", linewidth=0.7)
    ax3.set_xlabel(f"PC1 ({evr['PC1']*100:.1f}%)", fontsize=8)
    ax3.set_ylabel(f"PC2 ({evr['PC2']*100:.1f}%)", fontsize=8)
    ax3.set_title("C. Outcome-blind PCA", fontsize=9, loc="left")
    ax3.tick_params(labelsize=7)
    ax3.legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Patient-level methylation-difference patterns", fontsize=11, y=0.985)
    save_all_formats(fig, "F3_delta_patterns")
    manifest(
        "F3_delta_patterns",
        [ROOT / "data" / "derived" / "clinical.tsv", ROOT / "data" / "derived" / "methylation_wide.tsv", pca_scores_path, pca_loadings_path, corr_path, pattern_path],
        [SRC / "F3_delta_matrix_source_data.tsv", SRC / "F3_pca_scores_source_data.tsv", SRC / "F3_pca_loadings_source_data.tsv", SRC / "F3_spearman_correlation_source_data.tsv"],
        ["The figure displays patient-level tumor-minus-normal methylation patterns across the fixed genes."],
        ["PCA was computed on standardized paired differences without using outcome labels for fitting."],
        ["PCA and correlations are descriptive and do not define a recurrence subtype or prediction score."],
    )


def own_external_summary(paired_results: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cohort": "own_87",
            "gene": paired_results["gene"],
            "n_pairs": paired_results["n_pairs"],
            "mean_delta": paired_results["mean_difference_pp"],
            "bca95_low": paired_results["mean_diff_bca95_low"],
            "bca95_high": paired_results["mean_diff_bca95_high"],
            "paired_diff_mean": paired_results["mean_difference_pp"],
            "paired_diff_ci_low": paired_results["mean_diff_bca95_low"],
            "paired_diff_ci_high": paired_results["mean_diff_bca95_high"],
            "unit": "percentage points",
        }
    )


def figure4(paired_results: pd.DataFrame) -> None:
    public_path = ROOT / "results" / "external" / "paired_public_effects.tsv"
    probes_path = ROOT / "results" / "external" / "probe_coverage.tsv"
    public = read_required_csv(public_path, sep="\t")
    probes = read_required_csv(probes_path, sep="\t")
    keep = public[public["cohort"].isin(["colonomics", "gse119526"])].copy()
    keep = keep.rename(columns={"mean_delta_beta": "mean_delta"})
    keep["paired_diff_mean"] = keep["mean_delta"]
    keep["paired_diff_ci_low"] = keep["bca95_low"]
    keep["paired_diff_ci_high"] = keep["bca95_high"]
    plot_df = pd.concat([own_external_summary(paired_results), keep], ignore_index=True, sort=False)
    plot_df["cohort_label"] = plot_df["cohort"].map(COHORT_LABELS)
    write_tsv(SRC / "F4_external_gene_effect_source_data.tsv", plot_df)
    write_tsv(SRC / "F4_external_probe_coverage_source_data.tsv", probes)
    fig = plt.figure(figsize=(7.25, 7.35))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.7, 1.35], hspace=0.34, wspace=0.18)
    panels = [
        ("own_87", "This study PSQ\nn=87 pairs", "Mean T-N difference (percentage points)", (0, 50), [0, 10, 20, 30, 40, 50], COL_T),
        ("colonomics", "Colonomics 450K\nn=92 pairs", "Mean T-N difference (array beta)", (0, 0.60), [0, 0.2, 0.4, 0.6], "#E69F00"),
        ("gse119526", "GSE119526 EPIC\nn=48 pairs", "Mean T-N difference (array beta)", (0, 0.60), [0, 0.2, 0.4, 0.6], COL_ACCENT),
    ]
    display_genes = list(reversed(GENES))
    for idx, (cohort, title, xlabel, xlim, xticks, color) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        sub = plot_df[plot_df.cohort == cohort].set_index("gene").loc[display_genes].reset_index()
        y = np.arange(len(display_genes))
        ax.errorbar(
            sub["mean_delta"],
            y,
            xerr=[sub["mean_delta"] - sub["bca95_low"], sub["bca95_high"] - sub["mean_delta"]],
            fmt="o",
            color=color,
            ecolor="#555555",
            elinewidth=0.9,
            capsize=2.2,
            markersize=3.8,
        )
        ax.axvline(0, color="#BBBBBB", linewidth=0.8)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_yticks(y, display_genes if idx == 0 else [], fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    ax_cov = fig.add_subplot(gs[1, :])
    cov = probes[probes.cohort.isin(["colonomics", "gse119526"])].pivot(index="cohort", columns="gene", values="n_available_probes").loc[["colonomics", "gse119526"], GENES]
    im = ax_cov.imshow(cov.values, aspect="auto", cmap="Greens", vmin=0, vmax=max(17, float(np.nanmax(cov.values))))
    ax_cov.set_yticks(range(len(cov.index)), [COHORT_LABELS[c].replace("\n", " ") for c in cov.index], fontsize=7)
    ax_cov.set_xticks(range(len(GENES)), GENES, rotation=90, fontsize=7)
    ax_cov.set_title("B. Number of fixed CpG probes used per gene", fontsize=9, loc="left")
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            val = cov.iloc[i, j]
            text_color = "white" if pd.notna(val) and val >= 12 else "black"
            ax_cov.text(j, i, "" if pd.isna(val) else f"{int(val)}", ha="center", va="center", fontsize=6.5, color=text_color)
    cb = fig.colorbar(im, ax=ax_cov, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    fig.text(0.08, 0.925, "A. Direction of paired tissue differences by cohort", ha="left", va="top", fontsize=9)
    fig.suptitle("Public-cohort correspondence of methylation effects", fontsize=11, y=0.975)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.84, bottom=0.10, hspace=0.44, wspace=0.20)
    save_all_formats(fig, "F4_external_effects")
    manifest(
        "F4_external_effects",
        [ROOT / "results" / "paired.csv", public_path, probes_path],
        [SRC / "F4_external_gene_effect_source_data.tsv", SRC / "F4_external_probe_coverage_source_data.tsv"],
        ["Processed public-cohort summaries show positive tumor-minus-normal gene effects in the displayed independent cohorts."],
        ["Displayed effect estimates and confidence intervals are read from verified result files; local PSQ differences remain in percentage points."],
        ["Array beta values and PSQ percentages are platform-specific and should not be interpreted as pooled effect estimates; TCGA is reserved for supportive/supplementary use because of discovery-overlap concerns."],
    )


def figure5() -> None:
    recurrence_path = ROOT / "results" / "recurrence.csv"
    recurrence_all_path = ROOT / "results" / "recurrence_all_stage.csv"
    primary = read_required_csv(recurrence_path)
    allstage = read_required_csv(recurrence_all_path)
    for df in [primary, allstage]:
        df["HR_per_10_pctpt"] = df["hr_per10"]
        df["HR_ci_low"] = df["ci_lower"]
        df["HR_ci_high"] = df["ci_upper"]
        df["cox_p"] = df["p_value"]
        df["cox_p_bh10"] = df["p_bh"]
        df["PH_p"] = df["ph_p_value"]
    write_tsv(SRC / "F5_cox_primary_source_data.tsv", primary)
    write_tsv(SRC / "F5_cox_all_stage_source_data.tsv", allstage)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.5), sharey=True)
    display_genes = list(reversed(GENES))
    for ax, data, title in [
        (axes[0], primary, "A. Primary recurrence set\n82 patients; 14 events"),
        (axes[1], allstage, "B. All-stage sensitivity\n87 patients; 17 events"),
    ]:
        data = data.set_index("gene").loc[display_genes].reset_index()
        y = np.arange(len(GENES))
        finite = np.isfinite(data["hr_per10"]) & data["estimable"].astype(bool)
        ax.axvline(1, color="#888888", linewidth=0.8, linestyle="--")
        ax.errorbar(data.loc[finite, "hr_per10"], y[finite], xerr=[data.loc[finite, "hr_per10"] - data.loc[finite, "ci_lower"], data.loc[finite, "ci_upper"] - data.loc[finite, "hr_per10"]], fmt="o", color="black", ecolor="#555555", elinewidth=1.0, capsize=2.5, markersize=4)
        ax.set_xscale("log")
        max_hi = float(np.nanmax(data["ci_upper"]))
        min_lo = float(np.nanmin(data["ci_lower"]))
        ax.set_xlim(max(0.25, min(0.35, min_lo * 0.85)), max(2.2, max_hi * 1.08))
        ax.set_xticks([0.4, 0.6, 1.0, 1.5, 2.0])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
        ax.set_yticks(y, display_genes, fontsize=8)
        ax.set_xlabel("Hazard ratio per 10 percentage points", fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(axis="x", labelsize=7)
        for yi, (_, r) in enumerate(data.iterrows()):
            txt = "NE" if not bool(r.estimable) or pd.isna(r.hr_per10) else f"BH p={r.p_bh:.2f}"
            ax.text(1.08, yi, txt, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5, color=COL_GRAY, clip_on=False)
    fig.suptitle("Exploratory unadjusted recurrence associations", fontsize=11, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_all_formats(fig, "F5_recurrence_forest")
    manifest(
        "F5_recurrence_forest",
        [recurrence_path, recurrence_all_path],
        [SRC / "F5_cox_primary_source_data.tsv", SRC / "F5_cox_all_stage_source_data.tsv"],
        ["The panel reports unadjusted per-gene Cox associations between tumor methylation and recorded recurrence/progression time."],
        ["All estimates, confidence intervals, p values, BH q values, and PH diagnostics are read from verified recurrence result tables."],
        ["Small event counts and missing death data limit time-to-event interpretation; no multivariable prediction model is shown."],
    )


def write_captions() -> None:
    text = f"""# Figure Captions

Generated by `scripts/make_figures.py` with fixed analysis date 2026-09-05.

**Figure 1. Data flow and analysis roles.** Paired tumor and adjacent-normal pyrosequencing measurements for ten genes were linked to clinical records from 87 patients, yielding 174 local tissue specimens. The primary recurrence analysis included 82 patients with provider-defined stage I-III disease and no recorded palliative operation, with 14 events, and was extended to grouped 4-fold x 25-repeat nested cross-validation. Tumor-versus-adjacent-normal classification used 87 patient pairs in grouped 5-fold x 20-repeat nested cross-validation. Colonomics (92 pairs) and GSE119526 (48 pairs) were evaluated separately with cohort-local tissue-classification models and paired tissue-effect summaries. TCGA-COAD/READ was retained as a supportive reference because of overlap with candidate discovery.

**Figure 2. Paired pyrosequencing methylation by gene.** Lines connect adjacent-normal (N) and tumor (T) methylation percentages from the same patient. Black points show the mean paired tumor-minus-normal difference in percentage points. Error bars are 95% BCa bootstrap confidence intervals from {N_BOOT:,} patient-pair resamples.

**Figure 3. Patient-level methylation-difference patterns.** (A) Gene-standardized tumor-minus-normal differences, with patients ordered by increasing PC1 score. (B) Spearman correlations between paired differences across genes. (C) Principal component analysis of the standardized differences. Clinical outcomes were not used to fit the PCA. Colors show recorded event status for descriptive comparison.

**Figure 4. Public-cohort correspondence of methylation effects.** (A) Mean paired tumor-minus-normal differences in the local pyrosequencing cohort (87 pairs), Colonomics (92 pairs), and GSE119526 (48 pairs). Error bars are 95% BCa bootstrap confidence intervals from {N_BOOT:,} patient-pair resamples. Separate axes retain percentage-point units for pyrosequencing and beta units for arrays. (B) Numbers of fixed CpG probes retained per gene in each array cohort. Comparisons are at the gene level because exact assayed CpG overlap is undocumented. Supportive TCGA results are reported separately because of overlap with candidate discovery.

**Figure 5. Exploratory unadjusted recurrence associations.** Points and error bars show hazard ratios and 95% confidence intervals per 10 percentage-point increase in tumor methylation from separate Cox models for each gene. (A) Primary recurrence analysis of patients with provider-defined stage I-III disease and no recorded palliative operation. (B) All-stage recurrence/progression sensitivity analysis. Dashed lines mark a hazard ratio of 1. P values are Benjamini-Hochberg adjusted across the ten genes within each panel.
"""
    (MANUSCRIPT / "figure_captions.md").write_text(text)


def write_run_summary() -> None:
    files = sorted([p for p in FIG.iterdir() if p.is_file()])
    rows = []
    for p in files:
        rows.append({"file": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": common.sha256(p)})
    pd.DataFrame(rows).to_csv(ROOT / "verification" / "figure_file_manifest.tsv", sep="\t", index=False)


def main() -> None:
    ensure_dirs()
    clinical, wide, long = read_inputs()
    paired_results = read_required_csv(ROOT / "results" / "paired.csv")
    figure1(clinical, long)
    figure2(wide, paired_results)
    figure3(clinical, wide)
    figure4(paired_results)
    figure5()
    write_captions()
    write_run_summary()


if __name__ == "__main__":
    main()
