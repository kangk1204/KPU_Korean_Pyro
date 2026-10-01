#!/usr/bin/env python3
"""Render recurrence revision supplementary figures from frozen 119/124 source tables.

Outputs are confined to 124_Integrated_Revision_20260909/figures/Recurrence_Revision.
The script copies the exact source tables used for plotting into source_data/
and records SHA-256 hashes for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib



def require(condition, message="Integrity check failed"):
    if not condition:
        raise RuntimeError(message)

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
OUT = ROOT / "figures" / "Recurrence_Revision"
SOURCE_OUT = OUT / "source_data"
REV_REC = ROOT / "results" / "recurrence_revision"
SRC119 = WORKSPACE / "119_Figure1_ABC_Revision_20260908"

GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
MODEL_COLORS = {
    "Clinical": "#7A7A7A",
    "Tumor methylation": "#2A7FB8",
    "Clinical + methylation": "#D07A2D",
}
STAGE_COLORS = {
    "Recorded stage": "#5B5B5B",
    "TNM-derived stage": "#2A7FB8",
    "Stage omitted": "#D07A2D",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "legend.frameon": False,
    "savefig.dpi": 600,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_sources() -> dict[str, dict[str, str]]:
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "S2_cox_primary_source_data.tsv": SRC119 / "figures" / "source_data" / "F5_cox_primary_source_data.tsv",
        "S2_cox_all_stage_source_data.tsv": SRC119 / "figures" / "source_data" / "F5_cox_all_stage_source_data.tsv",
        "S3_table_s3_recurrence_primary_corrected.tsv": REV_REC / "table_s3_recurrence_primary_corrected.tsv",
        "S3_table_s3_delta_ci.tsv": REV_REC / "table_s3_delta_ci.tsv",
        "S3_null_km_benchmark.tsv": REV_REC / "null_km_benchmark.tsv",
        "S3_horizon_support_counts.tsv": REV_REC / "horizon_support_counts.tsv",
        "S4_stage_plot_ready_summary.tsv": SRC119 / "results" / "reviewer" / "stage_plot_ready_summary.tsv",
        "S4_stage_plot_ready_deltas.tsv": SRC119 / "results" / "reviewer" / "stage_plot_ready_deltas.tsv",
        "S4_stage_patient_audit.tsv": SRC119 / "private" / "stage_patient_audit.tsv",
    }
    manifest = {}
    for dest_name, src in sources.items():
        if not src.exists():
            raise FileNotFoundError(src)
        dest = SOURCE_OUT / dest_name
        shutil.copy2(src, dest)
        manifest[dest_name] = {
            "source": str(src.relative_to(WORKSPACE)),
            "sha256": sha256(dest),
        }
    return manifest


def letter(ax, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def save_figure(fig, stem: str) -> dict[str, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": OUT / f"{stem}.pdf",
        "svg": OUT / f"{stem}.svg",
        "png": OUT / f"{stem}.png",
    }
    fig.savefig(paths["pdf"], facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(paths["svg"], facecolor="white", metadata={"Date": None})
    fig.savefig(paths["png"], dpi=600, facecolor="white")
    plt.close(fig)
    return {k: sha256(v) for k, v in paths.items()}


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def plot_hr_panel(ax, frame: pd.DataFrame, title: str, color: str) -> None:
    df = frame.set_index("gene").reindex(GENES).reset_index()
    y = np.arange(len(df))
    hr = df["HR_per_10_pctpt"].astype(float)
    lo = df["HR_ci_low"].astype(float)
    hi = df["HR_ci_high"].astype(float)
    require((lo > 0).all() and (hi > 0).all() and (hr > 0).all(), 'HR forest log axis requires positive intervals')
    ax.hlines(y, lo, hi, color=color, lw=1.0, zorder=2)
    ax.scatter(hr, y, color=color, edgecolor="white", linewidth=0.4, s=22, zorder=3)
    ax.axvline(1.0, color="#B8B8B8", lw=0.8, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.38, 1.85)
    ax.set_xticks([0.5, 0.75, 1.0, 1.5])
    ax.set_xticklabels(["0.5", "0.75", "1.0", "1.5"], fontsize=8.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["gene"], fontstyle="italic", fontsize=8.6)
    ax.grid(axis="x", color="#ECECEC", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, pad=8)


def figure_s1() -> dict[str, str]:
    primary = pd.read_csv(SOURCE_OUT / "S2_cox_primary_source_data.tsv", sep="\t")
    allstage = pd.read_csv(SOURCE_OUT / "S2_cox_all_stage_source_data.tsv", sep="\t")
    require(list(primary['gene']) == GENES, "Integrity check failed: list(primary['gene']) == GENES")
    require(list(allstage['gene']) == GENES, "Integrity check failed: list(allstage['gene']) == GENES")
    require(int(primary['n'].iloc[0]) == 82 and int(primary['events'].iloc[0]) == 14, "Integrity check failed: int(primary['n'].iloc[0]) == 82 and int(primary['events'].iloc[0]) == 14")
    require(int(allstage['n'].iloc[0]) == 87 and int(allstage['events'].iloc[0]) == 17, "Integrity check failed: int(allstage['n'].iloc[0]) == 87 and int(allstage['events'].iloc[0]) == 17")
    fig, axes = plt.subplots(1, 2, figsize=(175 / 25.4, 120 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.82, wspace=0.16)
    plot_hr_panel(axes[0], primary, "Primary recurrence set\n(n=82; 14 events)", "#2A7FB8")
    plot_hr_panel(axes[1], allstage, "All-stage sensitivity\n(n=87; 17 events)", "#D07A2D")
    axes[0].set_ylim(len(GENES) - 0.5, -0.5)
    axes[1].tick_params(axis="y", length=0, labelleft=False)
    for ax, lab in zip(axes, "AB"):
        letter(ax, lab)
        ax.set_xlabel("Hazard ratio per 10 pp (log scale)")
    return save_figure(fig, "S2_Cox_gene_HR_forest")


def figure_s2() -> dict[str, str]:
    table = pd.read_csv(SOURCE_OUT / "S3_table_s3_recurrence_primary_corrected.tsv", sep="\t")
    delta = pd.read_csv(SOURCE_OUT / "S3_table_s3_delta_ci.tsv", sep="\t")
    support = pd.read_csv(SOURCE_OUT / "S3_horizon_support_counts.tsv", sep="\t")
    null = pd.read_csv(SOURCE_OUT / "S3_null_km_benchmark.tsv", sep="\t")
    require(list(table['block']) == ['clinical', 'tumor10', 'combined'], "Integrity check failed: list(table['block']) == ['clinical', 'tumor10', 'combined']")
    support5 = support[support["horizon_days"].eq(1825)].iloc[0]
    drow = delta[delta["metric"].eq("delta_uno_c")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(175 / 25.4, 86 / 25.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.22, top=0.76, wspace=0.34)
    y = np.arange(len(table))
    for i, row in table.iterrows():
        label = row["predictor_block"]
        color = MODEL_COLORS[label]
        axes[0].hlines(i, row["uno_c_5y_ci_low"], row["uno_c_5y_ci_high"], color=color, lw=1.15)
        axes[0].scatter(row["uno_c_5y"], i, s=28, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    axes[0].axvline(0.5, color="#B8B8B8", lw=0.8, ls="--")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(table["predictor_block"])
    axes[0].set_ylim(2.5, -0.5)
    axes[0].set_xlim(0.34, 0.84)
    axes[0].set_xlabel("Five-year Uno C")
    axes[0].set_title("Ridge Cox model blocks", pad=8)
    axes[0].grid(axis="x", color="#ECECEC", lw=0.7)
    axes[0].set_axisbelow(True)
    axes[1].hlines(0, drow["ci_low"], drow["ci_high"], color=MODEL_COLORS["Clinical + methylation"], lw=1.2)
    axes[1].scatter(drow["estimate"], 0, s=30, color=MODEL_COLORS["Clinical + methylation"], edgecolor="white", linewidth=0.4, zorder=3)
    axes[1].axvline(0, color="#B8B8B8", lw=0.8, ls="--")
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(["Combined − clinical"])
    axes[1].set_ylim(-0.7, 0.7)
    axes[1].set_xlim(-0.25, 0.25)
    axes[1].set_xlabel("Δ Uno C")
    axes[1].set_title("Increment from methylation", pad=8)
    axes[1].grid(axis="x", color="#ECECEC", lw=0.7)
    for ax, lab in zip(axes, "AB"):
        letter(ax, lab)
    subtitle = f"Five-year horizon | {int(support5['n_patients'])} patients, {int(support5['total_events'])} events"
    fig.text(0.10, 0.86, subtitle, fontsize=6.5, ha="left", color="#444444")
    return save_figure(fig, "S3_recurrence_uno_delta")


def figure_s3() -> dict[str, str]:
    summary = pd.read_csv(SOURCE_OUT / "S4_stage_plot_ready_summary.tsv", sep="\t")
    deltas = pd.read_csv(SOURCE_OUT / "S4_stage_plot_ready_deltas.tsv", sep="\t")
    audit = pd.read_csv(SOURCE_OUT / "S4_stage_patient_audit.tsv", sep="\t")
    common = audit[audit["analysis_included"].astype(bool)]
    require(len(common) == 79 and int(common['event'].sum()) == 14, "Integrity check failed: len(common) == 79 and int(common['event'].sum()) == 14")
    stages = [("provider", "Recorded stage"), ("tnm", "TNM-derived stage"), ("nostage", "Stage omitted")]
    fig, axes = plt.subplots(1, 2, figsize=(175 / 25.4, 90 / 25.4), gridspec_kw={"width_ratios": [1.32, 1.0]})
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.23, top=0.78, wspace=0.37)
    offsets = {"clinical": 0.12, "combined": -0.12}
    markers = {"clinical": "s", "combined": "o"}
    labels = {"clinical": "Clinical", "combined": "Clinical + methylation"}
    for i, (key, label) in enumerate(stages):
        for block in ["clinical", "combined"]:
            row = summary[summary["stage_definition"].eq(key) & summary["model_block"].eq(block)].iloc[0]
            y = i + offsets[block]
            color = MODEL_COLORS[labels[block]]
            axes[0].hlines(y, row["ci_low.uno_c"], row["ci_high.uno_c"], color=color, lw=1.05)
            axes[0].scatter(row["estimate.uno_c"], y, color=color, marker=markers[block], s=24, edgecolor="white", linewidth=0.4, zorder=3)
        drow = deltas[deltas["stage_definition"].eq(key) & deltas["metric"].eq("delta_uno_c")].iloc[0]
        delta_color = MODEL_COLORS["Clinical + methylation"]
        axes[1].hlines(i, drow["ci_low"], drow["ci_high"], color=delta_color, lw=1.1)
        axes[1].scatter(drow["estimate"], i, color=delta_color, s=25, edgecolor="white", linewidth=0.4, zorder=3)
    for ax in axes:
        ax.set_yticks(range(3))
        ax.set_yticklabels([x[1] for x in stages])
        ax.set_ylim(2.5, -0.5)
        ax.grid(axis="x", color="#ECECEC", lw=0.7)
        ax.set_axisbelow(True)
    axes[0].axvline(0.5, color="#B8B8B8", lw=0.8, ls="--")
    axes[0].set_xlim(0.32, 0.86)
    axes[0].set_xlabel("Five-year Uno C")
    axes[0].set_title("Common primary subset", pad=8)
    axes[1].axvline(0, color="#B8B8B8", lw=0.8, ls="--")
    axes[1].set_xlim(-0.25, 0.25)
    axes[1].set_xlabel("Δ Uno C after adding methylation")
    axes[1].set_title("Increment across stage definitions", pad=8)
    for ax, lab in zip(axes, "AB"):
        letter(ax, lab)
    handles = [
        Line2D([], [], color=MODEL_COLORS["Clinical"], marker="s", lw=1.0, markersize=4, label="Clinical"),
        Line2D([], [], color=MODEL_COLORS["Clinical + methylation"], marker="o", lw=1.0, markersize=4, label="Clinical + methylation"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.48, 0.04), ncol=2)
    fig.text(0.14, 0.86, "Five-year horizon | common explicit-stage subset n=79, 14 events", fontsize=6.5, ha="left", color="#444444")
    return save_figure(fig, "S4_stage_concordance_sensitivity")


def write_legends() -> None:
    legends = """# Recurrence revision figure legends\n\n**Supplementary Figure S2 | Single-gene recurrence Cox associations.** Forest plots show hazard ratios per 10 percentage-point increase in tumor methylation for the ten prespecified genes, ordered as in the manuscript. Panel A uses the primary stage I–III non-palliative recurrence set (n=82, 14 events). Panel B shows the all-stage recurrence-or-progression sensitivity set (n=87, 17 events). Horizontal bars are Wald 95% confidence intervals from the saved aggregate Cox analyses; Cox p values, Benjamini-Hochberg adjusted q values, and proportional-hazards diagnostics are provided in the source data.\n\n**Supplementary Figure S3 | Internal recurrence prediction did not show a clear increment from methylation.** Panel A shows 5-year Uno concordance for ridge Cox models using clinical predictors, tumor methylation predictors, or both. Panel B shows the paired patient-cluster bootstrap difference between the combined and clinical ridge models. Points are preserved nested-CV estimates and bars are 95% CIs. The primary set included 82 patients and 14 total events; at five years, 11 events and 35 controls were observed, while 36 patients were censored before the horizon. Null-KM Brier was 0.1303 and the ridge model Brier values were essentially similar; these Brier values are source-data diagnostics rather than plotted panels.\n\n**Supplementary Figure S4 | Stage definition sensitivity of recurrence concordance.** Analyses use the common primary subset with explicit stage information (n=79, 14 events). Panel A shows 5-year Uno concordance for clinical and combined ridge Cox models under recorded stage, TNM-derived stage, and stage-omitted definitions. Panel B shows the paired combined-minus-clinical Uno C difference. Points are preserved nested-CV estimates from 25 repeats of fourfold cross-validation and bars are 95% bootstrap CIs.\n"""
    (OUT / "recurrence_revision_figure_legends.md").write_text(legends)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = copy_sources()
    output_hashes = {
        "S2_Cox_gene_HR_forest": figure_s1(),
        "S3_recurrence_uno_delta": figure_s2(),
        "S4_stage_concordance_sensitivity": figure_s3(),
    }
    write_legends()
    manifest = {
        "figure_contract": {
            "core_conclusion": "Recurrence Cox and nested-CV analyses remain exploratory and do not demonstrate a robust methylation increment beyond clinical predictors.",
            "archetype": "quantitative grid",
            "backend": "python/matplotlib",
            "width_mm_max": 180,
            "height_mm_max": 247,
            "minimum_label_pt_requested": 6,
            "pdf_fonttype": mpl.rcParams["pdf.fonttype"],
            "svg_fonttype": mpl.rcParams["svg.fonttype"],
            "png_dpi": 600,
        },
        "source_files": source_manifest,
        "outputs": output_hashes,
        "script_sha256": sha256(Path(__file__)),
        "panel_plan": {
            "S2": "single-gene Cox HR primary versus all-stage sensitivity",
            "S3": "primary 5-year Uno C plus paired combined-clinical delta; Brier only in legend/source benchmark",
            "S4": "common 79/14 stage-definition sensitivity for Uno C and delta",
        },
        "legends": "recurrence_revision_figure_legends.md",
    }
    (OUT / "recurrence_revision_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "complete", "figures": sorted(output_hashes)}, indent=2))


if __name__ == "__main__":
    main()
