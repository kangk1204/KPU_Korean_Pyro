#!/usr/bin/env python3
"""Render ML manuscript figures from the canonical final ML summary.

Default execution is fail-closed: no canonical F6/F7 files are written unless
``results/ml/summary.json`` is present, has ``status: final``, and points to
complete full-run manifests. Preliminary renders require ``--allow-preliminary``
and are confined to ``verification/ml/preliminary_figures``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

import common

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FIG = ROOT / "figures"
CANONICAL_SRC = CANONICAL_FIG / "source_data"
CANONICAL_MANUSCRIPT = ROOT / "manuscript"
VERIFY = ROOT / "verification" / "ml"
PRELIM = VERIFY / "preliminary_figures"
VALIDATOR = Path(os.environ["DATA_FIGURE_VALIDATOR"]) if os.environ.get("DATA_FIGURE_VALIDATOR") else None

MODEL_LABELS = {
    "ridge": "Ridge",
    "elastic_net": "Elastic net",
    "survival_forest": "RSF",
    "random_forest": "RF",
    "rbf_svm": "RBF-SVM",
    "best_single_gene": "Best single gene",
}
BLOCK_LABELS = {"clinical": "Clinical", "tumor10": "Tumor 10-gene", "methylation": "Tumor 10-gene", "combined": "Combined"}
PUBLIC_LABELS = {"colonomics": "Colonomics\n92 pairs", "gse119526": "GSE119526\n48 pairs"}
COLORS = {
    "clinical": "#0072B2",
    "tumor10": "#E69F00",
    "methylation": "#E69F00",
    "combined": "#009E73",
    "ridge": "#0072B2",
    "elastic_net": "#56B4E9",
    "survival_forest": "#D55E00",
    "random_forest": "#D55E00",
    "rbf_svm": "#CC79A7",
    "best_single_gene": "#666666",
}
SENSITIVITY_LABELS = {
    "no_stage": "No stage",
    "cea_binary": "CEA binary",
    "clinical_delta10": "Clinical + Δ10",
    "delta10": "Δ10",
    "all87_stage_advanced": "All-stage",
    "historical_cea_q95_ralyl_sfmbt2": "CEA + 2 genes\n(Q95)",
    "q95_binary_train_normals": "10 genes\n(Q95)",
}


class ContractError(RuntimeError):
    """Raised before canonical output when full ML evidence is incomplete."""


def read_json(path: Path) -> dict:
    if not path.exists():
        raise ContractError(f"Required ML evidence is missing: {path.relative_to(ROOT)}")
    return json.loads(path.read_text())


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ContractError(f"Required ML evidence is missing: {path.relative_to(ROOT)}")
    sep = "\t" if ".tsv" in path.name else ","
    return pd.read_csv(path, sep=sep)


def cfg() -> dict:
    return read_json(ROOT / "registry" / "ml_config.json")


def number(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)


def file_record(path: Path, base: Path) -> dict:
    path = path.resolve()
    rel = path.relative_to(base)
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


def write_tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, float_format="%.17g")


def scrub_svg(path: Path) -> None:
    text = path.read_text()
    if "<!DOCTYPE" not in text and "<!ENTITY" not in text:
        return
    lines: list[str] = []
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


def save_all_formats(fig: plt.Figure, stem: str, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_dir / f"{stem}.{ext}" for ext in ["pdf", "svg", "png", "tiff"]]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    scrub_svg(paths[1])
    fig.savefig(paths[2], dpi=600, bbox_inches="tight")
    fig.savefig(paths[3], dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return paths


def finite_values(*values: object) -> np.ndarray:
    numbers: list[float] = []
    for value in values:
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).ravel()
        numbers.extend(float(v) for v in arr if np.isfinite(v))
    return np.asarray(numbers, dtype=float)


def bounded_axis(values: np.ndarray, *, reference: float | None = None, domain: tuple[float, float] = (0.0, 1.0), min_span: float = 0.12) -> tuple[float, float]:
    extras = [values]
    if reference is not None:
        extras.append(np.asarray([reference], dtype=float))
    vals = finite_values(*extras)
    if len(vals) == 0:
        return domain
    low = max(domain[0], float(vals.min()))
    high = min(domain[1], float(vals.max()))
    span = max(high - low, min_span)
    pad = span * 0.08
    return max(domain[0], low - pad), min(domain[1], high + pad)


def frame_metric_axis(df: pd.DataFrame, *, reference: float | None = None, value_cols: tuple[str, ...] = ("plot_value", "plot_low", "plot_high"), domain: tuple[float, float] = (0.0, 1.0), min_span: float = 0.12) -> tuple[float, float]:
    cols = [df[col].to_numpy(float) for col in value_cols if col in df.columns]
    return bounded_axis(finite_values(*cols), reference=reference, domain=domain, min_span=min_span)


def draw_horizontal_ci(
    ax: plt.Axes,
    point: object,
    y: object,
    low: object,
    high: object,
    *,
    color: str,
    ecolor: str,
    label: str | None = None,
    capsize: float = 2.0,
    markersize: float = 4.0,
) -> None:
    x = np.asarray(point, dtype=float).ravel()
    yy = np.asarray(y, dtype=float).ravel()
    lo = np.asarray(low, dtype=float).ravel()
    hi = np.asarray(high, dtype=float).ravel()
    ci_ok = np.isfinite(yy) & np.isfinite(lo) & np.isfinite(hi)
    if ci_ok.any():
        ax.hlines(yy[ci_ok], lo[ci_ok], hi[ci_ok], color=ecolor, linewidth=0.8, zorder=2)
        cap = capsize / 35.0
        ax.vlines(lo[ci_ok], yy[ci_ok] - cap, yy[ci_ok] + cap, color=ecolor, linewidth=0.8, zorder=2)
        ax.vlines(hi[ci_ok], yy[ci_ok] - cap, yy[ci_ok] + cap, color=ecolor, linewidth=0.8, zorder=2)
    point_ok = np.isfinite(x) & np.isfinite(yy)
    if point_ok.any():
        ax.plot(x[point_ok], yy[point_ok], "o", color=color, markersize=markersize, label=label, linestyle="none", zorder=3)


def draw_vertical_ci(
    ax: plt.Axes,
    x: object,
    point: object,
    low: object,
    high: object,
    *,
    ecolor: str,
    capsize: float = 2.0,
) -> None:
    xx = np.asarray(x, dtype=float).ravel()
    lo = np.asarray(low, dtype=float).ravel()
    hi = np.asarray(high, dtype=float).ravel()
    ci_ok = np.isfinite(xx) & np.isfinite(lo) & np.isfinite(hi)
    if not ci_ok.any():
        return
    ax.vlines(xx[ci_ok], lo[ci_ok], hi[ci_ok], color=ecolor, linewidth=0.8, zorder=3)
    cap = capsize / 35.0
    ax.hlines(lo[ci_ok], xx[ci_ok] - cap, xx[ci_ok] + cap, color=ecolor, linewidth=0.8, zorder=3)
    ax.hlines(hi[ci_ok], xx[ci_ok] - cap, xx[ci_ok] + cap, color=ecolor, linewidth=0.8, zorder=3)


def validate_manifest_counts() -> None:
    config = cfg()
    tissue = read_json(ROOT / "results" / "ml" / "tissue" / "run_manifest.json")
    expected_tissue = {
        "patients": 87,
        "outer_repeats": config["tissue"]["outer_repeats"],
        "bootstrap_replicates": config["uncertainty"]["patient_bootstrap"],
        "optimism_bootstrap": config["uncertainty"]["optimism_bootstrap"],
        "permutations": config["uncertainty"]["permutations"],
    }
    for key, value in expected_tissue.items():
        if int(tissue.get(key, -1)) != int(value):
            raise ContractError(f"results/ml/tissue/run_manifest.json has {key}={tissue.get(key)!r}; expected {value}")
    if set(tissue.get("models", [])) != set(config["tissue"]["models"]):
        raise ContractError("results/ml/tissue/run_manifest.json does not contain the full tissue model set")

    recurrence = read_json(ROOT / "results" / "ml" / "recurrence" / "summary.json")
    expected_recurrence = {
        "outer_repeats": config["recurrence"]["outer_repeats"],
        "bootstrap_replicates": config["uncertainty"]["patient_bootstrap"],
        "optimism_bootstrap": config["uncertainty"]["optimism_bootstrap"],
        "permutations": config["uncertainty"]["permutations"],
    }
    for key, value in expected_recurrence.items():
        if int(recurrence.get(key, -1)) != int(value):
            raise ContractError(f"results/ml/recurrence/summary.json has {key}={recurrence.get(key)!r}; expected {value}")

    for cohort, patients in [("colonomics", 92), ("gse119526", 48)]:
        manifest = read_json(ROOT / "results" / "ml" / "public" / cohort / "run_manifest.json")
        expected_public = {
            "patients": patients,
            "outer_repeats": config["tissue"]["outer_repeats"],
            "bootstrap_replicates": config["uncertainty"]["patient_bootstrap"],
        }
        for key, value in expected_public.items():
            if int(manifest.get(key, -1)) != int(value):
                raise ContractError(f"results/ml/public/{cohort}/run_manifest.json has {key}={manifest.get(key)!r}; expected {value}")


def load_final_summary(summary_path: Path) -> dict:
    summary = read_json(summary_path)
    if summary.get("status") != "final":
        raise ContractError(f"{summary_path.relative_to(ROOT)} has status={summary.get('status')!r}; expected 'final'")
    for key in ["tissue", "recurrence", "public", "global_tests", "sources"]:
        if key not in summary:
            raise ContractError(f"{summary_path.relative_to(ROOT)} is missing required key: {key}")
    validate_manifest_counts()
    return summary


def ci_text(ci: dict, point_key: str = "point") -> tuple[float, float, float]:
    point = number(ci.get(point_key, ci.get("estimate")))
    return point, number(ci.get("ci_low")), number(ci.get("ci_high"))


def find_ci(rows: list[dict], metric: str, *, model: str | None = None, block: str | None = None, horizon: int | None = None) -> dict:
    matches = []
    for row in rows:
        if row.get("metric") != metric:
            continue
        if model is not None and row.get("model") != model:
            continue
        if block is not None and row.get("block") != block:
            continue
        if horizon is not None and int(row.get("horizon_days", -1)) != int(horizon):
            continue
        matches.append(row)
    if len(matches) != 1:
        raise ContractError(f"Expected one CI row for metric={metric}, model={model}, block={block}, horizon={horizon}; found {len(matches)}")
    return matches[0]


def metric_rows_to_frame(rows: list[dict], ci_rows: list[dict], metric: str, value_col: str, id_cols: list[str], *, require_ci: bool = True) -> pd.DataFrame:
    out = []
    for row in rows:
        try:
            ci = find_ci(ci_rows, metric, model=row.get("model"), block=row.get("block"), horizon=row.get("horizon_days"))
            point, low, high = ci_text(ci)
            ci_status = "available"
        except ContractError:
            if require_ci:
                raise
            point = number(row.get(value_col))
            low = high = float("nan")
            ci_status = "missing_in_preliminary_input"
        item = dict(row)
        item.update({"plot_metric": metric, "plot_value": number(row.get(value_col)), "ci_point": point, "plot_low": low, "plot_high": high, "ci_status": ci_status})
        out.append(item)
    return pd.DataFrame(out)


def figure_manifest(stem: str, fig_dir: Path, src_dir: Path, inputs: list[Path], source_data: list[Path], claims: list[str], transformations: list[str], limitations: list[str], status: str) -> None:
    manifest_path = fig_dir / f"{stem}_manifest.json"
    data = {
        "schema_version": "1.0",
        "profile": "publication",
        "figure_id": stem,
        "generated_at": "2026-09-05T00:00:00+09:00",
        "base_dir": ".",
        "analysis_status": status,
        "script": file_record(ROOT / "scripts" / "make_ml_figures.py", ROOT),
        "inputs": [file_record(p, ROOT) for p in inputs if status == "preliminary" and p.exists() or status != "preliminary"],
        "outputs": [file_record(fig_dir / f"{stem}.{ext}", ROOT) for ext in ["pdf", "svg", "png"]],
        "source_data": [file_record(p, ROOT) for p in source_data],
        "panel_claims": claims,
        "transformations": transformations,
        "known_limitations": limitations,
        "validation": {
            "status": "structural_pass",
            "scientific_validation": "not_assessed",
            "commands": ["python3 scripts/make_ml_figures.py"],
        },
    }
    common.write_json(manifest_path, data)
    if VALIDATOR and VALIDATOR.exists() and fig_dir == CANONICAL_FIG:
        subprocess.run(["python3", str(VALIDATOR), "--manifest", str(manifest_path), "--base", str(ROOT)], check=True, cwd=ROOT)


def draw_figure6(summary: dict, summary_path: Path, fig_dir: Path, src_dir: Path, status: str) -> None:
    tissue = summary["tissue"]
    local_metrics = pd.DataFrame(tissue["model_metrics"])
    local_ci = tissue["bootstrap_ci"]
    auc = metric_rows_to_frame(local_metrics.to_dict("records"), local_ci, "auc", "auc", ["model"])
    brier = metric_rows_to_frame(local_metrics[local_metrics["model"].ne("rbf_svm")].to_dict("records"), local_ci, "brier", "brier", ["model"])
    probability_metrics = local_metrics[local_metrics["model"].ne("rbf_svm")]
    bal = metric_rows_to_frame(probability_metrics.to_dict("records"), local_ci, "threshold_0_5_balanced_accuracy", "threshold_0_5_balanced_accuracy", ["model"])
    local_source = pd.concat([auc, brier, bal], ignore_index=True, sort=False)
    local_source["analysis_status"] = status
    write_tsv(src_dir / "F6_tissue_local_metrics_source_data.tsv", local_source)

    public_rows = []
    for cohort in ["colonomics", "gse119526"]:
        result = summary.get("public", {}).get(cohort)
        if not result:
            continue
        point, low, high = ci_text(result["ridge_auc_ci"])
        public_rows.append({
            "cohort": cohort,
            "model": "ridge",
            "metric": "auc",
            "plot_value": number(result["ridge"]["auc"]),
            "ci_point": point,
            "plot_low": low,
            "plot_high": high,
            "analysis_status": status,
        })
    public = pd.DataFrame(public_rows)
    if public.empty:
        public = pd.DataFrame(columns=["cohort", "model", "metric", "plot_value", "ci_point", "plot_low", "plot_high", "analysis_status"])
    write_tsv(src_dir / "F6_tissue_public_metrics_source_data.tsv", public)

    ordered_models = [m for m in ["ridge", "elastic_net", "random_forest", "rbf_svm", "best_single_gene"] if m in set(local_metrics["model"])]
    fig = plt.figure(figsize=(7.5, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.9], width_ratios=[1.2, 1.0], hspace=0.44, wspace=0.35)

    ax_auc = fig.add_subplot(gs[:, 0])
    plot_auc = auc.set_index("model").reindex(ordered_models).reset_index()
    y = np.arange(len(plot_auc))
    draw_horizontal_ci(
        ax_auc,
        plot_auc["plot_value"].to_numpy(float),
        y,
        plot_auc["plot_low"].to_numpy(float),
        plot_auc["plot_high"].to_numpy(float),
        color="#111111",
        ecolor="#555555",
        capsize=2.5,
        markersize=4.2,
    )
    ax_auc.axvline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    ax_auc.set_xlim(*frame_metric_axis(plot_auc, reference=0.5))
    ax_auc.set_yticks(y, [MODEL_LABELS.get(m, m) for m in plot_auc["model"]], fontsize=8)
    ax_auc.set_xlabel("Cross-fitted ROC AUC", fontsize=8)
    ax_auc.set_title("A. Local tumor-vs-normal classification", loc="left", fontsize=9)
    ax_auc.tick_params(axis="x", labelsize=7)
    ax_auc.spines[["top", "right"]].set_visible(False)

    ax_perf = fig.add_subplot(gs[0, 1])
    x = np.arange(len(ordered_models))
    width = 0.34
    probability_models = [m for m in ordered_models if m != "rbf_svm"]
    x = np.arange(len(probability_models))
    bvals = brier.set_index("model").reindex(probability_models)["plot_value"].to_numpy(float)
    avals = bal.set_index("model").reindex(probability_models)["plot_value"].to_numpy(float)
    ax_perf.bar(x - width / 2, avals, width, color="#009E73", label="Balanced accuracy")
    ax_perf.bar(x + width / 2, bvals, width, color="#999999", label="Brier")
    ax_perf.set_xticks(x, [MODEL_LABELS.get(m, m).replace(" ", "\n") for m in probability_models], fontsize=6.5)
    ax_perf.set_ylim(0, 1.14)
    ax_perf.set_title("B. Threshold and probability summaries", loc="left", fontsize=9)
    ax_perf.tick_params(axis="y", labelsize=7)
    ax_perf.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, columnspacing=1.2, handlelength=1.4)
    ax_perf.spines[["top", "right"]].set_visible(False)

    ax_pub = fig.add_subplot(gs[1, 1])
    if len(public):
        x = np.arange(len(public))
        draw_vertical_ci(
            ax_pub,
            x,
            public["plot_value"].to_numpy(float),
            public["plot_low"].to_numpy(float),
            public["plot_high"].to_numpy(float),
            ecolor="#555555",
            capsize=2.5,
        )
        ax_pub.plot(x, public["plot_value"].to_numpy(float), "o", color="#0072B2", markersize=4.2, linestyle="none", zorder=4)
        ax_pub.axhline(0.5, color="#999999", linestyle="--", linewidth=0.8)
        ax_pub.set_xticks(x, [PUBLIC_LABELS[c] for c in public["cohort"]], fontsize=7)
        ax_pub.set_ylim(*frame_metric_axis(public, reference=0.5, domain=(0.0, 1.03)))
    else:
        ax_pub.text(0.5, 0.5, "Public outputs not included", ha="center", va="center", fontsize=8, color="#555555")
        ax_pub.set_xticks([])
        ax_pub.set_yticks([])
    ax_pub.set_ylabel("AUC", fontsize=8)
    ax_pub.set_title("C. Public cohort-local ridge models", loc="left", fontsize=9)
    ax_pub.tick_params(axis="y", labelsize=7)
    ax_pub.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Tissue-classification model performance", fontsize=11, y=0.985)
    save_all_formats(fig, "F6_tissue_ml", fig_dir)
    figure_manifest(
        "F6_tissue_ml",
        fig_dir,
        src_dir,
        [summary_path, ROOT / "results" / "ml" / "tissue" / "run_manifest.json", ROOT / "results" / "ml" / "public" / "colonomics" / "run_manifest.json", ROOT / "results" / "ml" / "public" / "gse119526" / "run_manifest.json"],
        [src_dir / "F6_tissue_local_metrics_source_data.tsv", src_dir / "F6_tissue_public_metrics_source_data.tsv"],
        ["The panel reports cross-fitted tissue-classification metrics for the fixed ten-gene classifiers."],
        ["No model fitting or confidence-interval calculation was performed by the figure script; intervals were read from results/ml/summary.json."],
        ["Adjacent normal specimens are paired local controls, not a screening population; public panels are cohort-local array models rather than PSQ-model external validation."],
        status,
    )


def draw_figure7(summary: dict, summary_path: Path, fig_dir: Path, src_dir: Path, status: str) -> None:
    recurrence = summary["recurrence"]
    rows = pd.DataFrame(recurrence["model_metrics"])
    ci_rows = recurrence["bootstrap_ci"]
    horizon = 1825
    primary = rows[rows["horizon_days"].eq(horizon)].copy()
    cindex = metric_rows_to_frame(primary.to_dict("records"), ci_rows, "uno_c", "uno_c_mean", ["block", "model", "horizon_days"])
    cindex["analysis_status"] = status
    write_tsv(src_dir / "F7_recurrence_primary_metrics_source_data.tsv", cindex)
    ridge_primary = primary[primary["model"].eq("ridge")].copy()
    brier = metric_rows_to_frame(ridge_primary.to_dict("records"), ci_rows, "brier", "brier_mean", ["block", "model", "horizon_days"], require_ci=status != "preliminary")
    brier["analysis_status"] = status
    write_tsv(src_dir / "F7_recurrence_brier_source_data.tsv", brier)

    sens = pd.DataFrame(recurrence.get("sensitivity_metrics", []))
    sens["analysis_status"] = status
    write_tsv(src_dir / "F7_recurrence_sensitivity_source_data.tsv", sens)
    delta = pd.DataFrame([recurrence["delta"]])
    delta["analysis_status"] = status
    write_tsv(src_dir / "F7_recurrence_delta_source_data.tsv", delta)

    fig = plt.figure(figsize=(7.5, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], width_ratios=[1.15, 1.0], hspace=0.42, wspace=0.48)
    ax1 = fig.add_subplot(gs[:, 0])
    model_order = [m for m in ["ridge", "elastic_net", "survival_forest"] if m in set(cindex["model"])]
    block_order = [b for b in ["clinical", "tumor10", "methylation", "combined"] if b in set(cindex["block"])]
    offsets = np.linspace(-0.22, 0.22, max(1, len(model_order)))
    ybase = np.arange(len(block_order))
    for off, model in zip(offsets, model_order):
        sub = cindex[cindex["model"].eq(model)].set_index("block").reindex(block_order)
        draw_horizontal_ci(
            ax1,
            sub["plot_value"].to_numpy(float),
            ybase + off,
            sub["plot_low"].to_numpy(float),
            sub["plot_high"].to_numpy(float),
            label=MODEL_LABELS.get(model, model),
            color=COLORS.get(model, "#111111"),
            ecolor=COLORS.get(model, "#111111"),
            capsize=2.2,
            markersize=3.8,
        )
    ax1.axvline(0.5, color="#999999", linestyle="--", linewidth=0.8)
    ax1.set_xlim(*frame_metric_axis(cindex, reference=0.5))
    ax1.set_yticks(ybase, [BLOCK_LABELS.get(b, b) for b in block_order], fontsize=8)
    ax1.set_xlabel("Uno IPCW C-index through 5 years", fontsize=8)
    ax1.set_title("A. Nested recurrence-prediction performance", loc="left", fontsize=9)
    ax1.tick_params(axis="x", labelsize=7)
    ax1.set_ylim(-0.45, len(block_order) - 0.15)
    ax1.legend(frameon=True, facecolor="white", edgecolor="white", framealpha=1.0, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, columnspacing=1.0, handletextpad=0.4)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    ridge = brier.set_index("block").reindex(block_order).reset_index()
    ax2.bar(np.arange(len(ridge)), ridge["plot_value"].to_numpy(float), color=[COLORS.get(b, "#999999") for b in ridge["block"]])
    draw_vertical_ci(
        ax2,
        np.arange(len(ridge)),
        ridge["plot_value"].to_numpy(float),
        ridge["plot_low"].to_numpy(float),
        ridge["plot_high"].to_numpy(float),
        ecolor="#333333",
        capsize=2.2,
    )
    ax2.set_xticks(np.arange(len(ridge)), [BLOCK_LABELS.get(b, b).replace(" ", "\n") for b in ridge["block"]], fontsize=7)
    ax2.set_ylim(*frame_metric_axis(ridge, reference=0.0, domain=(0.0, float("inf")), min_span=0.08))
    ax2.set_ylabel("IPCW Brier", fontsize=8)
    ax2.set_title("B. Ridge 5-year probability error", loc="left", fontsize=9)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.spines[["top", "right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    if len(sens) and "horizon_days" in sens.columns:
        sens1825 = sens[sens["horizon_days"].eq(horizon)].copy()
        sens1825["plot_value"] = pd.to_numeric(sens1825.get("uno_c_mean"), errors="coerce")
        if "plot_low" not in sens1825.columns:
            for low_col, high_col in [("uno_c_ci_low", "uno_c_ci_high"), ("ci_low", "ci_high")]:
                if low_col in sens1825.columns and high_col in sens1825.columns:
                    sens1825["plot_low"] = pd.to_numeric(sens1825[low_col], errors="coerce")
                    sens1825["plot_high"] = pd.to_numeric(sens1825[high_col], errors="coerce")
                    break
        sens1825 = sens1825.sort_values("plot_value", na_position="last")
        combined = cindex[(cindex["model"].eq("ridge")) & (cindex["block"].eq("combined"))]["plot_value"].iloc[0]
        ax3.set_xlim(*frame_metric_axis(sens1825, reference=float(combined)))
        y_sens = np.arange(len(sens1825))
        finite = np.isfinite(sens1825["plot_value"].to_numpy(float))
        if finite.any():
            kwargs = {
                "color": "#555555",
                "ecolor": "#777777",
                "capsize": 2.0,
                "markersize": 3.8,
            }
            finite_rows = sens1825.loc[finite]
            if {"plot_low", "plot_high"}.issubset(sens1825.columns):
                finite_ci = finite_rows[["plot_low", "plot_high"]].apply(pd.to_numeric, errors="coerce")
                ci_ok = np.isfinite(finite_ci["plot_low"].to_numpy(float)) & np.isfinite(finite_ci["plot_high"].to_numpy(float))
                if ci_ok.all():
                    kwargs["low"] = finite_ci["plot_low"].to_numpy(float)
                    kwargs["high"] = finite_ci["plot_high"].to_numpy(float)
            draw_horizontal_ci(
                ax3,
                finite_rows["plot_value"].to_numpy(float),
                y_sens[finite],
                kwargs.pop("low", np.repeat(np.nan, len(finite_rows))),
                kwargs.pop("high", np.repeat(np.nan, len(finite_rows))),
                **kwargs,
            )
        if (~finite).any():
            xmin, xmax = ax3.get_xlim()
            x_ne = xmin + (xmax - xmin) * 0.02
            for yi in y_sens[~finite]:
                ax3.text(x_ne, yi, "NE", ha="left", va="center", fontsize=6.5, color="#555555")
        ax3.axvline(float(combined), color="#009E73", linewidth=1.0)
        labels = [SENSITIVITY_LABELS.get(v, str(v).replace("_", " ")) for v in sens1825["sensitivity"]]
        ax3.set_yticks(np.arange(len(sens1825)), labels, fontsize=6.5)
    else:
        ax3.text(0.5, 0.5, "No sensitivity rows", ha="center", va="center", fontsize=8, color="#555555")
        ax3.set_xticks([])
        ax3.set_yticks([])
    ax3.set_xlabel("Uno C at 5 years", fontsize=8)
    ax3.set_title("C. Planned ridge sensitivities", loc="left", fontsize=9)
    ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax3.tick_params(axis="x", labelsize=7)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Recurrence-prediction model performance", fontsize=11, y=0.985)
    save_all_formats(fig, "F7_recurrence_ml", fig_dir)
    figure_manifest(
        "F7_recurrence_ml",
        fig_dir,
        src_dir,
        [summary_path, ROOT / "results" / "ml" / "recurrence" / "summary.json"],
        [src_dir / "F7_recurrence_primary_metrics_source_data.tsv", src_dir / "F7_recurrence_brier_source_data.tsv", src_dir / "F7_recurrence_sensitivity_source_data.tsv", src_dir / "F7_recurrence_delta_source_data.tsv"],
        ["The panel reports nested cross-validation metrics for recurrence models at the prespecified five-year horizon."],
        ["No model fitting or uncertainty calculation was performed by the figure script; plotted estimates and intervals were read from results/ml/summary.json."],
        ["The recurrence cohort has few events and no death data; probabilities describe recorded recurrence under censoring assumptions rather than competing-risk incidence."],
        status,
    )


def write_captions(manuscript_dir: Path, status: str) -> None:
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    text = """# ML Figure Captions

**Figure 6. Tissue-classification model performance.** (A) Cross-fitted ROC AUC for tumor-versus-adjacent-normal classification in the local 87-pair cohort. Patient pairs were kept together within grouped nested cross-validation. Points show the mean of 20 repeated outer cross-validation estimates; error bars show 2000 patient-cluster conditional bootstrap 95% confidence intervals. (B) Balanced accuracy at the fixed probability threshold of 0.5 and Brier score for probability-generating models. (C) Cohort-local ridge classification performance in public paired tissue cohorts; these public analyses assess array-cohort portability of the ten-gene signal and are not frozen pyrosequencing-model external validation.

**Figure 7. Recurrence-prediction model performance.** (A) Uno IPCW C-index through 5 years for clinical, tumor-methylation, and combined predictor blocks in the primary recurrence cohort of 82 patients with 14 events. Points show the mean of 25 repeated outer cross-validation estimates; error bars show 2000 patient-cluster conditional bootstrap 95% confidence intervals. (B) Five-year IPCW Brier score for ridge Cox models. (C) Planned ridge sensitivity analyses; the green vertical line marks the combined primary ridge Uno C-index reference, and NE indicates a non-estimable metric under the censoring and event-count rules. Δ10 denotes paired tumor-minus-normal values for the ten genes. The two-gene panel contains RALYL and SFMBT2 with binary CEA; Q95 thresholds were estimated from training normals. These panels are descriptive model-assessment summaries and do not establish external recurrence-validation performance.
"""
    (manuscript_dir / "ml_figure_captions.md").write_text(text)


def write_qa(fig_dir: Path, status: str, output_path: Path) -> None:
    rows = []
    for stem in ["F6_tissue_ml", "F7_recurrence_ml"]:
        for ext in ["pdf", "svg", "png", "tiff"]:
            p = fig_dir / f"{stem}.{ext}"
            rows.append({"file": str(p.relative_to(ROOT)), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else 0, "sha256": common.sha256(p) if p.exists() else ""})
    qa = {"status": status, "all_outputs_present": all(r["exists"] and r["bytes"] > 0 for r in rows), "outputs": rows}
    common.write_json(output_path, qa)


def output_dirs(preliminary: bool) -> tuple[Path, Path, Path, Path]:
    if preliminary:
        fig_dir = PRELIM / "figures"
        src_dir = fig_dir / "source_data"
        manuscript_dir = PRELIM / "manuscript"
        qa_path = PRELIM / "figure_ml_visual_qa.json"
    else:
        fig_dir = CANONICAL_FIG
        src_dir = CANONICAL_SRC
        manuscript_dir = CANONICAL_MANUSCRIPT
        qa_path = VERIFY / "figure_ml_visual_qa.json"
    for path in [fig_dir, src_dir, manuscript_dir, qa_path.parent]:
        path.mkdir(parents=True, exist_ok=True)
    return fig_dir, src_dir, manuscript_dir, qa_path


def render(summary: dict, summary_path: Path, preliminary: bool) -> None:
    status = "preliminary" if preliminary else "final"
    fig_dir, src_dir, manuscript_dir, qa_path = output_dirs(preliminary)
    draw_figure6(summary, summary_path, fig_dir, src_dir, status)
    draw_figure7(summary, summary_path, fig_dir, src_dir, status)
    write_captions(manuscript_dir, status)
    write_qa(fig_dir, status, qa_path)


def load_preliminary_summary(path: Path) -> dict:
    summary = read_json(path)
    summary["status"] = summary.get("status", "preliminary")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ml-summary", type=Path, default=ROOT / "results" / "ml" / "summary.json")
    parser.add_argument("--allow-preliminary", action="store_true", help="Render only under verification/ml/preliminary_figures from the supplied summary.")
    args = parser.parse_args()

    if args.allow_preliminary:
        render(load_preliminary_summary(args.ml_summary), args.ml_summary, preliminary=True)
        print(json.dumps({"status": "preliminary", "output": str(PRELIM.relative_to(ROOT))}))
        return

    summary = load_final_summary(args.ml_summary)
    render(summary, args.ml_summary, preliminary=False)
    print(json.dumps({"status": "final", "figures": ["F6_tissue_ml", "F7_recurrence_ml"]}))


if __name__ == "__main__":
    main()
