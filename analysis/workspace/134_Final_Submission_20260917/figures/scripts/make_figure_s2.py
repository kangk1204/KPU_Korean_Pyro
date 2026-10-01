#!/usr/bin/env python3
"""Figure S2 - panel-size and external-transfer summary.

This script is the 133-local rebuild of the former Figure_S1 layout from the
129 package. It writes Figure_S2.{pdf,svg,png}, a manifest, and a QA table. The
only intended scientific change is that panel A reads the corrected A1 output
tables from the 133 workspace.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


BASE = Path(__file__).resolve().parents[2]
WORKSPACE_CANDIDATES = [
    BASE / "source" / "Analysis_Code" / "workspace",
    BASE.parent,
]


def _first_existing(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate {label}: " + ", ".join(str(p) for p in candidates))


WORKSPACE = _first_existing(WORKSPACE_CANDIDATES, "analysis workspace")
FIGSTYLE_DIR = _first_existing(
    [
        WORKSPACE / "129_Submission_Revised_20260909" / "figures_redesign",
        BASE.parent / "129_Submission_Revised_20260909" / "figures_redesign",
    ],
    "129 figure style directory",
)
sys.path.insert(0, str(FIGSTYLE_DIR))
import figstyle as fs  # noqa: E402


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": fs.FS["body"],
        "svg.fonttype": "none",
        "svg.hashsalt": "crc-133-figure-s2",
        "pdf.fonttype": 42,
    }
)

WORKSPACE_RESULT_CANDIDATES = [
    WORKSPACE / "127_Exploratory_PanelSize_Coordination_20260909" / "A_panel_size" / "results",
    BASE.parent / "127_Exploratory_PanelSize_Coordination_20260909" / "A_panel_size" / "results",
]
WORKSPACE_RESULTS = next((path for path in WORKSPACE_RESULT_CANDIDATES if path.exists()), WORKSPACE_RESULT_CANDIDATES[0])
SOURCE_DATA = BASE / "figures" / "source_data"
RESULTS = SOURCE_DATA / "FigureS2_panel_size"
OUT = BASE / "figures"
QA_DIR = BASE / "figures" / "qa"

MANUSCRIPT_SINGLE = 0.943
MANUSCRIPT_FULL = 0.968
TRANSFER_REF = 0.90

TARGET_ORDER = ["CMCBSN", "SNUH", "ASAN", "GSE119526", "GSE193535", "GSE77718", "GSE42752"]
TRAIN_ORDER = ["Colonomics", "CMCBSN", "SNUH", "ASAN"]
COL_ORDER = ["Colonomics"] + TARGET_ORDER

TARGET_STYLE = {
    "CMCBSN": (fs.C["cohort"]["CMCBSN"], "solid"),
    "SNUH": (fs.C["cohort"]["SNUH"], "solid"),
    "ASAN": (fs.C["cohort"]["ASAN"], "solid"),
    "GSE119526": (fs.C["ink"], "solid"),
    "GSE193535": (fs.C["ink2"], (0, (3.5, 1.5))),
    "GSE77718": (fs.C["muted"], "solid"),
    "GSE42752": (fs.C["na_mark"], (0, (1.3, 1.3))),
}

SMALL = fs.FS["small"]
BODY = fs.FS["body"]
FIGURE_NAME = "Figure_S2"


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(BASE.resolve()))
    except ValueError:
        return path.name


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_paths(results_dir: Path) -> dict[str, Path]:
    return {
        "perk": results_dir / "a1_per_k_summary.tsv",
        "bytarget": results_dir / "a2_per_k_by_target.tsv",
        "frozen": results_dir / "a2_frozen_transfer.tsv",
        "loco": results_dir / "a2_loco_matrix.tsv",
        "locolong": results_dir / "a2_leave_one_cohort_out.tsv",
        "coverage": results_dir / "a2_cohort_coverage.tsv",
    }


def prepare_source_data(workspace_results: Path = WORKSPACE_RESULTS, source_dir: Path = RESULTS) -> list[dict[str, str]]:
    """Copy the six aggregate tables needed for Figure S2 into figures/source_data."""
    source_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in _source_paths(workspace_results).values():
        target = source_dir / path.name
        target.write_bytes(path.read_bytes())
        copied.append({"source": _rel(path), "target": _rel(target), "sha256": sha256(target)})
    (source_dir / "manifest.json").write_text(json.dumps({"files": copied}, indent=2) + "\n")
    return copied


def build(results_dir: Path = RESULTS):
    """Draw the figure and return source paths plus drawn values for QA."""
    src = _source_paths(results_dir)
    d = pd.read_csv(src["perk"], sep="\t")
    per_k = pd.read_csv(src["bytarget"], sep="\t")
    frozen = pd.read_csv(src["frozen"], sep="\t")
    frozen = frozen.loc[frozen["model"] == "full_10gene"].set_index("target")
    loco = pd.read_csv(src["loco"], sep="\t").set_index("train")

    fig = fs.figure(180, 150)

    ax = fs.ax_mm(fig, 15, 11, 67, 54)
    k = d["k"].to_numpy()
    band = ax.fill_between(
        k,
        d["min_auc"],
        d["max_auc"],
        color=fs.C["local"],
        alpha=0.16,
        linewidth=0,
        zorder=1,
    )
    (line,) = ax.plot(k, d["median_auc"], color=fs.C["local"], lw=1.0, zorder=3)
    bars = ax.errorbar(
        k,
        d["nested_mean_auc"],
        yerr=np.vstack([d["nested_mean_auc"] - d["ci_lo"], d["ci_hi"] - d["nested_mean_auc"]]),
        fmt="o",
        ms=2.8,
        mfc="white",
        mec=fs.C["ink"],
        ecolor=fs.C["ink2"],
        elinewidth=0.6,
        capsize=1.3,
        capthick=0.6,
        lw=0,
        zorder=4,
    )
    ref_full = ax.axhline(MANUSCRIPT_FULL, color=fs.C["rule"], lw=0.6, ls=(0, (4, 2)), zorder=2)
    ref_single = ax.axhline(MANUSCRIPT_SINGLE, color=fs.C["rule"], lw=0.6, ls=(0, (1.4, 1.6)), zorder=2)
    ax.set_xlim(0.55, 10.45)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.80, 1.002)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Genes in panel ($k$)", fontsize=BODY)
    ax.set_ylabel("Tissue-discrimination AUC", fontsize=BODY)
    fs.despine(ax)
    fs.light_grid(ax, axis="y")
    leg = ax.legend(
        [(band, line), bars, ref_full, ref_single],
        [
            "median across C(10,$k$) subsets (min-max band)",
            "nested best-$k$ (95% patient-cluster CI)",
            "manuscript full panel (0.968)",
            "manuscript single gene (0.943)",
        ],
        loc="lower right",
        fontsize=SMALL,
        borderpad=0.2,
        labelspacing=0.35,
        handlelength=1.7,
        borderaxespad=0.25,
    )
    leg.set_zorder(6)
    fs.panel_label(fig, 2.5, 3.5, "A")

    axb = fs.ax_mm(fig, 104, 11, 67, 54)
    handles, labels, b_markers = [], [], {}
    for i, target in enumerate(TARGET_ORDER):
        color, linestyle = TARGET_STYLE[target]
        lw = 1.05 if target in ("GSE42752", "GSE77718") else 0.9
        sub = per_k.loc[per_k["target"] == target].sort_values("k")
        (ln,) = axb.plot(sub["k"], sub["median_auc"], color=color, lw=lw, ls=linestyle, zorder=3)
        handles.append(ln)
        labels.append(target)
        row = frozen.loc[target]
        xm = 10.34 + 0.10 * i
        axb.errorbar(
            xm,
            row["auc"],
            yerr=[[row["auc"] - row["ci_lo"]], [row["ci_hi"] - row["auc"]]],
            fmt="D",
            ms=2.4,
            mfc="white",
            mec=color,
            ecolor=color,
            elinewidth=0.6,
            capsize=1.1,
            capthick=0.5,
            lw=0,
            zorder=4,
        )
        b_markers[target] = {"x": xm, "auc": float(row["auc"]), "ci": [float(row["ci_lo"]), float(row["ci_hi"])]}
    axb.axhline(TRANSFER_REF, color=fs.C["rule"], lw=0.5, ls=(0, (4, 2)), zorder=2)
    axb.text(0.70, TRANSFER_REF - 0.004, "0.90", fontsize=SMALL, color=fs.C["muted"], va="top", ha="left")
    axb.set_xlim(0.55, 11.15)
    axb.set_xticks(range(1, 11))
    axb.set_ylim(0.75, 1.003)
    axb.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    axb.set_xlabel("Genes in panel ($k$)", fontsize=BODY)
    axb.set_ylabel("External AUC (models fitted in Colonomics)", fontsize=BODY)
    fs.despine(axb)
    fs.light_grid(axb, axis="y")
    legb = axb.legend(handles, labels, loc="lower right", ncol=1, fontsize=SMALL, borderpad=0.2, labelspacing=0.28, handlelength=1.7, borderaxespad=0.25)
    legb.set_zorder(6)
    axb.text(
        0.012,
        0.015,
        "lines: median across C(10,$k$) subsets\n"
        "diamonds: 10-gene models at $k$=10 (95% CI)\n"
        "dashed: transfer reference 0.90",
        transform=axb.transAxes,
        fontsize=SMALL,
        color=fs.C["ink2"],
        va="bottom",
        ha="left",
        linespacing=1.35,
    )
    fs.panel_label(fig, 91, 3.5, "B")

    mat = np.full((len(TRAIN_ORDER), len(COL_ORDER)), np.nan)
    for i, train in enumerate(TRAIN_ORDER):
        for j, target in enumerate(COL_ORDER):
            value = loco.loc[train, target]
            mat[i, j] = float(value) if pd.notna(value) else np.nan

    c_text = []
    axh = fs.ax_mm(fig, 27, 86, 117, 32)
    norm = plt.Normalize(vmin=0.85, vmax=1.00)
    axh.imshow(np.ma.masked_invalid(mat), cmap=fs.CMAP_SEQ, norm=norm, aspect="auto")
    for i in range(len(TRAIN_ORDER)):
        for j in range(len(COL_ORDER)):
            if np.isnan(mat[i, j]):
                fs.na_cell(axh, j, i)
                axh.texts[-1].set_fontsize(SMALL)
                c_text.append((i, j, axh.texts[-1].get_text()))
            else:
                text = axh.text(
                    j,
                    i,
                    f"{mat[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=SMALL,
                    color="white" if mat[i, j] > 0.955 else fs.C["ink"],
                )
                c_text.append((i, j, text.get_text()))
    axh.set_xticks(range(len(COL_ORDER)))
    axh.set_xticklabels(COL_ORDER, rotation=45, ha="right", rotation_mode="anchor", fontsize=SMALL)
    axh.set_yticks(range(len(TRAIN_ORDER)))
    axh.set_yticklabels(TRAIN_ORDER, fontsize=SMALL)
    axh.set_ylabel("Training cohort", fontsize=BODY, labelpad=2)
    axh.set_xlabel("Target cohort", fontsize=BODY, labelpad=3)
    axh.xaxis.set_label_position("top")
    axh.tick_params(length=0, pad=1.5)
    for spine in ("top", "right", "left", "bottom"):
        axh.spines[spine].set_visible(False)
    fs.colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=fs.CMAP_SEQ), 27, 138, 52, 2.6, "Specimen-level AUC", ticks=[0.85, 0.90, 0.95, 1.00])
    fs.panel_label(fig, 2.5, 78, "C")
    width_in, height_in = fig.get_size_inches()
    fig.text(
        92 * fs.MM / width_in,
        1 - 137 * fs.MM / height_in,
        "Cross-cohort evaluation: models were fitted in each training cohort using the\n"
        "candidate CpGs measured in the target cohort, then applied without refitting.\n"
        "Grey × = self cell (not estimated).",
        fontsize=SMALL,
        color=fs.C["ink2"],
        va="top",
        ha="left",
        linespacing=1.35,
    )

    registry = {
        "a_line": {"x": list(line.get_xdata()), "y": list(line.get_ydata())},
        "a_band_lo": list(d["min_auc"]),
        "a_band_hi": list(d["max_auc"]),
        "a_points": {"x": list(bars[0].get_xdata()), "y": list(bars[0].get_ydata())},
        "a_ci": [[float(v) for v in seg[:, 1]] for seg in bars[2][0].get_segments()],
        "a_ref": [MANUSCRIPT_SINGLE, MANUSCRIPT_FULL],
        "a_ylim": list(ax.get_ylim()),
        "b_lines": {target: list(handle.get_ydata()) for target, handle in zip(labels, handles)},
        "b_lines_x": {target: list(handle.get_xdata()) for target, handle in zip(labels, handles)},
        "b_markers": b_markers,
        "b_ref": TRANSFER_REF,
        "b_ylim": list(axb.get_ylim()),
        "c_matrix": mat.tolist(),
        "c_rows": TRAIN_ORDER,
        "c_cols": COL_ORDER,
        "c_text": {f"{i},{j}": text for (i, j, text) in c_text},
        "c_norm": [norm.vmin, norm.vmax],
    }
    sources = [src["perk"], src["bytarget"], src["frozen"], src["loco"], src["locolong"], src["coverage"]]
    return fig, sources, registry


def save(fig, name: str, out_dir: Path, sources: list[Path], notes: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(
        out_dir / f"{name}.pdf",
        dpi=600,
        pad_inches=0,
        metadata={"Title": name, "CreationDate": None, "ModDate": None},
    )
    fig.savefig(out_dir / f"{name}.svg", dpi=600, pad_inches=0, metadata={"Title": name, "Date": None})
    fig.savefig(out_dir / f"{name}.png", dpi=600, pad_inches=0, metadata=None)
    width_in, height_in = fig.get_size_inches()
    manifest = {
        "figure": name,
        "width_mm": round(width_in / fs.MM, 1),
        "height_mm": round(height_in / fs.MM, 1),
        "generated_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "style_module": _rel(Path(fs.__file__)),
        "style_sha256": sha256(Path(fs.__file__)),
        "plotting_script": _rel(Path(__file__)),
        "plotting_script_sha256": sha256(Path(__file__)),
        "sources": [{"path": _rel(path), "sha256": sha256(path)} for path in sources],
        "notes": notes,
        "outputs": {ext: sha256(out_dir / f"{name}.{ext}") for ext in ("pdf", "svg", "png")},
    }
    (out_dir / f"{name}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plt.close(fig)
    return manifest


def _read_tsv(results_dir: Path, name: str) -> list[dict[str, str]]:
    with open(results_dir / name, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    return str(value)


def write_qa(results_dir: Path, out_dir: Path, registry: dict, manifest: dict) -> dict:
    checks: list[tuple[str, str, object, object, bool]] = []

    def chk(panel: str, what: str, tsv, drawn, tol: float = 1e-9) -> None:
        ok = abs(tsv - drawn) <= tol if isinstance(tsv, float) and isinstance(drawn, float) else tsv == drawn
        checks.append((panel, what, tsv, drawn, ok))

    perk = {int(row["k"]): row for row in _read_tsv(results_dir, "a1_per_k_summary.tsv")}
    bytarget = _read_tsv(results_dir, "a2_per_k_by_target.tsv")
    frozen = {row["target"]: row for row in _read_tsv(results_dir, "a2_frozen_transfer.tsv") if row["model"] == "full_10gene"}
    loco = {row["train"]: row for row in _read_tsv(results_dir, "a2_loco_matrix.tsv")}
    coverage = {row["cohort"]: row for row in _read_tsv(results_dir, "a2_cohort_coverage.tsv")}

    ay = dict(zip(registry["a_line"]["x"], registry["a_line"]["y"]))
    py = dict(zip(registry["a_points"]["x"], registry["a_points"]["y"]))
    ci = {int(round(x)): seg for x, seg in zip(registry["a_points"]["x"], registry["a_ci"])}
    for k in range(1, 11):
        chk("A", f"median AUC across C(10,{k}) subsets", float(perk[k]["median_auc"]), float(ay[k]))
        chk("A", f"min of k={k} min-max band", float(perk[k]["min_auc"]), float(registry["a_band_lo"][k - 1]))
        chk("A", f"max of k={k} min-max band", float(perk[k]["max_auc"]), float(registry["a_band_hi"][k - 1]))
        chk("A", f"nested best-k point, k={k}", float(perk[k]["nested_mean_auc"]), float(py[k]))
        chk("A", f"nested 95% CI low, k={k}", float(perk[k]["ci_lo"]), float(min(ci[k])))
        chk("A", f"nested 95% CI high, k={k}", float(perk[k]["ci_hi"]), float(max(ci[k])))
    chk("A", "n subsets drawn", 10, len(registry["a_line"]["x"]))
    chk("A", "reference lines", [0.943, 0.968], registry["a_ref"])

    chk("B", "target cohorts drawn", 7, len(registry["b_lines"]))
    for target in TARGET_ORDER:
        rows = {int(row["k"]): row for row in bytarget if row["target"] == target}
        yy = dict(zip(registry["b_lines_x"][target], registry["b_lines"][target]))
        for k in range(1, 11):
            chk("B", f"{target} median external AUC, k={k}", float(rows[k]["median_auc"]), float(yy[k]))
        marker = registry["b_markers"][target]
        chk("B", f"{target} full-panel model AUC", float(frozen[target]["auc"]), marker["auc"])
        chk("B", f"{target} frozen 95% CI", [float(frozen[target]["ci_lo"]), float(frozen[target]["ci_hi"])], marker["ci"])
        chk("B", f"{target} frozen AUC equals k=10 subset AUC", float(rows[10]["median_auc"]), float(frozen[target]["auc"]))
        chk("B", f"{target} n pairs", int(coverage[target]["n_pairs"]), int(frozen[target]["n_pairs"]))
        chk("B", f"{target} n CpGs", int(coverage[target]["n_cpgs_available"]), int(frozen[target]["n_cpgs"]))
    chk("B", "reference line", 0.90, registry["b_ref"])

    for i, train in enumerate(registry["c_rows"]):
        for j, target in enumerate(registry["c_cols"]):
            raw = loco[train][target]
            text = registry["c_text"][f"{i},{j}"]
            if raw.strip() == "":
                chk("C", f"self cell {train}->{target} marked ×", "×", text)
            else:
                chk("C", f"unrounded cell {train}->{target}", float(raw), float(registry["c_matrix"][i][j]))
                chk("C", f"cell {train}->{target}", f"{float(raw):.3f}", text)
    chk("C", "colour scale limits", [0.85, 1.00], registry["c_norm"])
    chk("C", "matrix shape", [4, 8], [len(registry["c_rows"]), len(registry["c_cols"])])

    n_ok = sum(1 for check in checks if check[4])
    lines = [
        "# Figure S2 QA record",
        "",
        f"Generated: {_dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC  ",
        f"Checks passed: **{n_ok} / {len(checks)}**",
        "",
        "Every value below was re-read from the source TSV and compared with the value held by the matplotlib artist.",
        "",
        "| Panel | Quantity | Source TSV | Drawn | Pass |",
        "|---|---|---|---|---|",
    ]
    for panel, what, tsv, drawn, ok in checks:
        lines.append(f"| {panel} | {what} | {_format_value(tsv)} | {_format_value(drawn)} | {'PASS' if ok else 'FAIL'} |")
    lines += [
        "",
        "## Source Files",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    for source in manifest["sources"]:
        lines.append(f"| `{Path(source['path']).name}` | `{source['sha256']}` |")
    lines += [
        "",
        "## Rendering Audit",
        "",
        "- Figure size is 180.0 x 150.0 mm.",
        "- Text sizes are inherited from the manuscript figure style: 6 pt annotations, 7 pt body text, and 8 pt panel letters.",
        "- Panel A reads corrected A1 local panel-size results from the 133 workspace.",
        "- Panels B and C read the unchanged external-transfer and cross-cohort result tables.",
        "- All quantitative source tables are aggregate files; no patient-level records are used.",
    ]
    QA_DIR.mkdir(parents=True, exist_ok=True)
    qa_path = QA_DIR / "Figure_S2_qa.md"
    qa_path.write_text("\n".join(lines) + "\n")
    status = {"checks_passed": n_ok, "checks_total": len(checks), "qa_path": _rel(qa_path)}
    (QA_DIR / "Figure_S2_validation.json").write_text(json.dumps(status, indent=2) + "\n")
    if n_ok != len(checks):
        failed = [check for check in checks if not check[4]]
        raise RuntimeError(f"Figure S2 QA failed: {failed[:3]}")
    return status


def render(results_dir: Path = RESULTS, out_dir: Path = OUT, qa: bool = True) -> dict:
    if not results_dir.exists():
        prepare_source_data(WORKSPACE_RESULTS, results_dir)
    fig, sources, registry = build(results_dir)
    manifest = save(
        fig,
        FIGURE_NAME,
        out_dir,
        sources,
        "Panel A uses corrected A1 local panel-size aggregates. Panels B and C use the unchanged "
        "external-transfer and cross-cohort aggregate tables from the 133 workspace.",
    )
    if qa:
        manifest["qa"] = write_qa(results_dir, out_dir, registry, manifest)
        (out_dir / f"{FIGURE_NAME}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--prepare-source-data", action="store_true")
    parser.add_argument("--no-qa", action="store_true")
    args = parser.parse_args()
    if args.prepare_source_data:
        print(json.dumps({"copied": prepare_source_data(WORKSPACE_RESULTS, args.results_dir)}, indent=2))
        return
    print(json.dumps(render(args.results_dir, args.output_dir, qa=not args.no_qa), indent=2))


if __name__ == "__main__":
    main()
