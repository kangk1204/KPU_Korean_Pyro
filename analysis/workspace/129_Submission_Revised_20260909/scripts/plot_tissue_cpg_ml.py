#!/usr/bin/env python3
"""Forest plot for tissue-classification AUC across PSQ and public array ML."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures" / "Tissue_ML"
LOCAL_F6 = ROOT.parent / "114_ML_DataDriven_20260905" / "figures" / "source_data" / "F6_tissue_local_metrics_source_data.tsv"
PUBLIC_ML = ROOT / "results" / "public_ml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_auc_rows() -> pd.DataFrame:
    local = pd.read_csv(LOCAL_F6, sep="\t")
    rows = []
    local_specs = [
        ("Local PSQ, 87 patients", "PSQ gene-level", "ridge", "ridge"),
        ("Local PSQ, 87 patients", "PSQ gene-level", "inner-selected single gene", "best_single_gene"),
    ]
    for cohort, feature_unit, display_model, source_model in local_specs:
        hit = local.loc[
            local["model"].eq(source_model)
            & local["scope"].eq("all_repeats_pooled")
            & local["plot_metric"].eq("auc")
        ]
        if len(hit) != 1:
            raise ValueError(f"expected exactly one local F6 AUC row for {source_model}, found {len(hit)}")
        r = hit.iloc[0]
        rows.append(
            {
                "cohort": cohort,
                "feature_unit": feature_unit,
                "model": display_model,
                "auc": float(r["plot_value"]),
                "ci_low": float(r["plot_low"]),
                "ci_high": float(r["plot_high"]),
                "source_file": str(LOCAL_F6),
                "source_model": source_model,
                "analysis_scope": "cohort-local nested CV tissue classification",
            }
        )
    for cohort, label in [("colonomics", "Colonomics, 92 patients"), ("gse119526", "GSE119526, 48 patients")]:
        path = PUBLIC_ML / cohort / "bootstrap_ci.tsv"
        public = pd.read_csv(path, sep="\t")
        hit = public.loc[public["model"].eq("ridge_all_cpg") & public["metric"].eq("auc")]
        if len(hit) != 1:
            raise ValueError(f"expected exactly one public CpG ML AUC row for {cohort}, found {len(hit)}")
        r = hit.iloc[0]
        rows.append(
            {
                "cohort": label,
                "feature_unit": "array individual-CpG",
                "model": "ridge",
                "auc": float(r["point"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
                "source_file": str(path),
                "source_model": "ridge_all_cpg",
                "analysis_scope": "cohort-local nested CV tissue classification",
            }
        )
    out = pd.DataFrame(rows)
    out["plot_label"] = out["cohort"] + "\n" + out["feature_unit"] + "; " + out["model"]
    out["metric"] = "AUC"
    out["note"] = "Not external model validation; each row is cohort-specific nested cross-validation."
    return out


def write_source_records(outdir: Path, rows: pd.DataFrame) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(outdir / "tissue_ml_auc_forest_source.tsv", sep="\t", index=False)
    inputs = []
    for path in sorted({Path(p) for p in rows["source_file"]}):
        inputs.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(inputs).to_csv(outdir / "input_files.tsv", sep="\t", index=False)


def render(outdir: Path, rows: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    plot = rows.iloc[::-1].reset_index(drop=True)
    y = range(len(plot))
    colors = ["#4C78A8" if "PSQ" in unit else "#F58518" for unit in plot["feature_unit"]]
    fig, ax = plt.subplots(figsize=(7.05, 3.25))
    for i, r in plot.iterrows():
        ax.errorbar(
            r["auc"],
            i,
            xerr=[[r["auc"] - r["ci_low"]], [r["ci_high"] - r["auc"]]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            elinewidth=1.8,
            capsize=3,
            markersize=6,
            zorder=3,
        )
    ax.axvline(0.5, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_yticks(list(y), plot["plot_label"])
    ax.set_xlim(0.86, 1.075)
    ax.set_xlabel("AUC with patient-cluster bootstrap 95% CI", fontsize=6.2)
    ax.set_title("Tissue classification AUC", loc="left", fontsize=7.4, fontweight="bold")
    for i, r in plot.iterrows():
        ax.text(1.073, i, f"{r['auc']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f})", va="center", ha="right", fontsize=5.8)
    ax.text(
        0.86,
        -0.82,
        "Rows are cohort-local nested CV estimates; public array rows are individual-CpG reruns, not external transfer validation.",
        ha="left",
        va="top",
        fontsize=5.6,
        color="#444444",
    )
    fig.subplots_adjust(left=0.34, right=0.985, bottom=0.25, top=0.86)
    for suffix in (".png", ".pdf", ".svg"):
        target = outdir / f"Tissue_ML{suffix}"
        fig.savefig(target, dpi=600 if suffix == ".png" else None, facecolor="white")
        if suffix == ".svg":
            target.write_text(re.sub(r"<!DOCTYPE[^>]*>", "", target.read_text(), count=1, flags=re.S))
    plt.close(fig)


def write_legend(outdir: Path) -> None:
    (outdir / "legend.md").write_text(
        "\n".join(
            [
                "Tissue-classification AUC forest plot.",
                "",
                "The local cohort rows summarize 87-patient pyrosequencing models using gene-level PSQ features from the existing F6 source table. The public rows summarize the new Colonomics and GSE119526 array reruns using individual fixed CpG beta values. Points are mean repeat-level AUCs from patient-grouped nested cross-validation; intervals are patient-cluster bootstrap 95% CIs conditional on the fitted cross-validation models. These public rows are cohort-local tissue-classification estimates and are not external validation of a frozen PSQ model.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest_stub(outdir: Path, rows: pd.DataFrame) -> None:
    manifest = {
        "figure_id": "Tissue_ML",
        "profile": "publication",
        "created_by": str(Path(__file__)),
        "claims": [
            "Local PSQ ridge and inner-selected single-gene AUCs are read from the existing F6 source table.",
            "Public Colonomics and GSE119526 AUCs are read from the individual-CpG public ML rerun bootstrap CI files.",
        ],
        "limitations": [
            "Rows use cohort-specific nested cross-validation; the public array rows are not external frozen-model validation.",
            "Feature units differ between local PSQ gene-level measurements and public array individual CpG beta values.",
        ],
        "source_rows": int(len(rows)),
    }
    (outdir / "manifest_notes.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()
    rows = load_auc_rows()
    write_source_records(args.outdir, rows)
    render(args.outdir, rows)
    write_legend(args.outdir)
    write_manifest_stub(args.outdir, rows)
    shutil.copy2(Path(__file__), args.outdir / "plot_tissue_cpg_ml.py")
    print(args.outdir.resolve())


if __name__ == "__main__":
    main()
