#!/usr/bin/env python3
"""Render Figure 1 from documented candidate-selection and study-flow records."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "crc_public_discovery_20260906",
        "savefig.dpi": 300,
        "figure.dpi": 140,
    }
)


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def file_record(path: str) -> dict:
    p = ROOT / path
    data = p.read_bytes()
    record = {
        "path": path,
        "bytes": p.stat().st_size,
        "sha256": sha256(data).hexdigest(),
        "media_type": media_type(p),
    }
    if p.suffix.lower() == ".png":
        with Image.open(p) as image:
            record["png_width"] = image.width
            record["png_height"] = image.height
    if p.suffix.lower() == ".pdf":
        record["pdf_check"] = "header_and_eof"
    if p.suffix.lower() == ".svg":
        record["svg_check"] = "parsed_no_active_content"
    if p.suffix.lower() in {".tsv", ".csv"}:
        lines = p.read_text().splitlines()
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        header = lines[0].split(sep) if lines else []
        record["table"] = {"rows": max(len(lines) - 1, 0), "columns": len(header), "header": header}
    return record


def media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".py": "text/x-python",
        ".tsv": "text/tab-separated-values",
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".svg": "image/svg+xml",
    }.get(suffix, "application/octet-stream")


def validate_manifest(manifest: dict) -> None:
    required = ["schema_version", "profile", "figure_id", "generated_at", "base_dir", "script", "inputs", "outputs", "source_data", "panel_claims", "transformations", "known_limitations", "validation"]
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"F1 manifest missing keys: {missing}")
    if manifest["schema_version"] != "1.0" or manifest["profile"] != "publication" or manifest["figure_id"] != "F1_data_flow":
        raise ValueError("F1 manifest identity fields are invalid")
    for group in ["inputs", "outputs", "source_data"]:
        if not manifest[group]:
            raise ValueError(f"F1 manifest has empty {group}")
    for record in [manifest["script"], *manifest["inputs"], *manifest["outputs"], *manifest["source_data"]]:
        path = ROOT / record["path"]
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"F1 manifest path escapes package root: {record['path']}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(record["path"])
        if sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"F1 manifest hash mismatch: {record['path']}")
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                if image.width != record["png_width"] or image.height != record["png_height"]:
                    raise ValueError(f"F1 PNG dimensions mismatch: {record['path']}")
        if path.suffix.lower() == ".svg":
            root = ET.parse(path).getroot()
            if root.tag.split("}")[-1].lower() != "svg":
                raise ValueError("F1 SVG root is not svg")
            svg_text = path.read_text()
            if re.search(r"<script|javascript:|<foreignObject", svg_text, flags=re.I):
                raise ValueError("F1 SVG contains active or external-capable content")
        if path.suffix.lower() == ".pdf":
            data = path.read_bytes()
            if not data.startswith(b"%PDF") or b"%%EOF" not in data[-1024:]:
                raise ValueError("F1 PDF failed header/EOF check")
    if manifest["validation"].get("status") != "structural_pass":
        raise ValueError("F1 manifest validation status is not structural_pass")


def write_manifest() -> None:
    manifest = {
        "schema_version": "1.0",
        "profile": "publication",
        "figure_id": "F1_data_flow",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "base_dir": ".",
        "script": file_record("scripts/make_discovery_figure.py"),
        "inputs": [
            file_record("registry/original_gene_selection.json"),
            file_record("baseline_114/results/data_summary.json"),
            file_record("baseline_114/registry/ml_config.json"),
            file_record("registry/cohort_sources.tsv"),
        ],
        "outputs": [
            file_record("figures/F1_data_flow.png"),
            file_record("figures/F1_data_flow.pdf"),
            file_record("figures/F1_data_flow.svg"),
        ],
        "source_data": [
            file_record("figures/source_data/F1_discovery_source_data.tsv"),
            file_record("figures/source_data/F1_flow_source_data.tsv"),
        ],
        "panel_claims": [
            "Original candidate selection used four public Illumina HumanMethylation450 datasets, minfi q<0.001, delta beta >0.3, 1,088 hypermethylated CpGs, 390 annotated genes, literature prioritisation, and ten genes selected for pyrosequencing.",
            "The current reanalysis used 87 Korean paired patients, 174 local specimens, nested cross-validation for tissue classification and recurrence modelling, public biology cohorts, and no recurrence outcomes for candidate-gene selection.",
        ],
        "transformations": [
            "Counts were read from registry/original_gene_selection.json, baseline_114/results/data_summary.json, baseline_114/registry/ml_config.json, and registry/cohort_sources.tsv, then rendered as a schematic without rerunning the historical genome-wide screen."
        ],
        "known_limitations": [
            "Historical CpG and gene counts were not independently recomputed; the original 390-gene list and quantitative reduction rule from 390 genes to ten genes were unavailable.",
            "This figure is a provenance and study-design schematic, not a statistical result panel.",
        ],
        "validation": {
            "status": "structural_pass",
            "scientific_validation": "not_assessed",
            "commands": ["PYTHONDONTWRITEBYTECODE=1 python3 scripts/make_discovery_figure.py"],
        },
    }
    validate_manifest(manifest)
    path = ROOT / "figures" / "F1_data_flow_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_manifest(json.loads(path.read_text()))


def box(ax, xy, wh, title, body="", fc="#f8fbff", ec="#4a6572", lw=1.0, fs=9.0, body_fs=None):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.012", linewidth=lw,
        edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h*(0.79 if body else 0.5), title,
            ha="center", va="center", fontsize=fs, fontweight="bold")
    if body:
        ax.text(x+w/2, y+h*0.35, body, ha="center", va="center",
                fontsize=body_fs or fs-0.6, linespacing=1.25)
    return patch


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.0,
            color="#56616b",
            shrinkA=5,
            shrinkB=5,
        )
    )


def wrap_genes() -> str:
    return "\n".join([", ".join(GENES[:5]), ", ".join(GENES[5:])])


def count_public_methylation_cohorts() -> int:
    path = ROOT / "registry" / "cohort_sources.tsv"
    rows = path.read_text().splitlines()
    return sum(1 for line in rows[1:] if line and not line.startswith("KPU_87\t"))


def write_source_tables(discovery: dict, data_summary: dict, ml_config: dict) -> None:
    source_dir = ROOT / "figures" / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    d = discovery["discovery"]
    discovery_rows = [
        ("screen_platform", d["platform"], "Historical screen reported in original manuscript/Figure 1"),
        ("datasets", ";".join(d["datasets"]), "GSE139404 and GSE129364 adenoma data; GSE101764 and GSE68838 cancer data"),
        ("tissue_groups", ";".join(d["tissue_groups"]), "Original public methylome comparisons"),
        ("software", d["software"], "Reported analysis software"),
        ("q_cutoff_exclusive", d["q_cutoff_exclusive"], "Differential methylation criterion"),
        ("hypermethylation_delta_beta_exclusive", d["hypermethylation_delta_beta_exclusive"], "Differential methylation criterion"),
        ("hypomethylation_delta_beta_exclusive", d["hypomethylation_delta_beta_exclusive"], "Differential methylation criterion"),
        ("hypermethylated_cpgs", d["hypermethylated_cpgs"], "Historical reported count, not recomputed"),
        ("hypomethylated_cpgs", d["hypomethylated_cpgs"], "Historical reported count, not used for the current fixed panel"),
        ("genes_annotated_to_hypermethylated_cpgs", d["genes_annotated_to_hypermethylated_cpgs"], "Historical reported count"),
        ("prioritisation", d["prioritisation"], "No quantitative top-10 ranking rule was available"),
        ("selected_genes", ";".join(d["selected_genes"]), "Genes taken forward for pyrosequencing"),
        ("recurrence_used_for_selection", d["recurrence_used_for_selection"], "Selection used tissue methylation differences"),
    ]
    (source_dir / "F1_discovery_source_data.tsv").write_text(
        "item\tvalue\tnote\n"
        + "\n".join(f"{item}\t{value}\t{note}" for item, value, note in discovery_rows)
        + "\n"
    )

    flow_rows = [
        ("PSQ records", 88, "Paired pyrosequencing records before final clinical-linkage exclusion"),
        ("Excluded PSQ records", len(data_summary["excluded_psq_ids"]), "ID92 lacks final clinical linkage"),
        ("Clinical-matched patients", data_summary["n_patients"], "Korean patients retained for paired local analyses"),
        ("Local tissue specimens", data_summary["n_patients"] * 2, "Tumor and adjacent-mucosa specimens"),
        ("Genes assayed", data_summary["n_genes"], "Fixed candidate genes selected before current recurrence modelling"),
        ("Tissue classification", data_summary["n_patients"], f"{ml_config['tissue']['outer_folds']}-fold x {ml_config['tissue']['outer_repeats']}-repeat grouped nested CV"),
        ("Primary recurrence set", data_summary["n_recurrence_primary"], f"{data_summary['n_events_primary']} events; stage I-III without recorded palliative surgery"),
        ("Recurrence modelling", data_summary["n_recurrence_primary"], f"{ml_config['recurrence']['outer_folds']}-fold x {ml_config['recurrence']['outer_repeats']}-repeat nested CV"),
        ("Public methylation cohorts", count_public_methylation_cohorts(), "Tissue and molecular-context analyses"),
        ("Matched expression resources", 2, "Colonomics and supportive ColoCare discovery-overlap analysis"),
        ("Public tissue classifiers", 140, "92 Colonomics pairs plus 48 GSE119526 pairs; cohort-local models"),
        ("Recurrence used for gene selection", 0, "No recurrence outcomes were used to select the ten genes"),
    ]
    (source_dir / "F1_flow_source_data.tsv").write_text(
        "item\tn\tnote\n" + "\n".join(f"{item}\t{n}\t{note}" for item, n, note in flow_rows) + "\n"
    )


def draw_panel_a(ax, d: dict) -> None:
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.0, 1.02, "A", fontsize=13, fontweight="bold", va="bottom")
    ax.text(0.065, 1.02, "Original candidate-gene selection", fontsize=11, fontweight="bold", va="bottom")
    datasets = "; ".join(d["datasets"][:2])+"\n"+"; ".join(d["datasets"][2:])
    box(ax, (0.04, 0.77), (0.92, 0.18), "Four public Illumina 450K datasets", datasets, fc="#edf4f8")
    box(ax, (0.04, 0.57), (0.92, 0.14), "Differential methylation screen",
        f"{d['software']}: q < {d['q_cutoff_exclusive']}; Δβ > {d['hypermethylation_delta_beta_exclusive']} for hypermethylation", fc="#fff6e5")
    box(ax, (0.04, 0.41), (0.92, 0.10),
        f"{d['hypermethylated_cpgs']:,} hypermethylated CpGs → {d['genes_annotated_to_hypermethylated_cpgs']:,} annotated genes", fc="#edf6ef")
    box(ax, (0.04, 0.25), (0.92, 0.10), "Literature review", "Relevance to colorectal cancer", fc="#f3eef8")
    genes = d["selected_genes"]
    gene_lines = ", ".join(genes[:5])+"\n"+", ".join(genes[5:])
    box(ax, (0.04, 0.02), (0.92, 0.17), "Ten genes selected for pyrosequencing", gene_lines, fc="#edf6ef")
    for start, end in [(0.77,0.71),(0.57,0.51),(0.41,0.35),(0.25,0.19)]:
        arrow(ax, (0.5,start), (0.5,end))


def draw_panel_b(ax, data_summary: dict, ml_config: dict) -> None:
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.0, 1.02, "B", fontsize=13, fontweight="bold", va="bottom")
    ax.text(0.065, 1.02, "Current study design", fontsize=11, fontweight="bold", va="bottom")
    n=data_summary['n_patients'];ex=len(data_summary['excluded_psq_ids'])
    box(ax, (0.02,0.51), (0.31,0.42), "Korean cohort",
        f"{n+ex} paired records\n{ex} unlinked record excluded\n{n} patients / {2*n} specimens\n{data_summary['n_genes']} genes", fc="#edf4f8", fs=8.7, body_fs=8.0)
    tissue=ml_config['tissue'];rec=ml_config['recurrence']
    box(ax, (0.40,0.51), (0.58,0.42), "Local analyses",
        "Paired methylation effects and patterns\n"
        f"Tissue classification: {n} pairs\n"
        f"Nested {tissue['outer_folds']}-fold × {tissue['outer_repeats']}-repeat CV\n"
        f"Recurrence: {data_summary['n_recurrence_primary']} patients, {data_summary['n_events_primary']} events\n"
        f"Nested {rec['outer_folds']}-fold × {rec['outer_repeats']}-repeat CV", fc="#edf6ef", fs=8.7, body_fs=8.0)
    box(ax, (0.02,0.08), (0.31,0.35), "Public cohorts",
        "Biology: 7 methylation cohorts\n2 matched expression resources\nTissue models: 2 paired cohorts", fc="#edf4f8", fs=8.7, body_fs=7.7)
    box(ax, (0.40,0.08), (0.58,0.35), "Public analyses",
        "Healthy mucosa, adenoma and tumor context\n"
        "Expression, stromal score and CMS\n"
        "Cohort-local tissue classifiers\n"
        "Colonomics: 92 pairs; GSE119526: 48 pairs", fc="#f3eef8", fs=8.7, body_fs=8.0)
    # Separate lanes: neither the local models nor their outcomes feed public analyses.
    arrow(ax, (0.33,0.72), (0.40,0.72))
    arrow(ax, (0.33,0.255), (0.40,0.255))
    ax.text(0.5,0.005,"Gene selection did not use recurrence outcomes.",
            ha="center",va="bottom",fontsize=8,color="#4a4a4a")


def render() -> None:
    discovery = load_json("registry/original_gene_selection.json")
    data_summary = load_json("baseline_114/results/data_summary.json")
    ml_config = load_json("baseline_114/registry/ml_config.json")
    write_source_tables(discovery, data_summary, ml_config)

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 8.3), gridspec_kw={"height_ratios": [1.4, 1]}, layout="constrained")
    draw_panel_a(axes[0], discovery["discovery"])
    draw_panel_b(axes[1], data_summary, ml_config)
    stem = ROOT / "figures" / "F1_data_flow"
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    svg = stem.with_suffix(".svg")
    svg.write_text(re.sub(r"<!DOCTYPE svg PUBLIC.*?>\s*", "", svg.read_text(), count=1, flags=re.S))
    plt.close(fig)
    with Image.open(stem.with_suffix(".png")) as image:
        image.save(stem.with_suffix(".tiff"), compression="tiff_lzw")
    write_manifest()


def main() -> None:
    render()


if __name__ == "__main__":
    main()
