#!/usr/bin/env python3
"""Assemble revised Figure 1A-C from preserved original panels and vector study flow."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
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
WORKSPACE = ROOT.parent
SOURCE_JPEG = WORKSPACE / "101_Manuscript_20260902" / "Figure1_scheme.jpg"
OUT_STEM = ROOT / "figures" / "F1_data_flow"
SOURCE_DIR = ROOT / "figures" / "source_data"
REVIEW_DIR = ROOT / "review" / "figure1_abc"
ORIGINAL_COPY = SOURCE_DIR / "F1_original_panels.jpg"
DESIGN_COUNTS = SOURCE_DIR / "F1_original_design_counts.json"
EXPECTED_ORIGINAL_SHA256 = "6f96f26e012bb5f2a0245c944446b54f87a817be7619ae044911f4c34b56a20f"
NATURE_FIGURE_DIR = Path(os.environ.get("NATURE_FIGURE_DIR", "tools/nature-figure"))

WIDTH_MM = 183
HEIGHT_MM = 230
WIDTH_IN = WIDTH_MM / 25.4
HEIGHT_IN = HEIGHT_MM / 25.4

# Source JPEG coordinates in pixel space. These crop only the original panel
# rows and preserve the original scientific marks, values, axes and legends.
PANEL_CROPS = {
    "A": {"xyxy": [0, 0, 1941, 380], "description": "original discovery workflow"},
    "B": {"xyxy": [0, 410, 1941, 1609], "description": "original TCGA methylation-expression plots"},
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "crc_figure1_abc_20260908",
        "savefig.dpi": 600,
        "figure.dpi": 150,
    }
)


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def file_record(path: Path) -> dict:
    record = {
        "path": rel(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha(path),
        "media_type": {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".json": "application/json",
            ".py": "text/x-python",
            ".png": "image/png",
            ".pdf": "application/pdf",
            ".svg": "image/svg+xml",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }.get(path.suffix.lower(), "application/octet-stream"),
    }
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
        with Image.open(path) as image:
            record["width_px"] = image.width
            record["height_px"] = image.height
            record["mode"] = image.mode
            record["dpi"] = normalize_dpi(image.info.get("dpi"))
    return record


def normalize_dpi(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [normalize_dpi(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def copy_original_source() -> tuple[Path, dict]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_JPEG.exists():
        shutil.copyfile(SOURCE_JPEG, ORIGINAL_COPY)
        source = {
            "available": True,
            "path": str(SOURCE_JPEG.relative_to(WORKSPACE)),
            "sha256": file_sha(SOURCE_JPEG),
        }
    elif ORIGINAL_COPY.exists():
        source = {
            "available": False,
            "path": None,
            "sha256": None,
            "fallback": rel(ORIGINAL_COPY),
        }
    else:
        raise FileNotFoundError(f"Neither {SOURCE_JPEG} nor {ORIGINAL_COPY} is available")
    copied_sha = file_sha(ORIGINAL_COPY)
    if copied_sha != EXPECTED_ORIGINAL_SHA256:
        raise RuntimeError(f"Original Figure 1 JPEG hash mismatch: {copied_sha}")
    if source["available"] and copied_sha != source["sha256"]:
        raise RuntimeError("Copied original Figure 1 JPEG hash does not match source")
    return ORIGINAL_COPY, source


def write_design_counts() -> Path:
    if (ROOT / "baseline_114" / "results" / "data_summary.json").exists() and (ROOT / "baseline_114" / "registry" / "ml_config.json").exists():
        summary = load_json(ROOT / "baseline_114" / "results" / "data_summary.json")
        ml_config = load_json(ROOT / "baseline_114" / "registry" / "ml_config.json")
        counts = {
            "schema_version": "1.0",
            "source": "frozen aggregate counts extracted from current analysis metadata",
            "korean": {
                "patients": summary["n_patients"],
                "specimens": summary["n_patients"] * 2,
                "genes": summary["n_genes"],
            },
            "local_analyses": {
                "recurrence_patients": summary["n_recurrence_primary"],
                "recurrence_events": summary["n_events_primary"],
                "tissue_cv": {
                    "outer_folds": ml_config["tissue"]["outer_folds"],
                    "outer_repeats": ml_config["tissue"]["outer_repeats"],
                },
                "recurrence_cv": {
                    "outer_folds": ml_config["recurrence"]["outer_folds"],
                    "outer_repeats": ml_config["recurrence"]["outer_repeats"],
                },
            },
            "public": {
                "methylation_cohorts": 7,
                "expression_resources": 2,
                "classifier_cohorts": 2,
                "colonomics_pairs": 92,
                "gse119526_pairs": 48,
            },
            "selection": {
                "recurrence_used_for_gene_selection": False,
            },
        }
        DESIGN_COUNTS.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
    if not DESIGN_COUNTS.exists():
        raise FileNotFoundError(DESIGN_COUNTS)
    return DESIGN_COUNTS


def write_crop_record(original_copy: Path) -> Path:
    with Image.open(original_copy) as image:
        source = {
            "source_file": rel(original_copy),
            "source_sha256": file_sha(original_copy),
            "source_width_px": image.width,
            "source_height_px": image.height,
            "source_mode": image.mode,
            "source_embedded_dpi": normalize_dpi(image.info.get("dpi")),
        }
    record = {
        "schema_version": "1.0",
        "note": "Panel A and B crops preserve the original raster marks and labels; no plotted value was redrawn.",
        "source": source,
        "crops": PANEL_CROPS,
        "native_effective_dpi": {
            "at_183mm_full_width": round(source["source_width_px"] / WIDTH_IN, 1),
            "at_6_5in_document_width": round(source["source_width_px"] / 6.5, 1),
        },
        "typography": {
            "original_gene_labels": "retained",
            "reason": "Gene names in the original panels are embedded raster text adjacent to data graphics; selective replacement would risk obscuring original scientific marks.",
        },
    }
    path = SOURCE_DIR / "F1_original_crop_record.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def add_original_panel(ax: plt.Axes, source_image: Image.Image, panel: str) -> None:
    crop = PANEL_CROPS[panel]["xyxy"]
    arr = source_image.crop(tuple(crop)).convert("RGB")
    ax.imshow(arr)
    ax.set_axis_off()


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str = "#66717c",
    title_size: float = 7.2,
    body_size: float = 6.25,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.76, title, ha="center", va="center", fontsize=title_size, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=body_size, linespacing=1.12)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color="#5c6670",
            shrinkA=5,
            shrinkB=5,
        )
    )


def short_connector(ax: plt.Axes, y: float) -> None:
    ax.plot([0.235, 0.286], [y, y], color="#5c6670", linewidth=0.8, solid_capstyle="round", clip_on=False)
    ax.scatter([0.29], [y], marker=">", s=16, color="#5c6670", linewidths=0, clip_on=False)


def analysis_group(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, items: list[tuple[str, str, str]]) -> None:
    outer = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.016",
        linewidth=0.8,
        edgecolor="#66717c",
        facecolor="#ffffff",
    )
    ax.add_patch(outer)
    ax.text(x + w / 2, y + h * 0.88, title, ha="center", va="center", fontsize=6.6, fontweight="bold", color="#42484f")
    gap = 0.018
    inner_y = y + h * 0.08
    inner_h = h * 0.68
    inner_w = (w - gap * 4) / 3
    for i, (heading, body, face) in enumerate(items):
        ix = x + gap + i * (inner_w + gap)
        rounded_box(ax, ix, inner_y, inner_w, inner_h, heading, body, face, title_size=6.55, body_size=5.45)


def draw_panel_c(ax: plt.Axes, counts: dict) -> None:
    korean = counts["korean"]
    local = counts["local_analyses"]
    public = counts["public"]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.003, 0.965, "C", fontsize=9, fontweight="bold", va="top", ha="left")
    ax.text(0.044, 0.965, "Korean cohort validation and public-cohort analyses", fontsize=8.2, fontweight="bold", va="top", ha="left")

    rounded_box(
        ax,
        0.045,
        0.57,
        0.19,
        0.28,
        "Korean cohort",
        f"{korean['patients']} patients\n{korean['specimens']} specimens\nPyrosequencing: {korean['genes']} genes",
        "#edf4f8",
    )
    analysis_group(
        ax,
        0.295,
        0.54,
        0.66,
        0.33,
        "Korean analyses",
        [
            ("Paired methylation", "Tumor-adjacent\neffects and\ncoordination", "#edf6ef"),
            ("Tissue status", "Patient-grouped\nclassification", "#fff4df"),
            ("Recurrence", f"{local['recurrence_patients']} patients\n{local['recurrence_events']} events\nclinical+methylation", "#f7edf7"),
        ],
    )
    short_connector(ax, 0.71)

    rounded_box(
        ax,
        0.045,
        0.15,
        0.19,
        0.27,
        "Public cohorts",
        f"{public['methylation_cohorts']} methylation cohorts\n{public['expression_resources']} expression resources\n{public['classifier_cohorts']} classifier cohorts",
        "#edf4f8",
        body_size=5.8,
    )
    analysis_group(
        ax,
        0.295,
        0.12,
        0.66,
        0.32,
        "Public-cohort analyses",
        [
            ("Lesion context", "Healthy, adjacent,\nadenoma and tumor", "#edf6ef"),
            ("Expression/CMS", "Expression\nstromal score\nCMS labels", "#edf6ef"),
            ("Cohort-local classifiers", f"Colonomics: {public['colonomics_pairs']} pairs\nGSE119526: {public['gse119526_pairs']} pairs\nseparate retraining", "#fff4df"),
        ],
    )
    short_connector(ax, 0.285)
    ax.text(
        0.5,
        0.055,
        "Candidate selection used tissue methylation differences; recurrence outcomes were not used to select genes.",
        ha="center",
        va="center",
        fontsize=6.45,
        color="#42484f",
    )


def render_figure(original_copy: Path, counts: dict) -> None:
    with Image.open(original_copy) as image:
        source_image = image.convert("RGB")
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.42, 4.5, 1.74],
        top=0.985,
        bottom=0.035,
        left=0.03,
        right=0.985,
        hspace=0.07,
    )
    add_original_panel(fig.add_subplot(gs[0, 0]), source_image, "A")
    add_original_panel(fig.add_subplot(gs[1, 0]), source_image, "B")
    draw_panel_c(fig.add_subplot(gs[2, 0]), counts)

    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"bbox_inches": None}
        if suffix == ".pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
        if suffix == ".svg":
            kwargs["metadata"] = {"Date": None}
        fig.savefig(OUT_STEM.with_suffix(suffix), dpi=600, **kwargs)
    svg_path = OUT_STEM.with_suffix(".svg")
    svg_path.write_text(re.sub(r"<!DOCTYPE svg PUBLIC.*?>\s*", "", svg_path.read_text(), count=1, flags=re.S))
    plt.close(fig)

    with Image.open(OUT_STEM.with_suffix(".png")) as image:
        image.save(OUT_STEM.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")


def validate_outputs(manifest: dict) -> None:
    for rec in manifest["outputs"] + manifest["source_data"] + [manifest["script"]]:
        path = ROOT / rec["path"]
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        if file_sha(path) != rec["sha256"]:
            raise RuntimeError(f"Hash mismatch: {path}")
    svg_path = OUT_STEM.with_suffix(".svg")
    svg_text = svg_path.read_text()
    root = ET.parse(svg_path).getroot()
    if root.tag.split("}")[-1] != "svg":
        raise RuntimeError("SVG root is not svg")
    if re.search(r"<script|javascript:|<foreignObject", svg_text, flags=re.I):
        raise RuntimeError("SVG contains active content")
    pdf_bytes = OUT_STEM.with_suffix(".pdf").read_bytes()
    if not pdf_bytes.startswith(b"%PDF") or b"%%EOF" not in pdf_bytes[-2048:]:
        raise RuntimeError("PDF header/EOF check failed")
    with Image.open(OUT_STEM.with_suffix(".png")) as png:
        expected = (round(WIDTH_IN * 600), round(HEIGHT_IN * 600))
        if abs(png.width - expected[0]) > 2 or abs(png.height - expected[1]) > 2:
            raise RuntimeError(f"Unexpected PNG dimensions: {png.size}, expected near {expected}")


def write_manifest(original_copy: Path, crop_record: Path, counts_path: Path, original_source: dict) -> Path:
    source_meta = json.loads(crop_record.read_text())["source"]
    manifest = {
        "schema_version": "1.0",
        "profile": "publication",
        "figure_id": "F1_data_flow",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "canvas": {
            "width_mm": WIDTH_MM,
            "height_mm": HEIGHT_MM,
            "width_in": round(WIDTH_IN, 4),
            "height_in": round(HEIGHT_IN, 4),
            "aspect_ratio": round(HEIGHT_IN / WIDTH_IN, 4),
            "word_6_5in_height_in": round(6.5 * HEIGHT_IN / WIDTH_IN, 3),
            "export_dpi": 600,
        },
        "script": file_record(ROOT / "scripts" / "build_figure1_abc.py"),
        "inputs": [
            {
                "path": original_source["path"],
                "available": original_source["available"],
                "fallback": original_source.get("fallback"),
                "bytes": ORIGINAL_COPY.stat().st_size if not SOURCE_JPEG.exists() else SOURCE_JPEG.stat().st_size,
                "sha256": original_source["sha256"],
                "expected_sha256": EXPECTED_ORIGINAL_SHA256,
                "media_type": "image/jpeg",
                "width_px": source_meta["source_width_px"],
                "height_px": source_meta["source_height_px"],
                "mode": source_meta["source_mode"],
                "dpi": source_meta["source_embedded_dpi"],
            },
            file_record(counts_path),
        ],
        "outputs": [
            file_record(OUT_STEM.with_suffix(".png")),
            file_record(OUT_STEM.with_suffix(".pdf")),
            file_record(OUT_STEM.with_suffix(".svg")),
            file_record(OUT_STEM.with_suffix(".tiff")),
        ],
        "source_data": [
            file_record(original_copy),
            file_record(crop_record),
            file_record(counts_path),
        ],
        "panel_claims": [
            "Panel A preserves the original public methylome candidate-selection workflow.",
            "Panel B preserves the original TCGA pooled tumor/normal methylation-expression plots as historical supporting context.",
            "Panel C summarizes the current Korean pyrosequencing validation, recurrence analysis, public molecular context, and separate cohort-local public tissue classifiers.",
        ],
        "transformations": [
            "Original Figure 1A and 1B are raster crops from the preserved source JPEG; no plotted values, axes, points, heatmaps, or correlations were redrawn.",
            "Panel C is newly drawn vector text and shapes from aggregate counts recorded in F1_original_design_counts.json.",
        ],
        "known_limitations": [
            "The source JPEG has no embedded DPI metadata and contains CMYK raster text; original A/B gene-name typography was retained to avoid modifying embedded data panels.",
            "Exporting at 600 dpi increases output canvas density but does not increase the original A/B panel information content.",
            "Panel B combines tumor and normal samples in the preserved original TCGA plots and should be described as pooled historical supporting context, not tumor-only evidence.",
        ],
        "validation": {"status": "pending_until_figure_build_checks"},
    }
    path = ROOT / "figures" / "F1_data_flow_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def write_qa(manifest_path: Path, crop_record: Path, preflight: dict | None = None, pdf_text: dict | None = None) -> None:
    manifest = json.loads(manifest_path.read_text())
    checks = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "pass",
        "figure_contract": {
            "core_conclusion": "The ten-gene panel was selected from public tissue methylation differences, then validated in paired Korean pyrosequencing data and examined in public molecular-context analyses.",
            "results_level_question": "How were the candidate genes selected, and what current data streams support their tissue and molecular interpretation?",
            "archetype": "schematic-led composite",
            "panel_sequence": {
                "A": "historical candidate-selection source panel",
                "B": "historical pooled methylation-expression context source panel",
                "C": "current reanalysis design and cohort roles",
            },
        },
        "image_integrity": {
            "source_copy_sha_matches": SOURCE_JPEG.exists() and file_sha(SOURCE_JPEG) == file_sha(SOURCE_DIR / "F1_original_panels.jpg"),
            "included_source_sha_matches_expected": file_sha(SOURCE_DIR / "F1_original_panels.jpg") == EXPECTED_ORIGINAL_SHA256,
            "crop_record": rel(crop_record),
            "original_marks_redrawn": False,
            "typography_changes_to_original_panels": "none",
        },
        "layout": manifest["canvas"],
        "automated_checks": {
            "svg_parsed": True,
            "svg_active_content_absent": True,
            "pdf_header_eof": True,
            "png_tiff_exist": True,
            "source_validator": preflight,
            "pdf_text_audit": pdf_text,
        },
        "panel_audit": [
            {
                "panel": "A",
                "unique_claim": "Original public methylome screening and reduction to ten genes",
                "source": "preserved original JPEG crop",
                "data_area_preserved": True,
                "pass": True,
            },
            {
                "panel": "B",
                "unique_claim": "Original pooled TCGA methylation-expression context",
                "source": "preserved original JPEG crop",
                "data_area_preserved": True,
                "pass": True,
            },
            {
                "panel": "C",
                "unique_claim": "Current study and public follow-up roles without implying a frozen external classifier validation",
                "source": "current frozen counts",
                "labels_clear": True,
                "pass": True,
            },
        ],
        "known_issues": [
            "Original panels A/B are raster and their embedded text is not editable.",
            "Original panel B gene labels are not italicized because safe replacement would require overpainting embedded plot text.",
        ],
    }
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (REVIEW_DIR / "figure_build_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")

    contract = f"""# Figure 1 A/B/C Contract

Core conclusion: The ten-gene panel was selected from public colorectal tissue methylation differences, validated in paired Korean pyrosequencing data, and then examined in public molecular-context analyses.

Results-level question: How were the ten genes selected, and which current data streams support tissue-status and molecular-context interpretation?

Archetype: schematic-led composite.

Panel roles:

- A: preserved original public methylome candidate-selection workflow.
- B: preserved original pooled TCGA tumor/normal methylation-expression plots; this is historical supporting context and not the tumor-only analysis used in Figure 10.
- C: current study design, separating Korean pyrosequencing analyses from public molecular context and cohort-local tissue classifiers.

Source handling:

- Original source JPEG copied unchanged to `figures/source_data/F1_original_panels.jpg`.
- Crop coordinates and source metadata recorded in `figures/source_data/F1_original_crop_record.json`.
- Original A/B marks, axes, heatmaps, points, correlations, and labels were not redrawn.
- Gene-name typography in original A/B raster text was retained because selective replacement would risk covering embedded scientific marks.

Layout:

- Canvas: {WIDTH_MM} x {HEIGHT_MM} mm.
- Aspect ratio: {manifest['canvas']['aspect_ratio']}; at 6.5 inch document width, height is {manifest['canvas']['word_6_5in_height_in']} inches.
- Native source effective DPI at 183 mm full width: {json.loads(crop_record.read_text())['native_effective_dpi']['at_183mm_full_width']}.
- Native source effective DPI at 6.5 inch document width: {json.loads(crop_record.read_text())['native_effective_dpi']['at_6_5in_document_width']}.
"""
    (REVIEW_DIR / "figure_contract.md").write_text(contract)


def run_optional_json(command: list[str]) -> dict:
    tool_path = Path(command[1])
    if not tool_path.exists():
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "json": None,
            "status": "skipped",
            "reason": f"Optional QA tool is not available: {tool_path.name}",
        }
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    result = {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    try:
        result["json"] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result["json"] = None
    if proc.returncode != 0:
        raise RuntimeError(f"QA command failed: {' '.join(command)}\n{proc.stderr}")
    return result


def main() -> None:
    original_copy, original_source = copy_original_source()
    counts_path = write_design_counts()
    counts = load_json(counts_path)
    crop_record = write_crop_record(original_copy)
    render_figure(original_copy, counts)
    manifest_path = write_manifest(original_copy, crop_record, counts_path, original_source)
    manifest = json.loads(manifest_path.read_text())
    validate_outputs(manifest)
    manifest["validation"] = {"status": "structural_pass"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_outputs(json.loads(manifest_path.read_text()))
    preflight = run_optional_json(["python3", str(NATURE_FIGURE_DIR / "scripts" / "validate_figure.py"), "scripts/build_figure1_abc.py", "--json"])
    pdf_text = run_optional_json(["python3", str(NATURE_FIGURE_DIR / "scripts" / "audit_pdf_text.py"), "figures/F1_data_flow.pdf", "--min-pt", "5", "--json"])
    write_qa(manifest_path, crop_record, preflight, pdf_text)
    print(json.dumps({"status": "pass", "figure": rel(OUT_STEM.with_suffix(".png")), "manifest": rel(manifest_path)}))


if __name__ == "__main__":
    main()
