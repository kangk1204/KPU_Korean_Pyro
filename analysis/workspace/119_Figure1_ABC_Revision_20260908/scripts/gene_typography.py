"""DOCX OOXML postprocessor for bounded gene-symbol italics."""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lxml import etree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}

GENE_SYMBOLS = (
    "EYA4",
    "ZNF568",
    "ZNF793",
    "SFMBT2",
    "ADHFE1",
    "HOXA2",
    "BEND5",
    "UNC5C",
    "RALYL",
    "GFRA1",
    "SDC2",
    "PPP2R5C",
    "HOXA5",
    "HOXA6",
)

GENE_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, GENE_SYMBOLS)) + r")(?![A-Za-z0-9_])")
OTHER_CAPS_PATTERN = re.compile(r"(?<![A-Za-z0-9_])([A-Z][A-Z0-9]{2,})(?![A-Za-z0-9_])")
REFERENCE_HEADING_PATTERN = re.compile(r"^\s*references\s*$", re.IGNORECASE)
NUMBERED_REFERENCE_PATTERN = re.compile(r"^\s*(?:\[\d+\]|\d+\.)\s+")
PROTEIN_CONTEXT_PATTERN = re.compile(
    r"\b(?:protein|proteins|peptide|peptides|antibody|antibodies|enzyme|enzymes|immunoblot|western blot|transcription factor)\b",
    re.IGNORECASE,
)
SKIP_SPAN_PATTERN = re.compile(
    r"`[^`]*`"
    r"|https?://\S+"
    r"|www\.\S+"
    r"|doi:\S+"
    r"|10\.\d{4,9}/\S+"
    r"|(?<!\S)(?:~|\.\.?|[A-Za-z]:)?[\\/](?:[A-Za-z0-9_.-]+[\\/])*\S+"
    r"|(?<!\S)(?:[A-Za-z0-9_.-]+[\\/]){2,}\S+"
    r"|[A-Za-z0-9_.-]+\.(?:py|r|R|tsv|csv|xlsx?|docx?|pdf|json|xml|zip|gz)(?:[#:/\\][^\s]*)?",
    re.IGNORECASE,
)

WORD_PART_PATTERN = re.compile(r"word/(?:document|header\d+|footer\d+)\.xml$")
TEXT_TAG = f"{{{W_NS}}}t"
RUN_TAG = f"{{{W_NS}}}r"
RPR_TAG = f"{{{W_NS}}}rPr"
ITALIC_TAG = f"{{{W_NS}}}i"
ITALIC_CS_TAG = f"{{{W_NS}}}iCs"
VAL_ATTR = f"{{{W_NS}}}val"
SPACE_ATTR = f"{{{XML_NS}}}space"


@dataclass(frozen=True)
class TextSegment:
    run: ET.Element
    text: ET.Element
    start: int
    end: int
    run_start: int
    run_end: int


def _paragraph_text(paragraph: ET.Element) -> str:
    return "".join(text.text or "" for text in paragraph.iter(TEXT_TAG))


def _is_reference_paragraph(text: str, in_references: bool) -> tuple[bool, bool]:
    stripped = text.strip()
    if REFERENCE_HEADING_PATTERN.match(stripped):
        return True, True
    if in_references:
        return True, True
    if NUMBERED_REFERENCE_PATTERN.match(stripped) and (
        "doi" in stripped.lower() or "http" in stripped.lower() or re.search(r"\b(?:19|20)\d{2}\b", stripped)
    ):
        return True, in_references
    return False, in_references


def _has_true_italic(run: ET.Element) -> bool:
    rpr = run.find("w:rPr", NS)
    if rpr is None:
        return False
    italic = rpr.find("w:i", NS)
    if italic is None:
        return False
    return italic.get(VAL_ATTR, "true") not in {"0", "false", "False"}


def _ensure_true_italic(run: ET.Element) -> None:
    rpr = run.find("w:rPr", NS)
    if rpr is None:
        rpr = ET.Element(RPR_TAG, nsmap=run.nsmap)
        run.insert(0, rpr)
    for tag in (ITALIC_TAG, ITALIC_CS_TAG):
        child = rpr.find(f"w:{tag.rsplit('}', 1)[1]}", NS)
        if child is None:
            child = ET.SubElement(rpr, tag)
        child.set(VAL_ATTR, "true")


def _is_text_only_run(run: ET.Element) -> bool:
    for child in list(run):
        if child.tag in {RPR_TAG, TEXT_TAG}:
            continue
        return False
    return True


def _segments(paragraph: ET.Element) -> list[TextSegment]:
    out: list[TextSegment] = []
    offset = 0
    for run in paragraph.iter(RUN_TAG):
        run_offset = 0
        for text_node in run.findall("w:t", NS):
            value = text_node.text or ""
            out.append(TextSegment(run, text_node, offset, offset + len(value), run_offset, run_offset + len(value)))
            offset += len(value)
            run_offset += len(value)
    return out


def _intervals_by_run(segments: Iterable[TextSegment], intervals: list[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    by_run: dict[int, list[tuple[int, int]]] = {}
    for segment in segments:
        for start, end in intervals:
            local_start = max(start, segment.start)
            local_end = min(end, segment.end)
            if local_start < local_end:
                by_run.setdefault(id(segment.run), []).append(
                    (segment.run_start + local_start - segment.start, segment.run_start + local_end - segment.start)
                )
    return by_run


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _clone_text_run(source_run: ET.Element, text: str, italic: bool) -> ET.Element:
    new_run = ET.Element(source_run.tag, dict(source_run.attrib), nsmap=source_run.nsmap)
    rpr = source_run.find("w:rPr", NS)
    if rpr is not None:
        new_run.append(copy.deepcopy(rpr))
    if italic:
        _ensure_true_italic(new_run)
    text_node = ET.SubElement(new_run, TEXT_TAG)
    text_node.text = text
    if text[:1].isspace() or text[-1:].isspace():
        text_node.set(SPACE_ATTR, "preserve")
    return new_run


def _replace_run(paragraph: ET.Element, run: ET.Element, replacements: list[ET.Element]) -> None:
    for parent in paragraph.iter():
        children = list(parent)
        for index, child in enumerate(children):
            if child is run:
                parent.remove(run)
                for offset, replacement in enumerate(replacements):
                    parent.insert(index + offset, replacement)
                return
    raise ValueError("run parent not found")


def _is_skip_context(text: str, start: int, end: int) -> bool:
    for match in SKIP_SPAN_PATTERN.finditer(text):
        span_start, span_end = match.span()
        if span_start <= start and end <= span_end:
            return True
    return False


def _has_protein_context(text: str, start: int, end: int) -> bool:
    nearby = text[max(0, start - 55) : min(len(text), end + 55)]
    return bool(PROTEIN_CONTEXT_PATTERN.search(nearby))


def _run_text(run: ET.Element) -> str:
    return "".join(text.text or "" for text in run.findall("w:t", NS))


def _italicize_paragraph(paragraph: ET.Element, part_name: str, paragraph_index: int, report: dict) -> int:
    before_text = _paragraph_text(paragraph)
    if not before_text:
        return 0

    intervals: list[tuple[int, int]] = []
    for match in GENE_PATTERN.finditer(before_text):
        gene = match.group(1)
        start, end = match.span(1)
        if _is_skip_context(before_text, start, end):
            report["skipped_context"].append({"part": part_name, "paragraph": paragraph_index, "gene": gene, "text": before_text})
            continue
        if _has_protein_context(before_text, start, end):
            report["ambiguities"].append(
                {"type": "protein_context", "part": part_name, "paragraph": paragraph_index, "gene": gene, "text": before_text}
            )
            continue
        intervals.append((start, end))

    for match in OTHER_CAPS_PATTERN.finditer(before_text):
        symbol = match.group(1)
        if symbol not in GENE_SYMBOLS and len(symbol) >= 3:
            report["other_symbol_candidates"].append(
                {"part": part_name, "paragraph": paragraph_index, "symbol": symbol, "text": before_text}
            )

    if not intervals:
        return 0

    segments = _segments(paragraph)
    by_run = _intervals_by_run(segments, intervals)
    changed = 0
    seen_runs: set[int] = set()
    for segment in segments:
        run_id = id(segment.run)
        if run_id in seen_runs or run_id not in by_run:
            continue
        seen_runs.add(run_id)
        run = segment.run
        run_text = _run_text(run)
        local_intervals = _merge_intervals(sorted(by_run[run_id]))
        if all(_has_true_italic(run) for _ in local_intervals):
            continue
        if not _is_text_only_run(run):
            report["ambiguities"].append(
                {"type": "non_text_run_skipped", "part": part_name, "paragraph": paragraph_index, "text": before_text}
            )
            continue
        replacements: list[ET.Element] = []
        cursor = 0
        for start, end in local_intervals:
            if cursor < start:
                replacements.append(_clone_text_run(run, run_text[cursor:start], _has_true_italic(run)))
            replacements.append(_clone_text_run(run, run_text[start:end], True))
            changed += 1
            cursor = end
        if cursor < len(run_text):
            replacements.append(_clone_text_run(run, run_text[cursor:], _has_true_italic(run)))
        _replace_run(paragraph, run, replacements)

    after_text = _paragraph_text(paragraph)
    if before_text != after_text:
        raise ValueError(f"text invariant failed in {part_name} paragraph {paragraph_index}")
    if changed:
        report["changed_paragraphs"].append(
            {"part": part_name, "paragraph": paragraph_index, "before_text": before_text, "after_text": after_text, "changes": changed}
        )
    return changed


def _process_xml(data: bytes, part_name: str) -> tuple[bytes, int, dict]:
    parser = ET.XMLParser(remove_blank_text=False, resolve_entities=False)
    root = ET.fromstring(data, parser=parser)
    report = {"changed_paragraphs": [], "skipped_references": [], "skipped_context": [], "ambiguities": [], "other_symbol_candidates": []}
    changes = 0
    in_references = False
    for paragraph_index, paragraph in enumerate(root.findall(".//w:p", NS), start=1):
        paragraph_text = _paragraph_text(paragraph)
        skip_reference, in_references = _is_reference_paragraph(paragraph_text, in_references)
        if skip_reference:
            if paragraph_text.strip():
                report["skipped_references"].append({"part": part_name, "paragraph": paragraph_index, "text": paragraph_text})
            continue
        changes += _italicize_paragraph(paragraph, part_name, paragraph_index, report)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True, standalone=root.getroottree().docinfo.standalone), changes, report


def format_docx(path: str | Path) -> dict:
    """Italicize the approved standalone gene symbols in a DOCX in place."""
    docx_path = Path(path)
    if not docx_path.is_file():
        raise FileNotFoundError(docx_path)

    counts = {"parts_changed": 0, "runs_italicized": 0}
    report = {"files": [], "skipped_references": [], "skipped_context": [], "ambiguities": [], "other_symbol_candidates": []}

    with tempfile.NamedTemporaryFile(prefix=docx_path.name, suffix=".tmp", delete=False) as handle:
        tmp_path = Path(handle.name)

    try:
        with zipfile.ZipFile(docx_path, "r") as zin, zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if WORD_PART_PATTERN.match(item.filename):
                    data, changes, part_report = _process_xml(data, item.filename)
                    if changes:
                        counts["parts_changed"] += 1
                        counts["runs_italicized"] += changes
                    for key in ("skipped_references", "skipped_context", "ambiguities", "other_symbol_candidates"):
                        report[key].extend(part_report[key])
                    report["files"].extend(part_report["changed_paragraphs"])
                zout.writestr(item, data)
        shutil.move(str(tmp_path), docx_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {"path": str(docx_path), "counts": counts, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Italicize approved gene symbols in DOCX files.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    results = [format_docx(path) for path in args.paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
