"""Prepare portable public methylation inputs for the 124 integrated revision.

The source 119 public matrices are copied into 124. GSE77954 metadata are then
corrected from the preserved GEO series matrix so that the four carcinoma / normal
colon-adjacent-to-carcinoma pairs are marked as verified patient pairs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PUBLIC_ROOT = ROOT.parent / "119_Figure1_ABC_Revision_20260908"
DEFAULT_SOURCE_GSE77954_MATRIX = ROOT.parent / "115_Public_Biology_20260905" / "data" / "raw" / "GSE77954_series_matrix.txt.gz"
DEFAULT_OUT_ROOT = ROOT / "data" / "public_inputs"
COHORTS = ["Colonomics", "GSE48684", "GSE42752", "GSE193535", "GSE77718", "GSE77954", "GSE164811"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_geo_line(line: str) -> list[str]:
    return [field.strip().strip('"') for field in line.rstrip("\n").split("\t")][1:]


def read_gse77954_geo_sample_metadata(matrix_path: Path) -> pd.DataFrame:
    fields: dict[str, list[str]] = {}
    with gzip.open(matrix_path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_title"):
                fields["title"] = split_geo_line(line)
            elif line.startswith("!Sample_geo_accession"):
                fields["sample"] = split_geo_line(line)
            elif line.startswith("!Sample_source_name_ch1"):
                fields["source_name_ch1"] = split_geo_line(line)
            elif line.startswith("!Sample_description"):
                fields["description"] = split_geo_line(line)
            if {"title", "sample", "source_name_ch1", "description"}.issubset(fields):
                break
    missing = {"title", "sample", "source_name_ch1", "description"} - set(fields)
    if missing:
        raise ValueError(f"{matrix_path}: missing GEO sample fields {sorted(missing)}")
    lengths = {k: len(v) for k, v in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{matrix_path}: inconsistent sample metadata lengths {lengths}")
    return pd.DataFrame(fields)


def correct_gse77954_pairing(
    samples: pd.DataFrame,
    geo: pd.DataFrame,
    matrix_copy: Path | str,
    matrix_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return GSE77954 samples corrected from fresh GEO sample metadata."""
    required_samples = {"sample", "tissue"}
    required_geo = {"sample", "title", "source_name_ch1", "description"}
    missing_samples = required_samples - set(samples.columns)
    missing_geo = required_geo - set(geo.columns)
    if missing_samples or missing_geo:
        raise ValueError(
            "GSE77954 pairing requires sample/tissue columns and GEO title/source/description; "
            f"missing samples={sorted(missing_samples)} geo={sorted(missing_geo)}"
        )

    source = geo.drop_duplicates("sample").set_index("sample")
    missing_geo_samples = sorted(set(samples["sample"]) - set(source.index))
    if missing_geo_samples:
        raise ValueError(f"GSE77954 samples absent from GEO metadata: {missing_geo_samples[:5]}")

    samples = samples.copy()
    samples = samples.drop(columns=[c for c in ["source_pair_code"] if c in samples.columns])
    for column in ["title", "source_name_ch1", "description"]:
        samples[column] = source[column].reindex(samples["sample"]).to_numpy()
    samples["pair_verified"] = False
    samples["patient"] = samples["sample"]
    samples["pair_rule"] = "GSE77954 tissue type; liver/metastasis excluded; no same-patient T-N source label"
    samples["source_pair_code"] = ""

    code_re = re.compile(r"^(N?CCX)(\d+)$")
    descriptions = samples["description"].astype(str)
    samples["source_pair_code"] = [
        (match.group(2) if (match := code_re.match(desc)) else "") for desc in descriptions
    ]

    tumor_mask = descriptions.str.match(r"^CCX\d+$", na=False)
    normal_mask = descriptions.str.match(r"^NCCX\d+$", na=False)
    tumor_codes = set(samples.loc[tumor_mask, "source_pair_code"])
    normal_codes = set(samples.loc[normal_mask, "source_pair_code"])
    paired_codes = sorted((tumor_codes & normal_codes), key=lambda x: int(x))
    if paired_codes != ["02", "04", "05", "06"]:
        raise ValueError(f"Unexpected GSE77954 paired carcinoma-normal codes: {paired_codes}")

    pair_rows = []
    for code in paired_codes:
        patient_id = f"GSE77954_CCX{code}"
        is_pair_member = samples["source_pair_code"].eq(code) & (tumor_mask | normal_mask)
        samples.loc[is_pair_member, "patient"] = patient_id
        samples.loc[is_pair_member, "pair_verified"] = True
        samples.loc[is_pair_member, "pair_rule"] = (
            "GEO Sample_description CCX## carcinoma matched to NCCX## normal colon adjacent to carcinoma"
        )
        tumor = samples.loc[descriptions.eq(f"CCX{code}")].iloc[0]
        normal = samples.loc[descriptions.eq(f"NCCX{code}")].iloc[0]
        pair_rows.append(
            {
                "patient": patient_id,
                "source_pair_code": code,
                "tumor_sample": tumor["sample"],
                "normal_sample": normal["sample"],
                "tumor_title": tumor["title"],
                "normal_title": normal["title"],
                "tumor_description": tumor["description"],
                "normal_description": normal["description"],
                "tumor_source_name_ch1": tumor["source_name_ch1"],
                "normal_source_name_ch1": normal["source_name_ch1"],
                "source_matrix_copy": str(matrix_copy),
                "source_matrix_sha256": matrix_sha256,
            }
        )

    return samples, pd.DataFrame(pair_rows), paired_codes


def copy_public_inputs(source_public_root: Path, source_gse77954_matrix: Path, out_root: Path) -> dict[str, object]:
    derived = out_root / "data" / "derived"
    raw = out_root / "data" / "raw"
    evidence_dir = ROOT / "review" / "evidence" / "public_revision"
    derived.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, dict[str, str]] = {}
    for cohort in COHORTS:
        for suffix in ["beta.tsv.gz", "samples.tsv"]:
            src = source_public_root / "data" / "derived" / f"{cohort}_{suffix}"
            dst = derived / src.name
            shutil.copy2(src, dst)
            copied[f"{cohort}_{suffix}"] = {
                "source_path": str(src),
                "source_sha256": file_sha256(src),
                "copied_path": str(dst),
                "copied_sha256_initial": file_sha256(dst),
            }

    matrix_copy = raw / source_gse77954_matrix.name
    shutil.copy2(source_gse77954_matrix, matrix_copy)
    geo = read_gse77954_geo_sample_metadata(matrix_copy)
    geo.to_csv(derived / "GSE77954_geo_sample_labels.tsv", sep="\t", index=False)

    samples_path = derived / "GSE77954_samples.tsv"
    samples = pd.read_csv(samples_path, sep="\t", keep_default_na=False, na_values=["", "NaN", "nan"])
    matrix_sha = file_sha256(matrix_copy)
    samples, pair_evidence, paired_codes = correct_gse77954_pairing(
        samples, geo, matrix_copy, matrix_sha
    )

    samples.to_csv(samples_path, sep="\t", index=False)
    pair_evidence.to_csv(derived / "GSE77954_pairing_evidence.tsv", sep="\t", index=False)
    pair_evidence.to_csv(evidence_dir / "gse77954_pairing_evidence.tsv", sep="\t", index=False)

    copied["GSE77954_samples.tsv"]["copied_sha256_corrected"] = file_sha256(samples_path)
    copied["GSE77954_series_matrix.txt.gz"] = {
        "source_path": str(source_gse77954_matrix),
        "source_sha256": file_sha256(source_gse77954_matrix),
        "copied_path": str(matrix_copy),
        "copied_sha256": matrix_sha,
    }

    readme = out_root / "README.md"
    readme.write_text(
        "# Public inputs for 124 integrated revision\n\n"
        "This folder is the portable public input root used by `scripts/analyze_public_cpg_contrasts.py`. "
        "It contains copied beta matrices/sample tables from the 119 public revision plus a corrected "
        "GSE77954 sample table. The correction is limited to four verified carcinoma/adjacent-normal "
        "pairs identified from GEO `Sample_description` labels: CCX02/NCCX02, CCX04/NCCX04, "
        "CCX05/NCCX05, and CCX06/NCCX06. The older 119 metadata are not edited.\n",
        encoding="utf-8",
    )

    manifest = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "out_root": str(out_root),
        "paired_policy": "GSE77954 primary T-N uses only four GEO description-verified CCX/NCCX pairs; remaining carcinoma samples stay available for unpaired non-T-N contrasts.",
        "paired_codes": paired_codes,
        "n_verified_pairs": len(paired_codes),
        "files": copied,
        "evidence": {
            "geo_sample_labels": str(derived / "GSE77954_geo_sample_labels.tsv"),
            "pairing_evidence": str(derived / "GSE77954_pairing_evidence.tsv"),
            "review_pairing_evidence": str(evidence_dir / "gse77954_pairing_evidence.tsv"),
        },
    }
    (out_root / "public_inputs_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    manifest = copy_public_inputs(DEFAULT_SOURCE_PUBLIC_ROOT, DEFAULT_SOURCE_GSE77954_MATRIX, DEFAULT_OUT_ROOT)
    print(json.dumps({"status": manifest["status"], "out_root": manifest["out_root"], "n_verified_pairs": manifest["n_verified_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
