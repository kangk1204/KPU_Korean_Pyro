#!/usr/bin/env python3
"""Prepare the manuscript's restricted local PSQ/clinical inputs portably.

This entry point prepares data only. It does not train models, publish data, or
read historical manuscript DOCX files. Outputs contain participant-level data
and must remain under the same access restrictions as the supplied inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise ValueError(message)

GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
SHEETS = {"tumor": "종양조직PSQ", "normal": "정상조직PSQ", "legacy": "종양조직"}
CLINICAL_COLUMNS = {
    "age": "Age", "tnm": "Stage", "stage": "Stage.1", "lvi": "LVI",
    "cea_ng_ml": "CEA_law_data(ng/ml)(normal 0-7)",
    "ca199_u_ml": "CA19-9_law_data(unit/ml)(normal: 0-37)",
    "pni": "PNI", "preop_treatment": "Preop treatment", "surgery": "OP",
    "postop_chemo": "postop chemo(3 cycle 이상인 경우 1: yes)",
    "event": "Recur (progression) or not",
}
PRIMER_COLUMNS = ["gene", "forward_primer", "reverse_primer", "sequencing_primer", "amplicon_bp"]


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    require(not missing, f"{label}: missing required columns: {missing}")


def numeric(frame, columns, label):
    try:
        result = frame.loc[:, columns].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: nonnumeric value in required numeric columns") from exc
    require(np.isfinite(result.to_numpy(dtype=float)).all(), f"{label}: missing or nonfinite numeric value")
    return result


def indexed_source(frame, id_column, label, allow_blank_summary=False):
    frame = frame.copy()
    require_columns(frame, [id_column], label)
    if allow_blank_summary:
        # The archived legacy-call sheet ends with two unlabeled aggregate rows.
        # Nonblank malformed identifiers are never silently discarded.
        frame = frame.loc[frame[id_column].notna()].copy()
    ids = numeric(frame, [id_column], label)[id_column]
    require(ids.gt(0).all() and ids.mod(1).eq(0).all(), f"{label}: IDs must be positive integers")
    require(not ids.duplicated().any(), f"{label}: duplicate patient IDs")
    frame[id_column] = ids.astype(int)
    return frame.set_index(id_column).sort_index()


def validate_provider_reply(text):
    statements = [
        "재발 환자의 경우 마지막 추적일이 재발 날짜입니다.",
        "재발하지 않은 분은 마지막 추적일입니다.",
        "수치 및 reference 기준이 맞습니다",
    ]
    require(all(s in text for s in statements),
            "Provider reply does not contain the archived endpoint and concentration confirmations; do not infer event-date semantics.")


def build_local_frames(clinical_raw, psq_sheets, primers):
    """Validate and transform source frames; no filesystem writes or fixed ROOT."""
    required = ["ListNo.", "Gender(M:1, F:2)", "Date of OP", "Date of last follow-up",
                "CEA_1elevation", "CA19-9elevation", *CLINICAL_COLUMNS.values()]
    required += [prefix + g for prefix in ["CP_", "NP_", "C_"] for g in GENES]
    require_columns(clinical_raw, required, "Clinical workbook")
    x = indexed_source(clinical_raw, "ListNo.", "Clinical workbook")
    require(len(x) > 0, "Clinical workbook: no patients")
    sources, source_ids = {}, {}
    for role, name in SHEETS.items():
        require(role in psq_sheets, f"Missing PSQ sheet: {name}")
        frame = psq_sheets[role]
        require_columns(frame, ["Samples", *GENES], f"PSQ {name}")
        d = indexed_source(frame, "Samples", f"PSQ {name}", allow_blank_summary=role == "legacy")
        source_ids[role] = set(d.index)
        missing = sorted(set(x.index) - set(d.index))
        require(not missing, f"PSQ {name}: clinical patients absent from sheet: {missing}")
        # Clinical eligibility precedes assay-value validation. The archived
        # PSQ-only ID 92 has specimen-exhaustion text and is not an analysis row.
        values = numeric(d.loc[x.index], GENES, f"PSQ {name}")
        if role == "legacy":
            require(values.isin([0, 1]).all().all(), f"PSQ {name}: legacy calls must be binary")
        else:
            require(values.ge(0).all().all() and values.le(100).all().all(), f"PSQ {name}: methylation must be within 0 to 100 percent")
        sources[role] = values
    require(source_ids["tumor"] == source_ids["normal"], "Tumor and normal PSQ sheets have different patient-ID sets")
    require(source_ids["tumor"] == source_ids["legacy"], "PSQ and legacy-call sheets have different patient-ID sets")
    t, n, old = [sources[role].loc[x.index, GENES] for role in ["tumor", "normal", "legacy"]]
    for source, prefix in [(t, "CP_"), (n, "NP_"), (old, "C_")]:
        clinical_values = numeric(x, [prefix + g for g in GENES], "Clinical methylation values")
        require(np.array_equal(source.to_numpy(), clinical_values.to_numpy()),
                f"Clinical {prefix} values disagree with the PSQ workbook")

    c = pd.DataFrame({"patient_id": x.index.astype(int), "study_id": [f"P{i:03d}" for i in range(1, len(x) + 1)]})
    for dest, source in CLINICAL_COLUMNS.items():
        c[dest] = x[source].to_numpy()
    quantitative = ["age", "stage", "lvi", "cea_ng_ml", "ca199_u_ml", "pni", "preop_treatment", "postop_chemo", "event"]
    c[quantitative] = numeric(c, quantitative, "Clinical variables")
    require(c.stage.isin([1, 2, 3, 4]).all(), "Clinical stage must use recorded codes 1 to 4")
    for col in ["lvi", "pni", "preop_treatment", "postop_chemo", "event"]:
        require(c[col].isin([0, 1]).all(), f"Clinical {col} must be binary")
    require(c.age.gt(0).all() and c.age.le(120).all(), "Clinical age is outside the supported range")
    require(c.cea_ng_ml.ge(0).all() and c.ca199_u_ml.ge(0).all(), "Clinical marker concentrations must be nonnegative")
    require(c.tnm.notna().all() and c.surgery.notna().all(), "Clinical TNM and surgery fields must not be missing")
    c["sex"] = x["Gender(M:1, F:2)"].map({1: "M", 2: "F"}).to_numpy()
    require(c.sex.notna().all(), "Clinical sex must use recorded codes 1 or 2")
    c["cea_elevated"] = c.cea_ng_ml.gt(7).astype(int)
    c["ca199_elevated"] = c.ca199_u_ml.gt(37).astype(int)
    try:
        dates1 = pd.to_datetime(x["Date of OP"], errors="raise")
        dates2 = pd.to_datetime(x["Date of last follow-up"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Clinical dates cannot be parsed") from exc
    require(dates1.notna().all() and dates2.notna().all(), "Clinical dates must not be missing")
    c["operation_date"] = dates1.dt.strftime("%Y-%m-%d").to_numpy()
    c["endpoint_date"] = dates2.dt.strftime("%Y-%m-%d").to_numpy()
    c["duration_days"] = (dates2 - dates1).dt.days.to_numpy()
    require(c.duration_days.gt(0).all(), "Every endpoint date must follow surgery")
    c["recurrence_primary"] = (c.stage.isin([1, 2, 3]) & ~c.surgery.str.contains("palliative", case=False, na=False)).astype(int)

    wide = c[["patient_id", "study_id"]].copy()
    for g in GENES:
        wide["T_" + g] = t[g].to_numpy()
        wide["N_" + g] = n[g].to_numpy()
    long = pd.DataFrame([
        {"patient_id": row.patient_id, "study_id": row.study_id, "gene": g,
         "tissue": tissue, "methylation_pct": getattr(row, tissue + "_" + g)}
        for row in wide.itertuples(index=False) for g in GENES for tissue in ["T", "N"]
    ])
    legacy = c[["patient_id", "study_id"]].copy()
    for g in GENES:
        legacy[g] = old[g].astype(int).to_numpy()
    flags = []
    multiple = c.groupby("tnm").stage.nunique()
    for row in c[c.tnm.isin(multiple[multiple > 1].index)].itertuples():
        flags.append({"patient_id": row.patient_id, "study_id": row.study_id,
                      "type": "same_TNM_different_provider_stages",
                      "detail": f"TNM={row.tnm}; provider stage={row.stage}; retained without automatic restaging"})
    for row in c[c.surgery.str.contains("palliative", case=False, na=False)].itertuples():
        flags.append({"patient_id": row.patient_id, "study_id": row.study_id, "type": "palliative_operation_record", "detail": row.surgery})
    for row in c[c.event.eq(0) & c.duration_days.lt(5 * 365.25)].itertuples():
        flags.append({"patient_id": row.patient_id, "study_id": row.study_id,
                      "type": "non_event_followup_under_five_years",
                      "detail": f"{row.duration_days} days; do not describe universal minimum5year followup"})
    audit_flags = numeric(x, ["CEA_1elevation", "CA19-9elevation"], "Clinical supplied marker flags")
    require(audit_flags.isin([0, 1]).all().all(), "Clinical supplied marker flags must be binary")
    cea_audit = pd.DataFrame({"patient_id": c.patient_id, "cea_ng_ml": c.cea_ng_ml,
                              "derived_elevated": c.cea_elevated, "source_sheet_elevated": x["CEA_1elevation"].to_numpy()})
    ca_audit = pd.DataFrame({"patient_id": c.patient_id, "ca199_u_ml": c.ca199_u_ml,
                             "derived_elevated": c.ca199_elevated, "source_sheet_elevated": x["CA19-9elevation"].to_numpy()})
    excluded = sorted(source_ids["tumor"] - set(x.index))
    exclusions = pd.DataFrame({"patient_id": excluded, "reason": ["PSQ-only ID absent from final clinical workbook"] * len(excluded)})

    require_columns(primers, PRIMER_COLUMNS, "Primer TSV")
    primers = primers.loc[:, PRIMER_COLUMNS].copy()
    require(len(primers) == len(GENES) and set(primers.gene) == set(GENES) and not primers.gene.duplicated().any(), "Primer TSV must contain the ten genes exactly once")
    for col in PRIMER_COLUMNS[1:4]:
        require(primers[col].notna().all() and primers[col].astype(str).str.fullmatch(r"\*?[ACGTRYSWKMBDHVN]+").all(), f"Primer TSV: invalid or missing {col}")
    length = numeric(primers, ["amplicon_bp"], "Primer TSV").amplicon_bp
    require(length.gt(0).all() and length.mod(1).eq(0).all(), "Primer lengths must be positive integers")
    primers["amplicon_bp"] = length.astype(int)

    d = c.merge(wide, on=["study_id", "patient_id"], validate="one_to_one", sort=False)
    recurrence = d[["study_id", "event", "duration_days", "recurrence_primary", "stage", "lvi"]].copy()
    recurrence["stage_iii"] = d.stage.eq(3).astype(int)
    recurrence["stage_advanced"] = d.stage.ge(3).astype(int)
    recurrence["cea_log1p"] = np.log1p(d.cea_ng_ml)
    recurrence["cea_binary"] = d.cea_elevated
    for g in GENES:
        recurrence["T_" + g] = d["T_" + g]
        recurrence["N_" + g] = d["N_" + g]
        recurrence["D_" + g] = d["T_" + g] - d["N_" + g]
    tissue = pd.DataFrame([
        {"study_id": row.study_id, "specimen_id": f"{row.study_id}_{label}", "tissue": label, "y": y,
         **{g: getattr(row, label + "_" + g) for g in GENES}}
        for row in d.itertuples(index=False) for label, y in [("N", 0), ("T", 1)]
    ])
    require(recurrence.notna().all().all() and tissue.notna().all().all(), "ML inputs contain missing values")
    require(tissue.groupby("study_id").y.agg(["size", "sum"]).eq([2, 1]).all().all(), "Each patient must have one normal and one tumor ML specimen")
    frames = {
        "data/derived/clinical.tsv": c, "data/derived/methylation_wide.tsv": wide,
        "data/derived/methylation_long.tsv": long, "data/derived/assay_primers.tsv": primers,
        "qc/legacy_calls.tsv": legacy,
        "qc/clinical_flags.tsv": pd.DataFrame(flags, columns=["patient_id", "study_id", "type", "detail"]),
        "qc/cea_audit.tsv": cea_audit, "qc/ca199_audit.tsv": ca_audit, "qc/exclusions.tsv": exclusions,
        "data/ml/recurrence.tsv": recurrence, "data/ml/tissue.tsv": tissue,
    }
    summary = {
        "n_patients": len(c), "n_long_measurements": len(long), "n_genes": len(GENES),
        "n_events_all_stage": int(c.event.sum()), "n_recurrence_primary": int(c.recurrence_primary.sum()),
        "n_events_primary": int(c.loc[c.recurrence_primary.eq(1), "event"].sum()),
        "cea_elevated_n": int(c.cea_elevated.sum()), "ca199_elevated_n": int(c.ca199_elevated.sum()),
        "stage_counts": {str(k): int(v) for k, v in c.stage.value_counts().sort_index().items()},
        "excluded_psq_ids": [int(z) for z in excluded], "all_psq_clinical_values_equal": True,
        "event_date_authority": "supplied provider reply: event last follow-up is recurrence date; non-event last follow-up is last contact",
        "deaths_available": False, "cpG_replicates_available": False,
        "min_duration_days": int(c.duration_days.min()), "max_duration_days": int(c.duration_days.max()),
        "clinical_flag_rows": len(flags), "source_stage_retained": True,
    }
    return frames, summary


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(psq, clinical, provider_reply, primers, outdir):
    """Prepare the fixed study into a new directory; never overwrite a run."""
    outdir = Path(outdir).absolute()
    require(not outdir.exists() and not outdir.is_symlink(), "Output directory must not exist; choose a fresh --outdir")
    paths = {role: Path(p).resolve(strict=True) for role, p in {
        "psq": psq, "clinical": clinical, "provider_reply": provider_reply, "primers": primers}.items()}
    require(all(p.is_file() for p in paths.values()), "All source arguments must be regular files")
    hashes = {role: sha256(p) for role, p in paths.items()}
    validate_provider_reply(paths["provider_reply"].read_text(encoding="utf-8-sig"))
    clinical_raw = pd.read_excel(paths["clinical"], sheet_name="data")
    with pd.ExcelFile(paths["psq"]) as book:
        sheets = {role: pd.read_excel(book, sheet_name=name) for role, name in SHEETS.items()}
    frames, summary = build_local_frames(clinical_raw, sheets, pd.read_csv(paths["primers"], sep="\t"))
    expected = {"n_patients": 87, "n_long_measurements": 1740, "n_events_all_stage": 17,
                "n_recurrence_primary": 82, "n_events_primary": 14, "cea_elevated_n": 18, "ca199_elevated_n": 6}
    for key, value in expected.items():
        require(summary[key] == value, f"Fixed manuscript cohort changed: {key}={summary[key]}, expected {value}")
    require(all(sha256(paths[role]) == h for role, h in hashes.items()), "A source file changed while it was being read")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(exist_ok=False)
    (outdir / "registry").mkdir()
    def write_json(relative, value):
        p = outdir / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    manifest = {"status": "incomplete", "originals_modified": False,
                "restricted_local_data": True,
                "sources": [{"role": role, "input_path": str(p), "sha256": hashes[role], "bytes": p.stat().st_size} for role, p in paths.items()]}
    write_json("registry/source_manifest.json", manifest)
    outputs = []
    for rel, frame in frames.items():
        p = outdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(p, sep="\t", index=False)
        outputs.append({"file": rel, "rows": len(frame), "sha256": sha256(p)})
    write_json("results/data_summary.json", summary)
    write_json("registry/ml_input_manifest.json", {
        "n_patients": 87, "n_specimens": 174, "n_primary": 82, "primary_events": 14,
        "all_events": 17, "cea_elevated_n": 18, "genes": GENES,
        "inputs": [{"file": row["file"], "sha256": row["sha256"]} for row in outputs if row["file"].startswith("data/ml/")],
        "no_original_ids_or_dates": True,
        "scope_note": "The no-original-identifiers claim applies only to data/ml; clinical/QC outputs retain restricted source identifiers and dates.",
    })
    write_json("registry/runtime.json", {"python": sys.version, "platform": platform.platform(),
                                         "numpy": np.__version__, "pandas": pd.__version__})
    manifest.update(status="complete", outputs=outputs)
    write_json("registry/source_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psq", type=Path, required=True, help="Author-supplied PSQ workbook with the three original Korean sheet names")
    parser.add_argument("--clinical", type=Path, required=True, help="Author-supplied clinical workbook, sheet data")
    parser.add_argument("--provider-reply", type=Path, required=True, help="Archived provider reply defining endpoint dates and raw-concentration authority")
    parser.add_argument("--primers", type=Path, required=True, help="Aggregate assay_primers.tsv supplied with the study")
    parser.add_argument("--outdir", type=Path, required=True, help="New restricted output directory; must not already exist")
    args = parser.parse_args()
    try:
        summary = prepare(args.psq, args.clinical, args.provider_reply, args.primers, args.outdir)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({"outdir": str(args.outdir.absolute()), **summary}, indent=2))


if __name__ == "__main__":
    main()
