"""Boundary tests for portable restricted local-input preparation."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_local_reproduction.py"
spec = importlib.util.spec_from_file_location("local_reproduction", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


@pytest.fixture
def source_frames():
    # Deliberately unsorted inputs demonstrate key-based pairing and study IDs.
    clinical, tumor, normal, legacy = [], [], [], []
    for i, pid in enumerate([30, 10, 20]):
        row = {
            "ListNo.": pid, "Age": 50 + i, "Gender(M:1, F:2)": 1 + i % 2,
            "Stage": ["T3N1M1", "T2N0M0", "T3N1M0"][i], "Stage.1": [4, 1, 3][i],
            "LVI": i % 2, "CEA_law_data(ng/ml)(normal 0-7)": [7.0, 7.1, 0.0][i],
            "CA19-9_law_data(unit/ml)(normal: 0-37)": [37.0, 37.1, 60.7][i],
            "CEA_1elevation": 0, "CA19-9elevation": 0, "PNI": 0,
            "Preop treatment": 0, "OP": "AR(palliative op)" if pid == 20 else "LAR",
            "postop chemo(3 cycle 이상인 경우 1: yes)": 0,
            "Recur (progression) or not": [1, 0, 1][i],
            "Date of OP": "2020-01-01", "Date of last follow-up": "2025-01-01",
        }
        tr, nr, lr = {"Samples": pid}, {"Samples": pid}, {"Samples": pid}
        for j, gene in enumerate(prep.GENES):
            tr[gene] = 10 + pid / 10 + j
            nr[gene] = j / 2
            lr[gene] = j % 2
            row["CP_" + gene], row["NP_" + gene], row["C_" + gene] = tr[gene], nr[gene], lr[gene]
        clinical.append(row)
        tumor.append(tr)
        normal.append(nr)
        legacy.append(lr)
    sheets = {"tumor": pd.DataFrame(tumor).iloc[::-1].copy(), "normal": pd.DataFrame(normal), "legacy": pd.DataFrame(legacy)}
    primers = pd.DataFrame({"gene": prep.GENES, "forward_primer": ["ACGT"] * 10,
                            "reverse_primer": ["*ACGT"] * 10,
                            "sequencing_primer": ["ACGT"] * 10, "amplicon_bp": [100] * 10})
    return pd.DataFrame(clinical), sheets, primers


def test_keyed_pairing_raw_cutoffs_and_ml_differences(source_frames):
    frames, summary = prep.build_local_frames(*source_frames)
    c = frames["data/derived/clinical.tsv"]
    assert c.patient_id.tolist() == [10, 20, 30]
    assert c.study_id.tolist() == ["P001", "P002", "P003"]
    assert c.cea_elevated.tolist() == [1, 0, 0]
    assert c.ca199_elevated.tolist() == [1, 1, 0]
    assert c.stage.tolist() == [1, 3, 4]
    assert c.recurrence_primary.tolist() == [1, 0, 0]
    r = frames["data/ml/recurrence.tsv"]
    t = frames["data/ml/tissue.tsv"]
    assert len(t) == 6 and len(frames["data/derived/methylation_long.tsv"]) == 60
    assert t.specimen_id.tolist() == ["P001_N", "P001_T", "P002_N", "P002_T", "P003_N", "P003_T"]
    assert t.groupby("study_id").y.sum().eq(1).all()
    for g in prep.GENES:
        np.testing.assert_array_equal(r["D_" + g], r["T_" + g] - r["N_" + g])
        np.testing.assert_array_equal(t.loc[t.tissue.eq("T"), g], r["T_" + g])
        np.testing.assert_array_equal(t.loc[t.tissue.eq("N"), g], r["N_" + g])
    assert "patient_id" not in r and "operation_date" not in r
    assert summary["n_recurrence_primary"] == 1


@pytest.mark.parametrize("location", ["clinical", "tumor", "normal", "legacy"])
def test_duplicate_ids_rejected(source_frames, location):
    c, sheets, primers = source_frames
    if location == "clinical":
        c = pd.concat([c, c.iloc[[0]]], ignore_index=True)
    else:
        sheets[location] = pd.concat([sheets[location], sheets[location].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate patient IDs"):
        prep.build_local_frames(c, sheets, primers)


def test_missing_clinical_patient_in_psq_rejected(source_frames):
    c, sheets, primers = source_frames
    sheets["normal"] = sheets["normal"].iloc[1:].copy()
    with pytest.raises(ValueError, match="clinical patients absent"):
        prep.build_local_frames(c, sheets, primers)


def test_psq_only_exhausted_specimen_is_excluded_before_analysis(source_frames):
    c, sheets, primers = source_frames
    extra = pd.DataFrame([{"Samples": 92, **{g: "시료고갈" for g in prep.GENES}}])
    for role in sheets:
        sheets[role] = pd.concat([sheets[role], extra], ignore_index=True)
    frames, summary = prep.build_local_frames(c, sheets, primers)
    assert summary["n_patients"] == 3 and summary["excluded_psq_ids"] == [92]
    assert frames["qc/exclusions.tsv"].patient_id.tolist() == [92]
    assert len(frames["data/ml/tissue.tsv"]) == 6


@pytest.mark.parametrize("value", [-0.01, 100.01, np.nan, np.inf])
def test_invalid_methylation_rejected(source_frames, value):
    c, sheets, primers = source_frames
    sheets["tumor"].iloc[0, sheets["tumor"].columns.get_loc("EYA4")] = value
    with pytest.raises(ValueError, match="methylation must be|missing or nonfinite"):
        prep.build_local_frames(c, sheets, primers)


def test_workbook_disagreement_rejected(source_frames):
    c, sheets, primers = source_frames
    c.loc[0, "CP_EYA4"] += 0.1
    with pytest.raises(ValueError, match="disagree with the PSQ workbook"):
        prep.build_local_frames(c, sheets, primers)


def test_legacy_summary_rows_allowed_but_bad_id_rejected(source_frames):
    c, sheets, primers = source_frames
    footer = {"Samples": np.nan, **{g: 12 for g in prep.GENES}}
    sheets["legacy"] = pd.concat([sheets["legacy"], pd.DataFrame([footer])], ignore_index=True)
    frames, _ = prep.build_local_frames(c, sheets, primers)
    assert len(frames["qc/legacy_calls.tsv"]) == 3
    sheets["legacy"]["Samples"] = sheets["legacy"]["Samples"].astype(object)
    sheets["legacy"].loc[0, "Samples"] = "bad-id"
    with pytest.raises(ValueError, match="nonnumeric"):
        prep.build_local_frames(c, sheets, primers)


@pytest.mark.parametrize("column,value,match", [
    ("CEA_law_data(ng/ml)(normal 0-7)", -1, "nonnegative"),
    ("Recur (progression) or not", 2, "event must be binary"),
    ("Date of last follow-up", "2019-12-31", "endpoint date must follow surgery"),
])
def test_invalid_clinical_values_rejected(source_frames, column, value, match):
    c, sheets, primers = source_frames
    c.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        prep.build_local_frames(c, sheets, primers)


def test_missing_primer_gene_rejected(source_frames):
    c, sheets, primers = source_frames
    with pytest.raises(ValueError, match="ten genes exactly once"):
        prep.build_local_frames(c, sheets, primers.iloc[:-1])


def test_existing_output_rejected_without_changes(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "untouched.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="must not exist"):
        prep.prepare("absent.xlsx", "absent.xlsx", "absent.txt", "absent.tsv", target)
    assert sentinel.read_text() == "keep"
    assert sorted(p.name for p in target.iterdir()) == ["untouched.txt"]


def test_missing_provider_authority_rejected():
    with pytest.raises(ValueError, match="do not infer event-date semantics"):
        prep.validate_provider_reply("The last follow-up date is available.")
