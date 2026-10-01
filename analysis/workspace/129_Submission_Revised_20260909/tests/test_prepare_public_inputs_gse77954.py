from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_public_inputs as ppi  # noqa: E402


def test_gse77954_pairing_uses_fresh_geo_description_not_stale_sample_copy():
    rows = []
    geo_rows = []
    for code in ["02", "04", "05", "06"]:
        for prefix, tissue in [("CCX", "T"), ("NCCX", "N")]:
            sample = f"GSM_{prefix}{code}"
            rows.append(
                {
                    "sample": sample,
                    "patient": sample,
                    "tissue": tissue,
                    "title": "stale title",
                    "source_name_ch1": "stale source",
                    "description": f"STALE{code}",
                }
            )
            geo_rows.append(
                {
                    "sample": sample,
                    "title": f"{prefix}{code} title",
                    "source_name_ch1": "Carcinoma" if tissue == "T" else "Normal colon adjacent to carcinoma",
                    "description": f"{prefix}{code}",
                }
            )

    samples, evidence, paired_codes = ppi.correct_gse77954_pairing(
        pd.DataFrame(rows),
        pd.DataFrame(geo_rows),
        "GSE77954_series_matrix.txt.gz",
        "dummy-sha256",
    )

    assert paired_codes == ["02", "04", "05", "06"]
    assert sorted(evidence["patient"]) == [
        "GSE77954_CCX02",
        "GSE77954_CCX04",
        "GSE77954_CCX05",
        "GSE77954_CCX06",
    ]
    assert not samples["description"].str.startswith("STALE").any()
    assert samples.loc[samples["sample"].eq("GSM_CCX02"), "patient"].iloc[0] == "GSE77954_CCX02"
    assert samples.loc[samples["sample"].eq("GSM_NCCX02"), "patient"].iloc[0] == "GSE77954_CCX02"
    assert samples["pair_verified"].sum() == 8
