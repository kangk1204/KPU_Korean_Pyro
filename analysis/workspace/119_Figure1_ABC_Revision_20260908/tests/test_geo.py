from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACCESSIONS = ["GSE48684", "GSE42752", "GSE193535", "GSE77718", "GSE77954", "GSE164811"]


def load_samples(accession):
    return pd.read_csv(ROOT / "data" / "derived" / f"{accession}_samples.tsv", sep="\t", keep_default_na=False)


def load_beta(accession):
    return pd.read_csv(ROOT / "data" / "derived" / f"{accession}_beta.tsv.gz", sep="\t", index_col=0)


def test_geo_outputs_exist_and_align():
    assert (ROOT / "registry" / "geo_downloads.json").exists()
    assert (ROOT / "registry" / "geo_curation.md").exists()
    for accession in ACCESSIONS:
        samples = load_samples(accession)
        beta = load_beta(accession)
        assert beta.shape[0] <= 77
        assert beta.shape[0] > 0
        assert list(beta.columns) == samples["sample"].tolist()
        assert samples["sample"].is_unique
        assert {"cohort", "sample", "patient", "tissue", "pair_verified", "age", "sex", "site", "cms"}.issubset(samples.columns)


def test_expected_tissue_counts():
    expected = {
        "GSE48684": {"H": 17, "N": 24, "A": 42, "T": 64},
        "GSE42752": {"H": 19, "N": 22, "T": 22},
        "GSE193535": {"N": 54, "T": 54},
        "GSE77718": {"N": 96, "T": 96},
        "GSE77954": {"A": 12, "T": 13, "N": 4, "NA": 7, "excluded": 12},
        "GSE164811": {"T": 146},
    }
    for accession, exp in expected.items():
        assert load_samples(accession)["tissue"].value_counts(dropna=False).to_dict() == exp


def test_verified_pair_rules_are_conservative():
    assert load_samples("GSE193535").query("pair_verified").patient.nunique() == 54
    assert load_samples("GSE77718").query("pair_verified").patient.nunique() == 95
    assert load_samples("GSE42752").query("pair_verified").patient.nunique() == 22
    assert not load_samples("GSE48684")["pair_verified"].any()
    assert not load_samples("GSE77954")["pair_verified"].any()
    assert not load_samples("GSE164811")["pair_verified"].any()


def test_beta_values_are_not_unit_converted():
    audit = pd.read_csv(ROOT / "registry" / "geo_beta_range_audit.tsv", sep="\t")
    assert set(audit["cohort"]) == set(ACCESSIONS)
    assert not audit["decision"].str.contains("divide", case=False, regex=False).any()
    for accession in ACCESSIONS:
        beta = load_beta(accession)
        assert beta.max(skipna=True).max() <= 1
        assert beta.min(skipna=True).min() >= 0
