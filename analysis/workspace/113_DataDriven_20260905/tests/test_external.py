from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_external_analysis_reruns_from_public_snapshot():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_external.py")], check=True, cwd=ROOT)
    effects = pd.read_csv(ROOT / "results" / "external" / "paired_public_effects.tsv", sep="\t")
    assert set(["colonomics", "gse119526", "tcga_coadread", "tcga_coad", "tcga_read"]).issubset(set(effects["cohort"]))
    primary = effects[effects["cohort"].isin(["colonomics", "gse119526"])]
    assert len(primary) == 20
    assert primary["n_pairs"].min() > 0
    assert primary["paired_t_q_BH10"].between(0, 1).all()
    assert np.isfinite(primary["bca95_low"]).all()
    assert np.isfinite(primary["bca95_high"]).all()
    assert (primary["bca95_low"] <= primary["mean_delta_beta"]).all()
    assert (primary["mean_delta_beta"] <= primary["bca95_high"]).all()


def test_fixed_probe_coverage_and_mapping_limits_are_explicit():
    coverage = pd.read_csv(ROOT / "results" / "external" / "probe_coverage.tsv", sep="\t")
    assert set(coverage["probe_source"]).issuperset({"fixed_table1_only", "historical_tcga_feature_snapshot_probe_availability_not_reverified"})
    assert not coverage["probe_source"].str.contains("promoter", case=False, na=False).any()
    assert coverage[coverage["cohort"].str.startswith("tcga")]["n_available_probes"].isna().all()
    gse = coverage[coverage["cohort"].eq("gse119526")]
    assert int(gse["n_target_probes"].sum()) == 77
    assert int(gse["n_available_probes"].sum()) == 71
    mapping = pd.read_csv(ROOT / "supplement" / "assay_mapping.tsv", sep="\t")
    assert len(mapping) == 10
    assert set(mapping["psq_array_mapping_level"]) == {"gene_or_region_level"}
    assert mapping["exact_psq_cpg_overlap_verified"].astype(str).str.lower().eq("false").all()
