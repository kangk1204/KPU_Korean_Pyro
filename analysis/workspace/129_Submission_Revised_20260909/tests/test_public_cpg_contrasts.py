from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pathlib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_public_cpg_contrasts as apc  # noqa: E402


def _write_toy_public_root(tmp_path: Path, missing_probe: str | None = None, gse77954_paired: bool = True) -> tuple[Path, Path]:
    public_root = tmp_path / "public"
    derived = public_root / "data" / "derived"
    derived.mkdir(parents=True)
    counts = {"EYA4": 17, "ZNF568": 8, "ZNF793": 8, "SFMBT2": 8, "ADHFE1": 8, "HOXA2": 8, "BEND5": 4, "UNC5C": 5, "RALYL": 5, "GFRA1": 6}
    fixed = {}
    probe_i = 1
    for gene, count in counts.items():
        fixed[gene] = [f"cg{probe_i + offset:08d}" for offset in range(count)]
        probe_i += count
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(json.dumps(fixed))
    sample_rows = []
    samples = []
    for cohort in apc.COHORTS:
        tissues = sorted({t for high, low, _ in apc.CONTRASTS[cohort] for t in (high, low)} | {"T"})
        for tissue in tissues:
            for i in range(4):
                patient = f"{cohort}_{i}" if tissue in {"T", "N"} else f"{cohort}_{tissue}_{i}"
                sample = f"{cohort}_{tissue}_{i}"
                samples.append(sample)
                sample_rows.append(
                    {
                        "cohort": cohort,
                        "sample": sample,
                        "patient": patient,
                        "tissue": tissue,
                        "pair_verified": tissue in {"T", "N"} and cohort not in {"GSE48684"} and (cohort != "GSE77954" or gse77954_paired),
                        "age": 60 + i,
                        "sex": "M" if i % 2 else "F",
                        "site": "Left" if i % 2 else "Right",
                    }
                )
        beta_rows = {}
        for gene_i, probes in enumerate(fixed.values()):
            for probe in probes:
                if probe == missing_probe:
                    continue
                values = []
                for row in sample_rows[-len(tissues) * 4 :]:
                    base = 0.10 + gene_i * 0.01 + int(row["sample"].rsplit("_", 1)[1]) * 0.001
                    shift = {"H": 0.00, "N": 0.02, "A": 0.05, "T": 0.10}.get(row["tissue"], 0.0)
                    if row["tissue"] == "T":
                        shift += int(row["sample"].rsplit("_", 1)[1]) * 0.002
                    values.append(base + shift)
                beta_rows[probe] = values
        pd.DataFrame(beta_rows, index=[r["sample"] for r in sample_rows[-len(tissues) * 4 :]]).T.to_csv(
            derived / f"{cohort}_beta.tsv.gz", sep="\t", compression="gzip"
        )
        pd.DataFrame(sample_rows[-len(tissues) * 4 :]).to_csv(derived / f"{cohort}_samples.tsv", sep="\t", index=False)
    return public_root, fixed_path



def _require_retained(path, what):
    """Skip when a retained input/evidence file is absent (clean archive checkouts; see reproducibility/INPUT_REQUIREMENTS.tsv)."""
    if not pathlib.Path(path).exists():
        pytest.skip(f"{what} not present in this checkout: {path}")

def test_real_fixed_probe_registry_has_77_unique_cpgs():
    fixed = apc.read_fixed_probes(ROOT / "registry" / "fixed_probes.json")
    targets = apc.probe_targets(fixed)

    assert len(targets) == 77
    assert len({target.probe for target in targets}) == 77
    assert not any(target.probe == target.gene for target in targets)


def test_planned_public_contrast_rows_expand_to_15_by_77():
    n_contrasts = sum(len(v) for v in apc.CONTRASTS.values())
    family_sizes = {
        family: sum(1 for values in apc.CONTRASTS.values() for _, _, fam in values if fam == family) * 77
        for family in {"healthy_reference", "tissue_replication", "lesion"}
    }

    assert n_contrasts == 15
    assert family_sizes == {"healthy_reference": 462, "tissue_replication": 462, "lesion": 231}


def test_absent_probe_keeps_planned_rows_and_uses_p_one_for_bh(tmp_path):
    public_root, fixed_path = _write_toy_public_root(tmp_path, missing_probe="cg00000077")
    contrasts, _, _, _ = apc.build_contrasts(public_root, fixed_path, n_boot=20)
    missing = contrasts.loc[contrasts["probe"].eq("cg00000077")]

    assert len(contrasts) == 15 * 77
    assert len(missing) == 15
    assert missing["status"].eq("probe_absent").all()
    assert missing["p"].isna().all()
    assert missing["p_for_bh"].eq(1.0).all()
    assert missing["q_BH_contrast_77"].notna().all()


def test_paired_tn_uses_verified_pairs_and_reports_percentage_points(tmp_path):
    public_root, fixed_path = _write_toy_public_root(tmp_path)
    contrasts, _, _, _ = apc.build_contrasts(public_root, fixed_path, n_boot=30)
    row = contrasts.loc[
        contrasts["cohort"].eq("Colonomics")
        & contrasts["contrast"].eq("T-N")
        & contrasts["probe"].eq("cg00000001")
    ].iloc[0]

    assert row["paired"]
    assert row["method"] == "paired_t"
    assert row["n_pairs"] == 4
    assert row["effect_pp"] == pytest.approx(8.3)


def test_meta_requires_three_eligible_public_cohorts_and_keeps_tn_paired_only(tmp_path):
    public_root, fixed_path = _write_toy_public_root(tmp_path)
    contrasts, _, _, _ = apc.build_contrasts(public_root, fixed_path, n_boot=30)
    # The toy cohorts have four pairs each, so the all-paired-cohort rule (>=3 pairs) is used for the
    # estimability assertions; the primary rule (>=20 pairs) is checked separately below.
    meta = apc.build_meta(contrasts, min_pairs=apc.MIN_PAIRS_SENSITIVITY)
    tn = meta.loc[meta["contrast"].eq("T-N")]
    nh = meta.loc[meta["contrast"].eq("N-H")]

    assert tn["status"].eq("estimated").all()
    assert nh["status"].eq("estimated").all()
    assert not tn["cohorts"].str.contains("GSE48684").any()
    assert tn["cohorts"].str.contains("GSE77954").all()
    assert tn["q_BH_contrast_77"].notna().all()
    # Primary rule: cohorts below the pair threshold are excluded from the T-N pool.
    assert apc.MIN_PAIRS_PRIMARY == 20
    tn_five = apc.build_meta(contrasts, min_pairs=5).loc[lambda d: d["contrast"].eq("T-N")]
    assert tn_five["status"].eq("insufficient_independent_cohorts").all()
    tn_primary = apc.build_meta(contrasts).loc[lambda d: d["contrast"].eq("T-N")]
    assert not tn_primary["cohorts"].str.contains("GSE77954").any()


def test_gse77954_pairing_comes_from_corrected_metadata_not_accession_name(tmp_path):
    old_root, fixed_path = _write_toy_public_root(tmp_path / "old", gse77954_paired=False)
    old_contrasts, _, _, _ = apc.build_contrasts(old_root, fixed_path, n_boot=20)
    old_row = old_contrasts.loc[
        old_contrasts["cohort"].eq("GSE77954")
        & old_contrasts["contrast"].eq("T-N")
        & old_contrasts["probe"].eq("cg00000001")
    ].iloc[0]
    assert not old_row["paired"]
    assert old_row["method"] == "patient_cluster_OLS"
    assert "GSE77954" not in ";".join(apc.build_meta(old_contrasts, min_pairs=apc.MIN_PAIRS_SENSITIVITY).loc[lambda d: d["contrast"].eq("T-N"), "cohorts"])

    corrected_root, fixed_path = _write_toy_public_root(tmp_path / "corrected", gse77954_paired=True)
    corrected_contrasts, _, _, _ = apc.build_contrasts(corrected_root, fixed_path, n_boot=20)
    corrected_row = corrected_contrasts.loc[
        corrected_contrasts["cohort"].eq("GSE77954")
        & corrected_contrasts["contrast"].eq("T-N")
        & corrected_contrasts["probe"].eq("cg00000001")
    ].iloc[0]
    assert corrected_row["paired"]
    assert corrected_row["method"] == "paired_t"
    assert corrected_row["n_pairs"] == 4
    assert "GSE77954" in ";".join(apc.build_meta(corrected_contrasts, min_pairs=apc.MIN_PAIRS_SENSITIVITY).loc[lambda d: d["contrast"].eq("T-N"), "cohorts"])
    assert "GSE77954" not in ";".join(apc.build_meta(corrected_contrasts, min_pairs=5).loc[lambda d: d["contrast"].eq("T-N"), "cohorts"])


def test_default_124_public_inputs_have_gse77954_four_verified_source_label_pairs():
    _require_retained(ROOT / "data" / "public_inputs" / "data" / "derived" / "GSE77954_beta.tsv.gz", "prepared GSE77954 beta matrix")
    beta, meta = apc.load_cohort("GSE77954")
    verified = meta.loc[meta["pair_verified"] & meta["tissue"].isin(["T", "N"])]
    shared = sorted(
        set(verified.loc[verified["tissue"].eq("T"), "patient"])
        & set(verified.loc[verified["tissue"].eq("N"), "patient"] )
    )
    assert beta.shape[1] == len(meta) == 48
    assert shared == ["GSE77954_CCX02", "GSE77954_CCX04", "GSE77954_CCX05", "GSE77954_CCX06"]
    assert apc.is_paired(meta, "T", "N")



def test_run_analysis_writes_contract_manifest_and_three_korean_reports(tmp_path):
    public_root, fixed_path = _write_toy_public_root(tmp_path)
    outdir = tmp_path / "out"
    contract = tmp_path / "contract.json"

    contrasts, meta = apc.run_analysis(public_root, outdir, fixed_path, contract, n_boot=15)
    manifest = json.loads((outdir / "public_contrast_manifest.json").read_text())
    contract_data = json.loads(contract.read_text())

    assert len(contrasts) == 15 * 77
    assert len(meta) > 0
    assert manifest["status"] == "complete"
    assert (outdir / apc.META_SENSITIVITY_NAME).exists()
    assert contract_data["estimand"] == "individual CpG beta difference; no gene means, no panel means"
    for name in [
        "공개코호트_CpG_분석요약.md",
        "공개코호트_CpG_meta_요약.md",
        "공개코호트_CpG_coverage_요약.md",
    ]:
        assert (outdir / name).exists()


def test_korean_only_meta_computes_se_without_rewriting_primary_inputs(tmp_path):
    root = tmp_path / "individual_cpg"
    rows = []
    for i in range(77):
        rows.append(
            {
                "cohort": "",
                "probe": f"cg{i + 1:08d}",
                "gene": "EYA4" if i < 17 else "ZNF568",
                "n_pairs": 25,
                "mean_delta_pp": 10.0 + i * 0.01,
                "sd_delta_pp": 5.0,
                "paired_t_p": 0.01,
                "status": "estimated",
            }
        )
    for cohort in ["CMCBSN", "SNUH", "ASAN"]:
        d = root / cohort
        d.mkdir(parents=True)
        df = pd.DataFrame(rows)
        df["cohort"] = cohort
        df.to_csv(d / "paired_cpg_effects.tsv", sep="\t", index=False)

    meta = apc.build_korean_only_meta(root)

    assert len(meta) == 77
    assert meta["status"].eq("estimated").all()
    assert meta["cohorts"].eq("CMCBSN;SNUH;ASAN").all()
    assert meta["q_BH_77"].notna().all()
    for cohort in ["CMCBSN", "SNUH", "ASAN"]:
        df = pd.read_csv(root / cohort / "paired_cpg_effects.tsv", sep="\t")
        assert "se_delta_pp" not in df.columns
