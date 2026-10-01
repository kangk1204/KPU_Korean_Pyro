#!/usr/bin/env python3
"""Part B (B1-B5): biological identity of the coordination axis.

Pre-specified in PLAN.md (127_Exploratory_PanelSize_Coordination_20260909).
Inputs are read-only; every output lands under B_coordination_axis/.
"""
from __future__ import annotations
import os

import datetime as dt
import json
import sys
import zipfile  # noqa: F401  (kept for provenance parity with 124 TCGA reader)
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from coordination_axis_lib import (  # noqa: E402
    MIN_GROUP, N_BOOT, SEED, bh, boot_spearman, build_axis, rec, sha256,
    test_kruskal, test_spearman, test_two_group,
)

WORKSPACE_ROOT = Path(os.environ.get("KPU_WORKSPACE_ROOT", Path(__file__).resolve().parents[3]))
SOURCE_ROOT = Path(os.environ.get("KPU_SOURCE_ROOT", os.environ.get("KPU_PROJECT_ROOT", WORKSPACE_ROOT)))
OUTDIR = WORKSPACE_ROOT / "127_Exploratory_PanelSize_Coordination_20260909" / "B_coordination_axis"
RESULTS = OUTDIR / "results"
D124 = SOURCE_ROOT / "124_Integrated_Revision_20260909" / "data" / "derived"
D120 = SOURCE_ROOT / "120_Korean_External_Validation_20260908"
D102 = SOURCE_ROOT / "102_ML_revised_20260902" / "external" / "data" / "tcga"
FIXED_PROBES = SOURCE_ROOT / "124_Integrated_Revision_20260909" / "registry" / "fixed_probes.json"
CLIN_RAW = SOURCE_ROOT / "115_Public_Biology_20260905" / "data" / "raw" / "Colonomics_CLX_ClinicalData.tab"


def resolve_public_input_derived() -> Path:
    explicit = os.environ.get("KPU_PUBLIC_INPUT_DERIVED")
    if explicit:
        candidate = Path(explicit)
        if (candidate / "GSE77954_samples.tsv").exists():
            return candidate
        raise FileNotFoundError(
            "KPU_PUBLIC_INPUT_DERIVED was set, but it does not contain GSE77954_samples.tsv: "
            f"{candidate}"
        )
    candidates = [
        SOURCE_ROOT
        / "129_Submission_Revised_20260909"
        / "data"
        / "public_inputs"
        / "data"
        / "derived",
        SOURCE_ROOT
        / "129_Submission_Revised_20260909"
        / "129_Submission_Revised_20260909"
        / "data"
        / "public_inputs"
        / "data"
        / "derived",
        WORKSPACE_ROOT
        / "129_Submission_Revised_20260909"
        / "data"
        / "public_inputs"
        / "data"
        / "derived",
    ]
    for candidate in candidates:
        if (candidate / "GSE77954_samples.tsv").exists():
            return candidate
    msg = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Corrected 129 public input root was not found. Set KPU_PUBLIC_INPUT_DERIVED "
        "to the data/public_inputs/data/derived directory.\nChecked:\n" + msg
    )


PUBLIC_INPUT_DERIVED = resolve_public_input_derived()

np.random.seed(SEED)

INPUTS: list[Path] = []
CHECKS: dict = {}


def track(p: Path) -> Path:
    if p not in INPUTS:
        INPUTS.append(p)
    return p


def logical_path(p: Path) -> str:
    resolved = p.resolve()
    for root in (SOURCE_ROOT.resolve(), WORKSPACE_ROOT.resolve()):
        try:
            return str(resolved.relative_to(root))
        except ValueError:
            continue
    return p.name


# --------------------------------------------------------------- site coding
RIGHT_TERMS = {"cecum", "ascending", "hepatic flexure", "transverse", "right"}
LEFT_TERMS = {"splenic flexure", "descending", "sigmoid", "rectosigmoid", "rectal",
              "rectum", "left"}


def code_site(value) -> float | str:
    if not isinstance(value, str):
        return np.nan
    v = value.strip().lower()
    if v in RIGHT_TERMS:
        return "Right"
    if v in LEFT_TERMS:
        return "Left"
    return np.nan


def code_stage(value) -> float | str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return np.nan
    s = str(value).strip().upper()
    if s in {"", "NAN", "NA"}:
        return np.nan
    if s.startswith("IV") or s in {"4"}:
        return "III-IV"
    if s.startswith("III") or s in {"3"}:
        return "III-IV"
    if s.startswith("II") or s in {"2"}:
        return "I-II"
    if s.startswith("I") or s in {"1"}:
        return "I-II"
    return np.nan


# ------------------------------------------------------------ axis machinery
def complete_cpgs(beta: pd.DataFrame, samples: list[str]) -> list[str]:
    sub = beta.loc[:, samples]
    ok = sub.notna().all(axis=1) & (sub.std(axis=1, ddof=1) > 0)
    return list(sub.index[ok])


def cohort_axes(name: str, beta: pd.DataFrame, meta: pd.DataFrame,
                sample_col: str, patient_col: str, tissue_col: str) -> dict:
    """beta: CpG x sample. meta: one row per sample.

    Returns per-sample tumor axis frame, per-patient delta axis frame (when
    pairs exist), and the axis diagnostics.
    """
    meta = meta[meta[sample_col].isin(beta.columns)].copy()
    tum = meta[meta[tissue_col] == "T"]
    nor = meta[meta[tissue_col] == "N"]

    out = {"cohort": name, "diagnostics": {}}

    # ---- tumor axis (all tumors) --------------------------------------
    tum_ids = list(tum[sample_col])
    cpgs_t = complete_cpgs(beta, tum_ids)
    mat_t = beta.loc[cpgs_t, tum_ids].T
    mean_t = mat_t.mean(axis=1).to_numpy()
    ax_t = build_axis(mat_t, mean_t)
    tumor_df = pd.DataFrame({
        sample_col: tum_ids,
        "patient": list(tum[patient_col]),
        "tumor_axis": ax_t["scores"],
        "mean_tumor_beta": mean_t,
    })
    out["tumor_df"] = tumor_df
    out["diagnostics"]["tumor_axis"] = {
        "n_sample": ax_t["n_sample"], "n_cpg": ax_t["n_cpg"],
        "pc1_var_explained": ax_t["var_explained"], "sign_flipped": ax_t["flipped"],
        "spearman_vs_mean_tumor_beta": ax_t["orient_rho"],
    }
    out["tumor_loadings"] = ax_t["loadings"]

    # ---- delta axis (pairs) -------------------------------------------
    shared = sorted(set(tum[patient_col]) & set(nor[patient_col]))
    # keep one tumor / one normal per patient (first by sample id) for the delta
    if len(shared) >= 10:
        t_first = tum.drop_duplicates(patient_col).set_index(patient_col)
        n_first = nor.drop_duplicates(patient_col).set_index(patient_col)
        t_ids = [t_first.loc[p, sample_col] for p in shared]
        n_ids = [n_first.loc[p, sample_col] for p in shared]
        cpgs_d = sorted(set(complete_cpgs(beta, t_ids)) & set(complete_cpgs(beta, n_ids)))
        d = (beta.loc[cpgs_d, t_ids].to_numpy().T - beta.loc[cpgs_d, n_ids].to_numpy().T)
        d = pd.DataFrame(d, index=shared, columns=cpgs_d)
        mean_tp = beta.loc[cpgs_d, t_ids].mean(axis=0).to_numpy()
        ax_d = build_axis(d, mean_tp)
        delta_df = pd.DataFrame({
            "patient": shared,
            "delta_axis": ax_d["scores"],
            "mean_tumor_beta": mean_tp,
            "mean_delta_beta": d.mean(axis=1).to_numpy(),
        })
        delta_df = delta_df.merge(
            tumor_df[["patient", "tumor_axis"]].drop_duplicates("patient"),
            on="patient", how="left")
        out["delta_df"] = delta_df
        out["diagnostics"]["delta_axis"] = {
            "n_sample": ax_d["n_sample"], "n_pair": ax_d["n_sample"], "n_cpg": ax_d["n_cpg"],
            "pc1_var_explained": ax_d["var_explained"], "sign_flipped": ax_d["flipped"],
            "spearman_vs_mean_tumor_beta": ax_d["orient_rho"],
        }
        out["delta_loadings"] = ax_d["loadings"]
    else:
        out["delta_df"] = None
        out["diagnostics"]["delta_axis"] = {
            "n_pair": len(shared),
            "note": "fewer than 10 verified pairs; delta axis not computed",
        }
    return out


def axis_agreement(name: str, df: pd.DataFrame, axes: list[str]) -> list[dict]:
    rows = []
    for i, a in enumerate(axes):
        for b in axes[i + 1:]:
            ok = df[a].notna() & df[b].notna()
            if ok.sum() < MIN_GROUP:
                continue
            r, lo, hi = boot_spearman(df.loc[ok, a].to_numpy(),
                                      df.loc[ok, b].to_numpy(), f"agree|{name}|{a}|{b}")
            rows.append({"cohort": name, "axis_a": a, "axis_b": b, "n": int(ok.sum()),
                         "spearman_rho": r, "ci_lo": lo, "ci_hi": hi})
    return rows


# ------------------------------------------------------------------ loaders
def load_beta(path: Path) -> pd.DataFrame:
    return pd.read_csv(track(path), sep="\t", index_col=0)


def validate_gse77954_corrected_pairs(meta: pd.DataFrame, path: Path) -> None:
    required = {"sample", "patient", "tissue", "pair_verified", "source_pair_code"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            f"{path}: GSE77954 metadata are stale or incomplete; missing {sorted(missing)}"
        )
    pair_verified = meta["pair_verified"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    verified = meta.loc[pair_verified & meta["tissue"].isin(["T", "N"])]
    shared = sorted(
        set(verified.loc[verified["tissue"].eq("T"), "patient"])
        & set(verified.loc[verified["tissue"].eq("N"), "patient"])
    )
    expected = ["GSE77954_CCX02", "GSE77954_CCX04", "GSE77954_CCX05", "GSE77954_CCX06"]
    if shared != expected:
        raise ValueError(
            f"{path}: expected four GEO description-verified GSE77954 pairs {expected}, "
            f"got {shared}"
        )


def load_colonomics():
    beta = load_beta(PUBLIC_INPUT_DERIVED / "Colonomics_beta.tsv.gz")
    meta = pd.read_csv(track(PUBLIC_INPUT_DERIVED / "Colonomics_samples.tsv"), sep="\t")
    meta = meta[(~meta["excluded"].astype(bool)) & meta["tissue"].isin(["T", "N"])].copy()
    meta["site2"] = meta["site"].map(code_site)
    meta["stage2"] = meta["stage"].map(code_stage)
    return beta, meta


def load_public(name: str):
    beta = load_beta(PUBLIC_INPUT_DERIVED / f"{name}_beta.tsv.gz")
    meta_path = PUBLIC_INPUT_DERIVED / f"{name}_samples.tsv"
    meta = pd.read_csv(track(meta_path), sep="\t")
    if name == "GSE77954":
        validate_gse77954_corrected_pairs(meta, meta_path)
    meta = meta[meta["tissue"].isin(["T", "N"])].copy()
    meta["site2"] = meta["site"].map(code_site)
    meta["stage2"] = meta["stage"].map(code_stage)
    return beta, meta


def load_cmcbsn():
    beta = pd.read_csv(track(D124 / "CMCBSN_beta.tsv"), sep="\t", index_col=0)
    meta = pd.read_csv(track(D124 / "CMCBSN_samples.tsv"), sep="\t")
    cms = pd.read_csv(
        track(D120 / "review" / "cmc_metadata" / "CMCBSN_cms_source_labels.tsv"), sep="\t")
    # join rule taken from 124/scripts/analyze_korean_context.py: exact sample_id match
    meta = meta.merge(cms[["sample_id", "CMS_source", "CMS_label_status"]],
                      on="sample_id", how="left")
    rna_path = D120 / "data" / "derived" / "CMCBSN_rna_samples.tsv"
    if rna_path.exists():
        rna = pd.read_csv(track(rna_path), sep="\t")
        keep = ["sample_id", "age", "sex", "site", "Stage", "MSI"]
        meta = meta.merge(rna[keep], on="sample_id", how="left")
    meta["cms"] = meta["CMS_source"].where(
        meta["CMS_source"].isin(["CMS1", "CMS2", "CMS3", "CMS4"]))
    meta["site2"] = meta.get("site", pd.Series(index=meta.index, dtype=object)).map(code_site)
    meta["stage2"] = meta.get("Stage", pd.Series(index=meta.index, dtype=object)).map(code_stage)
    meta["msi2"] = meta.get("MSI", pd.Series(index=meta.index, dtype=object)).map(
        lambda v: "MSI-H" if v == "MSI-H" else ("MSS/MSI-L" if v in ("MSS", "MSI-L") else np.nan))
    meta["sex"] = meta.get("sex", pd.Series(index=meta.index, dtype=object)).map(
        {"Male": "M", "Female": "F"})
    return beta, meta


def load_tcga():
    """Case x CpG tumor beta from the 124 derived table (streamed from beta_archived.zip)."""
    long = pd.read_csv(track(D124 / "TCGA_fixed77_case_tissue_beta.tsv"), sep="\t")
    wide = long.pivot_table(index=["case", "tissue"], columns="cpg", values="beta")
    wide = wide.reset_index()
    meta = wide[["case", "tissue"]].copy()
    meta.columns = ["patient", "tissue"]
    meta["sample"] = meta["patient"] + "|" + meta["tissue"]
    beta = wide.drop(columns=["case", "tissue"]).T
    beta.columns = meta["sample"]
    clin = pd.read_csv(track(D102 / "clinical_PANCAN_patient_with_followup.tsv"),
                       sep="\t", dtype=str, low_memory=False, encoding="latin-1")
    clin = clin[clin["acronym"].isin(["COAD", "READ"])]
    clin = clin.drop_duplicates("bcr_patient_barcode")
    miss = {"[Not Available]", "[Unknown]", "[Not Applicable]", "[Not Evaluated]",
            "[Discrepancy]", "", "NA"}
    def clean(s):
        return s.where(~s.isin(miss))
    lab = pd.DataFrame({
        "patient": clin["bcr_patient_barcode"],
        "msi": clean(clin["microsatellite_instability"]).map(
            {"YES": "MSI-positive", "NO": "MSI-negative"}),
        "braf": clean(clin["braf_gene_analysis_result"]).map(
            {"Abnormal": "BRAF abnormal", "Normal": "BRAF normal"}),
        "kras": clean(clin["kras_mutation_found"]).map(
            {"YES": "KRAS mutated", "NO": "KRAS wild-type"}),
        "sex": clean(clin["gender"]).map({"MALE": "M", "FEMALE": "F"}),
        "age_raw": pd.to_numeric(clin.get("age_at_initial_pathologic_diagnosis"),
                                 errors="coerce"),
        "stage_raw": clean(clin.get("pathologic_stage", pd.Series(dtype=str))),
        "site_raw": clean(clin.get("anatomic_neoplasm_subdivision", pd.Series(dtype=str))),
    })
    lab["site2"] = lab["site_raw"].map(code_site)
    lab["stage2"] = lab["stage_raw"].map(
        lambda v: code_stage(str(v).replace("Stage ", "")) if isinstance(v, str) else np.nan)
    # documented check: TCGA-CDR carries no MSI / CIMP / BRAF / hypermutation field
    cdr = pd.ExcelFile(track(D102 / "TCGA-CDR-SupplementalTableS1.xlsx"))
    cdr_hits = [c for sh in cdr.sheet_names for c in map(str, cdr.parse(sh, nrows=0).columns)
                if any(k in c.lower() for k in ("msi", "cimp", "braf", "microsat", "hypermut"))]
    CHECKS["tcga_cdr_msi_cimp_braf_columns"] = cdr_hits
    CHECKS["tcga_pancan_cimp_columns"] = [c for c in clin.columns if "cimp" in c.lower()]
    meta = meta.merge(lab, on="patient", how="left")
    meta["age"] = lab.set_index("patient")["age_raw"].reindex(meta["patient"]).to_numpy()
    return beta, meta


# ------------------------------------------------------------------ analyses
def colonomics_tests(df: pd.DataFrame, axis: str, in_family: bool, cohort="Colonomics"):
    v = df[axis].to_numpy(float)
    out = []
    out.append(test_two_group(v, df["BRAF_V600E"], "Yes", "No", cohort=cohort, axis=axis,
                              label="BRAF V600E", comparison="BRAF V600E mutant vs wild-type",
                              in_family=in_family))
    out.append(test_two_group(v, df["KRAS_mutated"], "Yes", "No", cohort=cohort, axis=axis,
                              label="KRAS", comparison="KRAS mutated vs wild-type",
                              in_family=in_family))
    out.append(test_kruskal(v, df["cms"], ["CMS1", "CMS2", "CMS3", "CMS4"], cohort=cohort,
                            axis=axis, label="CMS", comparison="CMS four groups",
                            in_family=in_family))
    cms1 = np.where(df["cms"].isin(["CMS1", "CMS2", "CMS3", "CMS4"]),
                    np.where(df["cms"].eq("CMS1"), "CMS1", "CMS2-4"), None)
    out.append(test_two_group(v, cms1, "CMS1", "CMS2-4", cohort=cohort, axis=axis,
                              label="CMS1", comparison="CMS1 vs CMS2-4", in_family=in_family))
    out.append(test_two_group(v, df["site2"], "Right", "Left", cohort=cohort, axis=axis,
                              label="Site", comparison="right- vs left-sided",
                              in_family=in_family))
    out.append(test_two_group(v, df["stage2"], "III-IV", "I-II", cohort=cohort, axis=axis,
                              label="Stage", comparison="stage III-IV vs I-II",
                              in_family=in_family))
    out.append(test_two_group(v, df["sex"], "M", "F", cohort=cohort, axis=axis,
                              label="Sex", comparison="male vs female", in_family=in_family))
    out.append(test_spearman(v, df["age"], cohort=cohort, axis=axis, label="Age",
                             comparison="age (years)", in_family=in_family))
    out.append(test_spearman(v, df["stromal_score"], cohort=cohort, axis=axis,
                             label="Stromal score", comparison="ESTIMATE-style stromal score",
                             in_family=in_family))
    return out


def multivariable(df: pd.DataFrame, axis: str, cohort: str, terms=None,
                  model_name="pre-specified"):
    d = df.copy()
    d["y"] = d[axis]
    d["braf"] = d["BRAF_V600E"].map({"Yes": 1.0, "No": 0.0})
    d["cms1"] = np.where(d["cms"].isin(["CMS1", "CMS2", "CMS3", "CMS4"]),
                         d["cms"].eq("CMS1").astype(float), np.nan)
    d["right"] = d["site2"].map({"Right": 1.0, "Left": 0.0})
    d["male"] = d["sex"].map({"M": 1.0, "F": 0.0})
    d["stromal_z"] = (d["stromal_score"] - d["stromal_score"].mean()) / d["stromal_score"].std(ddof=1)
    d["age_z"] = (d["age"] - d["age"].mean()) / d["age"].std(ddof=1)
    terms = terms or ["braf", "cms1", "right", "stromal_z", "age_z", "male"]
    cols = ["y"] + terms
    d = d[cols].dropna()
    X = sm.add_constant(d[terms])
    fit = sm.OLS(d["y"], X).fit(cov_type="HC3")
    rows = []
    ci = fit.conf_int()
    for term in X.columns:
        rows.append({
            "cohort": cohort, "axis": axis, "model": model_name, "term": term,
            "coef": float(fit.params[term]), "robust_se_HC3": float(fit.bse[term]),
            "t": float(fit.tvalues[term]), "p_value": float(fit.pvalues[term]),
            "ci_lo": float(ci.loc[term, 0]), "ci_hi": float(ci.loc[term, 1]),
            "n": int(fit.nobs), "r_squared": float(fit.rsquared),
        })
    return pd.DataFrame(rows), d


def main():
    started = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gene_by_cpg = {}
    probes = json.loads(track(FIXED_PROBES).read_text())
    for g, ps in probes.items():
        for p in ps:
            gene_by_cpg[p] = g

    tests: list[dict] = []
    agree: list[dict] = []
    diagnostics: dict = {}
    axis_tables: list[pd.DataFrame] = []
    loadings: list[pd.DataFrame] = []
    descriptives: list[dict] = []

    def add_loadings(cohort, axis, series):
        loadings.append(pd.DataFrame({
            "cohort": cohort, "axis": axis, "cpg": series.index,
            "gene": [gene_by_cpg.get(c, "NA") for c in series.index],
            "loading": series.to_numpy()}))

    # ================================================== B1 Colonomics
    beta, meta = load_colonomics()
    ax = cohort_axes("Colonomics", beta, meta, "sample", "patient", "tissue")
    diagnostics["Colonomics"] = ax["diagnostics"]
    add_loadings("Colonomics", "delta_axis", ax["delta_loadings"])
    add_loadings("Colonomics", "tumor_axis", ax["tumor_loadings"])
    tum_meta = meta[meta["tissue"] == "T"].drop_duplicates("patient").set_index("patient")
    col = ax["delta_df"].copy()
    for c in ["BRAF_V600E", "KRAS_mutated", "cms", "site2", "stage2", "sex", "age",
              "stromal_score"]:
        col[c] = tum_meta[c].reindex(col["patient"]).to_numpy()
    col["cohort"] = "Colonomics"
    col["axis_role"] = "delta_axis (coordination axis)"
    axis_tables.append(col)

    tests += colonomics_tests(col, "delta_axis", True)
    tests += colonomics_tests(col, "tumor_axis", False)
    tests += colonomics_tests(col, "mean_tumor_beta", False)
    agree += axis_agreement("Colonomics", col,
                            ["delta_axis", "tumor_axis", "mean_tumor_beta", "mean_delta_beta"])
    mv, mv_data = multivariable(col, "delta_axis", "Colonomics")
    mv_tum, _ = multivariable(col, "tumor_axis", "Colonomics")
    # sensitivity: BRAF (3 carriers, all right-sided, 2 of them CMS1) and CMS1 (n=5)
    # are near-collinear and inflate the robust SEs; refit without them.
    red_terms = ["right", "stromal_z", "age_z", "male"]
    mv_red, _ = multivariable(col, "delta_axis", "Colonomics", red_terms,
                              "sensitivity: BRAF and CMS1 dropped (near-collinear, n<5)")
    mv_red_t, _ = multivariable(col, "tumor_axis", "Colonomics", red_terms,
                                "sensitivity: BRAF and CMS1 dropped (near-collinear, n<5)")
    mv_all = pd.concat([mv, mv_tum, mv_red, mv_red_t], ignore_index=True)

    # cross-check the 119 clinical table against the Colonomics raw clinical file
    raw = pd.read_csv(track(CLIN_RAW), sep="\t")
    raw_t = raw[raw["type"] == "Tumor"].set_index("id_clx_individual")
    chk = []
    for src, dst in [("BRAF_V600E", "BRAF_V600E"), ("KRAS_mutated", "KRAS_mutated"),
                     ("CMS", "cms"), ("stromal_score", "stromal_score"), ("age", "age")]:
        a = raw_t[src].reindex(col["patient"])
        b = col[dst]
        if src in ("stromal_score", "age"):
            ok = np.isclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=1e-6)
        else:
            ok = (a.to_numpy().astype(str) == b.to_numpy().astype(str))
        chk.append({"field": src, "n": int(len(ok)), "n_match": int(np.sum(ok)),
                    "all_match": bool(np.all(ok))})
    crosscheck = pd.DataFrame(chk)

    # ================================================== B2 MATCH / GSE164811
    beta, meta = load_public("GSE164811")
    ax = cohort_axes("GSE164811", beta, meta, "sample", "patient", "tissue")
    diagnostics["GSE164811"] = ax["diagnostics"]
    add_loadings("GSE164811", "tumor_axis", ax["tumor_loadings"])
    tm = meta[meta["tissue"] == "T"].set_index("sample")
    g = ax["tumor_df"].copy()
    for c in ["cms", "site2", "stage2", "sex"]:
        g[c] = tm[c].reindex(g["sample"]).to_numpy()
    g["cohort"] = "GSE164811 (MATCH)"
    g["axis_role"] = "tumor_axis (coordination axis)"
    axis_tables.append(g)
    v = g["tumor_axis"].to_numpy(float)
    tests.append(test_two_group(v, g["cms"], "CMS3", "CMS2", cohort="GSE164811",
                                axis="tumor_axis", label="CMS", comparison="CMS3 vs CMS2"))
    tests.append(test_kruskal(v, g["cms"], ["CMS1", "CMS2", "CMS3", "CMS4"],
                              cohort="GSE164811", axis="tumor_axis", label="CMS",
                              comparison="CMS groups present"))
    tests.append(test_two_group(v, g["site2"], "Right", "Left", cohort="GSE164811",
                                axis="tumor_axis", label="Site",
                                comparison="right- vs left-sided"))
    tests.append(test_two_group(v, g["sex"], "M", "F", cohort="GSE164811",
                                axis="tumor_axis", label="Sex", comparison="male vs female"))
    tests.append(test_two_group(g["mean_tumor_beta"].to_numpy(float), g["cms"], "CMS3", "CMS2",
                                cohort="GSE164811", axis="mean_tumor_beta", label="CMS",
                                comparison="CMS3 vs CMS2", in_family=False))
    agree += axis_agreement("GSE164811", g, ["tumor_axis", "mean_tumor_beta"])

    # ================================================== B3 CMCBSN
    beta, meta = load_cmcbsn()
    ax = cohort_axes("CMCBSN", beta, meta, "sample_id", "patient_id", "tissue")
    diagnostics["CMCBSN"] = ax["diagnostics"]
    add_loadings("CMCBSN", "delta_axis", ax["delta_loadings"])
    add_loadings("CMCBSN", "tumor_axis", ax["tumor_loadings"])
    tmeta = meta[meta["tissue"] == "T"].drop_duplicates("patient_id").set_index("patient_id")
    cm = ax["delta_df"].copy()
    for c in ["cms", "site2", "stage2", "sex", "age", "msi2"]:
        cm[c] = tmeta[c].reindex(cm["patient"]).to_numpy()
    cm["cohort"] = "CMCBSN"
    cm["axis_role"] = "delta_axis (coordination axis)"
    axis_tables.append(cm)

    tm_s = meta[meta["tissue"] == "T"].set_index("sample_id")
    cmt = ax["tumor_df"].copy()
    for c in ["cms", "site2", "stage2", "sex", "age", "msi2"]:
        cmt[c] = tm_s[c].reindex(cmt["sample_id"]).to_numpy()
    cmt["cohort"] = "CMCBSN"
    cmt["axis_role"] = "tumor_axis (all tumors)"
    axis_tables.append(cmt)

    for frame, axname, fam in [(cm, "delta_axis", True), (cmt, "tumor_axis", False)]:
        v = frame[axname].to_numpy(float)
        tests.append(test_kruskal(v, frame["cms"], ["CMS1", "CMS2", "CMS3", "CMS4"],
                                  cohort="CMCBSN", axis=axname, label="CMS",
                                  comparison="CMS four groups", in_family=fam))
        c1 = np.where(frame["cms"].isin(["CMS1", "CMS2", "CMS3", "CMS4"]),
                      np.where(frame["cms"].eq("CMS1"), "CMS1", "CMS2-4"), None)
        tests.append(test_two_group(v, c1, "CMS1", "CMS2-4", cohort="CMCBSN", axis=axname,
                                    label="CMS1", comparison="CMS1 vs CMS2-4", in_family=fam))
        tests.append(test_two_group(v, frame["site2"], "Right", "Left", cohort="CMCBSN",
                                    axis=axname, label="Site",
                                    comparison="right- vs left-sided", in_family=False,
                                    note="not in the pre-specified B3 family; site added from "
                                         "120 CMCBSN_rna_samples.tsv"))
        tests.append(test_two_group(v, frame["msi2"], "MSI-H", "MSS/MSI-L", cohort="CMCBSN",
                                    axis=axname, label="MSI", comparison="MSI-H vs MSS/MSI-L",
                                    in_family=False,
                                    note="post-hoc: MSI present in 120 CMCBSN_rna_samples.tsv, "
                                         "not anticipated by PLAN B5"))
    agree += axis_agreement("CMCBSN", cm, ["delta_axis", "tumor_axis", "mean_tumor_beta",
                                           "mean_delta_beta"])

    # ================================================== B4 other public cohorts
    for name in ["GSE48684", "GSE77954", "GSE193535", "GSE77718", "GSE42752"]:
        beta, meta = load_public(name)
        ax = cohort_axes(name, beta, meta, "sample", "patient", "tissue")
        diagnostics[name] = ax["diagnostics"]
        add_loadings(name, "tumor_axis", ax["tumor_loadings"])
        has_delta = ax["delta_df"] is not None
        if has_delta:
            add_loadings(name, "delta_axis", ax["delta_loadings"])
        tm = meta[meta["tissue"] == "T"].drop_duplicates("patient").set_index("patient")
        if has_delta:
            frame = ax["delta_df"].copy()
            axname = "delta_axis"
            role = "delta_axis (coordination axis)"
            for c in ["site2", "stage2", "sex", "age"]:
                frame[c] = tm[c].reindex(frame["patient"]).to_numpy()
        else:
            frame = ax["tumor_df"].copy()
            axname = "tumor_axis"
            role = "tumor_axis (coordination axis)"
            tms = meta[meta["tissue"] == "T"].set_index("sample")
            for c in ["site2", "stage2", "sex", "age"]:
                frame[c] = tms[c].reindex(frame["sample"]).to_numpy()
        frame["cohort"] = name
        frame["axis_role"] = role
        axis_tables.append(frame)
        v = frame[axname].to_numpy(float)
        tests.append(test_two_group(v, frame["site2"], "Right", "Left", cohort=name,
                                    axis=axname, label="Site",
                                    comparison="right- vs left-sided"))
        tests.append(test_two_group(v, frame["stage2"], "III-IV", "I-II", cohort=name,
                                    axis=axname, label="Stage",
                                    comparison="stage III-IV vs I-II", in_family=False))
        tests.append(test_two_group(v, frame["sex"], "M", "F", cohort=name, axis=axname,
                                    label="Sex", comparison="male vs female", in_family=False))
        agree += axis_agreement(name, frame,
                                [axname, "mean_tumor_beta"] if not has_delta else
                                [axname, "tumor_axis", "mean_tumor_beta"])

    # ================================================== B4 TCGA
    beta, meta = load_tcga()
    ax = cohort_axes("TCGA-COADREAD", beta, meta, "sample", "patient", "tissue")
    diagnostics["TCGA-COADREAD"] = ax["diagnostics"]
    add_loadings("TCGA-COADREAD", "tumor_axis", ax["tumor_loadings"])
    if ax["delta_df"] is not None:
        add_loadings("TCGA-COADREAD", "delta_axis", ax["delta_loadings"])
    tms = meta[meta["tissue"] == "T"].set_index("sample")
    tg = ax["tumor_df"].copy()
    for c in ["msi", "braf", "kras", "sex", "site2", "stage2", "age"]:
        tg[c] = tms[c].reindex(tg["sample"]).to_numpy()
    tg["cohort"] = "TCGA-COADREAD"
    tg["axis_role"] = "tumor_axis (coordination axis)"
    axis_tables.append(tg)
    v = tg["tumor_axis"].to_numpy(float)
    tests.append(test_two_group(v, tg["msi"], "MSI-positive", "MSI-negative",
                                cohort="TCGA-COADREAD", axis="tumor_axis", label="MSI",
                                comparison="MSI-positive vs MSI-negative"))
    tests.append(test_two_group(v, tg["braf"], "BRAF abnormal", "BRAF normal",
                                cohort="TCGA-COADREAD", axis="tumor_axis", label="BRAF",
                                comparison="BRAF abnormal vs normal"))
    tests.append(test_two_group(v, tg["kras"], "KRAS mutated", "KRAS wild-type",
                                cohort="TCGA-COADREAD", axis="tumor_axis", label="KRAS",
                                comparison="KRAS mutated vs wild-type"))
    tests.append(test_two_group(v, tg["site2"], "Right", "Left", cohort="TCGA-COADREAD",
                                axis="tumor_axis", label="Site",
                                comparison="right- vs left-sided"))
    tests.append(test_two_group(v, tg["sex"], "M", "F", cohort="TCGA-COADREAD",
                                axis="tumor_axis", label="Sex", comparison="male vs female",
                                in_family=False))
    tests.append(test_spearman(v, tg["age"], cohort="TCGA-COADREAD", axis="tumor_axis",
                               label="Age", comparison="age (years)", in_family=False))
    tests.append(test_two_group(tg["mean_tumor_beta"].to_numpy(float), tg["msi"],
                                "MSI-positive", "MSI-negative", cohort="TCGA-COADREAD",
                                axis="mean_tumor_beta", label="MSI",
                                comparison="MSI-positive vs MSI-negative", in_family=False))
    agree += axis_agreement("TCGA-COADREAD", tg, ["tumor_axis", "mean_tumor_beta"])

    # descriptive summary of every label used
    for frame in axis_tables:
        coh = frame["cohort"].iloc[0]
        role = frame["axis_role"].iloc[0]
        for c in frame.columns:
            if c in {"cohort", "axis_role", "patient", "sample", "sample_id",
                     "delta_axis", "tumor_axis", "mean_tumor_beta", "mean_delta_beta"}:
                continue
            s = frame[c]
            if s.dtype.kind in "OSU":
                for lv, n in s.value_counts(dropna=True).items():
                    descriptives.append({"cohort": coh, "axis_role": role, "label": c,
                                         "level": str(lv), "n": int(n)})
                descriptives.append({"cohort": coh, "axis_role": role, "label": c,
                                     "level": "missing", "n": int(s.isna().sum())})
            else:
                descriptives.append({"cohort": coh, "axis_role": role, "label": c,
                                     "level": "n_non_missing", "n": int(s.notna().sum())})

    # ------------------------------------------------------------ BH within cohort
    tdf = pd.DataFrame(tests)
    tdf["q_value"] = np.nan
    for coh, idx in tdf.groupby("cohort").groups.items():
        sel = tdf.loc[idx]
        fam = sel[sel["in_family"] & sel["tested"]]
        if len(fam):
            tdf.loc[fam.index, "q_value"] = bh(fam["p_value"].to_numpy())
        sec = sel[(~sel["in_family"]) & sel["tested"]]
        if len(sec):
            tdf.loc[sec.index, "q_value"] = bh(sec["p_value"].to_numpy())
    tdf["family"] = np.where(tdf["in_family"], "pre-specified (BH within cohort)",
                             "secondary/exploratory (BH within cohort)")

    # ------------------------------------------------------------ write outputs
    RESULTS.mkdir(parents=True, exist_ok=True)
    order = ["cohort", "axis", "family", "label", "comparison", "test", "n_total",
             "group_sizes", "group_medians", "statistic", "p_value", "q_value",
             "effect_name", "effect", "ci_lo", "ci_hi", "effect_r", "tested", "note"]
    tdf[order].to_csv(RESULTS / "B_axis_tests.tsv", sep="\t", index=False)
    pd.DataFrame(agree).to_csv(RESULTS / "B_axis_agreement.tsv", sep="\t", index=False)
    mv_all.to_csv(RESULTS / "B_multivariable_colonomics.tsv", sep="\t", index=False)
    pd.DataFrame(descriptives).to_csv(RESULTS / "B_label_counts.tsv", sep="\t", index=False)
    pd.concat(loadings, ignore_index=True).to_csv(RESULTS / "B_axis_loadings.tsv",
                                                  sep="\t", index=False)
    crosscheck.to_csv(RESULTS / "B_colonomics_clinical_crosscheck.tsv", sep="\t", index=False)

    private_out = os.environ.get("KPU_B_PRIVATE_OUTDIR")
    private_outputs = []
    if private_out:
        private_dir = Path(private_out)
        private_dir.mkdir(parents=True, exist_ok=True)
        scores = []
        for frame in axis_tables:
            f = frame.copy()
            if "sample_id" in f.columns:
                f = f.rename(columns={"sample_id": "sample"})
            if "sample" not in f.columns:
                f["sample"] = np.nan
            scores.append(f)
        all_scores = pd.concat(scores, ignore_index=True, sort=False)
        front = ["cohort", "axis_role", "patient", "sample", "delta_axis", "tumor_axis",
                 "mean_tumor_beta", "mean_delta_beta"]
        cols = [c for c in front if c in all_scores.columns] + \
               [c for c in all_scores.columns if c not in front]
        private_scores = private_dir / "B_axis_scores.tsv"
        all_scores[cols].to_csv(private_scores, sep="\t", index=False)
        private_outputs.append({
            "name": private_scores.name,
            "status": "not distributed in public source; retained only as private evidence",
            "sha256": sha256(private_scores),
        })
        private_manifest = {
            "status": "private evidence only",
            "outputs": [{"path": str(private_scores), "sha256": sha256(private_scores)}],
        }
        (private_dir / "B_private_manifest.json").write_text(
            json.dumps(private_manifest, indent=2) + "\n"
        )

    diag_rows = []
    for coh, d in diagnostics.items():
        for axname, dd in d.items():
            row = {"cohort": coh, "axis": axname}
            row.update(dd)
            diag_rows.append(row)
    pd.DataFrame(diag_rows).to_csv(RESULTS / "B_axis_diagnostics.tsv", sep="\t", index=False)

    finished = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "analysis": "PLAN.md Part B (B1-B5), coordination-axis biology",
        "seed": SEED,
        "bootstrap_draws": N_BOOT,
        "min_group_size_for_testing": MIN_GROUP,
        "python": sys.version.split()[0],
        "packages": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": __import__("scipy").__version__,
                     "statsmodels": __import__("statsmodels").__version__},
        "commands": [
            "/usr/bin/python3 B_coordination_axis/scripts/analyze_coordination_axis.py",
            "/usr/bin/python3 B_coordination_axis/scripts/figure_coordination_axis.py",
        ],
        "started_utc": started,
        "finished_utc": finished,
        "inputs": [{"path": logical_path(p), "sha256": sha256(p)} for p in INPUTS],
        "input_notes": {
            "TCGA beta": "read from the read-only 124 derivative TCGA_fixed77_case_tissue_beta.tsv, "
                         "which 124/scripts/analyze_tcga_cpg.py streamed from "
                         "102/external/data/tcga/beta_archived.zip via gdc_sample_map.json "
                         "(the 2.5 GB archive itself was not re-read or hashed here)",
        },
        "label_availability_checks": CHECKS,
        "outputs": sorted(p.name for p in RESULTS.glob("*.tsv")) +
                   ["figures/B_coordination_axis_labels.{pdf,png,svg}"],
        "private_outputs": private_outputs,
        "diagnostics": diagnostics,
    }
    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print("tests:", len(tdf), "| cohorts:", tdf['cohort'].nunique())
    print(tdf[tdf.in_family & tdf.tested][
        ["cohort", "axis", "comparison", "n_total", "p_value", "q_value", "effect"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
