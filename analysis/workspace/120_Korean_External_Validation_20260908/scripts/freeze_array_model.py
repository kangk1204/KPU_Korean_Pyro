#!/usr/bin/env python3
"""Freeze a 10-gene array ridge transfer model from public colorectal cohorts.

This preflight is intentionally limited to historical public array cohorts.
It fits a StandardScaler + ridge logistic regression with C=1.0, uses
training-only preprocessing, and evaluates leave-one-cohort-out transfer
between Colonomics and GSE119526. The resulting fit is an exploratory public
array transfer model only; it is not a frozen validation of Korean pyrosequencing
outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SOURCE = ROOT.parent / "119_Figure1_ABC_Revision_20260908" / "submission" / "figure_source_data"
OUTDIR = ROOT / "registry" / "ml"

GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2", "BEND5", "UNC5C", "RALYL", "GFRA1"]
COHORT_FILES = {
    "colonomics": PUBLIC_SOURCE / "F4_input_colonomics_features_from_102.tsv",
    "gse119526": PUBLIC_SOURCE / "F4_input_gse119526_features_from_102.tsv",
}


@dataclass(frozen=True)
class FitResult:
    train_cohort: str
    test_cohort: str
    n_train_rows: int
    n_train_patients: int
    n_test_rows: int
    n_test_patients: int
    metrics: dict[str, float]
    raw_intercept: float
    raw_coefficients: dict[str, float]
    scaled_intercept: float
    scaled_coefficients: dict[str, float]
    scaler_mean: dict[str, float]
    scaler_scale: dict[str, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x)


def load_cohort(path: Path, cohort: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"sample", *{f"{gene}_beta" for gene in GENES}}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{cohort} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["cohort"] = cohort
    df["sample"] = df["sample"].astype(str)
    if "patient" not in df.columns:
        df["patient"] = ""
    if "tissue" not in df.columns:
        df["tissue"] = ""
    df["patient"] = df["patient"].astype("string").fillna("")
    df["tissue"] = df["tissue"].astype("string").fillna("")
    sample_match = df["sample"].str.extract(r"^(?P<patient>.+)_(?P<tissue>[TNM])$")
    blank_patient = df["patient"].isna() | df["patient"].astype(str).str.strip().eq("")
    blank_tissue = df["tissue"].isna() | df["tissue"].astype(str).str.strip().eq("")
    df.loc[blank_patient & sample_match["patient"].notna(), "patient"] = sample_match.loc[
        blank_patient & sample_match["patient"].notna(), "patient"
    ]
    df.loc[blank_tissue & sample_match["tissue"].notna(), "tissue"] = sample_match.loc[
        blank_tissue & sample_match["tissue"].notna(), "tissue"
    ]
    df["patient"] = df["patient"].astype(str).str.strip()
    df["tissue"] = df["tissue"].astype(str).str.strip()
    if not set(df["tissue"]).issubset({"T", "N", "M"}):
        raise ValueError(f"{cohort} has unexpected tissue codes: {sorted(set(df['tissue']) - {'T', 'N', 'M'})}")
    if df["patient"].eq("").any() or df["tissue"].eq("").any():
        bad = df.loc[df["patient"].eq("") | df["tissue"].eq(""), ["sample", "patient", "tissue"]].head()
        raise ValueError(f"{cohort} has unresolved sample annotations: {bad.to_dict('records')}")
    return df


def restrict_binary_tissues(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["tissue"].isin({"T", "N"})].copy()


def design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[[f"{gene}_beta" for gene in GENES]].to_numpy(dtype=float)
    y = (df["tissue"] == "T").astype(int).to_numpy(dtype=int)
    return x, y


def train_pipeline(train_df: pd.DataFrame) -> tuple[object, np.ndarray, np.ndarray]:
    x_train, y_train = design_matrix(train_df)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=0,
        ),
    )
    pipe.fit(x_train, y_train)
    return pipe, x_train, y_train


def evaluate_pipeline(pipe, test_df: pd.DataFrame) -> dict[str, float]:
    x_test, y_test = design_matrix(test_df)
    probs = pipe.predict_proba(x_test)[:, 1]
    pred = (probs >= 0.5).astype(int)
    n_pos = int(y_test.sum())
    n_neg = int(len(y_test) - n_pos)
    tp = int(((pred == 1) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())
    return {
        "auc": float(roc_auc_score(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
        "sensitivity_0_5": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity_0_5": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "balanced_accuracy_0_5": float(((tp / (tp + fn)) + (tn / (tn + fp))) / 2) if (tp + fn) and (tn + fp) else float("nan"),
        "n_test": int(len(y_test)),
        "n_test_t": n_pos,
        "n_test_n": n_neg,
        "pred_positive_rate_0_5": float(pred.mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def extract_raw_model(pipe, feature_names: list[str]) -> tuple[float, dict[str, float], float, dict[str, float], dict[str, float], dict[str, float]]:
    scaler: StandardScaler = pipe.named_steps["standardscaler"]
    lr: LogisticRegression = pipe.named_steps["logisticregression"]
    coef_scaled = lr.coef_[0].astype(float)
    intercept_scaled = float(lr.intercept_[0])
    mean = scaler.mean_.astype(float)
    scale = scaler.scale_.astype(float)
    raw_coef = coef_scaled / scale
    raw_intercept = intercept_scaled - float(np.sum(coef_scaled * mean / scale))

    raw = {gene: float(value) for gene, value in zip(feature_names, raw_coef)}
    scaled = {gene: float(value) for gene, value in zip(feature_names, coef_scaled)}
    mean_map = {gene: float(value) for gene, value in zip(feature_names, mean)}
    scale_map = {gene: float(value) for gene, value in zip(feature_names, scale)}
    return raw_intercept, raw, intercept_scaled, scaled, mean_map, scale_map


def predict_raw(df: pd.DataFrame, intercept: float, coef: dict[str, float]) -> np.ndarray:
    x = df[[f"{gene}_beta" for gene in GENES]].to_numpy(dtype=float)
    beta = np.array([coef[gene] for gene in GENES], dtype=float)
    return sigmoid(intercept + x @ beta)


def fit_transfer(train_df: pd.DataFrame, test_df: pd.DataFrame, train_cohort: str, test_cohort: str) -> FitResult:
    pipe, x_train, _ = train_pipeline(train_df)
    metrics = evaluate_pipeline(pipe, test_df)
    raw_intercept, raw_coef, scaled_intercept, scaled_coef, mean_map, scale_map = extract_raw_model(pipe, GENES)
    probs_pipeline = pipe.predict_proba(test_df[[f"{g}_beta" for g in GENES]].to_numpy(dtype=float))[:, 1]
    probs_raw = predict_raw(test_df, raw_intercept, raw_coef)
    if not np.allclose(probs_pipeline, probs_raw, atol=1e-12, rtol=1e-12):
        raise RuntimeError("raw-space coefficients do not reproduce pipeline probabilities")
    return FitResult(
        train_cohort=train_cohort,
        test_cohort=test_cohort,
        n_train_rows=int(len(train_df)),
        n_train_patients=int(train_df["patient"].nunique()),
        n_test_rows=int(len(test_df)),
        n_test_patients=int(test_df["patient"].nunique()),
        metrics=metrics,
        raw_intercept=float(raw_intercept),
        raw_coefficients=raw_coef,
        scaled_intercept=float(scaled_intercept),
        scaled_coefficients=scaled_coef,
        scaler_mean=mean_map,
        scaler_scale=scale_map,
    )


def overlap_registry(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    cohort_names = list(cohorts)
    for cohort, df in cohorts.items():
        binary = restrict_binary_tissues(df)
        rows.append(
            {
                "row_type": "cohort_summary",
                "cohort_a": cohort,
                "cohort_b": "",
                "n_rows_total": int(len(df)),
                "n_rows_tn": int(len(binary)),
                "n_tumor": int((binary["tissue"] == "T").sum()),
                "n_normal": int((binary["tissue"] == "N").sum()),
                "n_other": int((df["tissue"] == "M").sum()),
                "n_patients_total": int(df["patient"].nunique()),
                "n_patients_tn": int(binary["patient"].nunique()),
                "pair_overlap_n": "",
                "sample_overlap_n": "",
                "note": "binary model target uses T/N only; M rows excluded",
            }
        )
    if len(cohort_names) == 2:
        a, b = cohort_names
        pa = set(cohorts[a]["patient"].astype(str))
        pb = set(cohorts[b]["patient"].astype(str))
        sa = set(cohorts[a]["sample"].astype(str))
        sb = set(cohorts[b]["sample"].astype(str))
        rows.append(
            {
                "row_type": "pairwise_overlap",
                "cohort_a": a,
                "cohort_b": b,
                "n_rows_total": "",
                "n_rows_tn": "",
                "n_tumor": "",
                "n_normal": "",
                "n_other": "",
                "n_patients_total": "",
                "n_patients_tn": "",
                "pair_overlap_n": int(len(pa & pb)),
                "sample_overlap_n": int(len(sa & sb)),
                "note": "no exact overlap in available patient/sample identifiers; cross-study patient identity not independently adjudicated",
            }
        )
    return pd.DataFrame(rows)


def build_model_record(pooled_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    pipe, _, _ = train_pipeline(pooled_df)
    raw_intercept, raw_coef, scaled_intercept, scaled_coef, mean_map, scale_map = extract_raw_model(pipe, GENES)
    probs_pipeline = pipe.predict_proba(pooled_df[[f"{g}_beta" for g in GENES]].to_numpy(dtype=float))[:, 1]
    probs_raw = predict_raw(pooled_df, raw_intercept, raw_coef)
    if not np.allclose(probs_pipeline, probs_raw, atol=1e-12, rtol=1e-12):
        raise RuntimeError("pooled raw coefficients do not reproduce pipeline probabilities")

    model_rows = [
        {
            "term": "(Intercept)",
            "raw_coefficient": raw_intercept,
            "scaled_coefficient": scaled_intercept,
            "scaler_mean": "",
            "scaler_scale": "",
        }
    ]
    for gene in GENES:
        model_rows.append(
            {
                "term": gene,
                "raw_coefficient": raw_coef[gene],
                "scaled_coefficient": scaled_coef[gene],
                "scaler_mean": mean_map[gene],
                "scaler_scale": scale_map[gene],
            }
        )

    record = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "binary tissue classification",
        "analysis_class": "exploratory_public_array_transfer",
        "feature_compatibility_gate": {'eligible_for_korean_fixed_panel_test': False, 'original_target_probe_dictionary_equal': True, 'historical_samplewise_80_percent_coverage_verified': False, 'reason': 'Historical gene means used skipna=True and MIN_PROBES=3 selection; the current per-sample 80% coverage contract has not been verified on training inputs. Rebuild or verify source probe-level features before target testing.', 'source_probe_dictionary': '102_ML_revised_20260902/external/table1_probes.json', 'source_probe_dictionary_sha256': '30965dbf5c3434e184bb86dd303296325486f48743755ffc0899333a986bf155', 'source_feature_builder': '102_ML_revised_20260902/external/build_cohorts.py'},
        "positive_class": "T",
        "negative_class": "N",
        "excluded_tissue_codes": ["M"],
        "feature_space": "10-gene array beta means",
        "genes": GENES,
        "estimator": {
            "type": "logistic_regression",
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 5000,
            "preprocessing": "StandardScaler fit on training cohort only",
            "threshold": 0.5,
        },
        "final_fit": {
            "cohorts": sorted(pooled_df["cohort"].unique().tolist()),
            "n_rows": int(len(pooled_df)),
            "n_patients": int(pooled_df["patient"].nunique()),
            "n_tumor": int((pooled_df["tissue"] == "T").sum()),
            "n_normal": int((pooled_df["tissue"] == "N").sum()),
        },
        "intercept_raw": float(raw_intercept),
        "coefficients_raw": raw_coef,
        "intercept_scaled": float(scaled_intercept),
        "coefficients_scaled": scaled_coef,
        "scaler_mean": mean_map,
        "scaler_scale": scale_map,
        "notes": [
            "Cross-cohort transfer was checked by leave-one-public-cohort-out validation on historical public arrays before freezing the pooled public fit.",
            "This record is exploratory public-array transfer only; it does not validate Korean pyrosequencing outcomes.",
        ],
    }
    return record, pd.DataFrame(model_rows)


def write_tsv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, float_format="%.12g")
    return sha256_file(path)


def write_json(data: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    args = parser.parse_args()
    outdir = args.outdir

    cohorts = {name: load_cohort(path, name) for name, path in COHORT_FILES.items()}
    binary = {name: restrict_binary_tissues(df) for name, df in cohorts.items()}
    for name, df in binary.items():
        if df.empty:
            raise RuntimeError(f"{name} has no T/N rows after filtering")
        if df["tissue"].value_counts().to_dict().get("T", 0) == 0 or df["tissue"].value_counts().to_dict().get("N", 0) == 0:
            raise RuntimeError(f"{name} is missing one tissue class after filtering")

    validation = [
        fit_transfer(binary["colonomics"], binary["gse119526"], "colonomics", "gse119526"),
        fit_transfer(binary["gse119526"], binary["colonomics"], "gse119526", "colonomics"),
    ]
    validation_df = pd.DataFrame(
        [
            {
                "analysis": "leave_one_cohort_out",
                "train_cohort": item.train_cohort,
                "test_cohort": item.test_cohort,
                "n_train_rows": item.n_train_rows,
                "n_train_patients": item.n_train_patients,
                "n_test_rows": item.n_test_rows,
                "n_test_patients": item.n_test_patients,
                **item.metrics,
            }
            for item in validation
        ]
    )

    pooled_df = pd.concat([binary["colonomics"], binary["gse119526"]], ignore_index=True)
    model_record, coefficients_df = build_model_record(pooled_df)
    overlap_df = overlap_registry(cohorts)

    spec_record = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "freeze_array_model_preflight",
        "target": "binary tissue classification",
        "analysis_class": "exploratory_public_array_transfer",
        "feature_compatibility_gate": {'eligible_for_korean_fixed_panel_test': False, 'original_target_probe_dictionary_equal': True, 'historical_samplewise_80_percent_coverage_verified': False, 'reason': 'Historical gene means used skipna=True and MIN_PROBES=3 selection; the current per-sample 80% coverage contract has not been verified on training inputs. Rebuild or verify source probe-level features before target testing.', 'source_probe_dictionary': '102_ML_revised_20260902/external/table1_probes.json', 'source_probe_dictionary_sha256': '30965dbf5c3434e184bb86dd303296325486f48743755ffc0899333a986bf155', 'source_feature_builder': '102_ML_revised_20260902/external/build_cohorts.py'},
        "validation_design": "leave-one-public-cohort-out on historical public arrays, then pooled freeze",
        "feature_set": GENES,
        "excluded_tissue_codes": ["M"],
        "training_only_scaling": True,
        "threshold": 0.5,
        "public_sources": {
            name: {
                "path": str(path.relative_to(ROOT.parent)),
                "sha256": sha256_file(path),
                "rows_total": int(len(cohorts[name])),
                "rows_tn": int(len(binary[name])),
                "patients_total": int(cohorts[name]["patient"].nunique()),
                "patients_tn": int(binary[name]["patient"].nunique()),
            }
            for name, path in COHORT_FILES.items()
        },
        "notes": [
            "The public arrays are used as discovery/support cohorts only.",
            "This is an exploratory public-array transfer model for beta values, not a validation of Korean pyrosequencing outcomes.",
        ],
    }

    outdir.mkdir(parents=True, exist_ok=True)
    validation_path = outdir / "freeze_array_model_validation.tsv"
    coefficients_path = outdir / "freeze_array_model_coefficients.tsv"
    model_path = outdir / "freeze_array_model.json"
    spec_path = outdir / "freeze_array_model_spec.json"
    overlap_path = outdir / "freeze_array_patient_overlap.tsv"
    hashes_path = outdir / "freeze_array_model_hashes.json"

    validation_sha = write_tsv(validation_df, validation_path)
    coefficients_sha = write_tsv(coefficients_df, coefficients_path)
    model_sha = write_json(model_record, model_path)
    spec_sha = write_json(spec_record, spec_path)
    overlap_sha = write_tsv(overlap_df, overlap_path)

    hashes_record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            name: {
                "path": str(path.relative_to(ROOT.parent)),
                "sha256": sha256_file(path),
            }
            for name, path in COHORT_FILES.items()
        },
        "outputs": {
            "freeze_array_model_validation.tsv": validation_sha,
            "freeze_array_model_coefficients.tsv": coefficients_sha,
            "freeze_array_model.json": model_sha,
            "freeze_array_model_spec.json": spec_sha,
            "freeze_array_patient_overlap.tsv": overlap_sha,
        },
        "script": {
            "path": str(Path(__file__).relative_to(ROOT.parent)),
            "sha256": sha256_file(Path(__file__)),
        },
        "summary": {
            "validation_auc_min": float(validation_df["auc"].min()),
            "validation_auc_max": float(validation_df["auc"].max()),
            "validation_brier_min": float(validation_df["brier"].min()),
            "validation_brier_max": float(validation_df["brier"].max()),
            "final_fit_rows": int(model_record["final_fit"]["n_rows"]),
            "final_fit_patients": int(model_record["final_fit"]["n_patients"]),
        },
    }
    write_json(hashes_record, hashes_path)

    print("freeze_array_model: complete")
    print(validation_df.round(4).to_string(index=False))
    print()
    print(f"wrote: {validation_path}")
    print(f"wrote: {coefficients_path}")
    print(f"wrote: {model_path}")
    print(f"wrote: {spec_path}")
    print(f"wrote: {overlap_path}")
    print(f"wrote: {hashes_path}")


if __name__ == "__main__":
    main()
