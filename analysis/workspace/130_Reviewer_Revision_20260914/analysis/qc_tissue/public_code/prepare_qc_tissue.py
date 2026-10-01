#!/usr/bin/env python3
"""Prepare an isolated private run from authorised inputs, freeze its plan, and execute."""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return reader.fieldnames or [], list(reader)


def require_columns(columns, required, label):
    missing = set(required) - set(columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def read_genes(module):
    tree = ast.parse(module.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GENES" for t in node.targets):
            genes = ast.literal_eval(node.value)
            if isinstance(genes, list) and len(genes) == 10 and len(set(genes)) == 10:
                return genes
    raise ValueError("The original module must declare its ten-gene GENES list")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Authorised original paired tissue TSV")
    p.add_argument("--config", type=Path, required=True, help="Original model configuration JSON")
    p.add_argument("--module", type=Path, required=True, help="Original ml/tissue.py implementation")
    p.add_argument("--exclusions", type=Path, required=True, help="Authorised TSV with study_id, gene, summary")
    p.add_argument("--original-metrics", type=Path, required=True, help="Original metrics.tsv")
    p.add_argument("--original-ci", type=Path, required=True, help="Original bootstrap_ci.tsv")
    p.add_argument("--original-oof", type=Path, required=True, help="Original oof_predictions.tsv.gz")
    p.add_argument("--output", type=Path, required=True, help="New private directory; must not already exist")
    p.add_argument("--dry-run", action="store_true", help="Validate, copy inputs and freeze the plan without fitting")
    args = p.parse_args()
    for name in ("input", "config", "module", "exclusions", "original_metrics", "original_ci", "original_oof"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    supplied = {
        "source/data/ml/tissue.tsv": args.input,
        "source/registry/ml_config.json": args.config,
        "source/ml/tissue.py": args.module,
        "authorised_exclusions_PRIVATE.tsv": args.exclusions,
        "source/results/ml/tissue/metrics.tsv": args.original_metrics,
        "source/results/ml/tissue/bootstrap_ci.tsv": args.original_ci,
        "source/results/ml/tissue/oof_predictions.tsv.gz": args.original_oof,
    }
    supplied = {target: source.expanduser().resolve() for target, source in supplied.items()}
    for source in supplied.values():
        if not source.is_file():
            raise ValueError("Every supplied input must be an existing readable file")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Output directory already exists; choose a new private directory")
    runner = Path(__file__).with_name("run_qc_tissue.py")
    if not runner.is_file():
        raise ValueError("Keep the unchanged run_qc_tissue.py beside this helper")

    genes = read_genes(args.module)
    cfg = json.loads(args.config.read_text())
    tissue_cfg = cfg.get("tissue", cfg)
    for name, expected in (("outer_folds", 5), ("outer_repeats", 20), ("inner_folds", 4)):
        if tissue_cfg.get(name) != expected:
            raise ValueError(f"Original configuration must retain {name}={expected}")
    if [float(x) for x in tissue_cfg.get("logistic_C", [])] != [.01, .1, 1., 10.] or cfg.get("seed") != 20260905:
        raise ValueError("Original ridge C grid and random seed must be retained")

    columns, data = rows(args.input)
    require_columns(columns, ["study_id", "specimen_id", "tissue", "y", *genes], "Tissue input")
    by_patient = {}
    for row in data:
        by_patient.setdefault(row["study_id"], []).append(row)
        if not all(math.isfinite(float(row[g])) for g in genes):
            raise ValueError("All retained gene values must be finite")
    if len(data) != 174 or len(by_patient) != 87:
        raise ValueError("This fixed sensitivity analysis requires the original 87 pairs and 174 specimens")
    if any(len(pair) != 2 or sorted(int(r["y"]) for r in pair) != [0, 1] for pair in by_patient.values()):
        raise ValueError("Every original patient must have one tumour and one normal specimen")
    columns, exclusions = rows(args.exclusions)
    require_columns(columns, ["study_id", "gene", "summary"], "Exclusions crosswalk")
    ids = sorted({row["study_id"] for row in exclusions})
    if len(ids) != 2 or not set(ids).issubset(by_patient):
        raise ValueError("The crosswalk must designate exactly two patients present in the original input")
    for row in exclusions:
        if row["gene"] not in genes:
            raise ValueError("Every crosswalk gene must be in the original panel")
        tumour = next(r for r in by_patient[row["study_id"]] if int(r["y"]) == 1)
        if not math.isfinite(float(row["summary"])) or float(tumour[row["gene"]]) != float(row["summary"]):
            raise ValueError("The authorised crosswalk does not match the original tumour measurements")
    for path, required, label in (
        (args.original_metrics, ["model", "scope", "auc"], "Original metrics"),
        (args.original_ci, ["model", "metric", "point", "ci_low", "ci_high"], "Original confidence intervals"),
        (args.original_oof, ["model", "study_id", "sample_id", "y", "score", "repeat"], "Original OOF"),
    ):
        columns, records = rows(path)
        require_columns(columns, required, label)
        if not any(row["model"] == "ridge" for row in records):
            raise ValueError(f"{label} must contain the primary ridge model")

    # An isolated private copy preserves every original supplied file and the published runner.
    output.mkdir(parents=True, mode=0o700)
    execution = output / "execution"
    private = execution / "private"
    private.mkdir(parents=True, mode=0o700)
    (execution / "public").mkdir()
    origin = []
    for target, source in supplied.items():
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        shutil.copyfile(source, destination)
        if sha(destination) != before or sha(source) != before:
            raise RuntimeError("An authorised input changed while being copied")
        origin.append({"original_path": str(source), "staged_path": target, "sha256": before})
    (output / "source/ml/__init__.py").write_text("")
    shutil.copyfile(runner, execution / runner.name)
    if sha(runner) != sha(execution / runner.name):
        raise RuntimeError("Runner copy does not match the published runner")
    plan = {
        "analysis": "Post hoc complete-patient exclusion robustness for primary tissue ridge classifier",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": "source/data/ml/tissue.tsv", "source_module": "source/ml/tissue.py",
        "config_path": "source/registry/ml_config.json", "crosswalk_path": "authorised_exclusions_PRIVATE.tsv",
        "excluded_study_ids": ids, "genes": genes, "C_grid": [.01, .1, 1., 10.],
        "models": ["ridge"], "outer_folds": 5, "outer_repeats": 20, "inner_folds": 4,
        "seed": 20260905, "bootstrap_replicates": 2000, "optimism_bootstrap": 0, "permutations": 0,
        "exclusion_rule": "Remove both specimens from each authorised designated patient, retaining all other original measurements.",
        "expected_original_patients": 87, "expected_original_specimens": 174,
        "expected_sensitivity_patients": 85, "expected_sensitivity_specimens": 170,
        "estimator": "Mean of 20 repeat-specific specimen-level outer out-of-fold AUCs.",
        "conditional_ci": "2000 complete-patient bootstrap resamples; same sampled clusters across repeats; no bootstrap model refitting.",
        "contrast": "Descriptive original-versus-exclusion comparison; no difference confidence interval or significance test.",
        "claim_boundary": "Post hoc exclusion robustness; does not reconstruct historical assay QC.",
        "preflight_guarded_files": {target: sha(output / target) for target in supplied},
    }
    plan_file = private / "PLAN_PRIVATE.json"
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    plan_md = execution / "PLAN.md"
    plan_md.write_text(
        "# Frozen post hoc tissue exclusion sensitivity plan\n\n"
        "Use the original ten-gene ridge implementation and configuration. Remove both specimens from "
        "the two patients designated in the authorised private crosswalk, leaving 85 pairs. Retain five "
        "patient-grouped outer folds repeated 20 times, four patient-grouped inner folds, the original "
        "penalty grid and seed, and training-only preprocessing. Report the mean repeat-specific OOF AUC "
        "and a 2000-draw conditional patient-cluster bootstrap interval. No other predictive models, "
        "permutations, optimism bootstrap, or difference confidence interval/test are permitted. "
        "This is post hoc exclusion robustness, not restoration of historical assay QC. All patient-level "
        "inputs and outputs remain private; execution/public contains only aggregate outputs.\n")
    lock_file = execution / "PLAN_LOCK.json"
    lock_file.write_text(json.dumps({"plan_private_sha256": sha(plan_file), "plan_md_sha256": sha(plan_md),
                                     "frozen_utc": plan["frozen_utc"]}, indent=2) + "\n")
    for path in (plan_file, plan_md, lock_file):
        path.chmod(0o444)
    (private / "authorised_input_provenance_PRIVATE.json").write_text(json.dumps(origin, indent=2) + "\n")
    (output / "README_PRIVATE.md").write_text(
        "This entire directory contains authorised private inputs. Do not publish it. Only execution/public "
        "is designated for aggregate sharing. The original supplied files are unchanged. The execution plan "
        "was frozen before fitting. Set KPU_PROJECT_ROOT to this directory when invoking execution/run_qc_tissue.py.\n")
    env = os.environ.copy()
    env.update({"KPU_PROJECT_ROOT": str(output), "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    print(json.dumps({"state": "prepared_without_fitting" if args.dry_run else "prepared",
                      "input_patients": 87, "excluded_patients": 2, "retained_patients": 85,
                      "runner_sha256": sha(execution / runner.name), "plan_private_sha256": sha(plan_file)}), flush=True)
    if not args.dry_run:
        subprocess.run([sys.executable, str(execution / runner.name)], env=env, cwd=output, check=True)


if __name__ == "__main__":
    main()
