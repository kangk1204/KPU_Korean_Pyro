"""Compare aggregate-only TSV outputs with a separately retained reference run."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

FILES = ["background_score_adjustment.tsv", "candidate_tissue_decomposition.tsv",
         "normalized_covariance_decomposition.tsv", "set_effect_coordination_relationship.tsv",
         "ASAN_background_score_bootstrap.tsv", "SNUH_background_score_bootstrap.tsv",
         "ASAN_set_diagnostics.tsv", "SNUH_set_diagnostics.tsv",
         "local_qc_coordination_summary.tsv", "local_qc_coordination_pairs.tsv"]


def compare(expected_dir, actual_dir):
    results = []
    for filename in FILES:
        a = pd.read_csv(Path(expected_dir)/filename, sep="\t")
        b = pd.read_csv(Path(actual_dir)/filename, sep="\t")
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            raise AssertionError("Aggregate shape/schema changed: " + filename)
        errors = []
        for column in a:
            if pd.api.types.is_numeric_dtype(a[column]):
                np.testing.assert_allclose(a[column], b[column], atol=1e-12, rtol=0, equal_nan=True,
                                           err_msg=filename+" / "+column)
                difference = np.abs(a[column].to_numpy()-b[column].to_numpy())
                finite = difference[np.isfinite(difference)]
                errors.append(float(finite.max()) if len(finite) else 0.0)
            elif not a[column].fillna("").equals(b[column].fillna("")):
                raise AssertionError("Aggregate labels changed: " + filename+" / "+column)
        results.append({"file": filename, "rows": len(a), "columns": len(a.columns),
                        "max_absolute_numeric_difference": max(errors, default=0),
                        "within_1e_minus12": True,
                        "byte_identical": (Path(expected_dir)/filename).read_bytes() == (Path(actual_dir)/filename).read_bytes()})
    return {"all_pass": True, "comparisons": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-dir", required=True)
    p.add_argument("--actual-dir", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()
    result = compare(args.expected_dir, args.actual_dir)
    Path(args.report).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
