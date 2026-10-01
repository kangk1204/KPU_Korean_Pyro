"""Retrospective common-change diagnostics from explicitly authorized inputs.

No source lookup, matching, model selection, or patient-level output is performed.
The public registry membership and the reviewed 500-draw method remain fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.stats import rankdata, spearmanr

METRICS = ["raw_balanced_median", "background_adjusted_balanced_median",
           "adjusted_minus_raw", "median_probe_background_correlation",
           "mean_rank_panel_background_correlation"]
GENES = ["EYA4", "ZNF568", "ZNF793", "SFMBT2", "ADHFE1", "HOXA2",
         "BEND5", "UNC5C", "RALYL", "GFRA1"]
EXPECTED_PAIRS = {"ASAN": 128, "SNUH": 142, "CMCBSN": 103}
BOOTSTRAP_DRAWS = 500
SEED_BASE = 2026091491


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Inputs:
    """Record hashes under logical roles, never source paths or patient IDs."""
    def __init__(self):
        self.records = {}
        self.paths = {}

    def path(self, role, filename):
        p = Path(filename).resolve()
        require(p.is_file(), "Missing explicitly supplied input for " + role)
        self.records[role] = {"sha256": digest(p), "bytes": p.stat().st_size}
        self.paths[role] = p
        return p

    def table(self, role, filename):
        return pd.read_csv(self.path(role, filename), sep="\t")

    def verify_unchanged(self):
        require(all(digest(self.paths[k]) == v["sha256"]
                    for k, v in self.records.items()), "A source input changed during execution")


def unit(x):
    x = np.asarray(x, dtype=float)
    require(x.ndim == 2 and np.isfinite(x).all(), "Expected finite two-dimensional measurements")
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.sum(x*x, axis=1, keepdims=True))
    require((norm > 0).all(), "Constant rank/score vector; statistic undefined")
    return x / norm


def rankunit(x):
    return unit(rankdata(x, axis=1, method="average"))


def pair_values(corr, regions):
    return np.array([np.median(corr[np.ix_(a, b)])
                     for i, a in enumerate(regions) for b in regions[i+1:]])


def balanced(corr, regions):
    return float(np.median(pair_values(corr, regions)))


def covterm(a, b):
    a = a-a.mean(axis=1, keepdims=True)
    b = b-b.mean(axis=1, keepdims=True)
    return a @ b.T / (a.shape[1]-1)


def candidate_layout(candidate):
    require(len(candidate) == 10 and candidate.slot.is_unique,
            "Expected ten unique candidate slots")
    require(set(candidate.slot) == set(GENES), "Candidate gene set differs from fixed panel")
    probes = [p for s in candidate.probes for p in s.split(";")]
    require(len(probes) == 56 and len(set(probes)) == 56, "Expected 56 unique candidate probes")
    regions = []
    start = 0
    for s in candidate.probes:
        n = len(s.split(";")); regions.append(np.arange(start, start+n)); start += n
    return probes, regions


def background_groups(pool, lookup, candidate_probes):
    """Unite repeated slot windows within each distinct gene, then weight genes equally."""
    require(not set(pool.gene).intersection(GENES), "Candidate gene found in background pool")
    grouped = {gene: sorted(set(p for s in part.probes for p in s.split(";")))
               for gene, part in pool.groupby("gene", sort=True)}
    all_probes = {p for ps in grouped.values() for p in ps}
    require(not all_probes.intersection(candidate_probes), "Candidate probe found in background score")
    require(all(p in lookup for p in all_probes), "Missing fixed background probe")
    return [np.array([lookup[p] for p in ps]) for ps in grouped.values()], len(grouped), len(all_probes)


def paired_candidate_data(matrix, metadata, probes):
    require("ProbeID" in matrix.columns, "Beta table requires ProbeID")
    require(matrix.ProbeID.is_unique, "Duplicate beta probe identifiers")
    require(metadata.sample_id.is_unique, "Duplicate metadata sample identifiers")
    require(set(matrix.columns[1:]) == set(metadata.sample_id), "Matrix/metadata sample mismatch")
    require(metadata.patient_id.notna().all(), "Missing source patient identifiers")
    require(set(metadata.tissue) <= {"N", "T"}, "Use explicit N/T tissue labels")
    require(not metadata.duplicated(["patient_id", "tissue"]).any(), "Duplicate patient/tissue records")
    verified = metadata.pair_verified.astype(str).str.lower().eq("true")
    pairs = metadata.loc[verified].pivot(index="patient_id", columns="tissue", values="sample_id").sort_index()
    require({"N", "T"}.issubset(pairs.columns) and pairs.notna().all().all(), "Incomplete verified pairs")
    data = matrix.set_index("ProbeID")
    require(set(probes).issubset(data.index), "Missing candidate probe")
    t = data.loc[probes, pairs["T"]].to_numpy(float)
    n = data.loc[probes, pairs["N"]].to_numpy(float)
    require(np.isfinite(t).all() and np.isfinite(n).all(), "Nonfinite paired beta values")
    require(((t >= 0) & (t <= 1) & (n >= 0) & (n <= 1)).all(), "Beta values outside [0,1]")
    return t, n


def validate_cache(cache, probes, candidate_probes, paired_delta):
    require(len(probes) == len(set(probes)), "Duplicate cache probe identifiers")
    require(cache.shape == (len(probes), paired_delta.shape[1]), "Cache dimensions differ from paired inputs")
    require(np.isfinite(cache).all(), "Nonfinite cached deltas")
    lookup = {p: i for i, p in enumerate(probes)}
    require(set(candidate_probes).issubset(lookup), "Candidate probes missing from cache")
    idx = np.array([lookup[p] for p in candidate_probes])
    require(np.allclose(cache[idx], paired_delta, atol=1e-12, rtol=0),
            "Cache patient order or candidate values do not match source-defined pairs")
    return lookup, idx


def residual_stat(delta, candidate_idx, candidate_regions, control_groups):
    ranks = rankunit(delta)
    genes = unit(np.array([ranks[ix].mean(axis=0) for ix in control_groups]))
    score = rankunit(genes.mean(axis=0, keepdims=True))[0]
    candidate = ranks[candidate_idx]
    corr_score = candidate @ score
    residuals = unit(candidate - corr_score[:, None]*score[None, :])
    raw = balanced(candidate @ candidate.T, candidate_regions)
    conditioned = balanced(residuals @ residuals.T, candidate_regions)
    return np.array([raw, conditioned, conditioned-raw, float(np.median(corr_score)),
                     float(spearmanr(candidate.mean(axis=0), score).statistic)])


def bootstrap_stats(delta, candidate_idx, candidate_regions, control_groups, draws, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        ix = rng.integers(0, delta.shape[1], delta.shape[1])
        # Ranks, both score construction stages, and residual regression are refitted.
        values.append(residual_stat(delta[:, ix], candidate_idx, candidate_regions, control_groups))
    return np.asarray(values)


def tissue_decomposition(cohort, t, n, regions):
    d = t-n
    rows = []
    for name, values in [("delta", d), ("tumor", t), ("adjacent", n)]:
        r = rankunit(values); vals = pair_values(r @ r.T, regions)
        rows.append({"cohort": cohort, "measurement": name, "pairs": t.shape[1],
                     "balanced_median_spearman": np.median(vals),
                     "min_gene_pair_median": vals.min(), "max_gene_pair_median": vals.max()})
    sd = d.std(axis=1, ddof=1); denominator = sd[:, None]*sd[None, :]
    require((sd > 0).all(), "Constant candidate delta")
    terms = {"tumor_covariance": covterm(t, t)/denominator,
             "adjacent_covariance": covterm(n, n)/denominator,
             "negative_cross_tumor_adjacent": -covterm(t, n)/denominator,
             "negative_cross_adjacent_tumor": -covterm(n, t)/denominator,
             "delta_correlation": covterm(d, d)/denominator}
    error = np.max(np.abs(sum(terms[k] for k in list(terms)[:4])-terms["delta_correlation"]))
    require(error < 1e-12, "Covariance identity failed")
    decomposition = [{"cohort": cohort, "term": term,
                      "balanced_median_normalized_covariance": balanced(v, regions),
                      "identity_max_error": error} for term, v in terms.items()]
    rows.append({"cohort": cohort, "measurement": "adjacent_variance_divided_by_delta_variance",
                 "pairs": t.shape[1], "median_ratio": np.median(n.var(axis=1, ddof=1)/d.var(axis=1, ddof=1))})
    return rows, decomposition


def reference_diagnostics(cohort, cache, lookup, candidate, regions, pool, frozen, expected):
    require(pool.region_id.is_unique, "Duplicate control region identifiers")
    require(frozen.set_id.is_unique and len(frozen) == 5000, "Expected 5000 unique frozen reference sets")
    controls = [np.array([lookup[p] for p in s.split(";")]) for s in pool.probes]
    r = rankunit(cache); corr = r @ r.T
    block = np.array([[np.median(corr[np.ix_(a, b)]) for b in controls] for a in controls])
    ridx = {v: i for i, v in enumerate(pool.region_id)}
    setidx = np.array([[ridx[x] for x in record] for record in frozen[candidate.slot].to_numpy()])
    require(all(len(set(row)) == 10 for row in pool.gene.to_numpy()[setidx]), "Repeated control gene within set")
    ii, jj = np.triu_indices(10, 1)
    stat = np.median(block[setidx[:, ii], setidx[:, jj]], axis=1)
    expected = expected.set_index("set_id")
    error = np.max(np.abs(stat-expected.loc[frozen.set_id, "gene_pair_balanced_median_rho"].to_numpy()))
    require(error < 1e-12, "Frozen primary reference-set point statistics changed")
    src = pool.mean_delta.to_numpy()[setidx].mean(axis=1)
    target = np.array([cache[x].mean() for x in controls])[setidx].mean(axis=1)
    candidate_delta = cache[[lookup[p] for s in candidate.probes for p in s.split(";")]]
    row = {"cohort": cohort, "sets": len(stat), "point_statistic_max_abs_error": error,
           "rho_set_coordination_vs_source_mean_delta": spearmanr(stat, src).statistic,
           "rho_set_coordination_vs_target_mean_delta": spearmanr(stat, target).statistic,
           "candidate_source_mean_delta": candidate.mean_delta.mean(),
           "source_control_mean_delta_min": src.min(), "source_control_mean_delta_max": src.max(),
           "source_control_mean_delta_median": np.median(src),
           "candidate_target_mean_delta": np.mean([candidate_delta[ix].mean() for ix in regions]),
           "target_control_mean_delta_min": target.min(), "target_control_mean_delta_max": target.max(),
           "target_control_mean_delta_median": np.median(target)}
    sets = pd.DataFrame({"set_id": frozen.set_id, "independent_balanced_rho": stat,
                         "source_mean_delta": src, "target_mean_delta": target})
    return row, sets


def local_qc(data, excluded):
    require({"study_id", "tissue", *GENES}.issubset(data.columns), "Incomplete local tissue schema")
    require(not data.duplicated(["study_id", "tissue"]).any(), "Duplicate local patient/tissue records")
    require(set(data.tissue) == {"T", "N"}, "Local tissue labels must be T/N")
    require(data.study_id.nunique() == 87 and len(data) == 174, "Expected original 87 local pairs")
    require(len(excluded) == 2 and len(set(excluded)) == 2 and set(excluded) <= set(data.study_id),
            "Supply exactly the two authorized, unique QC exclusions")
    rows = []; tests = []
    for label, omissions in [("original_87", []), ("exclude_two_patients", excluded)]:
        use = data.loc[~data.study_id.isin(omissions)]
        t = use.loc[use.tissue.eq("T")].set_index("study_id")[GENES].sort_index()
        n = use.loc[use.tissue.eq("N")].set_index("study_id")[GENES].sort_index()
        require(t.index.equals(n.index), "Unpaired local tissue data")
        delta = t.to_numpy(float)-n.to_numpy(float)
        require(np.isfinite(delta).all(), "Nonfinite local measurements")
        corr, p = spearmanr(delta, axis=0); ix = np.triu_indices(10, 1); pv = p[ix]; r = corr[ix]
        order = np.argsort(pv)
        qsort = np.minimum.accumulate((pv[order]*45/np.arange(1, 46))[::-1])[::-1]
        q = np.empty(45); q[order] = np.minimum(qsort, 1)
        z = (delta-delta.mean(axis=0))/delta.std(axis=0, ddof=1)
        e = np.linalg.svd(z, compute_uv=False)**2
        rows.append({"scenario": label, "pairs": len(t), "median_spearman_rho": np.median(r),
                     "min_spearman_rho": r.min(), "max_spearman_rho": r.max(),
                     "positive_correlations": int((r > 0).sum()), "BH_q_lt_0_05": int((q < .05).sum()),
                     "PC1_variance_fraction": e[0]/e.sum()})
        for i, j, rho, pvalue, qvalue in zip(*ix, r, pv, q):
            tests.append({"scenario": label, "gene1": GENES[i], "gene2": GENES[j],
                          "rho": rho, "p": pvalue, "BH_q": qvalue})
    return rows, tests


def run(config, registry_dir, output_dir):
    output_dir = Path(output_dir).resolve(); registry_dir = Path(registry_dir).resolve()
    require(not output_dir.exists() or not any(output_dir.iterdir()), "Use a new or empty output directory")
    inputs = Inputs()
    candidate = inputs.table("candidate_registry", registry_dir/"candidate_regions.tsv")
    pool = inputs.table("control_pool_registry", registry_dir/"control_region_pool.tsv")
    frozen = inputs.table("frozen_set_registry", registry_dir/"frozen_control_sets.tsv")
    probes, regions = candidate_layout(candidate)
    rows = []; decomposition = []; scores = []; support = []; outputs = {}; seeds = {}
    require(set(config["cohorts"]) == set(EXPECTED_PAIRS), "Supply all three fixed Korean cohorts")
    for cohort in ["ASAN", "SNUH", "CMCBSN"]:
        spec = config["cohorts"][cohort]
        matrix = inputs.table(cohort+"_candidate_beta", spec["candidate_beta"])
        metadata = inputs.table(cohort+"_pair_metadata", spec["pair_metadata"])
        t, n = paired_candidate_data(matrix, metadata, probes)
        require(t.shape[1] == EXPECTED_PAIRS[cohort], "Unexpected paired-patient count for " + cohort)
        tissue_rows, terms = tissue_decomposition(cohort, t, n, regions)
        rows.extend(tissue_rows); decomposition.extend(terms)
        if cohort == "CMCBSN":
            continue
        with np.load(inputs.path(cohort+"_authorized_delta_cache", spec["delta_cache"]), allow_pickle=False) as z:
            cache = z["delta"]; names = z["probes"].tolist()
        lookup, ci = validate_cache(cache, names, probes, t-n)
        groups, gene_count, probe_count = background_groups(pool, lookup, probes)
        require((gene_count, probe_count) == (151, 965), "Frozen control membership changed")
        estimate = residual_stat(cache, ci, regions, groups)
        seeds[cohort] = SEED_BASE+t.shape[1]
        boot = bootstrap_stats(cache, ci, regions, groups, BOOTSTRAP_DRAWS, seeds[cohort])
        low, high = np.quantile(boot, [.025, .975], axis=0)
        for metric, val, lo, hi in zip(METRICS, estimate, low, high):
            scores.append({"cohort": cohort, "metric": metric, "estimate": val, "ci_low": lo, "ci_high": hi,
                           "bootstrap": BOOTSTRAP_DRAWS, "background_genes": gene_count,
                           "background_unique_probes": probe_count})
        outputs[cohort+"_background_score_bootstrap.tsv"] = pd.DataFrame(boot, columns=METRICS)
        expected = inputs.table(cohort+"_frozen_set_statistics", registry_dir/(cohort+"_set_statistics.tsv"))
        record, setdata = reference_diagnostics(cohort, cache, lookup, candidate, regions, pool, frozen, expected)
        support.append(record); outputs[cohort+"_set_diagnostics.tsv"] = setdata
        print(cohort+": 500 complete-patient bootstrap draws and 5000 frozen reference sets complete", flush=True)
    local = inputs.table("authorized_local_tissue", config["local"]["tissue"])
    exclusions = inputs.table("authorized_local_exclusions", config["local"]["exclusions"])
    require(list(exclusions.columns) == ["study_id"], "Exclusion table must have only a study_id column")
    local_rows, local_tests = local_qc(local, exclusions.study_id.tolist())
    outputs.update({"candidate_tissue_decomposition.tsv": pd.DataFrame(rows),
                    "normalized_covariance_decomposition.tsv": pd.DataFrame(decomposition),
                    "background_score_adjustment.tsv": pd.DataFrame(scores),
                    "set_effect_coordination_relationship.tsv": pd.DataFrame(support),
                    "local_qc_coordination_summary.tsv": pd.DataFrame(local_rows),
                    "local_qc_coordination_pairs.tsv": pd.DataFrame(local_tests)})
    inputs.verify_unchanged()
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(output_dir/filename, sep="\t", index=False)
    provenance = {"analysis": "Retrospective descriptive common-axis and QC diagnostics",
                  "inputs_by_logical_role": inputs.records, "inputs_unchanged_after_run": True,
                  "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seeds": seeds,
                  "scope": "Conditional on fixed candidates, source-frozen control membership, score definition, and deposited values; no purity/CIMP diagnosis",
                  "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                  "scipy": scipy_version, "script_sha256": digest(__file__),
                  "outputs": {name: {"rows": len(frame), "sha256": digest(output_dir/name)} for name, frame in outputs.items()}}
    (output_dir/"run_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Explicit authorized input-path JSON; never published")
    parser.add_argument("--registry-dir", required=True, type=Path, help="Public frozen registry folder")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty aggregate output folder")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.registry_dir, args.output_dir)


if __name__ == "__main__":
    main()
