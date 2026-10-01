# Portable numerical validation

The unchanged REML routine uses SciPy `minimize_scalar(method="bounded", bounds=(0,2000), xatol=1e-10)`. SciPy's bounded solver also includes a relative stopping term `sqrt(2.2e-16) * abs(tau²)`. Small changes in floating-point evaluation of the nearly flat log-likelihood can therefore alter the estimated optimum and derived Hartung–Knapp intervals/P values despite identical input estimates.

The initial Linux CI run (36937809013) failed the old generic comparison on q=0.553558795779051 versus 0.5535587979138761, an absolute difference of 2.1348251e-9. This observation alone does not establish all Linux outputs agree. The revised validator computes and emits diagnostics for both complete 385-row pools before raising, so any remaining difference is inspectable.

Controlled ±one-ULP perturbations of the original objective, evaluated over both meta-analysis pools, changed tau² by at most 1.80e-6 squared percentage points (maximum relative difference 4.83e-8), an interval endpoint by 3.66e-8 percentage points, effects by 2.05e-8 percentage points and P values by 2.60e-9. No significance decisions changed, and both reconstructed display tables remained identical. These probes characterize floating-point sensitivity; they do not alter the released original solver or canonical estimates.

An independent sensitivity probe perturbed each positive tau² by ±eight SciPy convergence units, where one unit is `sqrt(2.2e-16)*abs(tau²) + 1e-10/3`. Across the 462 estimated rows in both pools, maximum changes were 3.59e-5 squared percentage points for tau², 1.75e-7 percentage points for effects, 5.85e-7 for SE, 2.54e-6 for CI endpoints, 1.30e-8 for P and 1.15e-7 for Hartung–Knapp scale. The unit-specific absolute floors below cover this numerical sensitivity while the exact decision/display gates remain mandatory.

The second Linux CI run (36938905851, job 110625560572) provided complete diagnostics. It showed maximum CI endpoint difference 1.0007116e-6 percentage points, tau² difference 1.3478843e-5 squared percentage points, effect difference 3.827e-8 percentage points and SE difference 2.323e-7 percentage points. Some small tau² values differed relatively by 3.9178711e-7. Maximum relative P difference was 8.54e-8, while I²/Q passed their tight gate. These results supported using the independently justified unit floors rather than the earlier uniform 1e-8 floor. The complete diagnostics show all 15 previously failing fields and every reported per-column maximum fit the revised policy, with no earlier contract/decision violations. The full Linux TSVs were not supplied for local replay; a subsequent Linux run must still establish that all execution/display gates pass.

The comparison applies these explicit gates:

| Columns | Relative tolerance | Absolute tolerance | Unit |
| --- | --- | --- | --- |
| effect, SE and CI endpoints | 1e-7 | 1e-5 | percentage points |
| tau² | 1e-7 | 1e-4 | squared percentage points |
| Hartung–Knapp scale | 1e-7 | 1e-6 | dimensionless |
| P, P for BH and BH q | 1e-7 | 0 | probability |
| I² and heterogeneity Q | 1e-10 | 1e-10 | percent / dimensionless |

A zero absolute probability tolerance prevents an absolute floor from accepting orders-of-magnitude changes in very small P values. I² and Q do not depend on the REML optimum and retain their tight gate.

All planned keys, schemas, row counts, cohort counts/membership, status, method, other labels and missing-value masks must agree exactly. Every raw-P/q<0.05 decision, effect/CI endpoint three-way sign and tau² zero/nonzero boundary must agree. Table S6 display cells remain exact. The fixed-family BH recomputation and source-export comparisons retain 1e-12; these do not involve REML optimization.

Canonical outputs have substantial decision margins: nearest q to 0.05 is 0.0017787 in the primary pool and 0.0042189 in the sensitivity pool; the nearest CI endpoint to zero is 0.166738 percentage points. These margins contextualize the tiny observed numerical differences, while explicit decision gates reject a crossing even if it falls within numerical tolerance.

`meta_analysis/diagnostics.json` records maximum absolute/relative error and the applicable tolerances for every numerical column in both pools. Failing row keys, changed decisions and contract violations are retained. GitHub Actions uploads this diagnostic artifact even when verification fails. Local targeted tests accept the observed q roundoff and reject material estimate changes, tiny-P order changes, altered counts/cohort membership/missingness, q-threshold crossings, direction flips and tau² boundary changes. Consult the current Linux CI result to confirm these gates on that platform; local tests alone do not establish Linux success.
