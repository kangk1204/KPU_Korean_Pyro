# KPU Korean Pyro

Analysis code and aggregate inputs for a study of colorectal methylation using local paired pyrosequencing, Korean array cohorts and public validation cohorts.

This repository preserves the numerical analysis sources used for the final 2026-09-25 analysis package. The earlier private repository stopped at revision 129; final revision code from the delivered analysis archive is included here, with source hashes in [`provenance/sources.json`](provenance/sources.json). Paper documents and patient/sample-level inputs are excluded.

## Reproduce the available aggregate analyses

Use Python 3.12 and a fresh checkout:

```sh
git clone https://github.com/kangk1204/KPU_Korean_Pyro.git
cd KPU_Korean_Pyro
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify_repository.py
python reproduce.py --output reproduced
python run_tests.py
```

`reproduce.py` writes new results and `verification.json`. It stops on a failed comparison and preserves an existing output directory. It performs actual numerical recomputation:

- Executes the original REML/Hartung–Knapp meta-analysis routines on retained per-cohort CpG estimates, reconstructing both 385-row meta-analysis tables and comparing every field against the final retained outputs.
- Recomputes within-contrast BH q values for all 1,155 planned hypotheses. Family sizes are 231, 231, 462, 77 and 154 for N–H, T–H, T–N, A–H and T–A; unmeasured hypotheses contribute P=1.
- Regenerates final q/significance fields in two source exports (924 and 231 rows), checking row keys, effects, intervals, sample counts and all retained export fields. Historical source-folder labels are mapped to current figures in [`docs/figure_map.md`](docs/figure_map.md).
- Reconstructs Table S6 and the 18-row location-effect table from numerical summaries, then compares every display cell.

Meta-analysis files matched byte for byte in the release check. BH/source-export differences were at floating-point precision (maximum absolute difference below 1e-12). Other display tables are preserved exports; a file checksum is a provenance check, not independent statistical reconstruction.

`run_tests.py` runs synthetic and retained-input statistical regression checks for missing-site BH handling, corrected pairing, grouped train-only preprocessing, nested feature selection, matched reference sets, common-axis residualization and annotation-scope calculations. Seven historical-data tests are skipped because their separately restricted or retained source evidence is absent; the synthetic tests execute without patient inputs. The GitHub workflow runs these commands from a fresh checkout.

## Contents

| Path | Contents |
| --- | --- |
| `analysis/workspace/` | Numerical Python/R analysis sources and required registry definitions, with their original stage/dependency structure |
| `data/aggregate/` | Final aggregate results, source exports, fixed CpG registry and display tables |
| `data/candidate_screen.xlsx` | Aggregate initial candidate-screen results |
| `docs/analysis_inputs.md` | Input requirements and full-analysis execution boundaries |
| `provenance/` | Source identities, hashes and documented public-release adaptations |

The data files formerly supplied as supplementary data are available here as repository aggregate inputs. They contain detailed results beyond the supplementary display tables.

## Reproduction boundaries

The ready-to-run command reproduces **selected aggregate analyses**. Full sample-level reanalysis requires the specified public deposited matrices and authorized local clinical/pyrosequencing/OOF inputs. Local classifiers, recurrence models, paired patient bootstraps and panel-size refits cannot be reconstructed from aggregate summaries alone.

Current Figure 2 was extracted from an original TCGA raster: its underlying scatter values, exact sample/probe selection and regression source are absent from the delivered archive. This repository cannot independently regenerate that panel or recalculate its coefficients/P values. The candidate-screen workbook lacks delta-beta values, so the original >0.3 delta-beta screening threshold cannot be independently rechecked from that workbook. Clinical display Table 1 retains the final recorded CEA categories (16 elevated); continuous-CEA model inputs and historical measured-threshold summaries are distinct. These limitations are recorded rather than replaced with synthetic results.

Scientific scope, input access and detailed commands are in [`docs/analysis_inputs.md`](docs/analysis_inputs.md). No blanket claim of full raw-data reproducibility is made.
