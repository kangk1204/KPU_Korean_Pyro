# KPU Korean Pyro

Code and summary data for colorectal DNA methylation analyses using paired pyrosequencing, Korean methylation array cohorts and public validation datasets.

The Python and R scripts, summary results and CpG definitions used in the study are included here. Individual patient and sample data require separate access. File origins and checksums are listed in [`provenance/sources.json`](provenance/sources.json).

## Running the analyses

Use Python 3.12. Download the repository, install the required packages and run:

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

`reproduce.py` writes the results and a comparison report (`verification.json`) to the output folder. Choose a new folder for each run. The script stops if a comparison fails.

The script reruns the following analyses from the supplied summary data:

- REML/Hartung–Knapp meta-analysis of cohort-level CpG estimates, producing two tables with 385 rows each.
- Benjamini–Hochberg correction within each contrast for 1,155 planned hypotheses. The N–H, T–H, T–N, A–H and T–A families contain 231, 231, 462, 77 and 154 hypotheses, respectively. Unmeasured hypotheses are assigned P = 1.
- Calculation of q values and significance labels in two source-data tables with 924 and 231 rows.
- Calculation of Table S6 and the 18-row location-effect table from the numerical summaries.

The results are compared with the stored reference files, including row identifiers, effect estimates, confidence intervals, sample counts and other table fields. Small numerical differences between operating systems are allowed within the [documented tolerances](docs/portable_validation.md). The checks require agreement on significance, effect direction, confidence interval signs and whether the between-cohort variance (tau²) is zero. All Table S6 cells must match. The other display tables are supplied as saved study results.

`run_tests.py` checks the statistical routines using synthetic data and the available study inputs. Seven tests are skipped because their required source data are unavailable. GitHub Actions runs the same commands on a fresh copy of the repository.

## Contents

| Path | Contents |
| --- | --- |
| `analysis/workspace/` | Python and R analysis scripts, CpG definitions and supporting files |
| `data/aggregate/` | Summary results, figure source data, CpG registry and display tables |
| `data/candidate_screen.xlsx` | Results from the initial candidate screen |
| `docs/analysis_inputs.md` | Required inputs and instructions for the other analyses |
| `provenance/` | File origins, checksums and changes made for public access |

The data folders include detailed CpG results and candidate-screen data. The correspondence between source-data folders and figure numbers is listed in [`docs/figure_map.md`](docs/figure_map.md).

## Data needed for other analyses

The command above reruns the analyses listed under "Running the analyses." Repeating the sample-level analyses requires the deposited public matrices and access to the local clinical, pyrosequencing and out-of-fold prediction data. These inputs are needed to refit the local classifiers and recurrence models, repeat the paired patient bootstraps and compare panel sizes.

Two further analyses need data that were absent from the archived materials:

- Figure 2 was taken from a TCGA image. The underlying scatter values, exact sample and probe selection, and regression code are unavailable, so the panel and its coefficients and P values cannot be recalculated from this repository.
- The candidate-screen workbook does not contain delta-beta values. The original delta-beta > 0.3 screening criterion cannot be checked from this workbook alone.

Table 1 uses the final recorded CEA categories, with 16 patients classified as elevated. The models use continuous CEA values. Earlier summaries based on measured CEA thresholds are kept separately.

Details on data access and the commands for each analysis are in [`docs/analysis_inputs.md`](docs/analysis_inputs.md).
