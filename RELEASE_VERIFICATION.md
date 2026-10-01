# Public release verification — 2026-10-02

Checked using Python 3.12 with the pinned `requirements.txt` environment. Outputs were generated in a separate copy of the public package; original analysis and submission files were preserved.

| Check | Result |
| --- | --- |
| Original public meta-analysis, primary ≥20-pair rule | 385 rows, all fields matched; byte-identical TSV |
| Original public meta-analysis, sensitivity ≥3-pair rule | 385 rows, all fields matched; byte-identical TSV |
| Independent within-contrast BH recomputation | 1,155 planned rows; maximum q difference 1.11×10⁻¹⁶ |
| Source-export regeneration and comparison | 924+231 rows; all labels, effects, intervals, counts and final q fields consistent within 1e-12 |
| Table S6 reconstruction from regenerated meta-analysis | All 3 display rows/cells identical |
| Location-effect table reconstruction from aggregate summaries | All 18 display rows/cells identical |
| Scientific/security regression suites | 70 passed; 7 expected data-dependent skips |
| Restore into separate numerical-code workspace | 198 targets validated; 173 missing files copied; existing identical files preserved |

The seven skips comprise six original local tissue-model tests requiring the excluded patient table and one original GSE77954 test requiring retained source evidence. Synthetic patient-grouping, held-out perturbation/train-only scaling and invalid pairing tests execute successfully without those inputs.

The listed commands can be repeated with `python reproduce.py --output <new-directory>` and `python run_tests.py`. `python verify_repository.py` separately verifies the file manifest. GitHub Actions is configured to run these checks after checkout; this local report does not claim a CI run has completed.

This is selected aggregate reproduction plus numerical algorithm regression testing. Local raw-data refitting, R patient-level recurrence/stage runs, TCGA raster reconstruction and the original candidate delta-beta threshold remain outside the verified scope for the reasons in the README and `docs/analysis_inputs.md`. Retained display tables other than the two explicitly reconstructed tables are canonical exports, not independently regenerated patient estimates.
