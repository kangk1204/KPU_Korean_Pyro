# Numerical analysis sources and input requirements

## Environments

The public aggregate verification and synthetic Python tests were checked with Python 3.12 and `requirements.txt`. These pins describe the public release verification environment; they are not a claim that every historical analysis ran in this environment. Exact final archive/source hashes are in `provenance/sources.json`. Historical Python requirements retained beside analysis sources may describe earlier stages.

R analyses additionally require R and the scientific packages used by their source modules. The recurrence runner imports `survival`, `glmnet`, `ranger` and `jsonlite` as used by the relevant stages. Inspect the selected R script before installing its dependencies. R patient-level runs were not rerun for this public release because the required authorized inputs are excluded.

## Use an isolated workspace

Keep patient-level inputs and generated held-out predictions outside the public Git checkout. Copy the numerical sources to a separate analysis workspace, then restore the available aggregate inputs:

```sh
cp -R analysis/workspace /path/to/new_analysis_workspace
python reproduce.py --restore-workspace /path/to/new_analysis_workspace
```

The destination must exist. Every source checksum, relative path and existing destination is checked before copying. A different existing file is rejected without replacement. The public catalog maps actual final aggregate files to historical execution paths; it replaces the old restore script's absent `FILE_CATALOG.json` assumption. Some final presentation tables/exports have no historical numerical source destination and remain canonical exports under `data/aggregate`.

Consult `input_requirements.tsv` for retained source paths and access boundaries. That file is preserved historical input metadata; references to earlier supplementary archives or document builders in it do not constitute executable public-release instructions. Use the numerical commands below and the current `reproduce.py` entry point.

## Analysis order after inputs are available

Paths below are relative to the isolated analysis workspace. These are source-preserving analysis routes, not evidence that unavailable inputs have been recomputed.

1. **Local clinical/PSQ preparation.** Run `129_Submission_Revised_20260909/scripts/prepare_local_reproduction.py --help`. Pass authorized workbooks with its explicit `--psq`, `--clinical`, `--provider-reply`, `--primers` and `--outdir` arguments. The supplied aggregate primer input is `data/aggregate/results/local_pyrosequencing/assay_primers.tsv` in the repository. Install verified generated inputs at the retained `114_ML_DataDriven_20260905/data/derived/` and `data/ml/` paths. No per-CpG pyrogram reconstruction is available.
2. **Local paired analysis and tissue classification.** Run `114_ML_DataDriven_20260905/scripts/analyze_paired.py` and `run_tissue_ml.py`. `ml/tissue.py` implements patient grouping, train-only preprocessing and inner tuning. Paired sensitivity scripts are in `151_Discordant_Sample_Sensitivity/`; they require the same authorized local inputs.
3. **Recurrence and staging.** Run `Rscript 114_ML_DataDriven_20260905/scripts/run_recurrence_ml.R` for the configured full run; `--quick` is a smoke check and cannot reproduce reported estimates. Run `119_Figure1_ABC_Revision_20260908/scripts/run_stage_sensitivity.R` after installing its mirrored baseline/clinical sources. Final stored-prediction diagnostics use `129_Submission_Revised_20260909/scripts/revise_recurrence_metrics.py --bootstrap 300`. Individual predictions remain private.
4. **Public cohort preparation.** `119_Figure1_ABC_Revision_20260908/scripts/prepare_geo.py` and `prepare_colonomics.py` prepare the public inputs. Run `129_Submission_Revised_20260909/scripts/prepare_public_inputs.py` to transfer prepared matrices and repair GSE77954 pairing from its deposited source labels. Public accession records and exact required files are in `input_requirements.tsv` and final `data/aggregate/tables/Table_S9.csv`.
5. **Korean CpG cohorts.** Place the exact deposited `processed_beta.txt` matrices under `000_CMCBSN_catholic_beta`, `000_SNUH_seoul_beta` and `000_ASAN_seoul_beta` alongside the numbered workspaces. The retained source-size/hash checks reject changed inputs. Run current `129_Submission_Revised_20260909/scripts/prepare_korean_beta.py`, `build_korean_metadata.py` and `analyze_korean_cpg.py`. Source-linked sample metadata under stage 120 is also required; restoring aggregate flow counts cannot replace it. Create the intended output directories before running retained scripts that assume an existing analysis workspace.
6. **Public contrasts, molecular context and classification.** Run current `analyze_public_cpg_contrasts.py`, `analyze_public_cpg_context.py`, `analyze_korean_context.py`, `analyze_gse119526_cpg.py`, `analyze_tcga_cpg.py` and `analyze_public_cpg_ml.py`. Consult each script's CLI where supplied. Per-cohort estimates use their historical declared families; `reproduce.py` independently derives the final within-contrast correction for the displayed source exports. Meta-only aggregation can be run without matrices via the top-level entry point.
7. **Reference sets and conditioning.** Sources are in stage 130 `analysis/background`, stage 131 `analysis/common_axis/public_code` and stage 137 `analysis/specificity/scripts`. Probe lists are supplied. The stage-137 scripts accept `KPU_PROJECT_ROOT`, `KPU_COMMON_SCORE_REGISTRY` and `KPU_BACKGROUND_CACHE`; paired full-array difference caches and per-sample CIMP extracts remain private. The final reported CIMP strata use the 0.30 rule in `cimp_refine.py`; the earlier 0.20 first pass is superseded.
8. **Annotation scope and exploratory panel size.** Stage 134 `analysis/cpg_scope/public_code/cpg_scope.py` performs probe-location analysis with authorized matrix inputs. Stage 127 A-panel calculations require the local 174-specimen table (`A1_TISSUE_TSV`) and can store private restart predictions at `A1_CHECKPOINT_DIR`; do not distribute these files. Stage 127 B-axis calculations use corrected public pairing metadata and retain individual scores only when a private output directory is explicitly supplied.

## Display and inference boundaries

The 20 CSV display exports include final Tables 1–2 and S1–S9 parts, and two detailed aggregate tables. `reproduce.py` independently reconstructs Table S6 and the location-effect table. The remaining CSVs preserve authoritative final display values and are not recomputed from underlying patient measurements. Older table/prose/submission builders are excluded because they would generate superseded reporting and are unnecessary for numerical analysis.

Historical source-folder names differ from current figure numbers; see `figure_map.md`. Current local PSQ paired-effect points, heatmap/PCA ordering, classifiers, recurrence intervals and exploratory refits require their original restricted patient/OOF inputs. Aggregate figures can be redrawn from available numerical source exports; source-data consistency is distinct from upstream refitting.

Current Figure 2 has no recoverable scatter-level TCGA source/regression code in the final archive. The initial candidate workbook contains 1,088 unique CpGs across 619 genes with P/q and annotation fields but lacks delta-beta, so the original >0.3 screening filter is not independently executable. No original patient workbooks, pyrograms, individual predictions, private Git history, paper text, cover letters or document builders are supplied.
