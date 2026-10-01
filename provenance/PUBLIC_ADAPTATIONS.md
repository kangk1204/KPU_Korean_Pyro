# Public-release adaptations

The final delivered analysis archive (2026-09-25) supplements private repository commit `9fc2848efb935643b34c83a813d0e53bf463fbe4`. `sources.json` records the original source hash for every retained code/data file and whether numerical source files also match that private snapshot. It does not imply the older private commit contained later stage 130–151 changes.

Original numerical algorithms are retained. Original directory structure is preserved for module imports. Paper files, document/prose/submission builders, editorial materials, generated figure binaries, patient inputs and private Git history are excluded. Superseded clinical/prose exporters `prepare_review_metadata.py` and `build_cpg_tables.py` are deliberately omitted.

The original `restore_aggregate_inputs.py` assumed a `FILE_CATALOG.json` absent from the final aggregate bundle. The public version delegates to `reproduce.restore_aggregates` and a new `RESTORE_CATALOG.json` constructed from the actual final aggregate files. It validates all relative paths, source hashes and existing destinations before copying and rejects symlink traversal. The statistical calculations are unaffected. Its original and public hashes are recorded in `adaptations.json`.

`reproduce.py`, `run_tests.py`, `verify_repository.py`, top-level synthetic regression tests, execution docs, dependency pins and the CI workflow are new public-release support files. The top-level numerical verifier explicitly separates retained display exports from the two reconstructed display tables. Final q/significance source-export fields are recomputed from P values rather than depending on excluded Word files or the original DOCX synchronization tool.

`public_manifest.json` records the public release files and their current hashes, excluding itself, Git metadata, environments, test caches and generated outputs. `verify_repository.py` checks file integrity; numerical reproducibility is checked independently by `reproduce.py` and the scientific tests.
