# SCHM reproducibility-packaging report

Date: 2026-08-16 (Asia/Shanghai)

## A. Branch and freeze state

- Target journal: Structural Control and Health Monitoring (SCHM).
- Source repository branch: `paper0-jcshm-manuscript`.
- Scientific base commit: `c78af2c5d024a86071e8f6afb50f3661ae8f01fb`.
- Submission repository branch: `paper0-schm-submission`.
- Tested pre-report packaging commit: `b793755d86725468e96d595227bc870e17addf2a`.
- Archived release tag: `schm-submission-v1.0.0`.
- Archived release commit: `e1c89dc9fb54fd7cf777b16c0746f15138324885`.
- GitHub Release: `SCHM Submission Reproducibility Snapshot v1.0.0`.
- Zenodo DOI: `10.5281/zenodo.21967188`.
- The source repository was not modified. Its two pre-existing untracked files remain there: `environment_submission.txt` and the internal literature note. The environment record was deliberately copied into the submission repository as `requirements-lock.txt`; the internal literature note was deliberately excluded.

## B. Files added

Public entry points and metadata:

- `CITATION.cff`
- `LICENSE`
- `requirements-lock.txt`
- `docs/EXPERIMENT_MAP.md`
- `docs/NUMERICAL_ANCHORS.md`
- `docs/REPRODUCTION_GUIDE.md`
- `docs/DATA_DICTIONARY.md`
- `docs/REPOSITORY_STRUCTURE.md`
- `docs/SUBMISSION_SNAPSHOT.md`
- `scripts/verify_manuscript_results.py`
- `scripts/build_manuscript_outputs.py`
- `scripts/run_smoke_test.py`
- this report

## C. Files modified

- `README.md`: replaced the starter text with the SCHM reproducibility entry point, protocol scope, evidence boundaries, environment, and commands.
- `.gitignore`: added ignored reproduction scratch/output directories.
- `requirements.txt`: replaced floating dependencies with the tested minimal pinned set.
- `environment.yml`: added a Python 3.13.6 Conda entry point using `requirements.txt`.

No experimental result, manuscript numerical anchor, model grid, split, seed, descriptor rule, sensor-layout rule, or bootstrap rule was changed.

## D. P1-P5 implementation map

| Protocol | Primary implementation | Frozen evidence |
|---|---|---|
| P1 | `scripts/sss_fast_revision/run_repeated_factorial_78d_92d_ridge_svr.py` | `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/` |
| P2 | `scripts/sss_fast_revision/run_78d_asymmetric_calibration_repeated_splits.py` | `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/` |
| P3 | `build_matched_noise_descriptor_datasets.py`, `run_clean_trained_matched_noise_robustness.py`, and `run_matched_noise_failure_mechanism_diagnostic.py` under `scripts/sss_fast_revision/` | `matched_noise_robustness_clean_trained/` and `matched_noise_failure_mechanism/` under `results/sss_fast_revision/` |
| P4 | `build_exhaustive_sensor_layout_descriptor_datasets.py` and `run_exhaustive_sensor_layout_svr.py` under `scripts/sss_fast_revision/` | `results/sss_fast_revision/exhaustive_sensor_layout_svr/` |
| P5 | `scripts/sss_fast_revision/run_sensor_layout_paired_bootstrap_closure.py` | `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/` |

The complete command order, prerequisites, and stop conditions are in `docs/REPRODUCTION_GUIDE.md`.

## E. Verification performed

### Static and packaging checks

- `git diff --check`: passed.
- New Python entry points compiled with `python -m py_compile`: passed.
- No separate pytest unit-test suite was found; `run_smoke_test.py` is the explicit execution-chain test.
- Public entry points were scanned for common secret patterns and `/Users/jerry`: no hits.
- Public entry points were scanned for the obsolete journal full name and `Online Resource`: no hits.
- The excluded internal literature note is absent from the submission repository.
- Git commit identity was normalized to the author's existing repository identity; the machine-local email was removed from the new commits.

### Level A: frozen numerical-anchor verification

Command:

```bash
python scripts/verify_manuscript_results.py
```

Status: passed. The command ended with `ALL FROZEN MANUSCRIPT ANCHORS PASSED`.

### End-to-end smoke test

Command:

```bash
python scripts/run_smoke_test.py
```

Status: passed. Twenty short OpenSeesPy cases were generated; the response array was `(20, 200, 4)`, the descriptor matrix was `(20, 92)`, and finite Ridge validation/test MAEs were obtained. These MAEs are execution diagnostics and are not manuscript results.

### Level B: manuscript-product rebuild

Command, executed in an isolated clone of the committed submission branch:

```bash
python scripts/build_manuscript_outputs.py
```

Status: passed. Final tables, Figs. 3-7, and all frozen numerical checks were regenerated without training, tuning, calibration fitting, statistical re-testing, or sensor-layout optimization.

The first isolated test correctly exposed that the older provenance-conversion script depended on a descriptor-audit CSV not distributed in Git. The public Level B wrapper was therefore corrected to begin from the tracked manuscript-source CSVs and frozen metric evidence. This boundary is documented in `docs/REPRODUCTION_GUIDE.md`.

The rebuilt plots matched all data and numerical checks, but their PDF creation timestamps and Matplotlib-generated SVG element identifiers were not byte-identical to the archived files. The frozen archived vectors remain unchanged; no claim of byte-for-byte graphical identity is made.

## F. Numerical verification summary

- P1: all four 78D/92D × Ridge/RBF-SVR MAEs passed; the reduction-style interaction magnitude and confidence interval passed.
- P2: the 7.62974596484% high-damage MAE improvement, 10/10 beneficial splits, and underestimation reduction passed.
- P3: 0/5/10/20% noise MAEs, signed-bias reversal, imposed-bound clipping fractions, and 20% high-damage median passed.
- P4: all 15 layouts, the 78-descriptor full layout, full-layout MAE, objective-dependent best two-sensor layouts, and equal-dimensional checks passed.
- P5: all 28 admissible additions were detected; 28/28 were beneficial for both overall and high-damage MAE under the frozen edge-wise bootstrap analysis.

## G. External release status and limits

- The submission snapshot is archived on Zenodo under DOI `10.5281/zenodo.21967188`.
- The corresponding GitHub release is tagged `schm-submission-v1.0.0`.
- The complete 3000-case raw/intermediate NPZ arrays remain excluded from Git and Zenodo; their regeneration path is documented.
- A full 3000-case scientific rerun was not performed during packaging.
- The existing repository history and large tracked historical output tree were preserved rather than destructively reorganized.
- The study remains simulation-only and retains the scientific limitations stated in the manuscript and README.

## H. Scientific-freeze confirmation

> No scientific experiment, random seed, model-selection rule, descriptor definition, sensor-layout definition, bootstrap rule, or frozen manuscript result was intentionally changed during the SCHM reproducibility-packaging task.
