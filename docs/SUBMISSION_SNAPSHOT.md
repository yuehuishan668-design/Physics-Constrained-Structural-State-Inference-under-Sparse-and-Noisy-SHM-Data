# SCHM submission snapshot

## Identity

- Manuscript: *Descriptor Observability and Failure Mechanisms of Physics-Guided Structural Damage Inference under Sparse and Noisy Sensing*
- Target journal: Structural Control and Health Monitoring
- Scientific source branch: `paper0-jcshm-manuscript`
- Submission branch: `paper0-schm-submission`
- Scientific base commit: `c78af2c5`
- Archived release tag: `schm-submission-v1.0.0`
- Archived release commit: `e1c89dc9fb54fd7cf777b16c0746f15138324885`
- Zenodo DOI: `10.5281/zenodo.21967188`
- Zenodo record: `https://doi.org/10.5281/zenodo.21967188`
- Packaging date: 2026-08-16

The submission snapshot was released on GitHub as
`schm-submission-v1.0.0` and archived on Zenodo under DOI
`10.5281/zenodo.21967188`.

The release/tag remains the immutable scientific submission snapshot.
Subsequent commits on `paper0-schm-submission` are metadata-only unless
explicitly documented otherwise.

## Scientific freeze status

The manuscript experiments, seeds, split assignments, descriptor definitions, estimator grids, clipping bounds, sensor dependencies, bootstrap rules, and headline values are frozen. Packaging adds reviewer-facing documentation and deterministic verification/build entry points only.

## Dataset and protocols

- 3000 simulated successful cases: 596 healthy, 1208 with one damaged storey, and 1196 with two damaged storeys.
- P1: repeated 78D/92D × Ridge/RBF-SVR factorial on the heterogeneous-noise population.
- P2: repeated training-derived asymmetric calibration.
- P3: clean-trained matched-noise stress test without retraining.
- P4: all 15 nonempty dependency-aware structural-sensor layouts.
- P5: all 28 admissible one-sensor additions with paired case bootstrap.

## Frozen environment

| Component | Verified version |
|---|---|
| Python | 3.13.6 |
| OpenSeesPy | 3.8.0.0 (reported as 3.8.0 in the supplementary record) |
| NumPy | 2.5.1 |
| Pandas | 3.0.5 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| Matplotlib | 3.11.1 |

## Verification

```bash
python scripts/verify_manuscript_results.py
python scripts/run_smoke_test.py
python scripts/build_manuscript_outputs.py
```

Level A verifies frozen evidence without training. The smoke test checks execution but not manuscript metrics. Level B rebuilds manuscript products from frozen evidence.

## Known limitations

- Full 3000-case reproduction was not performed during packaging.
- Large NPZ arrays are not tracked in Git and must be regenerated or supplied locally.
- Evidence is simulation based and limited to one structural topology.
- P5 intervals are conditional and edge-wise; no family-wise correction was applied.

## Data policy and license

Processed evidence needed for fast verification is tracked. Large regenerable arrays remain ignored. Code and documentation are provided under the MIT License.
