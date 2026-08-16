# Reproduction guide

Run commands from the repository root. Stop if any script produces a value outside the frozen anchors; do not change code, seeds, grids, descriptors, or manuscript values to force agreement.

## Level A — fast manuscript verification

```bash
python scripts/verify_manuscript_results.py
```

This reads tracked CSV evidence only. It does not train or tune a model.

## Level B — rebuild manuscript products from frozen evidence

```bash
python scripts/build_manuscript_outputs.py
```

The wrapper uses the tracked manuscript-source CSVs under
`manuscript/jcshm_reconstruction/figures/data/` and the tracked frozen metric
evidence, then calls these existing deterministic scripts:

1. `scripts/jcshm_manuscript/prepare_final_tables.py`
2. `scripts/jcshm_manuscript/figures/plot_F03_factorial.py`
3. `scripts/jcshm_manuscript/figures/plot_F04_severity_calibration.py`
4. `scripts/jcshm_manuscript/figures/plot_F05_noise_failure_mechanism.py`
5. `scripts/jcshm_manuscript/figures/plot_F06_sensor_layout_observability.py`
6. `scripts/jcshm_manuscript/figures/plot_F07_sensor_subset_lattice.py`
7. `scripts/verify_manuscript_results.py`

The earlier `prepare_jcshm_manuscript_data.py` provenance-conversion step is
not part of the submission-only Level B path because its upstream descriptor
audit source is not distributed in this Git snapshot; its resulting
manuscript-source CSVs are tracked instead. The historical internal directory
name is retained to preserve provenance. Level B performs no training. Figs.
1–2 are conceptual/vector objects and are not rebuilt from model training.

## Smoke test

```bash
python scripts/run_smoke_test.py
```

The default test uses 20 short cases and writes ignored files under `.reproduction_work/smoke_test/`. It checks OpenSeesPy execution, response shapes, descriptor extraction, a basic Ridge fit, and MAE computation. It is not a manuscript-metric test.

## Level C — full scientific reproduction

### Prerequisites

Large arrays are excluded from Git. Before P1–P5 can be rerun, provide or regenerate:

- `data_processed/debug_plus_3000_dataset.npz`
- `data_inputs/aes_frozen/debug_plus_3000_physics_features_mlp.npz`
- derived descriptor-set and split NPZ files under `data_processed/sss_fast_revision/`

The tracked index, feature names, SHA-256 record, and build manifests remain available for auditing. Verify the frozen inputs against `configs/sss_fast_revision/aes_frozen_sha256.txt` before proceeding.

### Numbered command sequence

The original 3000-case generation entry point is:

```bash
bash scripts/run_q2_generate_debug_plus_3000.sh
```

It uses `N_CASES=3000` and `SEED=20260701`. This historical wrapper also runs development-stage ablations; those ablations are not the current P1–P5 manuscript pathway.

Build the canonical descriptor sets:

```bash
python scripts/sss_fast_revision/build_descriptor_datasets.py
python scripts/sss_fast_revision/verify_descriptor_sets.py
```

Run P1:

```bash
python scripts/sss_fast_revision/run_repeated_factorial_78d_92d_ridge_svr.py
```

Run P2:

```bash
python scripts/sss_fast_revision/run_78d_asymmetric_calibration_repeated_splits.py
```

Build matched-noise descriptors and run P3. `RAW3000` must point to the frozen 3000-case raw dataset:

```bash
RAW3000=data_processed/debug_plus_3000_dataset.npz \
python scripts/sss_fast_revision/build_matched_noise_descriptor_datasets.py
python scripts/sss_fast_revision/run_clean_trained_matched_noise_robustness.py
python scripts/sss_fast_revision/run_matched_noise_failure_mechanism_diagnostic.py
```

Build all dependency-aware layouts and run P4:

```bash
python scripts/sss_fast_revision/build_exhaustive_sensor_layout_descriptor_datasets.py
python scripts/sss_fast_revision/run_exhaustive_sensor_layout_svr.py
```

Run P5:

```bash
python scripts/sss_fast_revision/run_sensor_layout_paired_bootstrap_closure.py
```

Rebuild manuscript products and verify:

```bash
python scripts/build_manuscript_outputs.py
python scripts/verify_manuscript_results.py
```

## Scientific stop conditions

Stop and document the discrepancy before changing anything if a rerun disagrees with a frozen anchor. In particular, do not:

- change random seeds or split assignments;
- expand a hyperparameter grid;
- use test data for selection;
- redefine descriptors, dependencies, noise, clipping, or bootstrap rules;
- replace a failed or unfavorable result;
- mix P1 repeated heterogeneous-noise results with the P3/P4 clean fixed-split anchor.

## Output locations

- P1: `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/`
- P2: `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/`
- P3: `results/sss_fast_revision/matched_noise_robustness_clean_trained/` and `matched_noise_failure_mechanism/`
- P4: `results/sss_fast_revision/exhaustive_sensor_layout_svr/`
- P5: `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/`
- Manuscript products: `manuscript/jcshm_reconstruction/figures/` and `tables/`
