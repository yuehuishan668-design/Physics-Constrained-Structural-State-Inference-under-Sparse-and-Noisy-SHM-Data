# Frozen experiment map

This map records the actual scripts and tracked evidence used for the SCHM submission snapshot. Historical development analyses remain in the repository but are not part of the P1–P5 manuscript pathway.

| Protocol | Question | Population / split | Main estimator | Actual script(s) | Tracked evidence / manuscript output |
|---|---|---|---|---|---|
| P1 | How do information representation and estimator flexibility jointly affect inference? | Historical heterogeneous-noise population; 10 predefined 2100/450/450 splits, seeds 0–9 | Ridge and RBF-SVR, 78D and 92D | `scripts/sss_fast_revision/run_repeated_factorial_78d_92d_ridge_svr.py` | `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/`; Fig. 3 and Table 2 |
| P2 | Can a training-derived directional correction reduce severe-damage underprediction in distribution? | Same repeated population and split identities as P1 | 78D RBF-SVR plus one-sided isotonic calibration | `scripts/sss_fast_revision/run_78d_asymmetric_calibration_repeated_splits.py` | `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/`; Fig. 4 |
| P3 | How does matched response noise alter the failure direction of a frozen clean-trained model? | One clean fixed 2100/450/450 split; 0/5/10/20% noise; five realizations for each nonzero level | Clean-trained 78D RBF-SVR, no retraining | `scripts/sss_fast_revision/build_matched_noise_descriptor_datasets.py`; `scripts/sss_fast_revision/run_clean_trained_matched_noise_robustness.py`; `scripts/sss_fast_revision/run_matched_noise_failure_mechanism_diagnostic.py` | `results/sss_fast_revision/matched_noise_robustness_clean_trained/`; `results/sss_fast_revision/matched_noise_failure_mechanism/`; Fig. 5 and Table 3 |
| P4 | Which descriptors remain computable for each structural-sensor layout, and how does inference change? | Same clean fixed split as P3; all 15 nonempty structural-sensor subsets; base-input sensor retained | Separate layout-specific RBF-SVR | `scripts/sss_fast_revision/build_exhaustive_sensor_layout_descriptor_datasets.py`; `scripts/sss_fast_revision/run_exhaustive_sensor_layout_svr.py` | `results/sss_fast_revision/exhaustive_sensor_layout_svr/`; Fig. 6 and Table 4 |
| P5 | What is the conditional marginal value of each admissible one-sensor addition? | Fixed 450-case P4 test population; 5000 paired case bootstraps; seed 20260810 | No training; paired bootstrap of frozen layout predictions | `scripts/sss_fast_revision/run_sensor_layout_paired_bootstrap_closure.py` | `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/`; Fig. 7 |

## Shared descriptor construction

- Initial 92D frozen feature archive: `data_inputs/aes_frozen/debug_plus_3000_physics_features_mlp.npz` (not distributed in Git).
- Frozen feature names and hashes: `configs/sss_fast_revision/aes_frozen_sha256.txt`.
- Canonical 92D/86D/78D/59D builder: `scripts/sss_fast_revision/build_descriptor_datasets.py`.
- Descriptor-set manifest: `configs/sss_fast_revision/descriptor_dataset_build_manifest.json`.
- Dependency-aware sensor-layout builder: `scripts/sss_fast_revision/build_exhaustive_sensor_layout_descriptor_datasets.py`.

## Mandatory protocol distinction

```text
P1 repeated heterogeneous-noise benchmark:
78D RBF-SVR overall MAE = 0.0314500006

P3/P4 clean fixed-split benchmark:
78D RBF-SVR overall MAE = 0.0220471401
```

The first value summarizes repeated partitions of the heterogeneous-noise population. The second is one clean fixed-split anchor shared by P3 and the full-layout P4 model. They are not interchangeable or contradictory.

## Interpretation locks

- 92D is a privileged-information reference, not a deployment-valid representation or theoretical upper bound.
- 78D is the primary signal-derived representation under an assumed available base-input history.
- Measurement noise perturbs still-available information; sensor loss removes descriptors with unsatisfied dependencies.
- Descriptor displacement is diagnostic and associative, not causal feature attribution.
- P5 intervals are conditional and edge-wise; no family-wise multiplicity correction was applied.

