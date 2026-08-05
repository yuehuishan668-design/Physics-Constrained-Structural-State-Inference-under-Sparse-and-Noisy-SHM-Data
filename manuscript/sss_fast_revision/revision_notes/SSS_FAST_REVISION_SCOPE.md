# SSS Fast Revision Scope

## Source version

- Frozen source branch/tag: paper0-aes-frozen-2026-08-05
- Working branch: paper0-sss-fast-revision
- Target journal: Smart Structures and Systems

## First-stage frozen inputs

- debug_plus_3000_physics_features_mlp.npz
- debug_plus_3000_physics_feature_names.csv
- debug_plus_3000_dataset_index.csv

## Immediate objectives

1. Audit all 92 descriptor names.
2. Separate Oracle and Deployable descriptor sets.
3. Remove the duplicated no_meta / physics_no_meta_core comparison.
4. Reproduce the original sklearn baseline from frozen inputs.
5. Add SVR and boosting only after baseline reproduction succeeds.

## Later experiments

1. Matched paired-noise evaluation.
2. Exhaustive sensor-layout evaluation.
3. Damage-severity and underestimation diagnostics.

## Excluded from this fast revision

- New deep neural architecture
- PINN or differentiable physics
- Cross-structure transfer
- UQ and OOD detection
- Digital-twin claims
