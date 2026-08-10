# JCSHM Figure / Table / Evidence Map

## Purpose

This document freezes the relationship among:

1. scientific questions;
2. manuscript claims;
3. experimental protocols;
4. numerical evidence;
5. main figures;
6. main tables;
7. supplementary material.

No figure or table should combine numerical results produced under incompatible
experimental protocols.

The manuscript must distinguish among:

- historical heterogeneous-noise benchmark;
- controlled clean-trained matched-noise experiment;
- repeated paired train/validation/test splits;
- fixed historical split;
- exhaustive clean sensor-layout experiment;
- paired case-level bootstrap uncertainty.

---

# 1. Evidence Hierarchy

## Tier A — Primary evidence

Evidence suitable for primary manuscript claims.

### A1. Repeated 2 × 2 information × estimator factorial

Protocol:

- 3000 simulated structural cases;
- 10 deterministic repeated train/validation/test splits;
- 2100 / 450 / 450 cases per split;
- train-only standardisation;
- validation-only hyperparameter selection;
- clipped predictions in [0, 0.5];
- paired comparisons because all four conditions use identical splits.

Primary conditions:

- 78D + Ridge;
- 78D + RBF-SVR;
- 92D + Ridge;
- 92D + RBF-SVR.

Primary source files:

- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_mean_table.csv`
- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_effect_summary.csv`
- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_seed_effects.csv`
- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_report.json`

Scientific role:

Separate:

1. information-availability effect;
2. estimator-flexibility effect;
3. information × estimator complementarity.

Primary interpretation:

The nonlinear estimator extracts substantially more value from the
signal-derived representation than Ridge, while access to privileged
simulation information further improves both estimator classes.

Permitted wording:

- information effect;
- estimator effect;
- repeated-split-supported complementarity;
- interaction / difference-in-differences;
- privileged-information reference.

Forbidden wording:

- causal interaction in the physical system;
- global upper bound;
- independent experimental replication.

---

### A2. Repeated 78D Ridge versus SVR comparison

Primary source files:

- `results/sss_fast_revision/repeated_split_78d_ridge_svr/repeated_split_metric_summary.csv`
- `results/sss_fast_revision/repeated_split_78d_ridge_svr/repeated_split_paired_statistics.csv`
- `results/sss_fast_revision/repeated_split_78d_ridge_svr/repeated_split_damage_bins.csv`
- `results/sss_fast_revision/repeated_split_78d_ridge_svr/repeated_split_seed_metrics.csv`
- `results/sss_fast_revision/repeated_split_78d_ridge_svr/repeated_split_report.json`

Scientific role:

Establish that the descriptor-to-damage mapping contains consequential
nonlinearity.

Key evidence already frozen:

- mean overall MAE:
  Ridge ≈ 0.04486;
  SVR ≈ 0.03145;

- SVR improvement in overall MAE:
  ≈ 29.9%;

- high-damage MAE reduction:
  ≈ 36.3%;

- high-damage absolute signed-bias reduction:
  ≈ 42.7%;

- improvement direction:
  10 / 10 repeated splits.

Primary interpretation:

RBF-SVR is not presented as a new algorithm.

It is used as evidence that estimator flexibility materially affects the
ability to exploit the deployment-oriented descriptor space.

---

### A3. Repeated asymmetric residual calibration

Primary source files:

- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_paired_statistics.csv`
- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_paired_test_results.csv`
- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_oof_residual_summary.csv`
- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_alpha_frequency.csv`
- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_repeated_split_report.json`

Scientific role:

Demonstrate that severe-damage underprediction is directional rather than
merely an increase in random error.

Key repeated-split evidence:

Calibration relative to standard SVR:

- overall MAE improved ≈ 1.4%;
- damaged MAE improved ≈ 4.1%;
- high-damage MAE improved ≈ 7.6%;
- high-damage absolute signed bias improved ≈ 29.3%;
- high-damage underestimation decreased ≈ 10.37 percentage points;
- high-damage improvements occurred in 10 / 10 repeated splits;
- low- and zero-damage MAE increased.

Scientific interpretation:

The calibration is a reliability-oriented directional correction.

It is not a new machine-learning algorithm and not a universally robust
post-processing method.

---

### A4. Controlled clean-trained matched measurement-noise experiment

Primary performance source files:

- `results/sss_fast_revision/matched_noise_robustness_clean_trained/noise_level_summary.csv`
- `results/sss_fast_revision/matched_noise_robustness_clean_trained/condition_metrics.csv`
- `results/sss_fast_revision/matched_noise_robustness_clean_trained/degradation_vs_clean.csv`
- `results/sss_fast_revision/matched_noise_robustness_clean_trained/calibration_benefit_summary.csv`
- `results/sss_fast_revision/matched_noise_robustness_clean_trained/matched_noise_robustness_report.json`

Primary mechanism source files:

- `results/sss_fast_revision/matched_noise_failure_mechanism/severity_directional_summary.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/high_damage_bias_reversal.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/prediction_distribution_summary.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/descriptor_shift_summary.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/descriptor_shift_error_association_summary.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/matched_noise_failure_mechanism_report.json`

Protocol:

- canonical structural source:
  clean structural acceleration histories;

- model training:
  clean train only;

- model selection:
  clean validation only;

- structural-response measurement-noise levels:
  0%, 5%, 10%, 20%;

- nonzero conditions:
  five independent deterministic matched noise realizations per level;

- same underlying cases, damage states and excitation properties;

- ground/base-input signal:
  kept unchanged;

- no noise-aware retraining;

- no test-driven retuning.

Key standard-SVR high-damage signed bias:

- 0%:
  approximately -0.02134;

- 5%:
  approximately -0.00479;

- 10%:
  approximately +0.04257;

- 20%:
  approximately +0.12853.

Scientific interpretation:

Measurement noise does not merely increase error magnitude.

Observed pathway:

measurement noise
→ descriptor-space shift
→ prediction inflation/broadening
→ high-damage bias reversal
→ high-noise saturation.

Permitted wording:

- controlled measurement-noise sensitivity;
- robustness boundary;
- descriptor-space distribution shift;
- noise-induced bias reversal;
- prediction saturation;
- mechanism-consistent association.

Forbidden wording:

- robust to substantial noise;
- all-sensor noise robustness;
- field measurement noise;
- proof that a specific descriptor causally causes failure.

---

### A5. Exhaustive dependency-aware sensor-layout experiment

Primary point-estimate source files:

- `results/sss_fast_revision/exhaustive_sensor_layout_svr/sensor_layout_results.csv`
- `results/sss_fast_revision/exhaustive_sensor_layout_svr/sensor_count_summary.csv`
- `results/sss_fast_revision/exhaustive_sensor_layout_svr/sensor_layout_ranking.csv`
- `results/sss_fast_revision/exhaustive_sensor_layout_svr/sensor_placement_spread.csv`
- `results/sss_fast_revision/exhaustive_sensor_layout_svr/marginal_sensor_value.csv`
- `results/sss_fast_revision/exhaustive_sensor_layout_svr/marginal_sensor_value_summary.csv`
- `results/sss_fast_revision/exhaustive_sensor_layout_svr/exhaustive_sensor_layout_svr_report.json`

Primary uncertainty source files:

- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/layout_bootstrap_ci.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/sensor_count_bootstrap_ci.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/sensor_count_step_contrasts.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/best_layout_bootstrap_stability.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/marginal_sensor_edge_bootstrap_ci.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/marginal_sensor_bootstrap_summary.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/within_count_pairwise_layout_contrasts.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/sensor_layout_paired_bootstrap_report.json`

Protocol:

- clean structural response;
- base-input / ground signal always available;
- four structural response sensors;
- all 15 non-empty structural sensor subsets;
- dependency-aware physically available descriptors;
- no zero masking;
- no descriptor redefinition;
- layout-specific train-only StandardScaler;
- same frozen 75-candidate RBF-SVR validation grid;
- test evaluated once.

Bootstrap closure:

- 450 test cases;
- complete case is bootstrap sampling unit;
- four storey outputs remain together;
- 5000 paired bootstrap replicates;
- same resampled cases across all layouts;
- percentile 95% confidence intervals.

Key evidence:

Mean overall MAE by sensor count:

- 1 sensor:
  ≈ 0.07337;

- 2 sensors:
  ≈ 0.05819;

- 3 sensors:
  ≈ 0.04004;

- 4 sensors:
  ≈ 0.02205.

All consecutive sensor-count improvements:

- 1 → 2;
- 2 → 3;
- 3 → 4;

have paired 95% bootstrap intervals entirely above zero for both:

- overall MAE improvement;
- high-damage MAE improvement.

Complete subset lattice:

- 28 / 28 one-sensor additions improve overall MAE at the point estimate;
- 28 / 28 have entirely beneficial paired 95% bootstrap intervals;
- 28 / 28 improve high-damage MAE at the point estimate;
- 28 / 28 have entirely beneficial paired 95% bootstrap intervals.

Scientific interpretation:

Sensor sparsity produces observability loss rather than the same
distribution-shift mechanism observed under measurement noise.

Permitted wording:

- sensor observability;
- dependency-aware descriptor availability;
- sensor-count effect;
- sensor-placement effect;
- marginal sensing value;
- paired bootstrap support.

Forbidden wording:

- universal optimal sensor placement;
- field-proven monitoring design;
- cross-structure statistical confidence;
- causal proof that descriptor count determines performance.

---

# 2. Main Figure Architecture

# Figure 1 — Scientific concept and experimental architecture

## Proposed title

**Deployment-aware structural damage inference and the two sensing-degradation pathways**

## Purpose

This is the conceptual anchor of the manuscript.

It must answer immediately:

1. what is measured;
2. what information reaches the estimator;
3. where information provenance enters;
4. where observability enters;
5. what reliability means;
6. how noise and sensor loss differ.

## Suggested structure

Left-to-right main pathway:

Structural system
→ structural/base-input signals
→ deployment-accessible information
→ physics-guided descriptors
→ estimator
→ storey-level damage state
→ reliability diagnostics.

Two lower branches:

### Measurement-noise branch

response noise
→ descriptor distribution shift
→ prediction-distribution shift
→ bias reversal / saturation.

### Sensor-sparsity branch

sensor removal
→ descriptor unavailability
→ cross-storey observability loss
→ severe-damage underprediction.

## Numerical evidence

No numerical results required.

This is a conceptual diagram.

## Scientific claim

Trustworthy structural damage inference depends on information provenance,
estimator capability, observability and distribution stability.

## Avoid

Do not show 92D as the proposed deployable method.

---

# Figure 2 — Information provenance and descriptor observability

## Proposed title

**Information provenance hierarchy and dependency-aware descriptor availability**

## Panel A — Descriptor information hierarchy

Show:

92D privileged-information reference
↓
86D legacy simulation-informed reference
↓
78D primary signal-derived representation
↓
59D structural-response-only pressure-test representation.

Include descriptor dimension and source categories.

Primary evidence:

- `results/sss_fast_revision/tables/T01_descriptor_audit.csv`
- canonical descriptor-set manifests under `configs/sss_fast_revision/`

## Panel B — Descriptor-source decomposition

Categories:

- generator metadata;
- generator-frequency-derived;
- measured/sensed base-input dependent;
- structural-response dependent.

## Panel C — Sensor dependency concept

Show examples:

- story_i statistics:
  requires story i;

- story_1 relative-to-lower:
  requires story 1 + ground;

- story_i relative-to-lower for i > 1:
  requires stories i-1 and i;

- adjacent-story ratio/correlation:
  requires both adjacent sensors;

- spatial fraction:
  requires all four structural sensors.

## Panel D — Example layouts

Show several representative layouts and retained descriptor dimensions:

- {1};
- {1,2};
- {2,4};
- {1,2,3};
- {1,2,3,4}.

## Scientific claim

Deployment accessibility and sensor layout determine the physically observable
descriptor space.

## Avoid

Do not interpret higher descriptor dimension automatically as better
information quality.

---

# Figure 3 — Information × estimator capability

## Proposed title

**Complementary effects of information availability and estimator flexibility**

## Primary data source

- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_mean_table.csv`
- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_effect_summary.csv`

## Panel A

Repeated-split mean overall MAE for:

- 78D + Ridge;
- 78D + SVR;
- 92D + Ridge;
- 92D + SVR.

Use uncertainty across the 10 repeated splits.

## Panel B

High-damage MAE for the same four conditions.

## Panel C

High-damage absolute signed bias or underestimation ratio.

## Panel D

Effect decomposition:

- information effect under Ridge;
- information effect under SVR;
- model effect under 78D;
- model effect under 92D;
- difference-in-differences interaction.

## Primary scientific claims

1. nonlinear estimation materially improves the 78D deployable representation;
2. privileged information also improves inference;
3. information availability and estimator flexibility show repeated-split
   complementarity.

## Evidence strength

Strongest for:

- overall MAE;
- high-damage MAE.

More cautious for:

- high-damage bias;
- underestimation interaction.

## Forbidden wording

Do not call 92D the global upper bound.

---

# Figure 4 — Severity-dependent reliability and directional calibration

## Proposed title

**Damage-severity-dependent underprediction and training-only directional calibration**

## Primary baseline source

- `results/sss_fast_revision/repeated_split_78d_ridge_svr/repeated_split_damage_bins.csv`
- `results/sss_fast_revision/repeated_split_78d_ridge_svr/repeated_split_metric_summary.csv`

## Primary calibration source

- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_paired_statistics.csv`
- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_paired_test_results.csv`
- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_oof_residual_summary.csv`

## Panel A

MAE by damage severity:

- zero;
- low;
- medium;
- high.

Compare:

- Ridge;
- SVR.

## Panel B

High-damage signed bias / absolute signed bias.

## Panel C

High-damage underestimation ratio.

## Panel D

Standard SVR versus calibrated SVR repeated-split effects.

Show:

- overall MAE;
- high-damage MAE;
- high |bias|;
- underestimation ratio.

## Scientific claims

1. good overall accuracy does not eliminate severe-damage directional failure;
2. nonlinear estimation reduces but does not eliminate severe-damage
   underprediction;
3. training-only asymmetric calibration reduces severe-damage directional bias
   under the in-distribution repeated-split benchmark;
4. the correction incurs small error penalties in zero/low-damage states.

## Avoid

Do not call calibration universally reliable.

---

# Figure 5 — Matched-noise failure mechanism

## Proposed title

**Measurement-noise-induced descriptor shift, bias reversal and prediction saturation**

## Panel A — Prediction error versus noise

Primary source:

- `results/sss_fast_revision/matched_noise_robustness_clean_trained/noise_level_summary.csv`

Show standard SVR:

- overall MAE;
- high-damage MAE;

at:

- 0%;
- 5%;
- 10%;
- 20%.

Use mean ± variability over matched noise realizations for nonzero levels.

## Panel B — High-damage signed bias reversal

Primary source:

- `results/sss_fast_revision/matched_noise_failure_mechanism/severity_directional_summary.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/high_damage_bias_reversal.csv`

Expected standard-SVR trend:

- negative at 0%;
- near zero at 5%;
- positive at 10%;
- strongly positive at 20%.

Include calibrated SVR as secondary curve if visual complexity remains acceptable.

## Panel C — Prediction saturation

Primary source:

- `results/sss_fast_revision/matched_noise_failure_mechanism/prediction_distribution_summary.csv`

Show:

- high-bound clipping ratio;
- optionally low-bound clipping ratio;

for:

- all cases;
- high-damage cases.

## Panel D — Descriptor-space shift

Primary source:

- `results/sss_fast_revision/matched_noise_failure_mechanism/descriptor_shift_summary.csv`

Show top shifted descriptor families at:

- 5%;
- 10%;
- 20%.

Candidate descriptors include:

- spectral centroid;
- relative-to-lower / relative amplitude measures;
- amplification;
- crest factor.

## Scientific claim

The observed noise failure is consistent with a descriptor-distribution-shift
pathway leading to prediction inflation, bias reversal and saturation.

## Evidence limitation

Descriptor-shift/error association is descriptive.

Do not claim feature-level causality.

---

# Figure 6 — Exhaustive sensor-layout observability

## Proposed title

**Inference performance across all 15 dependency-aware structural sensor layouts**

## Primary source

- `results/sss_fast_revision/exhaustive_sensor_layout_svr/sensor_layout_results.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/layout_bootstrap_ci.csv`

## Panel A

All 15 layouts grouped by sensor count.

X-axis:

sensor layout.

Y-axis:

overall MAE.

Include paired case-bootstrap 95% CI where practical.

## Panel B

High-damage MAE for the same layouts.

## Panel C

Available descriptor count by layout.

This panel is descriptive and should not imply that descriptor count alone
determines inference accuracy.

## Scientific claims

1. increasing structural sensing availability substantially improves inference;
2. layouts with identical sensor count can yield different inference quality;
3. sensor placement therefore contributes beyond sensor count alone.

## Important nuance

Not every within-count pair is statistically distinguishable.

---

# Figure 7 — Sensor subset lattice and marginal sensing value

## Proposed title

**Complete structural-sensor subset lattice and paired marginal value of additional sensing**

## Primary source

- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/marginal_sensor_edge_bootstrap_ci.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/marginal_sensor_bootstrap_summary.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/sensor_count_step_contrasts.csv`

## Structure

15 nodes:

all non-empty sensor subsets.

28 directed edges:

S → S ∪ {j}.

Node encoding:

- sensor layout;
- overall MAE or sensor count.

Edge encoding:

- marginal MAE improvement;
- optional line thickness by magnitude.

## Central visual result

All 28 one-sensor additions:

- reduce overall MAE;
- reduce high-damage MAE;
- retain entirely beneficial paired 95% bootstrap intervals.

## Panel B or inset

Sensor-count mean improvement:

1 → 2;
2 → 3;
3 → 4.

## Scientific claim

Additional structural sensing was monotonically beneficial across the complete
tested subset lattice under the current simulated structural system.

## Required limitation

This is not a universal sensor-placement theorem.

---

# 3. Main Table Architecture

# Table 1 — Descriptor provenance and deployment role

## Proposed title

**Physics-guided descriptor sets, information provenance and deployment interpretation**

## Sources

- `results/sss_fast_revision/tables/T01_descriptor_audit.csv`
- canonical descriptor-set manifests in `configs/sss_fast_revision/`
- five-set Ridge comparison for supporting predictive context.

## Columns

1. descriptor set;
2. dimension;
3. generator metadata included?;
4. exact generator-frequency-derived features included?;
5. base-input-dependent descriptors included?;
6. structural-response descriptors included?;
7. deployment interpretation;
8. manuscript role.

## Rows

- 92D privileged-information reference;
- 86D legacy simulation-informed reference;
- 78D primary signal-derived representation;
- 59D structural-response-only pressure test.

Measured-ground-augmented 86D:

move to Supplementary unless needed for reviewer clarification.

---

# Table 2 — Primary information × estimator results

## Proposed title

**Repeated-split performance of deployment-oriented and privileged-information representations**

## Primary sources

- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_mean_table.csv`
- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_effect_summary.csv`

## Main rows

- 78D Ridge;
- 78D SVR;
- 92D Ridge;
- 92D SVR.

## Columns

- overall MAE;
- RMSE;
- damaged MAE;
- high-damage MAE;
- high-damage |bias|;
- high-damage underestimation ratio.

Report:

mean ± SD across repeated splits.

Add a compact effect block:

- information effect;
- estimator effect;
- interaction.

Do not place full hyperparameters in this table.

---

# Table 3 — Controlled matched-noise robustness boundary

## Proposed title

**Clean-trained inference under controlled matched structural-response measurement noise**

## Primary sources

- `results/sss_fast_revision/matched_noise_robustness_clean_trained/noise_level_summary.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/severity_directional_summary.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/prediction_distribution_summary.csv`

## Rows

- 0%;
- 5%;
- 10%;
- 20%.

## Columns

- overall MAE;
- high-damage MAE;
- high-damage signed bias;
- high-damage underestimation ratio;
- high-damage overestimation ratio;
- high-bound clipping ratio.

Primary method:

standard clean-trained SVR.

Calibrated results may be shown in a secondary block or Supplementary to avoid
overloading the main table.

---

# Table 4 — Sensor observability and bootstrap support

## Proposed title

**Sensor-count, placement and marginal-observability effects**

## Primary sources

- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/sensor_count_bootstrap_ci.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/sensor_count_step_contrasts.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/best_layout_bootstrap_stability.csv`
- `results/sss_fast_revision/exhaustive_sensor_layout_svr/sensor_placement_spread.csv`

## Main rows

Sensor counts:

- 1;
- 2;
- 3;
- 4.

## Columns

- number of layouts;
- mean available descriptor count;
- mean overall MAE;
- 95% bootstrap CI;
- best layout;
- worst layout;
- placement spread;
- mean high-damage MAE;
- high-damage MAE 95% CI.

Add compact footnote:

All 28 one-sensor additions had beneficial point estimates and entirely
beneficial paired 95% bootstrap intervals for both overall and high-damage MAE.

---

# 4. Supplementary Material Architecture

# Supplementary Note S1 — Descriptor provenance audit

Include:

- complete 92-feature audit;
- source classifications;
- duplicate-column analysis;
- canonical descriptor-set manifests;
- exact feature names.

Sources:

- `results/sss_fast_revision/tables/T01_descriptor_audit.csv`
- `results/sss_fast_revision/tables/T02_exact_duplicate_descriptor_columns.csv`
- descriptor manifests under `configs/sss_fast_revision/`.

---

# Supplementary Note S2 — Canonical descriptor-set sensitivity

Include:

- 92D;
- legacy 86D;
- measured-ground augmented 86D;
- 78D;
- 59D.

Sources:

- `results/sss_fast_revision/five_set_ridge_comparison/`

Scientific role:

show that:

- measured-ground augmentation provides only a small gain over 78D;
- structural-response-only 59D experiences a substantial performance penalty.

---

# Supplementary Note S3 — Estimator screening

Include:

- Ridge;
- ElasticNet;
- Random Forest;
- HistGradientBoosting;
- RBF-SVR.

Sources:

- `results/sss_fast_revision/multimodel_78d/`

Main manuscript should not become a model-zoo comparison.

---

# Supplementary Note S4 — Ridge and SVR hyperparameter sensitivity

Include:

- extended Ridge grid;
- Ridge near-OLS response-only check;
- SVR local refinement;
- final boundary checks;
- repeated selected-hyperparameter frequencies.

Sources:

- `results/sss_fast_revision/fixed_split_ridge_extended_grid/`
- `results/sss_fast_revision/svr_validation_refinement_78d/`
- `results/sss_fast_revision/svr_local_search_78d/`
- `results/sss_fast_revision/svr_final_boundary_78d/`

Important wording:

Some selected parameters occur at frozen search-grid boundaries.

No post-test grid expansion was performed.

---

# Supplementary Note S5 — Damage-weighted SVR pilot

Sources:

- `results/sss_fast_revision/damage_weighted_svr_validation_pilot/`
- `results/sss_fast_revision/repeated_split_weighted_svr_78d/`

Scientific role:

document the tested alternative severe-damage intervention.

Final interpretation:

weighting improved severe-damage metrics but did not satisfy the predeclared
robustness threshold for formal method extension.

Do not present this as a main proposed method.

---

# Supplementary Note S6 — Asymmetric calibration details

Include:

- validation candidate grid;
- OOF residual statistics;
- correction curves;
- alpha selection frequency;
- repeated paired test statistics.

Sources:

- `results/sss_fast_revision/asymmetric_residual_calibration_validation_pilot/`
- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/`

---

# Supplementary Note S7 — Matched-noise construction and provenance

Include:

- historical noise provenance;
- historical 92D reproduction lock;
- clean-source lock;
- deterministic matched-noise protocol;
- replicate construction;
- realized noise ratios.

Scientific role:

demonstrate that clean/noisy conditions are truly paired.

Do not mix historical heterogeneous-noise datasets with the controlled
clean-trained experiment.

---

# Supplementary Note S8 — Full descriptor-shift diagnostics

Include:

- descriptor shift by condition;
- all descriptor rankings;
- case-level displacement;
- shift/error associations;
- prediction-distribution summaries.

Sources:

- `results/sss_fast_revision/matched_noise_failure_mechanism/`

---

# Supplementary Note S9 — Full exhaustive sensor-layout results

Include:

- all 15 layout point estimates;
- all selected SVR hyperparameters;
- all 27 within-count pairwise contrasts;
- all 28 marginal sensor-addition edges;
- best-layout bootstrap stability.

Sources:

- `results/sss_fast_revision/exhaustive_sensor_layout_svr/`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/`

---

# Supplementary Note S10 — Reproducibility and integrity checks

Include:

- frozen source hash;
- historical descriptor reproduction;
- canonical set dimension checks;
- full-layout sensor descriptor anchor;
- full-layout sensor model anchor;
- bootstrap input reconstruction anchor;
- no-training / no-retuning diagnostic declarations.

Purpose:

support reproducibility without interrupting the scientific narrative of the
main manuscript.

---

# 5. Protocol Separation Rules

## Rule 1

Do not mix:

historical heterogeneous-noise repeated-split results

with:

controlled clean-trained matched-noise results.

They answer different scientific questions.

---

## Rule 2

Use repeated-split factorial evidence for primary claims about:

- estimator effect;
- information effect;
- complementarity.

Do not use the fixed split as the main inferential basis for these claims.

---

## Rule 3

Use the fixed clean split only for:

- controlled matched-noise perturbation;
- exhaustive sensor-layout evaluation;

where identical cases and frozen models are required for paired degradation
analysis.

---

## Rule 4

Use paired 5000-case-bootstrap results for uncertainty statements about the
sensor-layout test set.

Do not interpret these intervals as independent-structure or field-validation
uncertainty.

---

## Rule 5

Do not report only underestimation ratio under noise.

Signed bias, MAE and overestimation must be interpreted jointly because noise
causes directional bias reversal.

---

# 6. Main Claims and Their Evidence Strength

## Claim C1 — Deployment accessibility matters

Evidence:

descriptor provenance hierarchy + predictive comparisons.

Strength:

strong methodological claim.

Do not claim universal quantitative effect size.

---

## Claim C2 — Estimator flexibility materially matters

Evidence:

10 repeated paired splits.

Strength:

strong within-current-simulation evidence.

---

## Claim C3 — Information and estimator flexibility are complementary

Evidence:

complete repeated 2 × 2 factorial.

Strength:

strong for overall MAE and high-damage MAE;
moderate for directional-bias interaction metrics.

---

## Claim C4 — Severe damage is systematically underpredicted in-distribution

Evidence:

severity diagnostics across repeated splits.

Strength:

strong within current simulated population.

---

## Claim C5 — Training-only asymmetric calibration mitigates severe underprediction

Evidence:

10 repeated paired splits.

Strength:

strong in-distribution evidence.

Required caveat:

not robust to the controlled noise-induced directional shift.

---

## Claim C6 — Measurement noise creates a bias-reversal failure pathway

Evidence:

matched clean/noisy cases;
five stochastic realizations at each nonzero noise level;
descriptor shift;
signed bias;
prediction saturation.

Strength:

strong mechanistic evidence within the specified Gaussian response-noise model.

Do not claim universal field-noise behaviour.

---

## Claim C7 — Sensor loss reduces observability and strongly degrades severe-damage inference

Evidence:

complete 15-layout experiment.

Strength:

strong within current structural system.

---

## Claim C8 — Adding a structural sensor is consistently beneficial across the tested subset lattice

Evidence:

28 / 28 point-improvement edges;
28 / 28 paired 95% bootstrap CI-positive edges for overall MAE;
28 / 28 for high-damage MAE.

Strength:

very strong within current structural system and simulated test population.

Required caveat:

not a universal theorem for arbitrary structures.

---

## Claim C9 — Placement matters independently of sensor count

Evidence:

within-count layout spreads and pairwise bootstrap contrasts.

Strength:

strong general within-study claim.

Refinement:

not every placement pair is statistically distinguishable.

---

## Claim C10 — Descriptor count alone cannot explain placement performance

Evidence:

some equal-dimensional layouts show stable performance separation.

Strength:

moderate.

Do not claim all equal-dimensional layouts differ.

---

# 7. Main-Text Economy Rules

The main manuscript should prioritize:

1. information provenance;
2. repeated factorial inference;
3. severity-dependent reliability;
4. matched-noise failure mechanism;
5. exhaustive sensor observability.

Move to Supplementary:

- model screening details;
- hyperparameter grids;
- failed weighting extension;
- full descriptor rankings;
- all pairwise sensor contrasts;
- reproducibility tables.

The manuscript should read as one scientific argument, not as a chronology of
experiments.

---

# 8. Frozen Main Manuscript Inventory

Main figures:

1. conceptual architecture;
2. information provenance and descriptor observability;
3. information × estimator factorial;
4. severity-dependent reliability and calibration;
5. matched-noise failure mechanism;
6. exhaustive sensor-layout observability;
7. sensor subset lattice and marginal sensing value.

Main tables:

1. descriptor provenance hierarchy;
2. repeated information × estimator results;
3. controlled measurement-noise results;
4. sensor observability and bootstrap summary.

Supplementary notes:

S1–S10 as defined above.

---

# 9. Current Decision

Figure/table/evidence architecture:

**FROZEN FOR INITIAL JCSHM MANUSCRIPT DRAFTING**

Changes should be made only if:

1. visual redundancy becomes evident during figure production;
2. the JCSHM manuscript format requires consolidation;
3. internal manuscript review identifies duplicated scientific claims;
4. peer review requests additional presentation changes.

No new experiment should be initiated merely to populate a figure or table.
