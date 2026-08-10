# JCSHM Numerical Claim Lock

## Purpose

This document freezes the headline numerical evidence permitted for use in the
main JCSHM manuscript.

It prevents numerical results generated under different experimental protocols
from being mixed into the same scientific claim.

This file does NOT replace the source CSV/JSON files.

The source result files remain the authoritative numerical records.

When manuscript tables and figures are generated, values must be read from the
source files rather than manually re-entered from this document.

---

# 1. Numerical Reporting Rules

## Rule N1 — Protocol identity must be preserved

Every numerical statement belongs to one experimental protocol.

Results from different protocols must not be presented as though they were
directly comparable unless the manuscript explicitly explains the difference.

---

## Rule N2 — Repeated-split results are primary for estimator and information claims

Primary claims concerning:

- estimator flexibility;
- information availability;
- information × estimator complementarity;
- repeated in-distribution severe-damage behaviour;
- repeated calibration effects;

must be based on the 10 repeated paired splits.

Fixed-split results may be shown as supporting or controlled-experiment anchors,
but should not replace repeated-split evidence for these claims.

---

## Rule N3 — Controlled clean-trained results are primary for matched-noise claims

The controlled matched-noise experiment uses:

- clean training;
- clean validation;
- frozen clean-trained model;
- paired noisy test conditions.

Its clean baseline is therefore NOT numerically interchangeable with the
historical repeated-split benchmark.

---

## Rule N4 — Sensor-layout results use the controlled clean fixed split

The exhaustive sensor-layout experiment uses:

- clean structural response;
- the fixed 2100 / 450 / 450 split;
- layout-specific physically available descriptors;
- layout-specific train-only scaling;
- validation-selected RBF-SVR.

The full four-sensor layout exactly reproduces the clean matched-noise
experiment anchor.

---

## Rule N5 — Bootstrap intervals are conditional case-resampling intervals

Sensor-layout confidence intervals are:

**paired case-level percentile bootstrap intervals conditional on the current
450-case simulated test population.**

They are not:

- cross-structure confidence intervals;
- independent physical-experiment uncertainty;
- field-validation uncertainty.

---

## Rule N6 — Round only at manuscript presentation stage

Source values should remain stored at full available precision.

Recommended manuscript presentation:

- MAE / RMSE / bias:
  normally 4 significant decimal places;

- percentages:
  normally 1 decimal place;

- percentage-point changes:
  normally 1–2 decimal places;

- confidence intervals:
  precision consistent with the corresponding effect size.

Do not manually round the underlying evidence files.

---

# 2. Protocol Registry

## P1 — Historical repeated-split information × estimator benchmark

Purpose:

Evaluate information availability, estimator flexibility and their interaction.

Cases:

3000.

Repeated splits:

10.

Per split:

- train: 2100;
- validation: 450;
- test: 450.

Representation:

- 78D signal-derived-with-ground;
- 92D privileged-information reference.

Estimators:

- Ridge;
- RBF-SVR.

Important context:

The historical benchmark was generated from the original response dataset and
contains the original heterogeneous measurement-noise conditions.

This protocol is the primary basis for:

- estimator comparison;
- information comparison;
- factorial interaction;
- repeated severe-damage behaviour.

Primary sources:

- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_mean_table.csv`
- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_effect_summary.csv`
- `results/sss_fast_revision/repeated_factorial_78d_92d_ridge_svr/repeated_factorial_seed_effects.csv`

---

## P2 — Historical repeated-split asymmetric calibration benchmark

Purpose:

Evaluate whether training-only directional correction mitigates systematic
severe-damage underprediction under the in-distribution repeated-split
benchmark.

Representation:

78D signal-derived-with-ground.

Estimator:

RBF-SVR.

Calibration:

training-only OOF residual construction + one-sided monotone isotonic
correction.

Repeated splits:

10.

Primary sources:

- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_paired_statistics.csv`
- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_paired_test_results.csv`
- `results/sss_fast_revision/repeated_split_asymmetric_calibration_78d/calibration_repeated_split_report.json`

---

## P3 — Controlled clean-trained matched-noise benchmark

Purpose:

Isolate response-measurement-noise sensitivity and failure mechanism.

Training source:

clean structural response only.

Validation source:

clean structural response only.

Test conditions:

- 0% noise;
- 5% noise;
- 10% noise;
- 20% noise.

For each nonzero noise level:

5 deterministic matched stochastic noise realizations.

The underlying:

- structural case;
- damage state;
- excitation;
- case identity;

remain unchanged.

Structural response channels are perturbed.

The assumed ground/base-input signal remains unchanged.

Primary sources:

- `results/sss_fast_revision/matched_noise_robustness_clean_trained/noise_level_summary.csv`
- `results/sss_fast_revision/matched_noise_robustness_clean_trained/condition_metrics.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/severity_directional_summary.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/high_damage_bias_reversal.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/prediction_distribution_summary.csv`
- `results/sss_fast_revision/matched_noise_failure_mechanism/descriptor_shift_summary.csv`

---

## P4 — Exhaustive clean sensor-layout benchmark

Purpose:

Evaluate structural sensing observability as a function of sensor number and
placement.

Structural sensors:

4.

Ground/base-input sensor:

assumed available throughout.

Layouts:

all 15 non-empty structural sensor subsets.

Descriptor rule:

retain a descriptor only if all structural sensors required by its original
physical definition are available.

No:

- zero masking;
- descriptor redefinition.

Primary sources:

- `results/sss_fast_revision/exhaustive_sensor_layout_svr/sensor_layout_results.csv`
- `results/sss_fast_revision/exhaustive_sensor_layout_svr/sensor_count_summary.csv`
- `results/sss_fast_revision/exhaustive_sensor_layout_svr/sensor_placement_spread.csv`
- `results/sss_fast_revision/exhaustive_sensor_layout_svr/marginal_sensor_value.csv`

---

## P5 — Paired sensor-layout bootstrap closure

Purpose:

Quantify conditional case-resampling uncertainty for the frozen 15-layout
predictions.

Cases:

450.

Bootstrap replicates:

5000.

Bootstrap seed:

20260810.

Sampling unit:

complete structural case.

All four storey outputs remain together.

All layouts use the same resampled cases within every bootstrap replicate.

Confidence interval:

95% percentile bootstrap.

Primary sources:

- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/sensor_count_bootstrap_ci.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/sensor_count_step_contrasts.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/best_layout_bootstrap_stability.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/marginal_sensor_edge_bootstrap_ci.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/marginal_sensor_bootstrap_summary.csv`
- `results/sss_fast_revision/sensor_layout_paired_bootstrap_closure/within_count_pairwise_layout_contrasts.csv`

---

# 3. Claim Lock C1 — Primary 78D Ridge versus SVR Comparison

Protocol:

P1.

## Repeated-split mean performance

### 78D + Ridge

Overall MAE:

`0.0448582946`

High-damage MAE:

`0.0947165385`

High-damage absolute signed bias:

`0.0828037902`

High-damage underestimation ratio:

`0.8742895725`

---

### 78D + RBF-SVR

Overall MAE:

`0.0314500006`

High-damage MAE:

`0.0603373847`

High-damage absolute signed bias:

`0.0474323965`

High-damage underestimation ratio:

`0.7610611425`

---

## Paired improvement: SVR relative to Ridge

Overall MAE reduction:

approximately `29.9%`

Direction:

`10 / 10` repeated splits.

High-damage MAE reduction:

approximately `36.3%`

Direction:

`10 / 10` repeated splits.

High-damage absolute signed-bias reduction:

approximately `42.7%`

Direction:

`10 / 10` repeated splits.

High-damage underestimation reduction:

approximately `11.32 percentage points`

Direction:

`10 / 10` repeated splits.

---

## Permitted claim

A nonlinear estimator materially improved inference from the 78D signal-derived
descriptor representation and substantially reduced, but did not eliminate,
severe-damage underprediction.

---

## Forbidden claim

Do not write:

- "SVR solved severe-damage underprediction";
- "SVR is a novel model proposed in this study";
- "SVR is universally superior to linear models."

---

# 4. Claim Lock C2 — Complete 2 × 2 Information × Estimator Factorial

Protocol:

P1.

## Repeated-split mean table

| Representation | Estimator | Overall MAE | High-damage MAE | High |bias| | High underestimation |
|---|---|---:|---:|---:|---:|
| 78D | Ridge | 0.0448582946 | 0.0947165385 | 0.0828037902 | 0.8742895725 |
| 78D | RBF-SVR | 0.0314500006 | 0.0603373847 | 0.0474323965 | 0.7610611425 |
| 92D | Ridge | 0.0395454944 | 0.0840811314 | 0.0683545741 | 0.8172680006 |
| 92D | RBF-SVR | 0.0231048479 | 0.0443346398 | 0.0270172309 | 0.6657217161 |

---

## Information effect under Ridge

Absolute MAE difference:

`-0.0053128002`

Interpretation:

92D lower than 78D.

Relative improvement:

approximately `11.84%`.

Direction:

`10 / 10` repeated splits.

---

## Information effect under RBF-SVR

Absolute MAE difference:

`-0.0083451527`

Relative improvement:

approximately `26.48%`.

Direction:

`10 / 10` repeated splits.

---

## Estimator effect under 78D

Absolute MAE difference:

`-0.0134082939`

Interpretation:

SVR lower than Ridge.

Relative improvement:

approximately `29.90%`.

Direction:

`10 / 10`.

---

## Estimator effect under 92D

Absolute MAE difference:

`-0.0164406465`

Relative improvement:

approximately `41.57%`.

Direction:

`10 / 10`.

---

## Overall-MAE interaction / difference-in-differences

Interaction:

`-0.0030323525`

Paired interval:

approximately:

`[-0.0038546, -0.0022101]`

Direction:

`10 / 10`.

---

## High-damage MAE interaction

Interaction:

approximately `-0.0053673378`

Paired interval:

approximately:

`[-0.008574, -0.002160]`

Direction:

`9 / 10`.

---

## High-damage absolute signed-bias interaction

Interaction:

approximately `-0.00596595`

Paired interval:

approximately:

`[-0.010842, -0.001090]`

Direction:

`8 / 10`.

---

## High-damage underestimation interaction

Interaction:

approximately `-0.03831785`

Paired interval:

approximately:

`[-0.07255, -0.004085]`

Direction:

`8 / 10`.

---

## Permitted main claim

Information availability and estimator flexibility exhibited repeated-split-
supported complementarity.

The strongest and most consistent interaction evidence concerns:

- overall MAE;
- high-damage MAE.

Directional-bias interaction results should be discussed more cautiously.

---

## Forbidden claim

Do not write:

- "92D is the global upper bound";
- "the interaction proves a physical causal mechanism";
- "the repeated splits are independent physical experiments."

---

# 5. Claim Lock C3 — Asymmetric Calibration

Protocol:

P2.

Primary comparison:

standard 78D RBF-SVR versus training-only asymmetric residual calibration.

## Repeated paired effects

Overall MSE:

approximately `2.35%` improvement.

Direction:

`10 / 10`.

Overall MAE:

approximately `1.41%` improvement.

Direction:

`10 / 10`.

Damaged-state MAE:

approximately `4.05%` improvement.

High-damage MAE:

approximately `7.63%` improvement.

Direction:

`10 / 10`.

High-damage absolute signed bias:

approximately `29.27%` improvement.

Direction:

`10 / 10`.

High-damage underestimation:

standard mean approximately:

`0.76106`

calibrated mean approximately:

`0.65736`

change:

approximately `-10.37 percentage points`.

Direction:

`10 / 10`.

---

## Cost of calibration

Zero-damage MAE:

approximately `1.02%` worse.

Low-damage MAE:

approximately `7.71%` worse.

Medium-damage MAE:

approximately `0.59%` better,
but this effect was not consistently separated from zero across repeated splits.

---

## Permitted main claim

A training-only one-sided residual correction reduced severe-damage directional
underprediction under the in-distribution repeated-split benchmark, while
introducing modest penalties in the zero- and low-damage regimes.

---

## Forbidden claim

Do not write:

- "calibration universally improves reliability";
- "calibration is robust to distribution shift";
- "calibration improves every damage regime."

---

# 6. Claim Lock C4 — Controlled Clean Baseline

Protocol:

P3.

Primary estimator:

78D RBF-SVR.

Selected clean-validation hyperparameters:

- C = `1000`;
- gamma = `0.003`;
- epsilon = `0.03`.

Fixed clean test overall MAE:

`0.0220471401`

Fixed clean test RMSE:

`0.0345135277`

Fixed clean test high-damage MAE:

`0.0367799556`

Fixed clean test high-damage signed bias:

`-0.0213373048`

Fixed clean test high-damage underestimation ratio:

`0.6824817518`

---

## Important protocol warning

These clean fixed-split values MUST NOT be substituted for:

- repeated P1 78D SVR mean MAE = `0.0314500006`.

The values answer different questions.

P1:

repeated in-distribution benchmark.

P3:

clean controlled degradation anchor.

---

# 7. Claim Lock C5 — Measurement-Noise Performance Boundary

Protocol:

P3.

Primary method for the main robustness claim:

standard clean-trained 78D RBF-SVR.

## Overall MAE

### 0%

`0.0220471401`

### 5%

mean:

`0.0443593523`

SD across matched noise realizations:

`0.0005291280`

Approximate degradation relative to clean:

`+101.2%`

### 10%

mean:

`0.1118056216`

SD:

`0.0009397833`

Approximate degradation relative to clean:

`+407.1%`

### 20%

mean:

`0.2838931183`

SD:

`0.0008917419`

Approximate degradation relative to clean:

`+1187.7%`

---

## High-damage MAE

### 0%

`0.0367799556`

### 5%

mean:

`0.0582184324`

SD:

`0.0016232556`

Approximate degradation relative to clean:

`+58.3%`

### 10%

mean:

`0.1140226844`

SD:

`0.0018596110`

Approximate degradation:

`+210.0%`

### 20%

mean:

`0.2069806408`

SD:

`0.0028129007`

Approximate degradation:

`+462.8%`

---

## Permitted main claim

The clean-trained descriptor estimator exhibited pronounced nonlinear
measurement-noise sensitivity.

Even mild response-channel noise materially degraded prediction accuracy, and
10–20% noise entered a strong failure regime.

---

## Forbidden claim

Do not describe the method as:

- robust to substantial measurement noise;
- robust under 20% noise;
- field-noise validated.

---

# 8. Claim Lock C6 — Noise-Induced High-Damage Bias Reversal

Protocol:

P3.

Primary method:

standard clean-trained RBF-SVR.

## High-damage signed bias

### 0%

`-0.0213373048`

Direction:

underprediction.

### 5%

mean:

`-0.0047894942`

SD:

`0.0012721583`

Direction:

near-zero residual negative bias.

### 10%

mean:

`+0.0425749208`

SD:

`0.0036659876`

Direction:

overprediction.

### 20%

mean:

`+0.1285342352`

SD:

`0.0012963057`

Direction:

strong overprediction.

---

## High-damage underestimation ratio

### 0%

`0.6824817518`

### 5%

mean:

`0.5000000000`

### 10%

mean:

`0.2970802920`

### 20%

mean:

`0.1817518248`

---

## High-damage overestimation ratio

### 0%

approximately:

`0.3175182482`

### 5%

approximately:

`0.5000000000`

### 10%

approximately:

`0.7029197080`

### 20%

approximately:

`0.8182481752`

---

## Permitted main claim

Controlled measurement noise induced a directional reversal of severe-damage
prediction error:

underprediction
→ approximately balanced prediction
→ overprediction.

The reversal occurred between the 5% and 10% tested noise levels.

---

## Forbidden claim

Do not interpret falling underestimation ratio under noise as improving
reliability.

It reflects a transition toward overprediction.

---

# 9. Claim Lock C7 — Noise-Induced Prediction Saturation

Protocol:

P3.

Primary method:

standard RBF-SVR.

## Overall high-bound clipping ratio

### 0%

`0.0000000000`

### 5%

approximately:

`0.0035555556`

### 10%

approximately:

`0.0318888889`

### 20%

approximately:

`0.3225555556`

---

## High-damage high-bound clipping ratio

### 0%

`0.0000000000`

### 5%

approximately:

`0.0204379562`

### 10%

approximately:

`0.1270072993`

### 20%

approximately:

`0.6408759124`

---

## Key 20% observation

At 20% measurement noise:

- approximately 32.3% of all predictions reach the upper clipping boundary;
- approximately 64.1% of high-damage predictions reach the upper boundary;
- the high-damage median prediction is approximately 0.5.

---

## Permitted main claim

The severe-noise regime is partly characterized by upper-bound prediction
saturation rather than by a simple linear increase in prediction variance.

---

# 10. Claim Lock C8 — Descriptor-Space Shift under Noise

Protocol:

P3.

Descriptor shift is expressed relative to the frozen clean-training
StandardScaler.

For each feature:

delta-z = z(noisy) - z(clean).

## Most shifted example

Feature:

`story_1_spectral_centroid`

### 5% noise

mean absolute standardized shift:

approximately `0.2918`

### 10% noise

mean absolute standardized shift:

approximately `0.7175`

### 20% noise

mean absolute standardized shift:

approximately `1.4610`

At 20% noise:

fraction with absolute shift > 1 clean-training SD:

approximately `0.5058`

fraction with absolute shift > 2 clean-training SD:

approximately `0.2933`

---

## Other repeatedly shifted descriptor families

Observed among the largest shifts:

- spectral centroids;
- relative-to-lower / relative-amplitude descriptors;
- ground-amplification descriptors;
- crest-factor descriptors.

---

## Permitted claim

The largest standardized noise-induced shifts were concentrated in selected
spectral-location, relative-response, amplification and crest-factor
descriptors.

These shifts are consistent with the observed distributional failure pathway.

---

## Forbidden claim

Do not write:

- "spectral centroid caused model failure";
- "these descriptors are universally noise-sensitive."

No feature-ablation causal experiment was performed.

---

# 11. Claim Lock C9 — Sensor-Count Performance

Protocols:

P4 + P5.

## Mean overall MAE by structural sensor count

### 1 sensor

point estimate:

`0.0733744166`

95% paired case-bootstrap interval:

`[0.0703695921, 0.0764760960]`

### 2 sensors

point estimate:

`0.0581866467`

95% interval:

`[0.0557581132, 0.0606744927]`

### 3 sensors

point estimate:

`0.0400412847`

95% interval:

`[0.0382065754, 0.0419538486]`

### 4 sensors

point estimate:

`0.0220471401`

95% interval:

`[0.0206004277, 0.0236043874]`

---

## Mean high-damage MAE by sensor count

### 1 sensor

`0.2187050540`

95% interval:

`[0.2134540075, 0.2240879002]`

### 2 sensors

`0.1455292988`

95% interval:

`[0.1395322718, 0.1515807284]`

### 3 sensors

`0.0840897275`

95% interval:

`[0.0786434555, 0.0895532939]`

### 4 sensors

`0.0367799556`

95% interval:

`[0.0322723251, 0.0414528974]`

---

## Permitted main claim

Increasing structural sensing availability consistently reduced both overall and
high-damage inference error under the dependency-aware observability framework.

---

# 12. Claim Lock C10 — Consecutive Sensor-Count Improvements

Protocols:

P4 + P5.

Improvement is defined as:

MAE(k sensors) - MAE(k+1 sensors).

Positive values therefore indicate lower error after adding sensing.

---

## 1 → 2 sensors

Overall MAE improvement:

`0.0151877699`

Relative improvement:

approximately `20.70%`

95% interval:

`[0.0137089926, 0.0167357484]`

Bootstrap positive fraction:

`1.0000`

High-damage MAE improvement:

`0.0731757552`

Relative improvement:

approximately `33.46%`

95% interval:

`[0.0689184053, 0.0777117379]`

Bootstrap positive fraction:

`1.0000`

---

## 2 → 3 sensors

Overall MAE improvement:

`0.0181453620`

Relative improvement:

approximately `31.18%`

95% interval:

`[0.0168152041, 0.0194642180]`

Bootstrap positive fraction:

`1.0000`

High-damage MAE improvement:

`0.0614395713`

Relative improvement:

approximately `42.22%`

95% interval:

`[0.0580121239, 0.0648756629]`

Bootstrap positive fraction:

`1.0000`

---

## 3 → 4 sensors

Overall MAE improvement:

`0.0179941446`

Relative improvement:

approximately `44.94%`

95% interval:

`[0.0165346532, 0.0194297973]`

Bootstrap positive fraction:

`1.0000`

High-damage MAE improvement:

`0.0473097719`

Relative improvement:

approximately `56.26%`

95% interval:

`[0.0430868705, 0.0515813079]`

Bootstrap positive fraction:

`1.0000`

---

## Permitted main claim

Every consecutive increase in the structural sensor count showed an entirely
beneficial paired 95% case-bootstrap interval for both overall and high-damage
MAE.

---

# 13. Claim Lock C11 — Complete 28-Edge Sensor-Addition Lattice

Protocols:

P4 + P5.

Number of non-empty structural sensor layouts:

`15`

Number of admissible one-sensor-addition edges:

`28`

---

## Overall MAE

Point-estimate beneficial edges:

`28 / 28`

Paired 95% bootstrap CI entirely beneficial:

`28 / 28`

---

## High-damage MAE

Point-estimate beneficial edges:

`28 / 28`

Paired 95% bootstrap CI entirely beneficial:

`28 / 28`

---

## Permitted headline claim

Across all 28 admissible one-sensor additions in the exhaustive structural
sensor-subset lattice, both overall MAE and high-damage MAE decreased, and every
paired 95% case-level bootstrap interval remained entirely on the beneficial
side of zero.

---

## Required limitation

This is strong within-study evidence for the current:

- structural topology;
- descriptor formulation;
- simulated case population.

It is not a universal theorem for arbitrary civil structures or monitoring
systems.

---

# 14. Claim Lock C12 — Sensor Placement

Protocols:

P4 + P5.

## Point-estimate placement spread

### 1 sensor

best overall layout:

`{1}`

worst:

`{4}`

relative overall-MAE spread:

approximately `6.06%`

---

### 2 sensors

best overall layout:

`{1,2}`

worst:

`{2,4}`

relative overall-MAE spread:

approximately `16.89%`

---

### 3 sensors

best overall layout:

`{1,2,3}`

worst:

`{1,2,4}`

relative overall-MAE spread:

approximately `25.42%`

---

## Bootstrap best-layout stability

### 1-sensor budget

Layout `{1}`:

overall-MAE best fraction:

`0.9634`

high-damage-MAE best fraction:

`0.9972`

---

### 2-sensor budget

Overall-MAE best fractions:

- `{1,2}`:
  `0.7518`;

- `{2,3}`:
  `0.2174`;

- `{3,4}`:
  `0.0236`;

- `{1,3}`:
  `0.0072`.

High-damage-MAE best fractions:

- `{1,3}`:
  `0.3674`;

- `{1,2}`:
  `0.3100`;

- `{3,4}`:
  `0.2908`;

- `{2,3}`:
  `0.0318`.

Interpretation:

the optimal two-sensor layout is objective-dependent.

---

### 3-sensor budget

Layout `{1,2,3}`:

overall-MAE best fraction:

`1.0000`

high-damage-MAE best fraction:

`0.7948`

---

## Permitted main claim

Sensor placement materially affects inference performance under a fixed sensing
budget, but not every within-budget layout pair is statistically separable.

The preferred layout can also depend on whether the objective is:

- overall prediction accuracy;
- severe-damage accuracy.

---

# 15. Claim Lock C13 — Placement Cannot Be Explained by Descriptor Count Alone

Protocols:

P4 + P5.

Equal-dimensional examples are used to test whether feature quantity alone
explains placement performance.

---

## Layout {1,3} versus {1,4}

Descriptor dimensions:

31 versus 31.

Overall MAE difference:

approximately `-0.0022552935`

95% paired interval:

`[-0.0040888163, -0.0004184770]`

High-damage MAE difference:

approximately `-0.0095032287`

95% interval:

`[-0.0148136910, -0.0043142579]`

Interpretation:

layout {1,3} is stably better than {1,4} under this comparison despite identical
descriptor dimensionality.

---

## Layout {2,3} versus {3,4}

Descriptor dimensions:

34 versus 34.

Overall MAE interval:

approximately:

`[-0.0048236266, 0.0013665842]`

This interval crosses zero.

---

## Layout {1,2,4} versus {1,3,4}

Descriptor dimensions:

48 versus 48.

Overall MAE interval:

approximately:

`[-0.0006693638, 0.0051890107]`

This interval crosses zero.

---

## Permitted claim

Descriptor count alone cannot fully explain sensor-placement performance,
because at least one equal-dimensional layout comparison remains clearly
separated under paired bootstrap resampling.

However, several equal-dimensional placement contrasts remain uncertain.

---

# 16. Numerical Claims Reserved for Supplementary Material

The following should normally not become headline Abstract claims.

## Five-set Ridge comparison

Use for descriptor-set sensitivity and provenance support.

Main qualitative ordering:

92D privileged-information reference
<
legacy 86D
<
measured-ground augmented 86D
≈
78D signal-derived
<
59D response-only

in terms of fixed-split Ridge error.

Use exact values from the source files when preparing Supplementary tables.

---

## Measured-ground augmentation

Primary conclusion:

the additional eight measured-ground-derived frequency-ratio descriptors provide
only a small improvement over the canonical 78D representation.

Do not use this as a headline contribution.

---

## Damage-weighted SVR

Primary conclusion:

severity weighting improved high-damage metrics but failed the predeclared
robustness criterion for formal method extension.

Do not present this as a proposed method.

---

## Hyperparameter boundary investigations

Use only to demonstrate:

- validation-only model selection;
- frozen search space;
- no post-test grid expansion.

Do not use hyperparameter-boundary behaviour as a scientific result.

---

# 17. Abstract-Level Numerical Claims

The Abstract should use only a small number of high-information numerical
results.

Recommended candidates:

## Abstract number A1

78D RBF-SVR versus Ridge repeated-split overall MAE:

approximately `29.9%` reduction.

Optional accompanying high-damage result:

approximately `36.3%` high-damage MAE reduction.

---

## Abstract number A2

Measurement-noise failure:

high-damage signed bias changes from approximately:

`-0.021`

at clean conditions

to:

`+0.129`

at 20% response-measurement noise.

This is preferable to listing all four noise-level MAEs because it expresses the
mechanistic result.

---

## Abstract number A3

Sensor observability:

`28 / 28` one-sensor additions reduce both overall and high-damage MAE with
entirely beneficial paired 95% bootstrap intervals.

---

## Abstract number A4 — Optional

If space allows:

best three-sensor overall layout `{1,2,3}` is the bootstrap-best overall-MAE
configuration in:

`100%`

of 5000 paired case-bootstrap resamples.

This is secondary to the 28/28 result and may be omitted for economy.

---

# 18. Numbers That Must Never Be Combined Without Protocol Explanation

## Example 1

Repeated 78D SVR mean MAE:

`0.0314500006`

versus controlled clean fixed-split MAE:

`0.0220471401`

These values belong to different protocols.

Do not write:

"SVR achieved MAEs of 0.0315 and 0.0220 under two evaluations"

without explaining:

- repeated historical benchmark;
- clean controlled benchmark.

---

## Example 2

Repeated calibration underestimation:

approximately:

`0.7611 → 0.6574`

must not be directly combined with the controlled clean fixed-split calibration
underestimation result without explicitly identifying the protocol.

---

## Example 3

Repeated information-factorial 92D performance must not be interpreted as a
deployable field performance estimate.

92D contains privileged simulation information.

---

# 19. Evidence-Strength Vocabulary Lock

## Strongest wording

Use when supported by:

- 10/10 repeated paired directions;
- entirely separated paired interval;
- 28/28 exhaustive sensor edges.

Allowed phrases:

- consistently reduced;
- remained beneficial across all paired resamples/edges;
- repeated-split-supported;
- systematically observed;
- exhibited a clear bias reversal;
- exhaustive within the tested four-sensor subset lattice.

---

## Moderate wording

Use when:

- point estimate is consistent;
- some pairwise intervals cross zero;
- mechanism evidence is associative rather than causal.

Allowed phrases:

- suggests;
- is consistent with;
- was associated with;
- tended to;
- provides evidence that;
- cannot be explained solely by.

---

## Forbidden overstatement

Avoid:

- proves universally;
- guarantees;
- universally optimal;
- field validated;
- causal mechanism established;
- globally robust;
- general upper bound;
- independent experimental replication.

---

# 20. Frozen Headline Results

The following results are currently approved as the central numerical backbone of
the JCSHM manuscript:

1. 78D RBF-SVR reduces repeated-split overall MAE by approximately 29.9% relative
   to Ridge.

2. 78D RBF-SVR reduces repeated-split high-damage MAE by approximately 36.3%
   relative to Ridge.

3. Information availability and estimator flexibility show a repeated-split
   interaction of approximately -0.00303 in overall MAE.

4. Severe-damage underprediction persists after nonlinear estimation.

5. Training-only asymmetric calibration reduces repeated high-damage MAE by
   approximately 7.6%, high-damage |bias| by approximately 29.3%, and
   underestimation by approximately 10.37 percentage points.

6. Controlled response-measurement noise changes standard-SVR high-damage signed
   bias from approximately -0.0213 at clean conditions to +0.1285 at 20% noise.

7. At 20% controlled response noise, approximately 64.1% of high-damage
   predictions reach the upper clipping boundary.

8. Mean overall MAE decreases from approximately 0.0734 with one structural
   sensor to 0.0220 with all four sensors.

9. Every consecutive sensor-count increase has an entirely beneficial paired 95%
   bootstrap interval for both overall and high-damage MAE.

10. All 28 one-sensor additions in the complete tested sensor subset lattice have
    beneficial point estimates and entirely beneficial paired 95% bootstrap
    intervals for both overall and high-damage MAE.

11. Sensor placement effects remain after controlling for descriptor dimension
    in at least one equal-dimensional layout comparison.

12. Noise and sensor sparsity exhibit different directional failure patterns:
    noise drives bias reversal toward overprediction, whereas sensor sparsity
    intensifies severe-damage underprediction.

---

# 21. Current Status

Numerical headline claims:

**FROZEN FOR INITIAL JCSHM MANUSCRIPT DRAFTING**

No headline number should be changed from memory during manuscript writing.

If a value is uncertain:

1. return to the named source CSV/JSON;
2. verify the protocol;
3. update this lock only if a transcription error is confirmed.

Experimental models remain frozen.

No model retraining or retuning is permitted for the purpose of obtaining a more
favourable manuscript number.
