# Literature Block 2 — Measurement Noise, Uncertainty, and Directional Failure

## Status

TARGETED LITERATURE CORE — BLOCK 2

Purpose:

Support Introduction Gap 2 and Discussion Sections 4.3–4.7.

This block establishes the literature-supported distinction between:

    conventional noise robustness / uncertainty assessment

and the present manuscript's focus on:

    directional failure-mode transition under controlled measurement noise.

The manuscript MUST NOT claim that measurement-noise analysis, damage
underestimation, or uncertainty quantification are themselves novel.

---

# 1. Literature-supported problem

Measurement noise, environmental variability, sensing uncertainty, and incomplete
measurements are established challenges in vibration-based SHM.

Existing studies commonly evaluate their effects using quantities such as:

- damage-detection accuracy;
- identification success rate;
- precision / recall / F1;
- false-positive and false-negative rates;
- damage-index distributions;
- uncertainty intervals;
- robustness of damage localization or severity estimation.

The present study addresses a narrower question:

> Does increasing measurement noise merely increase prediction error, or can it
> qualitatively change the direction and form of damage-inference failure?

---

# 2. Recommended gap statement

Preferred formulation:

> Measurement noise and other sensing uncertainties are widely recognized as
> important limitations in structural damage identification, and previous
> studies have quantified their effects through detection accuracy, false-alarm
> behavior, identification success rates, and uncertainty propagation.
> However, aggregate robustness metrics do not necessarily reveal whether the
> direction of continuous damage-estimation error remains stable as the sensing
> distribution shifts. The present study therefore tracks severity-dependent
> signed bias, descriptor-space displacement, and prediction saturation across
> controlled noise levels.

DO NOT write:

> Previous SHM studies only evaluate accuracy.

Too strong.

DO NOT write:

> Bias has never been considered in structural damage identification.

False / unsupported.

DO NOT write:

> Severe-damage underestimation has never previously been observed.

False.

---

# 3. Core literature

## L1 — Neves et al. (2017)

Neves, A. C., González, I., Leander, J., and Karoumi, R.

Structural health monitoring of bridges: a model-free ANN-based approach to
damage detection.

Journal of Civil Structural Health Monitoring, 7, 689–702, 2017.

DOI:
10.1007/s13349-017-0252-5

### Role

Target-journal foundation for uncertainty-aware SHM evaluation.

### Key relevance

The study emphasizes that damage detection is affected by multiple uncertainty
sources, including measurement noise and signal-processing error, and evaluates
false-positive and false-negative behavior using statistical decision tools.

### Use

Introduction motivation for looking beyond a single aggregate performance
metric.

### Distinction from current work

The present manuscript addresses continuous storey-level damage quantification
and signed severity-dependent regression failure rather than binary detection
errors alone.

---

## L2 — Liu et al. (2019)

Liu, A., Wang, L., Bornn, L., and Farrar, C.

Robust structural health monitoring under environmental and operational
uncertainty with switching state-space autoregressive models.

Structural Health Monitoring, 18(2), 435–453, 2019.

DOI:
10.1177/1475921718757721

### Role

Established background on monitoring-distribution variability.

### Key relevance

Environmental and operational changes can alter structural responses and obscure
damage-sensitive information.

### Use

Support the general principle that a failure mode identified in one monitoring
distribution should not automatically be assumed stable under shifted sensing
conditions.

### Boundary

Environmental/operational variability is not equivalent to the controlled
Gaussian measurement-noise experiment in the current study.

---

## L3 — Favarelli et al. (2022)

Favarelli, E., Testi, E., and Giorgetti, A.

The impact of sensing parameters on data management and anomaly detection in
structural health monitoring.

Journal of Civil Structural Health Monitoring, 12, 1413–1425, 2022.

DOI:
10.1007/s13349-022-00566-4

### Role

High-priority JCSHM reference.

### Key relevance

The work directly evaluates how sensing parameters including:

- number of sensors;
- sampling resolution;
- observation duration;
- measurement noise;

affect machine-learning anomaly-detection performance.

Performance is assessed through conventional classification metrics including
accuracy, precision, recall, and F1 score.

### Use

Strong evidence that sensing noise and sensing configuration are already
established deployment concerns in JCSHM.

### Distinction from current work

The current paper tracks continuous damage-regression error direction and
failure-mode transition rather than anomaly-classification accuracy alone.

---

## L4 — Fan et al. (2023)

Fan, Q., Chen, Z., Xia, Z., et al.

A novel structural damage detection strategy based on VMD-FastICA and ESSAWOA.

Journal of Civil Structural Health Monitoring, 13, 149–163, 2023.

DOI:
10.1007/s13349-022-00629-6

### Role

Direct JCSHM noise-robustness reference.

### Key relevance

Gaussian noise is added to structural acceleration signals at specified SNR
levels to test the applicability and robustness of the damage-identification
method.

### Use

Demonstrates the conventional SHM framing:

    add noise
        ->
    evaluate whether damage identification remains accurate.

### Distinction from current work

The current manuscript asks whether increasing noise changes the qualitative
direction of damage-estimation failure.

---

## L5 — Wolniak et al. (2023)

Wolniak, M., Hofmeister, B., Jonscher, C., Fankhänel, M., Loose, A.,
Hübler, C., et al.

Validation of an FE model updating procedure for damage assessment using a
modular laboratory experiment with a reversible damage mechanism.

Journal of Civil Structural Health Monitoring, 2023.

DOI:
10.1007/s13349-023-00701-9

### Role

Measurement-uncertainty and model-validation reference.

### Key relevance

The study explicitly identifies noisy and spatially sparse measurements as major
sources of uncertainty in structural model updating and damage assessment.

### Use

Supports the general deployment statement that measured structural data cannot
be assumed noise-free or complete.

---

## L6 — Modesti et al. (2025)

Modesti, M., Gentilini, C., Palermo, A., Reynders, E., et al.

A two-step procedure for damage detection in beam structures with incomplete
mode shapes.

Journal of Civil Structural Health Monitoring, 15, 287–306, 2025.

DOI:
10.1007/s13349-024-00839-0

### Role

Very important novelty-boundary reference.

### Key relevance

The study evaluates noise effects on damage identification and reports cases in
which damage severity is underestimated.

### Critical consequence for current manuscript

DO NOT claim:

> Severe-damage underestimation itself is novel.

The current contribution is instead the systematic diagnosis of how the
direction of continuous damage-regression error changes under controlled
measurement-noise shift.

---

## L7 — Wang et al. (2024)

Wang, X., Zhao, Y., Wang, Z., and Hu, N.

An ultrafast and robust structural damage identification framework enabled by an
optimized extreme learning machine.

Mechanical Systems and Signal Processing, 216, 111509, 2024.

DOI:
10.1016/j.ymssp.2024.111509

### Role

Representative modern AI noise-robustness paper.

### Key relevance

The method is explicitly designed and evaluated for structural damage
identification using noise-contaminated vibration data.

### Use

Position the current work relative to a common ML research objective:

    improve performance despite noise.

### Distinction

The present manuscript does not propose a noise-robust estimator.

It deliberately retains a clean-trained estimator to diagnose where and how
inference fails under noise shift.

---

## L8 — Dessi et al. (2025)

Dessi, D., Passacantilli, F., and Venturi, A.

Analysis and mitigation of uncertainties in damage identification by
modal-curvature based methods.

Journal of Sound and Vibration, 596, 118769, 2025.

DOI:
10.1016/j.jsv.2024.118769

### Role

High-value uncertainty-propagation reference.

### Key relevance

The study explicitly separates and propagates:

- measurement-noise error;
- sensor bias error;
- discretization error;

and uses Monte Carlo analysis to quantify uncertainty in damage location and
severity.

### Use

Important novelty boundary:

bias and uncertainty analysis are established topics.

### Distinction from current work

The present study examines the signed prediction bias of a trained damage
regressor as a function of sensing-noise distribution shift.

This is different from propagating sensor bias/noise uncertainty through a
physics-based damage index.

---

## L9 — del Pozo et al. (2025)

del Pozo, G. A., Svendsen, B. T., Petersen, Ø. W., et al.

Explicit uncertainty propagation in Mahalanobis squared distance for reliable
damage detection of bridges using experimental data.

Journal of Civil Structural Health Monitoring, 15, 4195–4218, 2025.

DOI:
10.1007/s13349-025-01035-4

### Role

Recent target-journal uncertainty-quantification reference.

### Key relevance

The work explicitly propagates feature uncertainty into the distribution of a
damage index and supplements damage-detection decisions with quantified
uncertainty.

### Use

Shows that current JCSHM literature increasingly treats uncertainty as an
explicit component of damage-detection credibility.

### Distinction

The current manuscript studies failure-mode evolution of continuous damage
quantification rather than uncertainty intervals of a detection index.

---

## L10 — Hormazábal et al. (2026)

Hormazábal, M. F., Barontini, A., Masciotta, M. G., et al.

Automated damage identification for condition monitoring using short and
time-varying structural responses.

Journal of Civil Structural Health Monitoring, 16, Article 33, 2026.

DOI:
10.1007/s13349-025-01064-z

### Role

Recent target-journal context.

### Key relevance

The study explicitly addresses changing acquisition conditions and poor
signal-to-noise ratios, including the possibility that low SNR produces modal
misclassification and false alarms.

### Use

Demonstrates that failure behavior under realistic acquisition conditions is an
active JCSHM concern.

---

# 4. What the literature already establishes

The targeted literature establishes that:

A. measurement noise can degrade structural damage identification;

B. sensing uncertainty can produce false detections and uncertain damage
estimates;

C. noise robustness is an established model-development objective;

D. damage severity can be underestimated;

E. sensor bias and measurement noise can be propagated statistically.

These points are background, NOT novelty claims.

---

# 5. What the present study adds

The present evidence supports a more specific failure-mechanism analysis.

For the same clean-trained estimator, controlled response-measurement noise
produces:

    clean / low noise
        ->
    predominantly negative high-damage bias
        ->
    bias approaches zero
        ->
    signed bias becomes positive
        ->
    prediction distribution inflates
        ->
    physical upper clipping becomes dominant.

The headline transition is:

    high-damage signed bias

        -0.0213
            ->
        +0.1285

from 0% to 20% response-measurement noise.

Therefore, noise affects not only:

    |prediction error|

but also:

    sign(prediction error).

---

# 6. Calibration implication

The training-only asymmetric calibration is deliberately constructed to correct
the negative in-distribution severe-damage residual bias.

Under clean / repeated in-distribution evaluation it reduces severe-damage
underprediction.

After measurement-noise shift reverses the underlying bias, the same upward-only
correction becomes directionally misaligned and increases prediction error.

This supports the statement:

> A failure-targeted calibration can lose validity when the direction of the
> underlying error changes under sensing distribution shift.

This should NOT be generalized into:

> calibration methods are not robust.

The result applies specifically to the frozen one-sided correction and the
present controlled measurement-noise shift.

---

# 7. Novelty boundary after literature review

The targeted search does NOT support claiming novelty for:

    measurement-noise testing;
    uncertainty analysis;
    damage underestimation;
    sensor-bias analysis;
    noise-robust machine learning.

The more defensible contribution is:

> tracing the transition of severity-dependent signed regression bias across
> controlled measurement-noise levels and linking that transition to
> descriptor-space shift, output saturation, and failure of a direction-specific
> calibration.

Even this should not be written as:

> the first study to demonstrate bias reversal.

unless a systematic review establishes priority.

Preferred wording:

> In contrast to evaluating noise sensitivity only through aggregate error or
> detection accuracy, the present analysis tracks how the direction of
> severe-damage regression error evolves under controlled sensing shift.

---

# 8. Literature-supported Introduction logic

The final Introduction Gap 2 should follow this sequence:

Measurement and environmental uncertainty are unavoidable in SHM.

Existing work has therefore developed:

    robust damage indicators;
    noise-tolerant ML models;
    uncertainty propagation;
    false-alarm-aware decision rules.

However, a model can retain an acceptable aggregate metric while its
severity-dependent error direction changes.

Therefore the present study evaluates:

    severity-specific MAE
        +
    signed bias
        +
    underestimation ratio
        +
    descriptor-space shift
        +
    prediction saturation.

---

# 9. Preferred gap paragraph

Recommended scientific core:

> Measurement noise and related sensing uncertainties are established sources of
> error in vibration-based SHM, and previous studies have examined their effects
> through detection accuracy, false-alarm behavior, robustness tests, and
> uncertainty propagation. These measures are essential but do not by themselves
> determine whether the directional structure of continuous damage-estimation
> error remains stable as the sensing distribution changes. This distinction is
> important when a model exhibits severity-dependent underprediction or when a
> post-hoc correction is designed around a particular error direction. The
> present study therefore evaluates both the magnitude and sign of damage-
> inference error under controlled measurement-noise shift.

---

# 10. Preferred Discussion comparison

Existing literature question:

    Can damage still be detected / quantified accurately under noise?

Current manuscript question:

    How does the failure mode itself change as noise increases?

The manuscript should present these as complementary questions rather than claim
that previous robustness research is inadequate.

---

# 11. Priority citations for final Introduction

Minimum core set:

Neves et al. (2017)
Favarelli et al. (2022)
Fan et al. (2023)
Modesti et al. (2025)
Dessi et al. (2025)
del Pozo et al. (2025)
Hormazábal et al. (2026)

Additional Discussion/background:

Liu et al. (2019)
Wang et al. (2024)
Wolniak et al. (2023)

---

# 12. Block 2 conclusion

Established literature:

    sensing uncertainty
        ->
    performance degradation / false alarms / uncertain severity.

Current manuscript extension:

    sensing distribution shift
        ->
    descriptor shift
        ->
    directional bias transition
        ->
    saturation
        ->
    calibration mismatch.

Preferred scientific distinction:

    conventional robustness question:
        "How much performance is retained?"

    present failure-mechanism question:
        "How does the direction and form of the error change?"

