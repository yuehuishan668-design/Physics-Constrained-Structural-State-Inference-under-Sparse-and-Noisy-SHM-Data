# JCSHM Paper Blueprint

## 1. Target Journal

**Journal of Civil Structural Health Monitoring (JCSHM)**

Article type:

**Original Research Article**

Current manuscript status:

This manuscript is a fundamental reconstruction of the previously desk-rejected
version rather than a direct incremental revision.

The new manuscript is built around deployment-aware information provenance,
structural sensing observability, directional inference reliability, and sensing-
degradation failure mechanisms.

---

# 2. Working Title

## Primary working title

**Observability and Failure Mechanisms of Physics-Guided Structural Damage Inference under Sparse and Noisy Sensing**

## Alternative title

**Deployment-Aware Physics-Guided Structural Damage Inference under Sparse and Noisy Sensing: Observability and Failure Mechanisms**

The first title is currently preferred.

---

# 3. Central Scientific Question

The manuscript addresses the following overarching question:

> What makes data-driven structural damage inference reliable under realistic
> sensing and information constraints?

The study treats structural damage inference as an information-constrained
inverse problem:

Structural sensing
→ physically available information
→ physics-guided descriptor representation
→ estimator
→ structural damage inference
→ reliability or failure.

The central argument is that predictive reliability cannot be understood from
prediction accuracy alone.

It depends jointly on:

1. information provenance;
2. estimator flexibility;
3. structural sensing observability;
4. stability of the descriptor distribution under sensing degradation.

---

# 4. Central Thesis

**Reliable storey-level damage inference is jointly governed by the legitimacy
of available information, the nonlinear capacity of the estimator, and structural
sensing observability; measurement noise and sensor sparsity degrade inference
through fundamentally different failure mechanisms.**

In conceptual form:

Information provenance
+
Estimator flexibility
+
Sensor observability
+
Distribution stability
→
Damage-inference reliability.

---

# 5. Research Questions

## RQ1 — Information validity and deployment accessibility

**What information can legitimately be used for deployment-oriented structural
damage inference, and how does information availability affect inference
capability?**

This question distinguishes between:

- simulation-privileged information;
- legacy simulation-informed descriptors;
- signal-derived descriptors available under sensed base-input information;
- structural-response-only descriptors.

The objective is not merely feature reduction.

The objective is to establish a defensible boundary between information that is
available during simulation and information that can realistically be obtained
during structural monitoring.

---

## RQ2 — Estimator capability and directional reliability

**Given deployment-accessible information, how much estimator flexibility is
required, and how does inference reliability vary with damage severity?**

This question investigates:

- linear versus nonlinear descriptor-to-damage mappings;
- information × estimator interaction;
- severity-dependent errors;
- signed bias;
- severe-damage underestimation.

The key principle is:

Average prediction accuracy is not equivalent to reliable severe-damage
inference.

---

## RQ3 — Measurement-noise failure mechanism

**How does controlled response-measurement noise alter the descriptor space and
the direction of structural damage inference errors?**

The study investigates the pathway:

Measurement noise
→ descriptor-space distribution shift
→ prediction-distribution change
→ bias reversal
→ prediction saturation.

The purpose is therefore not simply to quantify performance degradation with
noise, but to identify how and why the inference mechanism fails.

---

## RQ4 — Sensor observability and sparse sensing

**How do structural sensor number and placement control the physically observable
descriptor space and damage-inference reliability?**

Sensor sparsity is represented using dependency-aware descriptor availability,
rather than artificial zero masking.

All 15 non-empty subsets of the four structural response sensors are evaluated.

This enables independent investigation of:

- sensor-count effects;
- sensor-placement effects;
- descriptor observability;
- marginal value of adding individual sensors.

---

# 6. Scientific Contributions

## Contribution 1 — Deployment-aware information and observability formulation

A provenance-based hierarchy is established between privileged simulation
information and deployment-accessible signal-derived descriptors.

Sensor sparsity is represented through dependency-aware descriptor availability:

a descriptor is retained only when all structural measurements required by its
original physical definition are actually available.

This eliminates artificial zero-signal representations of missing sensors and
preserves the physical meaning of the descriptor system.

---

## Contribution 2 — Reliability-oriented inference diagnosis

The respective roles of information availability and estimator flexibility are
separated through controlled comparisons and repeated-split factorial analysis.

The results demonstrate that:

- estimator nonlinearity materially affects the ability to exploit the
  signal-derived descriptor space;
- additional information and estimator flexibility provide complementary gains;
- severe structural damage exhibits systematic directional underprediction that
  is not visible from global MAE alone.

A training-only asymmetric residual calibration is therefore investigated as a
reliability-oriented diagnostic extension rather than as a new predictive
algorithm.

---

## Contribution 3 — Distinct sensing-degradation failure mechanisms

Controlled matched measurement-noise perturbations and exhaustive sensor-layout
experiments demonstrate two fundamentally different degradation pathways.

### Measurement noise

Measurement noise
→ descriptor distribution shift
→ prediction inflation and broadening
→ bias reversal
→ high-noise prediction saturation.

### Sensor sparsity

Sensor removal
→ loss of physically computable descriptors
→ reduced cross-storey observability
→ increasingly negative severe-damage bias
→ systematic underestimation.

Across the exhaustive structural-sensor subset lattice, all 28 admissible
one-sensor additions reduced both overall MAE and high-damage MAE, and all
paired 95% case-level bootstrap intervals remained on the beneficial side of
zero.

---

# 7. Scientific Positioning

This paper is **not** positioned as:

- a new machine-learning algorithm;
- a new SVR method;
- a new isotonic-regression method;
- a general-purpose robust SHM algorithm;
- a new structural mechanics theory.

It is positioned as:

**a deployment-aware computational structural health monitoring methodology
study of information provenance, sensor observability, directional reliability,
and sensing-degradation failure mechanisms.**

The novelty is primarily:

- methodological;
- diagnostic;
- experimental-design based;
- mechanistic in interpretation.

---

# 8. Main Manuscript Architecture

## 1. Introduction

Purpose:

Establish that trustworthy structural damage inference requires more than high
average predictive accuracy.

Main gaps:

1. deployment accessibility of input information is often insufficiently
   distinguished from simulation information;
2. global accuracy metrics may hide directional severe-damage failures;
3. sensing degradation is often quantified only through error magnitude;
4. sensor number and sensor placement are often not separated through exhaustive
   observability analysis.

End the Introduction with RQ1–RQ4 and the three major contributions.

---

## 2. Structural Damage Inference and Deployment-Aware Information Formulation

### 2.1 Structural system and damage-inference problem

Include:

- four-storey two-bay frame;
- numerical model;
- structural damage definition;
- excitation generation;
- response simulation;
- 3000 cases;
- train/validation/test partition.

### 2.2 Physics-guided multi-domain descriptors

Describe descriptor families rather than presenting the descriptor vector as the
main novelty.

Families include:

- response statistics;
- floor-to-ground amplification;
- relative-to-lower responses;
- adjacent-storey response ratios;
- spatial-response fractions;
- spectral descriptors;
- response correlations.

### 2.3 Information provenance and deployment hierarchy

Define:

- 92D privileged-information reference;
- 86D legacy simulation-informed reference;
- 78D primary signal-derived representation;
- 59D structural-response-only pressure-test representation.

Measured-ground augmented 86D is supplementary.

### 2.4 Dependency-aware sensor observability

Define the descriptor-availability rule formally.

For structural sensor layout S and descriptor dependency set D_j:

f_j is available only when:

D_j ⊆ S.

Ground/base-input sensing is assumed available throughout this experiment.

No zero masking and no descriptor redefinition are used.

---

## 3. Reliability-Oriented Inference and Evaluation Methodology

### 3.1 Linear and nonlinear estimators

Main estimators:

- Ridge regression;
- RBF-SVR.

Other estimator screening results are supplementary.

### 3.2 Information × estimator factorial protocol

Primary factorial:

- 78D + Ridge;
- 78D + SVR;
- 92D + Ridge;
- 92D + SVR.

Include repeated paired train/validation/test splits.

### 3.3 Severity-aware reliability diagnostics

Damage categories:

- zero;
- low;
- medium;
- high.

Metrics:

- MAE;
- RMSE;
- signed bias;
- absolute signed bias;
- underestimation ratio.

### 3.4 Training-only asymmetric residual calibration

Describe:

- OOF residual construction;
- monotone isotonic correction;
- upward-only correction;
- validation-only alpha selection;
- predetermined acceptance constraints.

### 3.5 Controlled sensing-degradation protocols

#### 3.5.1 Matched measurement-noise experiment

Use the same underlying structural cases.

Perturb structural-response channels only.

Keep the assumed base-input signal unchanged.

Noise levels:

0%, 5%, 10%, 20%.

Use deterministic matched stochastic perturbations.

#### 3.5.2 Exhaustive sensor-layout experiment

Evaluate all 15 non-empty structural sensor subsets.

Use dependency-aware descriptor availability.

#### 3.5.3 Paired case-level bootstrap

Use:

- 450 complete test cases;
- four storey outputs retained together;
- 5000 paired bootstrap replicates;
- percentile 95% confidence intervals.

---

## 4. Results

### 4.1 Information availability and nonlinear inference capability

Answer RQ1 and part of RQ2.

Show:

- descriptor hierarchy;
- 78D versus 92D;
- Ridge versus SVR;
- information × estimator complementarity.

Primary deployable estimator:

78D signal-derived representation + RBF-SVR.

---

### 4.2 Damage-severity-dependent failure and directional calibration

Show:

- overall accuracy;
- damage-severity MAE;
- high-damage signed bias;
- underestimation ratio;
- calibration effect.

Central result:

good average performance does not eliminate systematic severe-damage
underprediction.

---

### 4.3 Measurement noise causes descriptor shift and bias reversal

Present the mechanism in causal order:

noise
→ descriptor shift
→ prediction distribution change
→ high-damage bias reversal
→ prediction saturation.

Do not describe the method as noise robust.

The result is a robustness boundary and failure mechanism.

---

### 4.4 Sensor sparsity progressively reduces structural observability

#### 4.4.1 Sensor-count effect

Compare 1-, 2-, 3-, and 4-sensor configurations.

#### 4.4.2 Sensor-placement effect

Compare layouts under identical sensor budgets.

Distinguish stable and uncertain pairwise differences.

#### 4.4.3 Marginal sensing value

Analyse all 28 one-sensor additions in the complete sensor subset lattice.

---

## 5. Discussion

### 5.1 Information provenance as a prerequisite for credible computational SHM

Discuss why privileged simulation information can generate optimistic estimates
that do not represent deployment capability.

### 5.2 Complementarity between observability and estimator flexibility

A more flexible estimator cannot reconstruct information that was never
observed.

Additional sensing information cannot fully compensate for an inadequately
flexible inverse mapping.

### 5.3 Different failure mechanisms of noise and sensor sparsity

Noise:

distributional corruption.

Sensor sparsity:

observability reduction.

Highlight the opposite directional-bias behaviour.

### 5.4 Implications for structural monitoring-system design

Discuss:

- sensor number;
- sensor placement;
- cross-storey descriptor observability;
- severe-damage versus overall-error objectives;
- limitations of calibration under distribution shift.

### 5.5 Limitations and future validation

Explicitly state:

- one simulated structural topology;
- no physical experimental validation;
- no field monitoring validation;
- Gaussian structural-response noise assumption;
- clean base-input sensing assumption;
- bootstrap conditional on the simulated test population;
- some SVR optima occur on the frozen hyperparameter-grid boundary;
- calibration depends on directional-bias stability.

---

## 6. Conclusions

The conclusion should answer four questions:

1. Why does information provenance matter?
2. Why is nonlinear inference necessary but insufficient for reliability?
3. Why do measurement noise and sensor sparsity produce different failure modes?
4. Why must sensor observability be explicitly considered in SHM deployment?

---

# 9. Main-Figure Architecture

## Figure 1 — Conceptual framework

Scientific task:

Show the complete manuscript logic:

sensed structural response
→ information provenance
→ physics-guided descriptors
→ estimator
→ damage inference
→ reliability diagnostics.

Include two degradation branches:

measurement noise
and
sensor sparsity.

---

## Figure 2 — Descriptor provenance and sensor observability

Scientific task:

Visualize:

92D privileged
→ 86D legacy
→ 78D signal-derived primary
→ 59D response-only.

Also show dependency-aware sensor-descriptor relationships.

---

## Figure 3 — Information × estimator factorial

Scientific task:

Demonstrate complementary effects of:

- information availability;
- nonlinear estimator flexibility.

Main conditions:

78D Ridge
78D SVR
92D Ridge
92D SVR.

---

## Figure 4 — Severity-dependent inference and calibration

Scientific task:

Show why overall MAE alone is insufficient.

Panels should include:

- MAE versus damage severity;
- high-damage signed bias;
- underestimation ratio;
- standard versus calibrated SVR.

---

## Figure 5 — Measurement-noise failure mechanism

Scientific task:

Show:

noise
→ descriptor shift
→ bias reversal
→ saturation.

Candidate panels:

A. overall/high-damage MAE versus noise;
B. high-damage signed bias versus noise;
C. prediction clipping/saturation ratio;
D. largest standardized descriptor shifts.

---

## Figure 6 — Exhaustive sensor-layout observability

Scientific task:

Show all 15 structural sensor layouts and corresponding inference errors.

Group by sensor count.

Encode:

- layout;
- descriptor count;
- overall MAE;
- high-damage MAE.

---

## Figure 7 — Sensor subset lattice and marginal sensing value

Scientific task:

Visualize the complete 15-node sensor subset lattice and 28 one-sensor-addition
edges.

Highlight that every edge reduced both overall and high-damage MAE at the point
estimate level and retained an entirely beneficial paired bootstrap interval.

---

# 10. Main-Table Architecture

## Table 1 — Information provenance hierarchy

Include:

- descriptor set;
- dimension;
- information source;
- deployment accessibility;
- intended role in the manuscript.

---

## Table 2 — Primary estimator and factorial results

Include:

- representation;
- model;
- overall MAE;
- high-damage MAE;
- high-damage bias;
- underestimation ratio;
- repeated-split effect summary.

---

## Table 3 — Controlled measurement-noise robustness

Include:

- noise level;
- overall MAE;
- high-damage MAE;
- signed high-damage bias;
- under/overestimation;
- clipping ratio.

---

## Table 4 — Sensor observability summary

Include:

- sensor count;
- best layout;
- worst layout;
- mean MAE;
- placement spread;
- high-damage MAE;
- paired-bootstrap support.

---

# 11. Supplementary Material Strategy

Move supporting but non-central results out of the main manuscript:

- measured-ground augmented 86D analysis;
- ElasticNet;
- Random Forest;
- HistGradientBoosting;
- Ridge regularization sensitivity;
- matrix conditioning diagnostics;
- complete SVR hyperparameter grids;
- complete calibration-alpha grids;
- damage-weighted SVR pilot;
- complete 27 within-count sensor-layout contrasts;
- complete 28 marginal sensor-edge numerical results;
- feature provenance manifests;
- reproducibility and integrity checks.

---

# 12. Evidence and Wording Rules

## 92D

Always describe as:

**privileged-information reference**

or

**simulation-informed reference**.

Never describe as:

- global upper bound;
- deployable representation.

## 78D

Describe as:

**primary signal-derived representation under the assumed sensed/measured
base-input signal.**

Do not call it field measured.

## Noise

Do not claim:

**noise robustness**.

Use:

- controlled measurement-noise sensitivity;
- robustness boundary;
- noise-induced failure mechanism.

## Descriptor mechanism

Use:

**associated with**

or

**suggesting contribution from**.

Do not claim that a descriptor family causally caused failure without ablation
evidence.

## Sensor-layout bootstrap

Describe confidence intervals as:

**paired case-level bootstrap intervals conditional on the current simulated
test population.**

Do not claim:

- cross-structure confidence intervals;
- independent physical-experiment uncertainty.

---

# 13. Manuscript Writing Order

The manuscript will be written in the following order:

1. Freeze title, RQs, contributions and paper architecture.
2. Freeze main figures and tables.
3. Write Methods.
4. Write Results.
5. Write Discussion.
6. Write Introduction.
7. Write Conclusions.
8. Write Abstract.
9. Final terminology and evidence-strength audit.
10. JCSHM formatting and submission package preparation.

---

# 14. Current Status

Experimental development is frozen.

No additional model tuning, descriptor redesign, noise-model modification or
sensor-layout optimization should be performed unless a future reviewer
specifically requests additional evidence.

Current phase:

**JCSHM manuscript reconstruction.**
