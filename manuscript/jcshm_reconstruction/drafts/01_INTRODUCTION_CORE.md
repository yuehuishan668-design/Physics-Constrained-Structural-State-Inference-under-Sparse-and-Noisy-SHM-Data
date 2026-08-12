# 1. Introduction — Scientific Core

> STATUS: SCIENTIFIC CORE ONLY
>
> Purpose:
> freeze the motivation, research gap, research questions, contributions,
> scope, and novelty positioning of the JCSHM manuscript.
>
> This file is NOT the final literature-integrated Introduction.
>
> Later editing may:
>
> - add literature citations;
> - improve transitions;
> - compress or merge paragraphs;
> - improve journal style;
> - update terminology for consistency.
>
> Later editing MUST NOT:
>
> - introduce research questions not tested in the frozen experiments;
> - describe the work as a new machine-learning algorithm;
> - describe the 92D representation as a deployable upper bound;
> - claim field validation or cross-structure generalization;
> - collapse measurement noise and sensor sparsity into one generic robustness problem;
> - overstate sensor-placement conclusions.

---

# 1.1 Problem motivation

Structural health monitoring increasingly uses data-driven inference to estimate
structural condition from measured dynamic responses.

For structural damage quantification, the usual methodological focus is on the
mapping

\[
\mathbf x
\rightarrow
\mathbf y,
\]

where \(\mathbf x\) denotes measured or derived monitoring information and
\(\mathbf y\) denotes the target damage state.

However, prediction accuracy alone does not determine whether such an inference
pipeline is meaningful for deployment.

A deployment-oriented formulation must additionally answer:

1. whether the information supplied to the estimator is physically available;
2. whether that information remains computable under the actual sensor layout;
3. how the inference fails when the sensing information is perturbed or removed.

The present study therefore treats damage inference as the combined problem

\[
\boxed{
\text{information provenance}
\rightarrow
\text{observability}
\rightarrow
\text{inference}
\rightarrow
\text{failure mechanism}.
}
\]

---

# 1.2 Gap 1 — Simulation performance can depend on information unavailable at deployment

## Core problem

Simulation-based SHM studies can provide quantities that would not necessarily
be available to a deployed monitoring system.

Examples in the present benchmark include:

- exact excitation-generator frequency;
- excitation amplitude as a simulation parameter;
- prescribed noise level;
- other direct generator metadata.

If such quantities enter the predictor, low test error does not necessarily
represent performance achievable from the intended sensing configuration.

---

## Scientific gap

The relevant distinction is not simply:

\[
\text{many features}
\quad\text{versus}\quad
\text{few features}.
\]

The more important distinction is:

\[
\boxed{
\text{simulation-privileged information}
\quad\text{versus}\quad
\text{sensing-accessible information}.
}
\]

A deployment-oriented SHM assessment therefore requires explicit information
provenance.

---

## Literature integration required

[LITERATURE-LOCK-1]

Final Introduction should cite literature covering:

- data-driven structural damage identification;
- physics-guided / feature-based SHM;
- simulation-to-deployment discrepancy;
- information leakage or use of unavailable auxiliary variables where relevant.

Do not claim that no previous SHM study has considered information provenance
unless a systematic literature review supports that statement.

Preferred gap wording:

> Information provenance is often less explicit than model architecture and
> predictive accuracy in simulation-based damage-inference studies.

Avoid:

> No previous study has considered information provenance.

---

# 1.3 Gap 2 — Aggregate accuracy can conceal directionally important failure

## Core problem

Damage-inference models are commonly summarized by aggregate prediction metrics
such as MAE or RMSE.

However, an estimator with acceptable global error may still exhibit systematic
failure in the severe-damage regime.

For SHM applications, the distinction between

\[
\hat y > y
\]

and

\[
\hat y < y
\]

can be operationally important.

In particular, systematic severe-damage underprediction cannot be inferred from
global MAE alone.

---

## Scientific gap

Evaluation should therefore distinguish:

\[
\text{error magnitude}
\]

from

\[
\text{error direction}.
\]

The present work explicitly evaluates:

- damage-severity-specific MAE;
- signed prediction bias;
- absolute bias;
- underestimation frequency.

This enables failure diagnosis rather than only performance ranking.

---

## Distribution-shift extension

An additional issue is whether a directional failure observed under an
in-distribution benchmark remains directionally stable after sensing conditions
change.

A calibration designed to correct one directional bias may become unsuitable
if distribution shift reverses the bias.

Measurement-noise stress testing is therefore used here not to demonstrate
generic robustness, but to identify a robustness boundary and failure-mode
transition.

---

## Literature integration required

[LITERATURE-LOCK-2]

Final Introduction should cite work on:

- uncertainty and reliability of SHM inference;
- robustness of data-driven SHM to measurement noise;
- damage-severity-dependent prediction performance;
- distribution shift / domain shift in structural monitoring where available.

Do not write that directional failure is absent from the entire literature
without evidence.

---

# 1.4 Gap 3 — Sparse sensing changes what is observable, not only how much data are available

## Core problem

Structural sensor sparsity is not equivalent to simply reducing the number of
input variables.

Many physics-guided descriptors depend jointly on multiple sensing locations.

When a required structural response channel is unavailable, the corresponding
descriptor may become physically non-computable.

Artificially zero-filling the missing signal would retain a numerical feature
while changing its physical meaning.

---

## Dependency-aware formulation

For sensor layout

\[
S,
\]

and descriptor-specific structural dependency set

\[
D_j,
\]

descriptor \(f_j\) is available only when

\[
D_j\subseteq S.
\]

Thus sensing sparsity affects:

\[
\boxed{
\text{descriptor observability}
}
\]

rather than merely input dimension.

---

## Scientific gap

Sparse-sensing evaluation should therefore separate:

1. sensor count;
2. sensor placement;
3. resulting descriptor availability;
4. inference performance.

The present four-sensor system permits exhaustive evaluation of all

\[
2^4-1=15
\]

non-empty structural sensor layouts and all

\[
28
\]

admissible one-sensor additions.

---

## Literature integration required

[LITERATURE-LOCK-3]

Final Introduction should cite literature on:

- optimal sensor placement in SHM;
- sparse sensing;
- observability-informed monitoring;
- sensor-network design;
- data-driven damage inference under incomplete measurements.

Important terminology rule:

The present study uses “descriptor observability” or
“practical sensing observability”.

Do not imply that a classical state-space observability rank theorem is derived.

---

# 1.5 Unifying research problem

The three gaps are connected.

A damage estimator receives an effective information set determined by both:

\[
\text{information provenance}
\]

and

\[
\text{sensor availability}.
\]

The estimator then maps this available representation to damage.

Sensing degradation can affect this mapping through at least two fundamentally
different pathways:

### Measurement noise

\[
\text{available information}
\rightarrow
\text{perturbed information}.
\]

### Sensor sparsity

\[
\text{available information}
\rightarrow
\text{removed information}.
\]

The central premise of this study is that these two degradation modes need not
produce the same inference failure mechanism.

---

# 1.6 Research questions

The manuscript addresses three research questions.

---

## RQ1 — Information provenance and estimator capability

> **RQ1. How do deployment-accessible information and estimator flexibility
> jointly affect storey-level structural damage inference?**

This question is addressed through:

- the 92D privileged-information reference;
- the 78D signal-derived primary representation;
- Ridge regression;
- RBF-SVR;
- repeated ten-split \(2\times2\) factorial analysis.

Key quantities:

- overall MAE;
- high-damage MAE;
- high-damage bias;
- high-damage underestimation;
- information × estimator interaction.

RQ1 does NOT ask whether RBF-SVR is universally optimal.

---

## RQ2 — Directional failure under measurement-noise shift

> **RQ2. How does structural-response measurement noise alter the magnitude and
> direction of damage-inference failure, and does a calibration designed for
> in-distribution underprediction remain valid after that shift?**

This question is addressed through:

- severity-stratified repeated evaluation;
- training-only asymmetric residual calibration;
- controlled matched-noise levels of 0%, 5%, 10%, and 20%;
- descriptor-space shift diagnostics;
- signed high-damage bias;
- prediction-bound saturation.

Core distinction:

\[
\text{in-distribution underprediction}
\]

versus

\[
\text{noise-induced bias reversal}.
\]

RQ2 is a failure-mechanism question, not a claim of noise robustness.

---

## RQ3 — Dependency-aware observability under sensor sparsity

> **RQ3. How does structural sensor sparsity alter descriptor observability,
> severe-damage inference, and the marginal value of additional sensing?**

This question is addressed through:

- all 15 non-empty structural sensor layouts;
- dependency-aware descriptor availability;
- layout-specific model fitting and validation;
- 5000-replicate paired case bootstrap;
- all 28 admissible one-sensor additions.

Key distinctions:

- sensor count versus placement;
- descriptor count versus descriptor identity;
- overall accuracy versus severe-damage accuracy.

RQ3 does NOT seek a universal optimal sensor layout.

---

# 1.7 Study design

A controlled four-storey, two-bay linear-elastic OpenSeesPy frame is used as a
mechanism-isolation benchmark.

Storey-level damage is represented by column stiffness degradation:

\[
(EI)_{c,i}
=
(1-d_i)EI_c.
\]

A total of

\[
3000
\]

simulated structural cases are generated under randomized sinusoidal base
excitation.

The benchmark is deliberately controlled so that:

- damage states are exactly known;
- clean and noisy structural responses are available;
- information provenance can be audited;
- structural sensor subsets can be exhaustively enumerated.

The purpose of the benchmark is methodological failure diagnosis.

It is not presented as field validation.

---

# 1.8 Main contributions

The manuscript makes four principal contributions.

---

## Contribution 1 — Deployment-aware information provenance

A nested descriptor hierarchy distinguishes:

\[
92D
\rightarrow
86D
\rightarrow
78D
\rightarrow
59D
\]

according to information source.

The primary 78D representation excludes direct generator metadata and exact
generator-frequency-derived descriptors.

Contribution statement:

> An explicit deployment-aware information boundary is established between
> simulation-privileged and signal-derived damage-inference information.

Novelty is NOT the number “78”.

---

## Contribution 2 — Joint diagnosis of estimator capability and directional failure

The study separates:

\[
\text{average predictive accuracy}
\]

from

\[
\text{severity-dependent directional inference failure}.
\]

Repeated factorial analysis quantifies the complementary effects of information
availability and estimator flexibility.

A training-only asymmetric calibration experiment then tests whether the
persistent high-damage underprediction can be partially corrected.

Contribution statement:

> The analysis identifies severe-damage underprediction as a directional
> inference limitation that persists after substantial improvement in overall
> predictive accuracy.

---

## Contribution 3 — Identification of distinct sensing-degradation mechanisms

Controlled matched-noise analysis reveals the sequence

\[
\text{descriptor shift}
\rightarrow
\text{prediction inflation}
\rightarrow
\text{bias reversal}
\rightarrow
\text{saturation}.
\]

Sparse sensing instead produces

\[
\text{descriptor unavailability}
\rightarrow
\text{observability loss}
\rightarrow
\text{severe underprediction}.
\]

Contribution statement:

> Measurement noise and sensor loss are shown to produce qualitatively different
> failure pathways rather than a single generic degradation in model
> robustness.

This is one of the manuscript's principal scientific contributions.

---

## Contribution 4 — Exhaustive dependency-aware sensing analysis

All 15 non-empty layouts in the four-sensor system are evaluated without
zero-masking unavailable measurements or redefining descriptors.

Across the resulting 28 one-sensor additions, both overall and high-damage MAE
decrease in every tested transition, with paired case-bootstrap support.

Contribution statement:

> The complete tested structural-sensor subset lattice quantifies both
> placement-dependent observability and the marginal inference value of
> additional sensing.

This result is benchmark-specific and must not be presented as a universal
sensor-placement law.

---

# 1.9 What is deliberately NOT claimed

The manuscript does NOT propose:

- a new machine-learning algorithm;
- a new SVR formulation;
- a new isotonic-regression method;
- a fundamentally new individual handcrafted descriptor;
- a universal sensor-placement strategy.

The manuscript does NOT demonstrate:

- field-monitoring performance;
- laboratory validation;
- real-earthquake validation;
- cross-structure generalization;
- nonlinear collapse prediction;
- universal noise robustness.

---

# 1.10 Novelty positioning for JCSHM

The manuscript should be positioned as:

> **a deployment-aware computational structural health monitoring study of
> information provenance, sensing observability, and failure mechanisms in
> storey-level damage inference.**

The scientific novelty is the integration of:

\[
\boxed{
\text{information provenance}
+
\text{dependency-aware observability}
+
\text{directional failure diagnosis}
+
\text{complete sensing-subset analysis}.
}
\]

The paper should NOT be positioned as:

> “a new physics-guided SVR damage-identification method”.

---

# 1.11 Introduction-to-Methods transition

The final Introduction should end by stating that the study first establishes
the deployment-aware information hierarchy and descriptor-dependency rules,
then evaluates:

1. information × estimator interactions;
2. severity-dependent directional failure;
3. matched measurement-noise degradation;
4. exhaustive structural sensor sparsity.

The Methods section then defines the controlled structural benchmark and each
evaluation protocol.

---

# Literature integration rules

The final Introduction will require targeted literature support around three
specific gaps:

1. simulation/data-driven structural damage inference and physics-guided
   representations;
2. noise robustness, uncertainty, and severity-dependent failure;
3. sparse sensing, sensor placement, and observability.

Literature should be used to establish context and gap.

Do NOT allow the literature review to redefine the paper around:

- deep learning architectures;
- PINNs;
- digital twins;
- sensor-placement optimization algorithms;

unless those topics directly support one of the three research gaps.

---

# Introduction claim lock

The strongest defensible problem statement is:

> Damage-inference performance should be interpreted relative to the
> provenance, availability, and degradation mode of the information supplied to
> the estimator, rather than predictive accuracy alone.

The strongest defensible manuscript-level research proposition is:

\[
\boxed{
\text{Different sensing degradations can produce different damage-inference
failure mechanisms even within the same estimator and descriptor framework.}
}
\]

This proposition is evaluated within the present controlled numerical benchmark
and must not be stated as a universal physical law.
