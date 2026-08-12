# 2. Methodology — Scientific Core

> STATUS: SCIENTIFIC CORE ONLY
>
> This document freezes the methodological content, definitions, equations,
> protocols, assumptions, and reporting boundaries of the JCSHM manuscript.
>
> Later language editing may improve grammar, transitions, concision, and
> journal style, but MUST NOT alter:
>
> - methodological definitions;
> - numerical values;
> - train/validation/test protocol;
> - information-provenance interpretation;
> - sensor-dependency rules;
> - frozen hyperparameter-selection procedures;
> - calibration protocol;
> - noise protocol;
> - bootstrap/statistical interpretation;
> - deployment/generalization limitations.

---

# 2.1 Problem formulation and deployment-aware information boundary

## Core inverse problem

The study considers storey-level structural damage inference from monitored dynamic response information.

For a four-storey structure, the inference target is

\[
\mathbf y =
[y_1,y_2,y_3,y_4]^\mathsf{T},
\]

where \(y_i\) denotes the damage state associated with storey \(i\).

The general inverse mapping is

\[
\mathcal{M}:
\mathcal{I}_{\mathrm{sensed}}
\rightarrow
\mathbf y,
\]

where \(\mathcal{I}_{\mathrm{sensed}}\) denotes information available from the assumed sensing system.

The central methodological issue is not only estimator accuracy, but whether the information supplied to the estimator is physically available under the assumed deployment configuration.

---

## Deployment-aware information principle

Descriptor availability is separated according to information provenance.

The study distinguishes:

1. structural-response-derived information;
2. descriptors requiring the sensed base/ground-input history;
3. descriptors derived from exact excitation-generator frequency;
4. direct simulation-generator metadata.

The main deployment-oriented representation excludes exact generator metadata and exact generator-frequency-derived information.

The primary representation therefore contains only quantities derived from:

- structural response histories; and
- the sensed/measured base-input history under the assumed base-input sensor.

---

## Main representation and references

Four nested information representations are distinguished:

| Representation | Dimension | Methodological role |
|---|---:|---|
| Privileged-information reference | 92 | simulation-privileged reference |
| Legacy simulation-informed reference | 86 | historical comparison |
| Signal-derived representation | 78 | primary representation |
| Structural-response-only representation | 59 | strict output-only pressure test |

Nested information decomposition:

\[
92 = 59 + 19 + 8 + 6.
\]

Specifically:

- 59 structural-response-derived descriptors;
- 19 base-input-dependent signal-derived descriptors;
- 8 exact generator-frequency-derived descriptors;
- 6 simulation-generator metadata descriptors.

The 92D representation is NOT interpreted as a deployment method or universal upper bound.

The 86D representation is NOT considered deployment-clean because it retains eight descriptors dependent on the exact excitation-generator frequency.

The 78D representation is the principal representation used for the deployment-oriented analyses.

The 59D representation evaluates inference without access to the base-input sensing channel.

Reference: Fig. 2(a,b), Table 1.

---

# 2.2 Structural simulation and damage-inference dataset

## Structural system

A numerical four-storey, two-bay frame is used as the controlled structural system.

The structural response simulations are performed using OpenSeesPy.

The present study uses a linear-elastic structural model with parametrically varied storey damage states.

[CODE-LOCK REQUIRED BEFORE FINAL MANUSCRIPT:
Codex must extract exact geometry, section/material properties, mass assignment,
boundary conditions, damping specification, time step, excitation duration,
and damage implementation directly from the frozen OpenSeesPy model files.
No value may be inferred or invented.]

---

## Dataset size

The frozen simulation dataset contains

\[
N=3000
\]

structural cases.

Each case provides four structural acceleration-response channels corresponding to the four storeys.

The damage target contains four storey-level components.

Frozen array structure includes:

\[
X_{\mathrm{abs}} \in
\mathbb R^{3000\times2000\times4},
\]

with corresponding clean structural-response histories and base-input histories retained for controlled robustness analyses.

---

## Fixed train/validation/test partition

For the principal fixed-split analyses:

\[
N_{\mathrm{train}}=2100,
\]

\[
N_{\mathrm{validation}}=450,
\]

\[
N_{\mathrm{test}}=450.
\]

The validation set is used for hyperparameter selection.

The test set is evaluated only after validation-based model selection within each frozen experimental protocol.

---

## Repeated-split protocol

For repeated comparisons:

- ten deterministic repeated train/validation/test splits are used;
- seeds are \(0,\ldots,9\);
- each split retains the same 2100/450/450 sample counts;
- competing methods use identical splits;
- preprocessing is performed independently within each split.

These repeated holdouts are NOT interpreted as ten independent structural experiments because all splits originate from the same 3000 simulated cases.

They provide repeated in-distribution holdout evidence rather than cross-structure generalization evidence.

---

## Damage-severity groups

Frozen damage strata are:

\[
\text{zero}: y\le10^{-12},
\]

\[
\text{low}: 10^{-12}<y\le0.10,
\]

\[
\text{medium}:0.10<y\le0.20,
\]

\[
\text{high}:y>0.20.
\]

These thresholds are fixed throughout the study.

---

# 2.3 Physics-guided descriptor representation and information provenance

## Descriptor philosophy

Instead of supplying raw histories directly to the estimator, the monitored response is transformed into a physics-guided descriptor representation.

The descriptors encode complementary information associated with:

- response amplitudes and statistical characteristics;
- temporal/spectral response characteristics;
- relationships between adjacent storeys;
- relationships between structural response and base input;
- spatial response distribution.

The methodological contribution is NOT that each individual handcrafted descriptor is intrinsically novel.

The contribution lies in:

1. explicitly auditing descriptor information provenance;
2. constructing a deployment-oriented representation;
3. maintaining descriptor definitions under sensing degradation;
4. evaluating inference under controlled information and observability constraints.

---

## Information-provenance hierarchy

The original complete representation contains 92 descriptors.

Six descriptors correspond directly to simulation-generator metadata:

- excitation amplitude;
- excitation frequency;
- assigned noise level;
- frequency-to-healthy-mode relationship descriptors.

An additional eight descriptors within the historical 86D set depend on the exact excitation-generator frequency.

Therefore,

\[
92D
\rightarrow
86D
\rightarrow
78D
\rightarrow
59D
\]

represents progressively stricter information availability.

The main 78D representation removes all 14 generator-dependent quantities:

\[
6+8=14.
\]

The 59D response-only representation additionally removes 19 descriptors dependent on the base-input signal:

\[
78-19=59.
\]

---

## Base-input sensing assumption

For the primary 78D representation:

> the base/ground-input history is assumed to be sensed and available.

Accordingly, descriptors computed from this history are considered signal-derived under this sensing assumption.

This must be described as:

“signal-derived from the sensed/measured ground-input history under the assumed base-input sensor.”

Do NOT describe the simulated input as “field measured”.

Reference: Fig. 2 and Table 1.

---

# 2.4 Dependency-aware descriptor observability under sensor sparsity

## Motivation

Sensor loss is treated as an information-availability problem rather than an artificial zero-response problem.

Unavailable structural channels are therefore NOT replaced by zeros.

Descriptor definitions are also NOT changed to accommodate a sparse layout.

Instead, a descriptor is retained only if every structural-response channel required by its original definition is available.

---

## Formal observability rule

For structural sensor layout

\[
S\subseteq\{1,2,3,4\},
\]

let \(D_j\) denote the structural-sensor dependency set required to compute descriptor \(f_j\).

Descriptor \(f_j\) is observable under layout \(S\) if and only if

\[
\boxed{
D_j\subseteq S
}
\]

or equivalently,

\[
A_j(S)=
\mathbf 1[D_j\subseteq S].
\]

The layout-specific observable representation is

\[
\mathbf f_S
=
\{f_j:A_j(S)=1\}.
\]

This preserves the physical meaning of every retained descriptor.

---

## Frozen dependency rules

Representative dependency rules include:

### Ground-only descriptors

\[
D_j=\varnothing.
\]

The base-input channel is assumed available and is not counted as one of the four structural sensors.

### Story-specific descriptors

For basic/statistical/spectral quantities at storey \(i\),

\[
D_j=\{i\}.
\]

### Story-1 relative-to-lower descriptors

These require:

- structural sensor 1; and
- the assumed base-input channel.

Thus the structural dependency is

\[
D_j=\{1\}.
\]

### Relative-to-lower descriptors for storey \(i>1\)

\[
D_j=\{i-1,i\}.
\]

### Adjacent-storey ratios and correlations

\[
D_j=\{i-1,i\}.
\]

### Spatial response-fraction descriptors

These require the complete structural sensing set:

\[
D_j=\{1,2,3,4\}.
\]

---

## Exhaustive structural layouts

All non-empty subsets of the four structural sensors are evaluated:

\[
2^4-1=15.
\]

The resulting observable descriptor dimensions are:

- 1: 19D
- 2: 17D
- 3: 17D
- 4: 17D

- 12: 36D
- 13: 31D
- 14: 31D
- 23: 34D
- 24: 29D
- 34: 34D

- 123: 53D
- 124: 48D
- 134: 48D
- 234: 51D

- 1234: 78D.

The complete layout exactly reproduces the clean 78D benchmark.

Reference: Fig. 2(d), Figs. 6–7.

---

# 2.5 Damage-inference models and evaluation protocol

## Linear reference estimator

Ridge regression is used as the principal linear reference.

The regularization parameter is selected using validation-set mean squared error.

Predictions are constrained to the physically defined output range

\[
0\le\hat y_i\le0.5.
\]

---

## Nonlinear estimator

RBF-kernel support vector regression is used as the primary nonlinear estimator.

The estimator is applied independently to the four storey-level damage outputs.

The frozen validation grid is:

\[
C\in
\{100,300,500,750,1000\},
\]

\[
\gamma\in
\{0.0005,0.00075,0.001,0.003,0.01\},
\]

\[
\epsilon\in
\{0.02,0.03,0.04\}.
\]

This gives

\[
5\times5\times3=75
\]

candidate configurations.

Selection criterion:

\[
\underset{\theta}{\operatorname{argmin}}
\;
\mathrm{MSE}_{\mathrm{validation}}(\theta).
\]

After selection, predictions are clipped to

\[
[0,0.5].
\]

The hyperparameter grid is frozen and is not expanded after observing test results.

---

## Preprocessing

For every train/validation/test split:

- descriptor standardization is fitted on the training data only;
- the same fitted transformation is applied to validation and test data;
- no test information enters preprocessing or hyperparameter selection.

For sensor-layout analyses, each layout has its own observable descriptor set and corresponding train-only scaler.

---

## Primary performance quantities

Primary error metrics include:

\[
MAE
=
\frac{1}{n}
\sum
|\hat y-y|,
\]

and RMSE.

Directional severe-damage diagnostics include signed bias

\[
Bias
=
\frac{1}{n}
\sum
(\hat y-y),
\]

absolute bias magnitude

\[
|Bias|,
\]

and underestimation ratio

\[
R_{\mathrm{under}}
=
\frac{
\#(\hat y<y)
}{
n
}.
\]

High-damage diagnostics are evaluated for

\[
y>0.20.
\]

---

## Information × estimator factorial comparison

Four principal repeated-split conditions are compared:

1. 78D + Ridge;
2. 78D + RBF-SVR;
3. 92D + Ridge;
4. 92D + RBF-SVR.

This forms a \(2\times2\) comparison between:

- information availability; and
- estimator flexibility.

The interaction is quantified using a difference-in-differences quantity:

\[
\Delta_{\mathrm{int}}
=
(M_{92,SVR}-M_{78,SVR})
-
(M_{92,Ridge}-M_{78,Ridge}),
\]

where \(M\) denotes the selected error metric.

Reference: Fig. 3 and Table 2.

---

# 2.6 Directional residual calibration and matched-noise evaluation

## Motivation for calibration

Repeated in-distribution evaluation reveals persistent severe-damage underprediction.

The calibration extension is therefore directional rather than a generic post-hoc error correction.

---

## Training-only residual calibration

For the selected 78D RBF-SVR model:

1. five-fold out-of-fold predictions are generated using training data only;
2. residuals are defined as

\[
r=y-\hat y_{\mathrm{OOF}};
\]

3. a non-negative monotonic correction function is estimated using isotonic regression;
4. the correction is applied as

\[
\hat y_{\mathrm{cal}}
=
\operatorname{clip}
[
\hat y
+
\alpha g(\hat y),
0,
0.5
].
\]

The correction is constrained upward:

\[
g(\hat y)\ge0.
\]

Frozen candidate calibration strengths are

\[
\alpha
\in
\{0.25,0.50,0.75,1.00,1.25,1.50\}.
\]

Calibration selection is validation-only.

The test set is not used to choose or expand the calibration range.

---

## Matched measurement-noise protocol

The controlled noise experiment begins from clean structural acceleration histories.

For each case:

\[
\sigma_{\mathrm{noise}}
=
\eta
\,
\sigma(X_{\mathrm{clean}}),
\]

where \(\eta\) is the prescribed relative noise level.

Gaussian measurement noise is generated as

\[
\epsilon
\sim
\mathcal N
(
0,
\sigma_{\mathrm{noise}}^2
).
\]

The perturbed structural response is

\[
X_{\eta}
=
X_{\mathrm{clean}}
+
\epsilon.
\]

Frozen noise levels are

\[
\eta
\in
\{0,0.05,0.10,0.20\}.
\]

For each non-zero noise level:

\[
5
\]

matched stochastic noise realizations are generated.

The 0% condition contains one deterministic clean realization.

Frozen random seed:

\[
20260809.
\]

---

## Important noise assumptions

Only the structural acceleration-response channels are perturbed.

The base/ground-input history remains clean.

Damage states, case identities, and excitation metadata are unchanged across matched conditions.

The trained estimator is NOT retrained on noisy data.

The clean training scaler is retained for all noise conditions.

Thus the experiment isolates inference degradation caused by measurement-induced descriptor distribution shift.

---

## Noise failure diagnostics

The controlled noise analysis evaluates:

1. overall and severity-specific MAE;
2. high-damage signed bias;
3. high-damage underestimation;
4. prediction-distribution inflation;
5. clipping at the physical prediction bounds;
6. standardized descriptor-space displacement.

Descriptor shift is measured relative to the clean training standardization.

The principal descriptor-shift quantity is

\[
E(|\Delta z_j|).
\]

Large descriptor shifts are interpreted as distribution-shift diagnostics.

They are NOT interpreted as feature-level causal proof because no causal feature-ablation analysis is performed.

Reference: Figs. 4–5 and Table 3.

---

# 2.7 Exhaustive sensor-layout and statistical evaluation

## Layout-specific inference

Each of the 15 non-empty structural sensor layouts is evaluated independently.

For each layout:

1. the physically observable descriptor subset is determined using the dependency rule;
2. a layout-specific training scaler is fitted;
3. the same frozen 75-candidate RBF-SVR grid is evaluated;
4. hyperparameters are selected on validation MSE;
5. the selected estimator is evaluated once on the fixed test set.

No sensor layout uses zero-masked unavailable response channels.

No descriptor is redefined for sparse sensing.

---

## Complete subset lattice

The 15 layouts form the complete non-empty subset lattice over four structural sensors.

A marginal sensor addition is an edge

\[
S
\rightarrow
S\cup\{j\},
\qquad
j\notin S.
\]

Across the complete lattice there are

\[
28
\]

admissible one-sensor additions.

Marginal improvement is evaluated as

\[
\Delta MAE
=
MAE(S)
-
MAE(S\cup\{j\}).
\]

Thus

\[
\Delta MAE>0
\]

denotes improvement after adding the sensor.

The same definition is used for high-damage MAE.

---

## Paired case-level bootstrap

Sensor-layout uncertainty is evaluated by paired bootstrap resampling of complete test cases.

Frozen protocol:

- test cases: 450;
- bootstrap replicates: 5000;
- random seed: 20260810;
- resampling unit: complete structural case;
- all four storey outputs for a sampled case remain together;
- all layouts use the same resampled cases in each bootstrap replicate.

This preserves:

- within-case dependence across storeys; and
- paired comparison across layouts.

Percentile 95% bootstrap intervals are reported.

---

## Interpretation boundary of bootstrap intervals

The bootstrap intervals quantify conditional uncertainty under resampling of the present simulated 450-case test population.

They do NOT quantify:

- cross-structure uncertainty;
- laboratory-to-field uncertainty;
- uncertainty across independently constructed structural systems.

No family-wise multiplicity adjustment is applied across the complete set of pairwise or marginal comparisons.

Accordingly, the manuscript refers to:

“paired bootstrap support”

rather than universal statistical proof.

---

## General statistical reporting principles

The following reporting rules are frozen:

1. effect magnitude and direction are primary;
2. paired confidence intervals are reported where available;
3. repeated-split consistency is reported where relevant;
4. p-values are supportive rather than the primary evidence;
5. repeated train/test partitions from the same 3000 simulated cases are not treated as independent physical experiments;
6. no post-test hyperparameter-grid expansion is performed;
7. no test-driven damage-bin, noise-level, sensor-layout, or calibration redesign is performed.

---

# Methods scientific boundary

The present methodology establishes a controlled computational framework for studying:

\[
\text{information provenance}
\rightarrow
\text{observability}
\rightarrow
\text{damage inference}
\rightarrow
\text{failure mechanism}.
\]

The present study does NOT establish:

- cross-structure generalization;
- laboratory validation;
- field validation;
- universal optimal sensor placement;
- universal noise robustness;
- feature-level causal mechanisms.

These boundaries must remain explicit in the final manuscript.
