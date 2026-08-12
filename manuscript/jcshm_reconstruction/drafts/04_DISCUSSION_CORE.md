# 4. Discussion — Scientific Core

> STATUS: SCIENTIFIC CORE ONLY
>
> This document freezes the interpretation of the experimental evidence.
>
> The final manuscript may improve language, transitions, paragraph structure,
> and literature integration, but MUST NOT strengthen the claims beyond the
> boundaries defined here.
>
> The Discussion should explain mechanisms and implications rather than repeat
> numerical Results.

---

# 4.1 Information provenance is part of the SHM inference problem

## Central interpretation

The results show that structural damage inference cannot be evaluated only in
terms of estimator architecture and prediction error.

A third component is equally important:

\[
\boxed{\text{Where does the estimator's information come from?}}
\]

The historical 92D representation combines:

- structural-response-derived information;
- base-input-dependent information;
- exact excitation-generator-frequency-derived quantities;
- direct generator metadata.

Only the first two categories are available under the sensing assumptions of
the primary deployment-oriented formulation.

Therefore, an estimator may achieve lower numerical error while relying on
information that would not exist at deployment time.

---

## Main methodological implication

Descriptor provenance should be treated as part of model specification.

The relevant comparison is not simply

\[
\text{more descriptors}
\quad\text{versus}\quad
\text{fewer descriptors}.
\]

Instead, it is

\[
\text{privileged simulation information}
\quad\text{versus}\quad
\text{sensing-accessible information}.
\]

The 92D representation is therefore useful as an information-rich reference,
but should not be interpreted as a deployable benchmark or an attainable field
upper bound.

The historical 86D representation is also not fully deployment-clean because
eight descriptors still depend on the exact excitation-generator frequency.

The 78D representation establishes the principal deployment-aware information
boundary used in the manuscript.

---

## Scientific contribution

The contribution is NOT the removal of 14 arbitrary features.

The contribution is the explicit separation between:

\[
\text{simulation privilege}
\]

and

\[
\text{physically sensed information}.
\]

This distinction prevents performance improvements obtained from unavailable
simulation metadata from being interpreted as improvements in deployable SHM
capability.

---

## Preferred manuscript claim

> Information provenance should be treated as an explicit component of
> structural damage-inference design because prediction performance can depend
> substantially on quantities that are available during simulation but absent
> from the intended sensing configuration.

---

## Important terminology

Avoid casually describing the 92D/78D comparison as:

“feature selection”.

Prefer:

- information-provenance restriction;
- deployment-aware information boundary;
- signal-derived representation;
- privileged-information reference.

---

# 4.2 Estimator flexibility and information availability are complementary

## Interpretation of the factorial evidence

RBF-SVR substantially improves the primary 78D inference problem relative to
Ridge.

However, access to additional privileged information also improves both
estimators.

The factorial interaction further indicates that the information gain is
larger under the nonlinear estimator.

This supports a complementary relationship:

\[
\boxed{
\text{information richness}
\times
\text{estimator flexibility}
}
\]

rather than a substitution relationship.

---

## What the result means

A flexible estimator can exploit nonlinear structure in the available
descriptor space.

It does NOT follow that estimator flexibility can reconstruct information that
was never observed.

Likewise, adding information does not remove the need for an estimator capable
of representing the corresponding nonlinear mapping.

Thus:

\[
\text{better model}
\not\Rightarrow
\text{missing information recovered},
\]

and

\[
\text{more information}
\not\Rightarrow
\text{model flexibility irrelevant}.
\]

---

## Implication for SHM methodology

Model development and sensing/information design should not be treated as
independent optimization problems.

An apparently weak estimator may be limited by representation capacity,
whereas an apparently weak sensing configuration may constrain even a flexible
estimator.

The present factorial evidence suggests that both dimensions should be
reported when assessing damage-inference capability.

---

## Boundary

The study does not establish that RBF-SVR is an optimal estimator.

RBF-SVR is used as a conventional nonlinear reference that reveals the
importance of estimator flexibility without introducing a new deep-learning
architecture.

Do NOT position the paper as an SVR-method paper.

---

# 4.3 Improved average accuracy does not imply trustworthy severe-damage inference

## Central interpretation

The transition from Ridge to RBF-SVR materially reduces overall MAE and
high-damage MAE.

Nevertheless, severe-damage predictions retain systematic negative bias.

Therefore:

\[
\boxed{
\text{lower average error}
\neq
\text{absence of directional failure}
}
\]

This distinction is particularly important in SHM because error direction may
have different engineering consequences.

Underprediction of severe damage may be more operationally problematic than a
numerically similar overprediction.

---

## Why aggregate MAE is insufficient

A single aggregate error metric mixes:

- undamaged states;
- low damage;
- medium damage;
- severe damage;
- positive error;
- negative error.

Two models can therefore obtain similar global MAE while exhibiting very
different behavior in the damage regime of greatest engineering interest.

The present results demonstrate the value of reporting:

\[
MAE,
\quad
Bias,
\quad
R_{\mathrm{under}},
\]

jointly with severity-stratified performance.

---

## Broader methodological implication

SHM regression should be diagnosed not only by “how much error” occurs but also

\[
\boxed{\text{where and in which direction the error occurs}.}
\]

This is one of the reasons that the present study frames severe-damage
underprediction as a failure mechanism rather than merely a high-error subset.

---

# 4.4 Asymmetric calibration demonstrates both the value and limitation of failure-targeted correction

## In-distribution interpretation

The one-sided training-only calibration reduces:

- high-damage MAE;
- high-damage absolute bias;
- severe-damage underestimation.

This confirms that the repeated clean/in-distribution benchmark contains a
directional residual structure that can be partially corrected.

However, the improvement is accompanied by deterioration in the low-damage
regime.

---

## Scientific implication

Calibration redistributes error rather than creating new information.

The correction is beneficial because the in-distribution residual field has a
consistent directional structure.

Its benefit is therefore conditional on that structure remaining stable.

This yields an important distinction:

\[
\text{estimator correction}
\neq
\text{failure-mechanism elimination}.
\]

---

## Deployment implication

A post-hoc correction fitted to one directional bias should not automatically
be assumed transferable to a shifted sensing environment.

The matched-noise results directly demonstrate this limitation.

Once high-damage bias changes sign, the same upward-only correction becomes
directionally inappropriate.

Thus calibration should be validated under the anticipated sensing
distribution rather than only on an in-distribution validation set.

---

# 4.5 Measurement noise and sensor sparsity are not equivalent forms of degraded sensing

## Central conceptual result

A major result of the study is that the two sensing-degradation modes produce
qualitatively different inference failures.

They should not be summarized as:

“worse sensing causes larger error.”

Instead:

\[
\boxed{
\text{measurement noise perturbs available information}
}
\]

whereas

\[
\boxed{
\text{sensor loss removes observable information}.
}
\]

These operations affect the inverse problem differently.

---

# 4.6 Measurement noise produces a distribution-shift pathway

## Observed pathway

The matched-noise evidence is consistent with the sequence

\[
\text{response perturbation}
\rightarrow
\text{descriptor distribution shift}
\rightarrow
\text{prediction inflation}
\rightarrow
\text{bias reversal}
\rightarrow
\text{upper-bound saturation}.
\]

The most important finding is not simply the increase in MAE.

The direction of severe-damage error changes.

Under the clean benchmark, severe damage is predominantly underestimated.

At higher noise levels, the signed bias becomes positive.

This demonstrates that the original failure mechanism is not invariant to
measurement-distribution shift.

---

## Descriptor-space interpretation

The largest standardized shifts occur mainly in:

- spectral-centroid descriptors;
- relative-to-lower / cross-level amplitude descriptors;
- related amplification-type quantities.

These observations provide a representation-level diagnostic of where the
measurement perturbation enters the inference chain.

However:

\[
\boxed{
\text{large descriptor shift}
\neq
\text{causal proof of prediction failure}.
}
\]

No controlled feature intervention or ablation was performed to establish
feature-level causality.

---

## Shift/error association

At moderate noise levels, case-level descriptor displacement and prediction
error are positively associated.

At 20% noise this association becomes weak despite very large prediction
failure.

This should not be interpreted as evidence that descriptor shift becomes
irrelevant.

A plausible interpretation is that prediction clipping/saturation compresses
the variation in the final prediction error after the estimator has already
entered a strongly shifted regime.

This interpretation is mechanistically consistent with the simultaneous rise
in upper-bound clipping.

### Claim strength

Use:

“consistent with saturation limiting the observable shift-error association”.

Do NOT write:

“saturation was proven to cause the correlation collapse”.

---

# 4.7 The noise experiment identifies a robustness boundary rather than robustness

## Critical positioning

The clean-trained estimator performs well in the clean controlled benchmark,
but error rises strongly under measurement noise.

Therefore, this experiment should not be presented as demonstrating noise
robustness.

Its scientific value is instead that it identifies where and how the inference
pipeline breaks down.

Preferred framing:

\[
\boxed{\text{robustness boundary / failure diagnosis}}
\]

rather than:

\[
\boxed{\text{robust model}}
\]

---

## Practical implication

If descriptor-based inference is transferred to real sensing systems, the
measurement-noise distribution should be treated explicitly during:

- training;
- validation;
- uncertainty analysis;
- calibration design.

The present clean-trained results suggest that an estimator calibrated only on
clean or low-noise response may behave qualitatively differently after a
substantial measurement-distribution shift.

The present study does NOT test noise-aware retraining or domain adaptation.

These are future extensions rather than missing controls required for the
current failure-mechanism study.

---

# 4.8 Sensor sparsity produces an observability-loss pathway

## Different mechanism from noise

When structural sensors are removed, the available response histories are not
merely noisier.

Entire descriptor relationships become physically unavailable.

Examples include:

- adjacent-storey quantities;
- relative-to-lower quantities;
- full-structure spatial fractions.

Therefore the representation itself changes according to the dependency rule

\[
D_j\subseteq S.
\]

---

## Consequence

As structural sensing becomes sparse:

\[
\text{cross-storey information decreases}
\]

and severe-damage underprediction becomes increasingly dominant.

At the single-sensor level, high-damage underestimation approaches unity for
all four placements.

This behavior contrasts with the noise pathway, where severe-damage bias
eventually reverses toward positive values.

---

## Central interpretation

The sensor-sparsity results support the view that severe-damage quantification
requires sufficient structural observability across multiple response
locations.

A flexible estimator cannot recover descriptor relationships that are not
computable under the available sensor configuration.

---

## Important terminology boundary

The manuscript may use:

- descriptor observability;
- sensing observability;
- practical observability of the inference representation.

Avoid claiming formal control-theoretic or mathematical system observability
unless explicitly defined.

The present study evaluates empirical descriptor/inference observability rather
than deriving a classical observability rank condition.

---

# 4.9 Sensor count and sensor placement encode different information

## Count effect

Average performance improves from one to four structural sensors.

More importantly, every admissible one-sensor addition in the tested subset
lattice improves both overall and high-damage MAE with paired case-bootstrap
support.

This provides unusually consistent evidence, within the present benchmark,
that added sensing improves the available inference representation.

---

## Placement effect

Sensor count alone does not determine performance.

Different layouts with the same number of sensors can produce:

- different observable descriptor dimensions;
- different physical descriptor relationships;
- different inference error.

Moreover, the best two-sensor layout differs depending on whether the objective
is:

- overall MAE; or
- high-damage MAE.

Therefore:

\[
\boxed{
\text{sensor placement is objective-dependent}.
}
\]

---

## Equal-dimensional evidence

The comparison between layouts {1,3} and {1,4} is particularly informative
because both produce 31 observable descriptors while exhibiting a supported
performance difference.

This shows that:

\[
\text{descriptor count}
\]

alone cannot fully represent sensing value.

The physical identity of the available information also matters.

Other equal-dimensional comparisons are weaker and should not be overgeneralized.

---

# 4.10 The 28-edge lattice supports monotonic sensing value only within the tested system

## Strong evidence

For all 28 admissible transitions

\[
S
\rightarrow
S\cup\{j\},
\]

overall MAE decreases.

The same is true for high-damage MAE.

Every paired 95% case-bootstrap interval remains on the beneficial side of
zero.

This is one of the strongest empirical results in the paper.

---

## Correct interpretation

The result supports:

> Within the complete tested four-sensor subset lattice, every additional
> structural sensor improved both overall and severe-damage inference.

It does NOT establish:

> Every additional sensor is universally beneficial in SHM.

---

## Why universal interpretation is inappropriate

Marginal sensor value depends on:

- structural topology;
- excitation population;
- damage population;
- sensor locations;
- descriptor definitions;
- estimator;
- prediction objective.

The paired bootstrap additionally conditions on the current 450-case simulated
test population.

No family-wise multiplicity correction was applied across the 28 edges.

Accordingly, the result should be described as strong paired-bootstrap support
within the current benchmark rather than a universal statistical law.

---

# 4.11 Relative and absolute marginal sensing value convey different information

Relative error reduction increases as the sensor count approaches the complete
four-sensor configuration.

However, absolute improvement does not increase monotonically.

For high-damage MAE in particular, absolute improvement decreases across the
three sensor-count transitions while relative improvement increases.

This occurs because the denominator of the relative metric decreases as the
baseline error becomes smaller.

Therefore:

\[
\boxed{
\text{large relative gain}
\neq
\text{large absolute gain}.
}
\]

The manuscript should report both where appropriate.

This prevents an incorrect claim that the physical marginal value of each new
sensor necessarily increases as the network becomes denser.

---

# 4.12 Information provenance and sensor observability form a unified deployment boundary

## Unified interpretation

The provenance analysis and sensor-layout analysis address two levels of the
same problem.

### Information provenance asks:

\[
\text{Is this quantity obtainable from the assumed sensing system at all?}
\]

### Dependency-aware observability asks:

\[
\text{Can this quantity still be computed under the current sensor layout?}
\]

Together they define the effective information set supplied to the estimator.

---

## Unified formulation

A descriptor should enter a deployment-oriented inference model only when:

1. its source information is physically accessible; and
2. its required sensing channels are currently available.

Conceptually:

\[
\mathcal F_{\mathrm{available}}
=
\left\{
f_j:
P_j\in\mathcal I_{\mathrm{deploy}}
\;\land\;
D_j\subseteq S
\right\},
\]

where:

- \(P_j\) denotes descriptor provenance;
- \(\mathcal I_{\mathrm{deploy}}\) denotes the deployment-accessible information
  sources;
- \(D_j\) denotes the structural-sensor dependency set;
- \(S\) denotes the available structural-sensor layout.

This equation may be used in Discussion if useful, but it should not replace
the simpler observability definition in Methods.

---

## Main conceptual contribution

The paper therefore moves the question from:

> “Which estimator gives the lowest MAE?”

toward:

> “What damage information is actually observable from the sensing system,
> how effectively can the estimator exploit it, and how does the inference fail
> when that information is perturbed or removed?”

This is the preferred conceptual framing of the manuscript.

---

# 4.13 Implications for deployment-oriented SHM studies

The present evidence suggests several methodological practices for future
simulation-based SHM studies.

## 1. Audit information provenance

Simulation metadata should not enter a deployment claim without explicit
justification.

## 2. Preserve physical descriptor definitions under sensor loss

Unavailable measurements should not automatically be replaced by artificial
zeros when the descriptor itself becomes physically undefined.

## 3. Diagnose directional error

Severity-specific bias and underestimation should accompany aggregate MAE.

## 4. Stress-test distribution shift

A model that performs well under an in-distribution split should not be assumed
to retain the same failure mode after sensing noise changes.

## 5. Separate sensing quantity from sensing location

Sensor count and sensor placement should be analyzed separately.

## 6. Treat calibration as distribution-dependent

A directional calibration rule should be validated under the sensing
conditions expected during deployment.

---

# 4.14 Role of the simplified structural benchmark

## Why the benchmark remains scientifically useful

The four-storey linear-elastic frame and randomized sinusoidal excitation form
a deliberately controlled environment.

This allows:

- exact information provenance to be known;
- damage to be prescribed independently;
- clean and noisy responses to be paired;
- all structural sensor subsets to be enumerated;
- failure pathways to be isolated.

These properties would be difficult to obtain simultaneously in field data.

Therefore, the simplified model is appropriate for mechanism isolation.

---

## What the benchmark does not provide

The same simplification limits external validity.

The present evidence does not address:

- material nonlinearity;
- geometric nonlinearity;
- cracking and hysteresis;
- yielding or collapse;
- environmental and operational variability;
- sensor bias/drift/dropout;
- colored or non-Gaussian measurement noise;
- real earthquake records;
- uncertainty in structural properties;
- cross-structure transfer;
- laboratory-to-field transfer.

The controlled benchmark should therefore be framed as a methodological
failure-mechanism study rather than a direct field-validation study.

---

# 4.15 Specific limitations that must remain explicit

## L1 — Single structural topology

Only one four-storey, two-bay frame is studied.

No cross-structure generalization is demonstrated.

---

## L2 — Linear-elastic model

Damage is represented through prescribed storey-level column stiffness
reduction.

The study does not model cracking, hysteresis, yielding, or progressive
nonlinear deterioration.

---

## L3 — Parametric excitation

The benchmark uses randomized sinusoidal base excitation.

It does not use recorded earthquake ground motions.

---

## L4 — Simulation-only evidence

No laboratory or field-monitoring validation is included.

---

## L5 — In-distribution repeated splits

Repeated holdouts originate from the same 3000 simulated structural cases.

They are not independent physical experiments.

---

## L6 — Controlled measurement noise

The matched-noise experiment uses additive zero-mean Gaussian structural
response noise.

The base-input history remains clean.

Other measurement-error processes are not tested.

---

## L7 — Base-input sensing assumption

The main 78D representation assumes that the base/ground-input history is
available from a sensing channel.

The 59D response-only analysis provides the stricter no-base-input reference.

---

## L8 — Frozen hyperparameter grids

Several estimator/calibration optima occur at or near the frozen candidate-grid
boundaries.

The grids were intentionally not expanded after test evaluation.

---

## L9 — Bootstrap interpretation

The 5000-replicate paired bootstrap represents uncertainty conditional on the
current 450 simulated test cases.

It is not a measure of cross-structure uncertainty.

---

## L10 — Multiplicity

No family-wise multiplicity correction is applied to the complete set of
sensor-layout comparisons.

---

## L11 — Descriptor causality

Shifted descriptor families are diagnostic observations.

Individual descriptors have not been demonstrated to causally generate the
observed model failure.

---

# 4.16 Future work should follow directly from the identified boundaries

Future work should NOT be written as a generic wish list.

Each extension should address a specific limitation identified by the present
results.

Priority directions:

### F1. Cross-structure evaluation

Repeat the provenance/observability framework across structures with different:

- heights;
- bay configurations;
- stiffness distributions;
- modal characteristics.

Purpose:

test whether the present failure pathways generalize beyond the current
topology.

---

### F2. Nonlinear structural response

Introduce controlled material/geometric nonlinearities and hysteretic damage.

Purpose:

test whether the severe-damage directional failure changes when the forward
model itself becomes nonlinear.

---

### F3. Recorded and broadband excitation

Evaluate real or spectrum-compatible ground motions.

Purpose:

test descriptor provenance and distribution-shift behavior under richer
excitation content.

---

### F4. Noise-aware training

Compare clean-trained inference with:

- noise augmentation;
- robust preprocessing;
- domain adaptation;
- uncertainty-aware estimators.

Purpose:

determine whether the noise robustness boundary can be shifted.

---

### F5. Fault-aware sensing

Extend beyond additive Gaussian noise to:

- sensor dropout;
- bias;
- drift;
- synchronization error;
- colored noise.

Purpose:

separate additional sensing-failure mechanisms.

---

### F6. Deployment-aware sensor design

Use the dependency-aware framework as a basis for multi-objective sensor design
considering:

- overall accuracy;
- severe-damage accuracy;
- sensing cost;
- redundancy;
- failure tolerance.

Purpose:

move from exhaustive diagnosis in a small topology toward practical
sensor-network design.

---

### F7. Laboratory / field validation

Evaluate whether the identified directional failure mechanisms persist in
physical measurements.

Purpose:

establish external validity rather than merely increasing simulation scale.

---

# 4.17 Manuscript novelty positioning

## The paper is NOT primarily novel because of

- RBF-SVR;
- Ridge regression;
- isotonic regression;
- handcrafted statistics;
- OpenSeesPy simulation;
- the absolute number 78 or 92.

---

## The paper's defensible novelty lies in the combined framework

### Contribution 1

A deployment-aware information-provenance formulation that separates
simulation-privileged quantities from signal-derived information.

### Contribution 2

A dependency-aware definition of descriptor availability under structural
sensor sparsity without zero masking or descriptor redefinition.

### Contribution 3

A failure-mechanism diagnosis showing that measurement noise and sensor loss
produce qualitatively different severe-damage error pathways.

### Contribution 4

An exhaustive complete structural-sensor subset analysis showing consistent
benefit across all 28 admissible one-sensor additions in the tested topology.

---

## Preferred overall positioning

> The manuscript is a deployment-aware computational SHM methodology and
> failure-diagnosis study rather than a new machine-learning algorithm paper.

This positioning must remain consistent in:

- title;
- abstract;
- introduction;
- discussion;
- conclusion;
- cover letter.

---

# 4.18 Central Discussion synthesis

The complete evidence supports the following interpretation:

\[
\boxed{
\text{Damage-inference performance is constrained by both}
\;
\text{what information exists}
\;
\text{and}
\;
\text{how that information is observed}.
}
\]

Estimator flexibility determines how effectively the available information is
mapped to damage.

Measurement noise perturbs that information and can alter the direction of the
resulting prediction error.

Sensor loss removes parts of the observable representation and intensifies
severe-damage underprediction.

Therefore, deployment-oriented structural damage inference should be assessed
through the combined lens of:

\[
\boxed{
\text{information provenance}
+
\text{observability}
+
\text{estimator capability}
+
\text{failure direction}.
}
\]

This is the principal Discussion-level scientific message of the manuscript.

---

# Discussion claim boundary

The strongest defensible conclusion is:

> Within the present controlled computational benchmark, damage-inference
> failure depends not only on estimator accuracy but on the provenance,
> availability, and degradation mode of the sensing information supplied to
> the estimator.

Do NOT elevate this to:

- a universal SHM law;
- proof of formal identifiability;
- a field-validated sensor-placement rule;
- proof that the identified descriptors are causally responsible;
- proof of robustness under realistic earthquakes.
