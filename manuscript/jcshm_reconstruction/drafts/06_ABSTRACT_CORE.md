# Abstract — Scientific Core

> STATUS: SCIENTIFIC CORE ONLY
>
> Purpose:
> freeze the scientific content of the manuscript abstract before final
> journal-style compression and language editing.
>
> The final abstract may improve:
>
> - sentence structure;
> - concision;
> - transitions;
> - terminology consistency;
> - JCSHM style.
>
> The final abstract MUST NOT:
>
> - mix repeated-split and fixed-split numerical results;
> - describe the 92D representation as a deployable upper bound;
> - claim field or experimental validation;
> - claim universal noise robustness;
> - claim universal sensor-placement optimality;
> - describe RBF-SVR as a novel algorithm;
> - interpret descriptor shift as feature-level causal proof.

---

# A1. Background / problem

Simulation-based structural damage inference can achieve low prediction error
while relying on information that may not be available from the intended
monitoring system.

Moreover, sensing degradation can affect an inference pipeline in fundamentally
different ways:

- measurement noise perturbs available response information;
- sensor loss removes information required to compute physically meaningful
  descriptors.

Aggregate prediction accuracy alone may therefore conceal important
deployment-related failure mechanisms.

---

# A2. Objective

This study develops a deployment-aware computational SHM framework for
storey-level damage inference that jointly examines:

\[
\text{information provenance},
\]

\[
\text{descriptor observability},
\]

\[
\text{estimator capability},
\]

and

\[
\text{directional inference failure}.
\]

---

# A3. Numerical benchmark

A four-storey, two-bay linear-elastic OpenSeesPy frame is used as a controlled
mechanism-isolation benchmark.

Storey-level damage is represented by column stiffness degradation.

The frozen dataset contains

\[
3000
\]

simulated structural cases subjected to randomized sinusoidal base excitation.

The study is computational and simulation-based.

It does NOT constitute laboratory or field validation.

---

# A4. Information-provenance formulation

Four nested information representations are distinguished:

\[
92D
\rightarrow
86D
\rightarrow
78D
\rightarrow
59D.
\]

The 92D representation contains simulation-privileged information.

The principal 78D representation excludes:

- direct generator metadata;
- exact generator-frequency-derived descriptors;

while retaining descriptors derived from:

- structural response histories; and
- the sensed base-input history under the assumed base-input sensor.

The 78D representation is therefore the primary deployment-oriented
representation.

The 92D representation is a privileged-information reference, NOT an upper
performance bound.

---

# A5. Estimator comparison

Ridge regression is used as the principal linear reference.

RBF-SVR is used as a conventional nonlinear estimator.

The 78D-versus-92D and Ridge-versus-RBF-SVR comparison is evaluated across ten
repeated train/validation/test splits.

Under the primary 78D representation, RBF-SVR reduces mean overall MAE relative
to Ridge by approximately

\[
29.9\%.
\]

High-damage MAE decreases by approximately

\[
36.3\%.
\]

The nonlinear estimator substantially improves severe-damage inference but does
not eliminate systematic high-damage underprediction.

---

# A6. Directional failure and calibration

Severity-stratified diagnostics identify persistent negative high-damage bias
under the repeated in-distribution benchmark.

A training-only one-sided residual calibration partially reduces:

- high-damage MAE;
- absolute bias;
- underestimation.

However, the calibration effect is distribution-dependent and does not
eliminate the underlying failure mechanism.

Calibration is therefore treated as a diagnostic extension rather than the
principal contribution.

---

# A7. Matched measurement-noise experiment

A separate controlled fixed-split experiment perturbs only the structural
response channels at noise levels

\[
0\%,5\%,10\%,20\%.
\]

The estimator is trained on clean data and is not retrained under noise.

This experiment must remain numerically distinct from the repeated-split
factorial experiment.

High-damage signed bias changes from approximately

\[
-0.021
\]

under the clean condition to

\[
+0.129
\]

at 20% response-measurement noise.

The observed failure sequence is:

\[
\text{descriptor distribution shift}
\rightarrow
\text{prediction inflation}
\rightarrow
\text{bias reversal}
\rightarrow
\text{upper-bound saturation}.
\]

This experiment identifies a measurement-noise robustness boundary.

It does NOT demonstrate noise robustness.

---

# A8. Dependency-aware sensor observability

For structural sensor layout \(S\), descriptor \(f_j\) is retained only when
its required structural sensor set \(D_j\) satisfies

\[
D_j\subseteq S.
\]

No unavailable structural response channel is zero-masked.

No descriptor definition is changed to accommodate sparse sensing.

All

\[
15
\]

non-empty subsets of the four structural sensors are evaluated.

The complete subset lattice contains

\[
28
\]

admissible one-sensor additions.

---

# A9. Sensor-layout headline result

Across all

\[
28/28
\]

admissible one-sensor additions:

- overall MAE decreases;
- high-damage MAE decreases.

For both metrics, every paired 95% case-bootstrap interval remains entirely on
the beneficial side of zero.

This result is conditional on:

- the present structural topology;
- the present descriptor dependencies;
- the present simulated test population.

It is NOT a universal sensor-placement law.

---

# A10. Integrated failure-mechanism result

Measurement noise and structural sensor loss produce qualitatively different
damage-inference failure pathways.

Measurement noise:

\[
\text{perturbs available information}
\rightarrow
\text{bias reversal / saturation}.
\]

Sensor sparsity:

\[
\text{removes observable information}
\rightarrow
\text{severe-damage underprediction}.
\]

Therefore the two sensing degradations should not be collapsed into one generic
notion of reduced model robustness.

---

# A11. Central conclusion

The principal scientific conclusion is:

> Within the present controlled computational benchmark, structural
> damage-inference capability depends not only on estimator accuracy but also on
> the provenance, availability, and degradation mode of the sensing information
> supplied to the estimator.

Preferred final implication:

> Deployment-oriented SHM inference should therefore be evaluated through the
> combined lens of information provenance, sensing observability, estimator
> capability, and failure direction.

---

# A12. Recommended final abstract architecture

The final journal abstract should contain approximately five logical units.

## Sentence group 1 — Problem

Simulation-based damage inference may use information unavailable during
deployment, while sensing degradation may alter not only error magnitude but
also the failure mechanism.

## Sentence group 2 — Method

Use the controlled four-storey OpenSeesPy benchmark, deployment-aware
information hierarchy, nonlinear-versus-linear comparison, matched-noise
stress test, and exhaustive dependency-aware sensor-layout analysis.

## Sentence group 3 — Estimator headline

Report:

\[
29.9\%
\]

overall MAE reduction and

\[
36.3\%
\]

high-damage MAE reduction for RBF-SVR versus Ridge under the 78D primary
representation across repeated splits.

## Sentence group 4 — Failure-mechanism headlines

Report:

\[
-0.021
\rightarrow
+0.129
\]

high-damage signed-bias reversal from clean to 20% response noise.

Report:

\[
28/28
\]

sensor additions beneficial for both overall and high-damage MAE with paired
95% case-bootstrap support.

## Sentence group 5 — Conclusion

State that measurement noise and sensor sparsity act through distinct
information-degradation mechanisms and that deployment-oriented SHM evaluation
should explicitly consider provenance and observability.

---

# A13. Protocol-separation lock

The following values belong to different experimental protocols and MUST NOT be
presented as though they were generated from one common evaluation.

## Repeated-split factorial benchmark

78D RBF-SVR mean overall MAE:

\[
0.0314500.
\]

This value supports the repeated estimator comparison.

## Controlled clean fixed-split benchmark

Clean 78D RBF-SVR overall MAE:

\[
0.0220471.
\]

This value is the clean anchor for:

- matched-noise analysis;
- exhaustive sensor-layout analysis.

The final abstract does not need to report either absolute MAE value.

Prefer relative estimator improvement and mechanism-level results to avoid
protocol confusion.

---

# A14. Numbers permitted in the final abstract

Preferred headline numerical set:

### Estimator effect

\[
29.9\%
\]

overall MAE reduction.

\[
36.3\%
\]

high-damage MAE reduction.

### Noise failure

\[
-0.021
\rightarrow
+0.129
\]

high-damage signed bias from 0% to 20% measurement noise.

### Sensor observability

\[
28/28
\]

admissible one-sensor additions beneficial for overall and high-damage MAE,
with all paired 95% intervals remaining beneficial.

Optional if word count permits:

\[
15
\]

non-empty sensor layouts.

Do NOT overload the abstract with:

- all four 92D/78D factorial values;
- calibration percentages;
- descriptor-shift rankings;
- sensor-count step percentages;
- bootstrap replicate count;
- all noise-level MAEs.

Those belong in the main text.

---

# A15. Prohibited abstract claims

Do NOT write:

> The proposed model is robust to measurement noise.

Evidence shows substantial degradation.

---

Do NOT write:

> The proposed sensor layout is optimal.

No universal layout optimization is established.

---

Do NOT write:

> The 92D model provides the theoretical upper bound.

It is only a privileged-information reference.

---

Do NOT write:

> The approach was validated for earthquake-induced structural damage.

The excitation is randomized sinusoidal base excitation, not recorded
earthquake ground motion.

---

Do NOT write:

> The shifted spectral descriptors caused prediction failure.

The descriptor-shift analysis is diagnostic, not causal.

---

Do NOT write:

> The proposed SVR method...

RBF-SVR is a conventional estimator and not the methodological novelty.

---

# A16. Draft abstract skeleton for later Codex expansion

BACKGROUND:
Simulation-based structural damage inference may rely on information unavailable
to deployed monitoring systems, while different sensing degradations may produce
different inference failures.

OBJECTIVE:
Establish a deployment-aware framework linking information provenance,
descriptor observability, estimator capability, and failure direction.

METHOD:
Use a controlled 3000-case four-storey OpenSeesPy benchmark; compare a 78D
signal-derived representation with a 92D privileged-information reference under
linear and nonlinear estimators; perform controlled matched-noise analysis; and
exhaustively evaluate all 15 non-empty structural sensor layouts using
dependency-aware descriptor availability.

RESULT 1:
Under the primary 78D representation, RBF-SVR reduces repeated-split overall and
high-damage MAE by approximately 29.9% and 36.3% relative to Ridge.

RESULT 2:
Controlled response noise reverses high-damage signed bias from approximately
-0.021 in the clean condition to +0.129 at 20% noise and progresses toward
upper-bound prediction saturation.

RESULT 3:
Across all 28 admissible one-sensor additions, both overall and high-damage MAE
decrease, with every paired 95% case-bootstrap interval remaining beneficial.

CONCLUSION:
Measurement noise perturbs available information whereas sensor loss removes
observable information, producing distinct damage-inference failure mechanisms.

---

# Abstract claim lock

The abstract must communicate three central findings:

1. estimator flexibility materially improves inference under the
   deployment-oriented 78D representation;

2. measurement-noise shift can reverse the direction of severe-damage error;

3. dependency-aware sensor additions consistently improve inference across the
   complete tested subset lattice.

The abstract's scientific identity is:

> deployment-aware SHM inference through information provenance, observability,
> and failure-mechanism diagnosis.

It is NOT:

> a new SVR damage-prediction algorithm.
