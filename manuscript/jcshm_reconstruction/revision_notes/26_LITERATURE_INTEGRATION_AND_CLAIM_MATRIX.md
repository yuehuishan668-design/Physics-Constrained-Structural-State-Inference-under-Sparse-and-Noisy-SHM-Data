# Literature Integration and Claim Matrix

## Status

**SCIENTIFIC CLAIM LOCK**

Purpose:

Integrate Literature Blocks 1–3 with the frozen research questions,
experimental evidence, novelty positioning, and manuscript claim boundaries.

This file controls how the literature is used in:

- Introduction;
- Discussion;
- Abstract;
- Conclusions;
- Cover letter.

Later prose editing may improve wording but MUST NOT strengthen claims beyond
the boundaries defined here.

---

# 1. Manuscript-level scientific identity

Preferred identity:

> A deployment-aware computational SHM study of information provenance,
> descriptor observability, and failure mechanisms in storey-level damage
> inference.

The manuscript is NOT primarily:

- a new machine-learning algorithm paper;
- an SVR paper;
- an optimal sensor placement paper;
- a noise-robust learning paper;
- a new handcrafted-feature paper.

---

# 2. Overall literature position

Three major research areas relevant to this manuscript are already established.

## Existing field A — Simulation/deployment discrepancy

Established approaches include:

- domain adaptation;
- transfer learning;
- population-based SHM;
- digital twins;
- physics-guided learning;
- multi-source simulation/experimental learning.

Therefore:

    simulation-to-deployment discrepancy

is NOT itself the novelty.

---

## Existing field B — Noise and uncertainty

Established work includes:

- noise-robust damage identification;
- uncertainty propagation;
- false-alarm analysis;
- noise sensitivity;
- incomplete/noisy measurement analysis;
- damage severity uncertainty.

Therefore:

    measurement-noise analysis

and

    damage underestimation

are NOT themselves the novelty.

---

## Existing field C — Sensor placement and sparse sensing

Established work includes:

- optimal sensor placement;
- Fisher-information criteria;
- information entropy;
- modal observability;
- modal independence;
- multi-objective sensing;
- damage-oriented sensor placement;
- incomplete measurement reconstruction.

Therefore:

    sensor number matters

and

    sensor location matters

are NOT themselves the novelty.

---

# 3. RQ1 claim matrix

## Research question

> How do deployment-accessible information and estimator flexibility jointly
> affect storey-level structural damage inference?

---

## Established literature

Representative literature:

- Gardner et al. (2020): domain adaptation in SHM;
- Bull et al. (2021): population-based SHM;
- Xu and Noh (2021): physics-informed domain transfer;
- Zhang et al. (2022): numerical/real discrepancy and modeling uncertainty;
- Teng et al. (2023): simulation/digital-twin transfer;
- Miele et al. (2025): computational and experimental information assimilation.

Established conclusion:

    source-domain information
        !=
    target/deployment-domain information.

---

## What is already known

Machine-learning models trained using simulated or source-domain data may not
transfer directly to a target structure or deployment domain.

Label scarcity and domain discrepancy are established SHM problems.

Transfer learning and domain adaptation are established approaches for reducing
these discrepancies.

---

## Remaining question addressed here

The current manuscript asks an upstream model-specification question:

> Before transfer or prediction is considered, is each descriptor input
> physically obtainable from the assumed deployment sensing system?

This is formulated through descriptor information provenance.

---

## Frozen evidence

Information hierarchy:

    92D privileged-information reference
        ->
    86D legacy simulation-informed reference
        ->
    78D signal-derived primary representation
        ->
    59D structural-response-only reference.

Primary repeated factorial:

    78D Ridge
    78D RBF-SVR
    92D Ridge
    92D RBF-SVR.

Headline 78D estimator effect:

    overall MAE reduction ≈ 29.9%

    high-damage MAE reduction ≈ 36.3%.

Repeated overall-MAE interaction:

    difference-in-differences ≈ -0.003032

with paired interval entirely on the beneficial side.

---

## Allowed novelty claim

Preferred:

> The study makes descriptor-level information provenance explicit by
> distinguishing signal-derived information from quantities dependent on
> privileged simulation-generator information.

Preferred:

> Descriptor provenance is treated as part of model specification rather than
> being left implicit in the simulation-to-deployment transition.

Preferred:

> Information availability and estimator flexibility show complementary effects
> within the repeated simulation benchmark.

---

## Claims requiring caution

Acceptable only with qualification:

> deployment-aware representation

because deployment is represented through an assumed sensing configuration,
not field validation.

Acceptable:

> privileged-information reference.

Avoid:

> theoretical upper bound.

---

## Prohibited claims

DO NOT write:

> This is the first SHM study to consider simulation-to-deployment discrepancy.

DO NOT write:

> Existing SHM studies ignore information provenance.

DO NOT write:

> 92D represents the maximum achievable damage-inference performance.

DO NOT write:

> RBF-SVR is the optimal estimator.

---

## Main manuscript locations

Introduction:
Gap 1 + RQ1.

Methods:
information hierarchy and provenance definition.

Results:
Fig. 3 / Table 2.

Discussion:
Sections 4.1–4.2.

Abstract:
29.9% / 36.3% estimator headline only.

---

# 4. RQ2 claim matrix

## Research question

> How does structural-response measurement noise alter the magnitude and
> direction of damage-inference failure, and does a calibration designed for
> in-distribution underprediction remain valid after that shift?

---

## Established literature

Representative literature:

- Neves et al. (2017): uncertainty and damage-detection errors;
- Favarelli et al. (2022): sensing parameters and anomaly detection;
- Fan et al. (2023): noise robustness;
- Modesti et al. (2025): noise, incomplete measurements, and damage
  identification;
- Dessi et al. (2025): uncertainty propagation;
- del Pozo et al. (2025): explicit damage-index uncertainty;
- Hormazábal et al. (2026): adverse acquisition conditions / low-SNR failure.

---

## What is already known

Measurement noise can:

- reduce damage-identification accuracy;
- increase false alarms;
- degrade localization;
- increase severity uncertainty;
- alter damage-sensitive measurements.

Noise robustness and uncertainty quantification are established SHM research
topics.

Damage underestimation has also been observed in previous structural damage
identification studies.

---

## Remaining question addressed here

The present manuscript does not ask only:

    How much accuracy is lost?

It asks:

> Does the direction and form of continuous severe-damage regression failure
> change as the sensing distribution shifts?

This is evaluated through:

- signed bias;
- underestimation ratio;
- descriptor-space displacement;
- prediction inflation;
- clipping/saturation;
- calibration transfer.

---

## Frozen evidence

Standard clean-trained 78D RBF-SVR:

High-damage signed bias:

    0%   = -0.021337
    5%   = -0.004789
    10%  = +0.042575
    20%  = +0.128534.

Thus:

    signed-bias reversal occurs between 5% and 10% noise.

At 20% noise:

    high-damage upper clipping ratio ≈ 0.640876.

The largest standardized descriptor shifts are concentrated in spectral and
inter-storey/amplitude-related descriptor families.

Training-only asymmetric calibration:

    beneficial under the repeated in-distribution benchmark,

but:

    directionally misaligned after the noise-induced bias reversal.

---

## Allowed novelty claim

Preferred:

> Beyond aggregate noise-sensitivity metrics, the study tracks how both the
> magnitude and direction of severe-damage regression error evolve under
> controlled measurement-noise shift.

Preferred:

> The matched-noise analysis identifies a transition from severe-damage
> underprediction toward positive bias and upper-bound saturation.

Preferred:

> The results demonstrate that a direction-specific calibration can become
> misaligned when sensing shift changes the underlying error direction.

---

## Mechanism wording

Strong but acceptable:

    noise-induced failure pathway

    descriptor shift
        ->
    prediction inflation
        ->
    bias reversal
        ->
    saturation.

Use:

> consistent with

when connecting descriptor shift to downstream prediction failure.

---

## Prohibited claims

DO NOT write:

> This is the first study of noise effects in structural damage identification.

DO NOT write:

> Severe-damage underestimation has never previously been observed.

DO NOT write:

> Spectral centroid descriptors cause model failure.

DO NOT write:

> The proposed model is robust to measurement noise.

DO NOT write:

> Calibration methods are inherently non-robust.

---

## Main manuscript locations

Introduction:
Gap 2 + RQ2.

Methods:
severity diagnostics, calibration, matched-noise protocol.

Results:
Figs. 4–5 / Table 3.

Discussion:
Sections 4.3–4.7.

Abstract:
bias transition

    -0.021 -> +0.129.

---

# 5. RQ3 claim matrix

## Research question

> How does structural sensor sparsity alter descriptor observability,
> severe-damage inference, and the marginal value of additional sensing?

---

## Established literature

Representative literature:

- Papadimitriou et al. (2000): entropy-based sensor placement;
- Papadimitriou (2004): sensor number/location and information entropy;
- Meo and Zumpano (2005): bridge sensor placement;
- Yi et al. (2011): multiple OSP strategies;
- Stephan (2012): modal identification and observability;
- Lin et al. (2019): multi-objective damage-oriented sensor placement;
- Wu et al. (2019): JCSHM sensor-placement optimization;
- Reichert et al. (2021): OSP and experimental verification;
- Bertola et al. (2023): information-gain-based measurement selection;
- Modesti et al. (2025): incomplete structural measurements.

---

## What is already known

Sensor number and sensor location affect structural information content and
identification quality.

Optimal sensor placement, modal observability, information entropy, Fisher
information, redundancy reduction, and multi-objective damage-oriented sensor
placement are established fields.

More sensing can improve information content under established sensor-design
frameworks.

---

## Remaining question addressed here

The present manuscript does NOT seek

    S* = optimal sensor layout.

Instead it asks:

> Given the sensors actually available, which physics-guided descriptors remain
> physically computable under their original definitions?

For descriptor f_j:

    observable under layout S

iff

    D_j subseteq S.

The sparse-sensing problem is therefore formulated as a dependency-aware
descriptor-availability problem.

---

## Frozen evidence

Complete structural sensor set:

    {1,2,3,4}.

All non-empty layouts:

    15.

All admissible one-sensor additions:

    28.

No:

- zero masking;
- descriptor redefinition.

Layout dimensions depend on physical descriptor dependencies.

Examples:

    layout 13 = 31D
    layout 14 = 31D,

but their downstream inference differs with paired-bootstrap support.

Two-sensor objective dependence:

    best overall MAE = layout 12

    best high-damage MAE = layout 13.

Single-sensor high-damage underestimation:

    approximately 98–100%.

Complete lattice result:

    28/28 overall-MAE improvements;

    28/28 high-damage-MAE improvements;

and every paired 95% case-bootstrap interval remains beneficial.

---

## Allowed novelty claim

Preferred:

> Sensor sparsity is formulated as a dependency-aware descriptor-observability
> problem rather than as zero-masked feature loss.

Preferred:

> The complete tested subset lattice links structural sensor availability,
> physical descriptor computability, and downstream damage-inference
> performance.

Preferred:

> Within the tested four-sensor lattice, every admissible one-sensor addition
> reduced both overall and high-damage MAE with paired case-bootstrap support.

Preferred:

> Sensor placement effects cannot be explained solely by descriptor dimension.

---

## Optional conceptual wording

Acceptable with qualification:

> monotonic empirical sensing benefit within the tested lattice.

Always retain:

- empirical;
- tested;
- current benchmark.

---

## Prohibited claims

DO NOT write:

> A new optimal sensor placement method is proposed.

DO NOT write:

> This is the first study to show that sensor location matters.

DO NOT write:

> More sensors always improve SHM.

DO NOT write:

> The optimal two-sensor configuration is 12.

Because the preferred layout depends on objective.

DO NOT write:

> Classical system observability is established.

The present concept is descriptor observability.

---

## Main manuscript locations

Introduction:
Gap 3 + RQ3.

Methods:
dependency rule and exhaustive layout protocol.

Results:
Figs. 6–7 / Table 4.

Discussion:
Sections 4.8–4.12.

Abstract:
28/28 result.

---

# 6. Integrated novelty hierarchy

The manuscript should not present four unrelated technical contributions.

Use the following hierarchy.

---

## Level 1 — Methodological formulation

### Deployment-aware information formulation

The effective input space is controlled by:

    information provenance
        +
    physical sensor dependencies.

Conceptually:

    F_available
        =
    {f_j :
        P_j belongs to deployment-accessible information
        AND
        D_j subseteq S}.

This is the organizing methodology of the paper.

---

## Level 2 — Scientific finding

### Distinct sensing degradation mechanisms

Measurement noise:

    perturbs available information
        ->
    descriptor distribution shift
        ->
    bias reversal
        ->
    saturation.

Sensor loss:

    removes observable information
        ->
    descriptor / cross-storey observability loss
        ->
    severe underprediction.

This distinction is the principal mechanism-level finding.

---

## Level 3 — Estimator finding

Estimator flexibility materially improves inference but does not remove the
information boundary or severe-damage directional failure.

Headline:

    RBF-SVR vs Ridge on 78D:
        29.9% lower overall MAE;
        36.3% lower high-damage MAE.

This is supporting evidence, not the manuscript's primary algorithmic novelty.

---

## Level 4 — Strong empirical sensing result

Within the complete tested four-sensor subset lattice:

    28/28

admissible sensor additions improve both:

    overall MAE

and

    high-damage MAE

with paired case-bootstrap support.

This is a strong benchmark-level empirical result.

It is NOT a new universal information-theory result.

---

# 7. Final contribution set for Introduction

Recommended final contributions:

## C1

> An explicit deployment-aware information formulation that distinguishes
> signal-derived descriptors from simulation-privileged information and
> preserves this provenance boundary during inference evaluation.

## C2

> A dependency-aware descriptor-observability formulation in which sparse
> sensor layouts retain only descriptors that remain physically computable
> under their original definitions.

## C3

> A directional failure-mechanism analysis showing qualitatively different
> inference pathways under measurement-noise perturbation and sensor-induced
> information removal.

## C4

> An exhaustive evaluation of all non-empty structural sensor layouts and all
> 28 admissible one-sensor additions, providing paired-bootstrap evidence of
> consistent downstream inference improvement within the tested topology.

---

# 8. One-sentence novelty statement

Preferred:

> The novelty lies not in a new estimator or sensor-placement algorithm, but in
> treating information provenance and descriptor observability as explicit
> constraints on damage inference and using them to diagnose how different
> sensing degradations produce distinct severe-damage failure mechanisms.

---

# 9. One-sentence literature gap statement

Preferred:

> Existing research has extensively addressed domain transfer, measurement-noise
> robustness, uncertainty, and optimal sensor placement; the present study
> addresses the complementary question of how the physical provenance and
> layout-dependent computability of the information supplied to an estimator
> constrain both its achievable performance and its mode of failure.

---

# 10. Abstract claim set

The Abstract should contain only three numerical headlines.

## A1 — estimator capability

    29.9% lower overall MAE;
    36.3% lower high-damage MAE.

## A2 — noise failure transition

    high-damage signed bias:
    -0.021 -> +0.129.

## A3 — sensing lattice

    28/28 additions beneficial
    for both overall and high-damage MAE.

Do not overload the Abstract with additional numbers.

---

# 11. Discussion claim set

The Discussion should emphasize:

1. information provenance is part of model specification;
2. estimator flexibility and information availability are complementary;
3. aggregate accuracy does not define severe-damage inference quality;
4. calibration validity depends on stability of error direction;
5. noise and sparsity alter information through different operations;
6. descriptor count is not equivalent to descriptor identity;
7. the 28-edge lattice is strong empirical evidence but not a universal law.

---

# 12. Cover-letter positioning

Preferred editorial positioning:

> The manuscript does not propose another damage-prediction architecture or
> sensor-placement optimizer. Instead, it examines a deployment problem that
> precedes both: what information is legitimately available to the estimator,
> what remains physically computable as sensing becomes sparse, and how the
> resulting inference fails when that information is perturbed or removed.

---

# 13. High-risk language blacklist

Codex / later manuscript editing should flag the following expressions unless
manually justified:

    first-ever
    for the first time
    unprecedented
    universally
    proves
    guarantees
    optimal sensor layout
    theoretical upper bound
    noise robust
    field-ready
    field validated
    real-world validated
    causal descriptor
    formal identifiability
    classical observability

Preferred replacements:

    within the present benchmark
    within the complete tested lattice
    paired-bootstrap support
    privileged-information reference
    robustness boundary
    diagnostic descriptor shift
    descriptor observability
    deployment-oriented
    consistent with

---

# 14. Codex expansion rules

When converting the CORE manuscript into complete prose, Codex may:

- improve grammar;
- add transitions;
- integrate citations;
- reduce repetition;
- merge paragraphs;
- adopt JCSHM terminology.

Codex MUST NOT:

- invent additional novelty claims;
- add "first" claims;
- change numerical anchors;
- change protocol identities;
- merge repeated-split and fixed-split results;
- call 92D an upper bound;
- claim noise robustness;
- call the sensor analysis a new OSP method;
- convert descriptor-shift association into causality;
- remove benchmark/generalization qualifications.

If generated prose conflicts with this matrix, this matrix takes precedence.

---

# 15. Final claim hierarchy

## Strongest methodological claim

    explicit deployment-aware information provenance
        +
    dependency-aware descriptor observability.

## Strongest mechanism claim

    noise perturbation and sensor-induced information removal
    produce qualitatively different severe-damage failure pathways.

## Strongest numerical estimator claim

    29.9% overall and 36.3% high-damage MAE reductions
    for RBF-SVR versus Ridge under the primary 78D representation.

## Strongest sensing claim

    all 28 admissible one-sensor additions improve both overall
    and high-damage MAE with paired case-bootstrap support
    within the complete tested lattice.

## Strongest overall conclusion

> Within the present controlled benchmark, damage-inference capability depends
> not only on estimator accuracy but on the provenance, availability, and
> degradation mode of the sensing information supplied to the estimator.

