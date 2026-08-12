# Literature Block 3 — Sensor Sparsity, Placement, and Descriptor Observability

## Status

TARGETED LITERATURE CORE — BLOCK 3

Purpose:

Support Introduction Gap 3 and Discussion Sections 4.8–4.12.

This block distinguishes established optimal sensor placement (OSP) research
from the present manuscript's dependency-aware descriptor-observability
analysis.

The manuscript MUST NOT claim novelty for:

- optimal sensor placement itself;
- the general importance of sensor number;
- the general importance of sensor location;
- Fisher-information-based sensing design;
- entropy-based sensing design;
- modal observability;
- multi-objective sensor placement;
- the general statement that additional sensing can increase information.

---

# 1. Established sensor-placement problem

Optimal sensor placement is an established structural dynamics and SHM problem.

The conventional problem can be written conceptually as

    S* = argmax_S J(S)

or

    S* = argmin_S J(S),

where S denotes a sensor configuration and J is an information,
identification, modal, damage-sensitivity, or uncertainty criterion.

Established objectives include:

- Fisher information;
- information entropy;
- modal independence;
- modal observability;
- modal assurance criteria;
- structural response coverage;
- parameter-estimation uncertainty;
- damage sensitivity;
- damage identification performance.

Therefore, the current manuscript is NOT an OSP-method paper.

---

# 2. Core distinction of the present manuscript

The present study asks an upstream question:

    Given sensor layout S,
    which physics-guided descriptors remain physically computable?

For descriptor f_j, let D_j denote the structural sensing channels required by
its original physical definition.

Then

    f_j observable under S

if and only if

    D_j subseteq S.

The resulting representation is

    F_S = {f_j : D_j subseteq S}.

This differs conceptually from simply selecting a reduced subset of a fixed
feature vector.

Sensor loss may make a descriptor physically undefined because one or more
signals required to calculate it no longer exist.

---

# 3. Why zero masking is not equivalent

Suppose an adjacent-storey descriptor requires responses from storeys i-1 and i.

If sensor i-1 is unavailable,

    D_j not subseteq S.

The descriptor is therefore unavailable.

Replacing the missing measurement with zero would produce a numerical value,
but that value would not represent the original physical descriptor.

Accordingly, the present analysis:

- does not zero-mask unavailable structural response channels;
- does not redefine descriptors for each sparse layout;
- retains a descriptor only when its original sensing dependencies are
  satisfied.

This rule defines dependency-aware descriptor observability.

---

# 4. Core literature

## L1 — Papadimitriou, Beck and Au (2000)

Papadimitriou, C., Beck, J. L., and Au, S. K.

Entropy-Based Optimal Sensor Location for Structural Model Updating.

Journal of Vibration and Control, 6(5), 781–800, 2000.

DOI:
10.1177/107754630000600508

### Role

Foundational information-theoretic sensor-placement reference.

### Key relevance

Sensor configuration is selected to maximize the information available for
structural parameter estimation by minimizing parameter-information entropy.

The framework explicitly accounts for measurement and model uncertainty.

### Use

Establish that information-based sensor design is not novel.

### Distinction

The present study does not optimize information entropy.

It maps physical signal dependencies to descriptor availability and evaluates
the resulting damage-inference consequences.

---

## L2 — Papadimitriou (2004)

Papadimitriou, C.

Optimal sensor placement methodology for parametric identification of structural
systems.

Journal of Sound and Vibration, 278(4–5), 923–947, 2004.

DOI:
10.1016/j.jsv.2003.10.063

### Role

Important theoretical novelty-boundary reference.

### Key relevance

The study formulates sensor placement using information entropy and examines how
information content depends on both sensor number and location.

Importantly, under that parametric-identification framework, lower and upper
bounds of information entropy decrease with increasing sensor number.

### Critical consequence

DO NOT claim:

    "This study is the first to show that more sensors provide more information."

That principle is already established in structural sensor-placement theory.

### Distinction

The current manuscript reports an empirical monotonic result for damage-
regression performance:

    all 28 admissible additions
        ->
    lower overall MAE
        AND
    lower high-damage MAE

under the current dependency-aware descriptor architecture.

---

## L3 — Meo and Zumpano (2005)

Meo, M., and Zumpano, G.

On the optimal sensor placement techniques for a bridge structure.

Engineering Structures, 27(10), 1488–1497, 2005.

DOI:
10.1016/j.engstruct.2005.03.015

### Role

Classical civil-structure OSP reference.

### Key relevance

Multiple OSP strategies are compared for a bridge, including:

- Fisher-information-based methods;
- covariance-based methods;
- energy-based approaches.

The objective is to choose sensor number and locations that retain sufficient
information for structural dynamic characterization.

### Use

Establish the traditional OSP framing:

    "How many sensors and where should they be placed?"

---

## L4 — Yi, Li and Gu (2011)

Yi, T.-H., Li, H.-N., and Gu, M.

Optimal sensor placement for structural health monitoring based on multiple
optimization strategies.

The Structural Design of Tall and Special Buildings,
20(7), 881–900, 2011.

DOI:
10.1002/tal.712

### Role

Representative optimization-based SHM sensor-placement study.

### Key relevance

The approach combines:

- QR-based initial placement;
- modal assurance criterion;
- forward/backward sequential placement;
- genetic optimization;

to determine sensor quantity and locations.

### Use

Demonstrates that optimizing sensor count and placement using modal-information
criteria is established SHM methodology.

---

## L5 — Stephan (2012)

Stephan, C.

Sensor placement for modal identification.

Mechanical Systems and Signal Processing, 27, 461–470, 2012.

DOI:
10.1016/j.ymssp.2011.07.022

### Role

Important observability terminology reference.

### Key relevance

Sensor selection explicitly considers:

- observability of mode shapes;
- information shared between sensors;
- Fisher information;
- information redundancy.

### Terminology consequence

Classical "observability" already has a formal modal/system-identification
meaning.

Therefore the present manuscript should preferentially use:

    descriptor observability

or

    practical sensing observability

rather than imply that a classical state-space observability theorem is being
derived.

---

## L6 — Lin, Xu and Zhan (2019)

Lin, J.-F., Xu, Y.-L., and Zhan, S.

Experimental investigation on multi-objective multi-type sensor optimal
placement for structural damage detection.

Structural Health Monitoring, 18(3), 882–901, 2019.

DOI:
10.1177/1475921718785182

### Role

Important multi-objective damage-oriented OSP reference.

### Key relevance

A response-covariance-based multi-objective sensor-placement strategy was
experimentally evaluated using multiple sensor types and structural damage
scenarios.

The study directly connects sensor placement with quantitative damage
identification performance.

### Critical consequence

DO NOT claim:

    "Existing sensor placement studies optimize only modal identification."

Damage-oriented and multi-objective placement already exist.

### Distinction

The present manuscript does not propose another multi-objective optimization
algorithm.

Instead, it exhaustively evaluates every layout in a small complete structural
sensor lattice and diagnoses how physically available descriptors and damage
inference change.

---

## L7 — Wu et al. (2019)

Wu, Z. Y., Zhou, K., Shenton, H. W., and Chajes, M. J.

Development of sensor placement optimization tool and application to large-span
cable-stayed bridge.

Journal of Civil Structural Health Monitoring, 9(1), 77–90, 2019.

DOI:
10.1007/s13349-018-0320-5

### Role

High-priority target-journal OSP reference.

### Key relevance

The study develops a multi-type sensor-placement optimization framework and
software tool and applies it to a large cable-stayed bridge.

### Use

Demonstrates that JCSHM already contains direct sensor-placement optimization
research.

### Distinction

The current paper must not be positioned as a new OSP algorithm.

---

## L8 — Reichert, Olney and Lahmer (2021)

Reichert, I., Olney, P., and Lahmer, T.

Combined approach for optimal sensor placement and experimental verification in
the context of tower-like structures.

Journal of Civil Structural Health Monitoring, 11, 223–234, 2021.

DOI:
10.1007/s13349-020-00448-7

### Role

Target-journal experiment-design / OSP reference.

### Key relevance

The study combines optimal experimental design, Fisher-information concepts,
modal analysis, and experimental validation for sensor placement.

### Use

Further confirms that optimal sensing design is established within JCSHM.

---

## L9 — Bertola et al. (2023)

Bertola, N., Wang, Z. Z., Cao, W.-J., et al.

Methodology for selecting measurement points that optimize information gain for
model updating.

Journal of Civil Structural Health Monitoring, 13, 1351–1367, 2023.

DOI:
10.1007/s13349-023-00711-7

### Role

High-priority information-gain reference in the target journal.

### Key relevance

The study selects measurement points according to their information content,
including shared information / entropy considerations, for structural model
updating.

### Use

Supports the statement that information-based measurement-system design is
well established.

### Distinction

The present work does not maximize an abstract information-gain criterion.

It determines which damage-sensitive descriptors are actually computable from
each sensing configuration and then measures the inference consequences.

---

## L10 — Modesti et al. (2025)

Modesti, M., Gentilini, C., Palermo, A., Reynders, E., and Lombaert, G.

A two-step procedure for damage detection in beam structures with incomplete
mode shapes.

Journal of Civil Structural Health Monitoring, 15, 287–306, 2025.

DOI:
10.1007/s13349-024-00839-0

### Role

Incomplete-measurement / target-journal context.

### Key relevance

Damage identification is explicitly performed under incomplete modal
information.

### Use

Supports the practical relevance of incomplete structural measurements.

### Distinction

The current paper treats sensor loss through descriptor-specific physical
dependencies rather than reconstruction of incomplete mode shapes.

---

# 5. What the literature already establishes

The literature clearly establishes that:

A. sensor number influences structural information content;

B. sensor location influences identification capability;

C. sensor redundancy should be considered;

D. modal/system observability is a major sensor-design criterion;

E. information entropy and Fisher information can guide sensor placement;

F. damage-sensitive and multi-objective sensor-placement criteria already exist;

G. incomplete measurements are an established SHM challenge.

These points are background rather than novelty.

---

# 6. What the present study adds

The present paper contributes a different sensing question:

    Which descriptors remain physically defined under each sensor layout?

This introduces the chain:

    structural sensor layout S
        ->
    physical dependency satisfaction D_j subseteq S
        ->
    observable descriptor set F_S
        ->
    layout-specific damage inference
        ->
    severity-dependent failure.

The distinction is important because two layouts containing the same number of
sensors need not provide:

- the same descriptor dimension;
- the same descriptor identity;
- the same cross-storey relationships;
- the same damage-inference performance.

---

# 7. Descriptor dimension versus descriptor identity

The exhaustive results contain layouts with equal observable dimensionality but
different sensing geometry.

The strongest current example is:

    layout 13
    versus
    layout 14

both with

    31 descriptors.

Yet their damage-inference performance differs with paired-bootstrap support.

This provides evidence that

    information quantity

as represented by feature count is insufficient.

The physical identity and dependency structure of the retained descriptors also
matter.

Other equal-dimensional comparisons are weaker and should not be generalized.

---

# 8. Objective dependence

At the two-sensor budget:

    best overall-MAE layout = 12

whereas

    best high-damage-MAE layout = 13.

Therefore the preferred sensing arrangement depends on the prediction objective.

This result should be interpreted as:

    objective-dependent sensing value

rather than

    universal optimal placement.

Multi-objective sensor placement already exists in the literature.

The contribution here is that objective dependence emerges from the complete
dependency-aware damage-inference lattice.

---

# 9. Complete subset-lattice contribution

With four structural sensors, every non-empty sensing subset can be evaluated:

    2^4 - 1 = 15 layouts.

The complete subset lattice contains:

    28 admissible one-sensor additions.

For every edge

    S -> S union {j},

the current experiment measures the paired change in:

- overall MAE;
- high-damage MAE.

Observed result:

    28 / 28 overall-MAE improvements;

    28 / 28 high-damage-MAE improvements;

with every paired 95% case-bootstrap interval remaining beneficial.

---

# 10. How the 28/28 result should be positioned

DO NOT write:

    "We prove that adding sensors always improves SHM."

DO NOT write:

    "This establishes monotonic sensor value in general."

DO NOT write:

    "More sensors are always optimal."

Preferred formulation:

> Within the complete tested four-sensor subset lattice, the empirical damage-
> inference error decreased for every admissible one-sensor addition, with paired
> case-bootstrap support for both overall and high-damage MAE.

Optional conceptual phrase:

    monotonic empirical sensing benefit within the tested lattice

but always retain:

    empirical
    tested
    current benchmark.

---

# 11. Important theoretical novelty boundary

Information-theoretic sensor-placement theory already establishes that increased
sensor number can improve parameter information under specific assumptions.

Therefore the novelty of the 28/28 result is NOT:

    "more sensors contain more information."

The meaningful result is instead:

> Every physically admissible sensor addition improves downstream continuous
> damage-regression performance under the current dependency-aware descriptor
> representation, including the severe-damage regime.

This is an inference-level empirical result rather than a new information-theory
theorem.

---

# 12. Recommended Introduction Gap 3

Preferred core:

> Sensor placement is an established SHM problem, with information-theoretic,
> modal, damage-sensitive, and multi-objective methods developed to determine
> informative sensor numbers and locations. For physics-guided feature-based
> inference, however, sensor removal can have an additional consequence: some
> descriptors cease to be physically computable because their original
> definitions require measurements from multiple structural locations.
> Accordingly, the present study treats sparse sensing as a dependency-aware
> descriptor-observability problem and evaluates its downstream effect on
> storey-level damage inference.

This avoids claiming that sensor placement itself is the gap.

---

# 13. Recommended Discussion distinction

Traditional sensor-design question:

    Which sensor configuration maximizes a chosen information or
    identification objective?

Present manuscript question:

    Given a specific sensor configuration, what physics-guided information
    remains physically observable, and how does its removal alter the
    direction and magnitude of damage-inference error?

These questions are complementary.

---

# 14. Terminology lock

Preferred:

- descriptor observability;
- dependency-aware descriptor availability;
- sensing-accessible representation;
- sensor-layout observability;
- marginal sensing value;
- complete tested subset lattice.

Use with caution:

- observability.

Avoid unless formally qualified:

- system observability;
- structural observability;
- identifiability.

Do NOT imply derivation of a classical observability rank condition.

---

# 15. Priority literature for final Introduction

Essential:

Papadimitriou (2004)
Meo and Zumpano (2005)
Stephan (2012)
Lin et al. (2019)
Wu et al. (2019)
Bertola et al. (2023)

Additional:

Papadimitriou et al. (2000)
Yi et al. (2011)
Reichert et al. (2021)
Modesti et al. (2025)

---

# 16. Literature Block 3 conclusion

Established OSP literature asks:

    "How many sensors should be installed, and where?"

Current manuscript adds:

    "Given the sensors actually available, which physics-guided descriptors
    remain legitimate and computable?"

Established sensor design:

    layout
        ->
    optimize information criterion.

Current dependency-aware analysis:

    layout
        ->
    physical descriptor availability
        ->
    damage-inference performance
        ->
    failure direction.

The paper should therefore be positioned as:

    dependency-aware sensing observability analysis

rather than:

    a new optimal sensor placement method.

