# Structural Model and Dataset-Generation Method Lock

## Status

**LOCKED — 2026-08-12**

This document freezes the structural model, damage implementation, dynamic-analysis protocol, excitation model, and original 3000-case dataset-generation settings used by the reconstructed JCSHM manuscript.

No further experimental modification should be made on the basis of manuscript drafting.

---

## 1. Structural model

The numerical benchmark is a four-storey, two-bay, two-dimensional linear-elastic frame implemented in OpenSeesPy.

### Geometry

- Number of storeys: 4
- Number of bays: 2
- Storey height: 3.2 m
- Bay width: 6.0 m

### Material and section parameters

- Young's modulus: E = 2.0 × 10^11 Pa
- Column area: A_c = 0.09 m²
- Column second moment of area: I_c = 6.75 × 10^-4 m⁴
- Beam area: A_b = 0.075 m²
- Beam second moment of area: I_b = 4.80 × 10^-4 m⁴
- Total lumped floor mass: 2.0 × 10^5 kg per floor

A negligible vertical and rotational mass is assigned only for numerical stability.

### Boundary conditions and element formulation

Each node contains three degrees of freedom:

1. horizontal translation u_x,
2. vertical translation u_y,
3. in-plane rotation r_z.

All base nodes are fully fixed.

Both beams and columns are represented using OpenSeesPy elasticBeamColumn elements with a linear geometric transformation.

---

## 2. Damage representation

Structural damage is represented as storey-level column stiffness degradation.

For storey i,

    E_i = (1 - d_i) E

or equivalently,

    (EI)_{c,i} = (1 - d_i) EI_c,

where d_i denotes the prescribed stiffness-loss ratio of storey i.

The degraded Young's modulus is applied to all columns belonging to the damaged storey.

Beam stiffness remains unchanged.

Therefore, the regression target is the four-component vector

    d = [d_1, d_2, d_3, d_4].

Damage is a controlled numerical stiffness-loss parameter and should not be described as experimentally observed cracking or material deterioration.

---

## 3. Original damage sampling

The frozen 3000-case dataset contains:

- 596 healthy cases,
- 1208 one-storey-damage cases,
- 1196 two-storey-damage cases.

Observed healthy-case proportion:

    596 / 3000 = 0.19867

The generator uses:

    healthy_probability = 0.20
    max_damaged_stories = 2
    damage_range = (0.05, 0.35)

For non-healthy cases, one or two distinct storeys are selected randomly without replacement.

The stiffness-loss ratio of each selected storey is sampled independently from

    d_i ~ Uniform(0.05, 0.35).

Observed nonzero damage values in the frozen dataset range from approximately

    0.0500035 to 0.3497993.

---

## 4. Dynamic analysis

Transient time-history analysis is performed in OpenSeesPy.

### Rayleigh damping

The damping matrix is

    C = alpha_M M + beta_Kinit K_0,

with a target damping ratio of

    xi = 0.02.

The first two modal frequencies are used to determine the Rayleigh coefficients:

    alpha_M = 2 xi omega_1 omega_2 / (omega_1 + omega_2)

and

    beta_Kinit = 2 xi / (omega_1 + omega_2).

### Numerical solution

- Constraint handler: Transformation
- DOF numbering: RCM
- Linear system: BandGeneral
- Convergence test: NormDispIncr
- Tolerance: 1 × 10^-8
- Maximum iterations: 20
- Solution algorithm: Newton
- Time integration: Newmark average-acceleration
- gamma = 0.5
- beta = 0.25
- Analysis type: Transient
- Time step: 0.01 s
- Nominal duration: 20 s
- Nominal samples per response history: 2000

---

## 5. Excitation model

The original 3000-case benchmark uses parameterized randomized sinusoidal base excitation rather than recorded earthquake motions.

For each case,

    a_g(t) = A sin(2 pi f t + phi),

where

    A ~ Uniform(0.05, 0.25) g,
    f ~ Uniform(0.5, 2.0) Hz,
    phi ~ Uniform(0, 2 pi).

Observed values in the frozen 3000-case dataset are:

Amplitude:
    min = 0.0501716 g
    max = 0.2499730 g
    mean = 0.1509346 g

Frequency:
    min = 0.5000794 Hz
    max = 1.9987604 Hz
    mean = 1.2476686 Hz

The excitation is applied in the horizontal direction using OpenSees UniformExcitation.

This excitation model must be described as a controlled parametric excitation ensemble, not as recorded earthquake ground motions.

---

## 6. Structural response extraction

The middle-column-line node of each floor is used as the representative floor response location.

The transient solver records:

- relative floor displacement,
- relative floor acceleration,
- approximate absolute floor acceleration,
- inter-storey drift ratio,
- base-input acceleration.

Under uniform support excitation,

    a_abs,i(t) ≈ a_rel,i(t) + a_g(t).

The primary response histories subsequently used for descriptor construction are the floor absolute accelerations, together with the base-input history when the descriptor definition permits it.

---

## 7. Original heterogeneous measurement-noise protocol

The original 3000-case dataset contains four noise levels:

    eta ∈ {0, 0.05, 0.10, 0.20}.

Observed case counts:

    eta = 0.00 : 744
    eta = 0.05 : 750
    eta = 0.10 : 788
    eta = 0.20 : 718

For a structural-response array s,

    sigma_n = eta * std(s),

and independent zero-mean Gaussian noise is added as

    s_noisy = s + epsilon,
    epsilon ~ N(0, sigma_n²).

The standard deviation std(s) is computed as a scalar over the complete response array for the case.

This heterogeneous original benchmark must remain conceptually separate from the later matched-noise stress test used in the JCSHM reconstruction.

---

## 8. Information-provenance warning

The generator stores simulation-side metadata including excitation amplitude, exact excitation frequency, phase, and noise level.

These values are available to the simulator but are not necessarily available to a deployment-stage SHM estimator.

Therefore:

- the historical 92D representation is treated as a privileged-information reference;
- the 78D signal-derived representation is the primary deployment-aware representation;
- exact generator parameters must not be described as measured sensor information unless they are explicitly reconstructed from measured signals.

---

## 9. Random-seed reporting

The current generator source defines a default random seed of

    20260625.

However, the exact command used to generate the frozen 3000-case dataset has not been independently recovered in the present audit.

Therefore, the manuscript should not state 20260625 as the confirmed 3000-case generation seed unless the original run metadata is later recovered.

The frozen dataset and its cryptographic manifest remain the reproducibility reference.

---

## 10. Locked methodological interpretation

The benchmark should be described as:

> a controlled numerical SHM benchmark based on a four-storey linear-elastic frame subjected to randomized parametric sinusoidal base excitation, with storey-level column stiffness degradation as the damage variable.

It should NOT be described as:

- experimental damage data,
- field-monitoring data,
- recorded-earthquake validation,
- nonlinear structural collapse analysis,
- calibrated real-building response,
- or evidence of cross-structure generalization.

