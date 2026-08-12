# 3. Results — Scientific Core

> STATUS: SCIENTIFIC CORE ONLY
>
> This document freezes the numerical evidence, result interpretation,
> claim strength, and reporting boundaries of the JCSHM manuscript.
>
> Later language editing may improve:
>
> - transitions;
> - paragraph structure;
> - grammar;
> - concision;
> - journal style.
>
> Later editing MUST NOT alter:
>
> - numerical values;
> - comparison protocols;
> - paired/repeated-split interpretation;
> - fixed-split versus repeated-split distinction;
> - sign conventions;
> - confidence-interval interpretation;
> - deployment/generalization boundaries;
> - claim strength.

---

# 3.1 Information availability and estimator flexibility jointly control inference capability

## Core evidence: repeated 10-split factorial comparison

The principal repeated-split comparison contains four conditions:

1. 78D + Ridge;
2. 78D + RBF-SVR;
3. 92D + Ridge;
4. 92D + RBF-SVR.

Mean performance across ten repeated train/validation/test splits:

| Representation | Estimator | Overall MAE | High-damage MAE | High-damage \(|bias|\) | High-damage underestimation |
|---|---|---:|---:|---:|---:|
| 78D | Ridge | 0.0448583 | 0.0947165 | 0.0828038 | 0.874290 |
| 78D | RBF-SVR | 0.0314500 | 0.0603374 | 0.0474324 | 0.761061 |
| 92D | Ridge | 0.0395455 | 0.0840811 | 0.0683546 | 0.817268 |
| 92D | RBF-SVR | 0.0231048 | 0.0443346 | 0.0270172 | 0.665722 |

Reference: Fig. 3 and Table 2.

---

## Estimator effect under the primary 78D representation

Relative to Ridge, RBF-SVR reduces mean overall MAE from

\[
0.0448583
\rightarrow
0.0314500,
\]

corresponding to an average reduction of

\[
29.90\%.
\]

The improvement occurs in

\[
10/10
\]

repeated splits.

High-damage MAE decreases from

\[
0.0947165
\rightarrow
0.0603374,
\]

corresponding to

\[
36.30\%
\]

mean improvement.

High-damage absolute bias decreases from

\[
0.0828038
\rightarrow
0.0474324,
\]

corresponding to

\[
42.72\%
\]

mean improvement.

High-damage underestimation decreases from

\[
0.874290
\rightarrow
0.761061.
\]

### Core claim

The deployable 78D descriptor representation contains a materially nonlinear relationship with storey-level damage that is not captured adequately by the linear Ridge reference.

### Safe wording

“Estimator flexibility materially improved both overall and severe-damage inference under the primary signal-derived representation.”

### Do NOT claim

- RBF-SVR is universally superior to linear models.
- Nonlinear estimation eliminates severe-damage failure.
- The result demonstrates cross-structure generalization.

---

## Information effect

Under Ridge:

\[
78D
\rightarrow
92D
\]

reduces overall MAE by

\[
0.0053128
\]

or approximately

\[
11.84\%.
\]

This direction occurs in

\[
10/10
\]

repeated splits.

Under RBF-SVR:

\[
78D
\rightarrow
92D
\]

reduces overall MAE by

\[
0.0083452
\]

or approximately

\[
26.48\%.
\]

Again, the direction occurs in

\[
10/10
\]

splits.

### Core claim

Inference performance depends materially on information availability, but the 92D result represents a simulation-privileged reference rather than a deployable benchmark.

### Safe wording

“Access to privileged simulation information improved inference under both estimators, with a larger information effect under RBF-SVR.”

### Do NOT claim

- 92D is a physical upper bound.
- 92D represents achievable field performance.
- removal of privileged information makes the 78D representation inadequate.

---

## Information × estimator interaction

For overall MAE, the repeated-split difference-in-differences interaction is

\[
\Delta_{\mathrm{int}}
=
-0.0030324,
\]

with paired 95% interval

\[
[-0.0038546,-0.0022101].
\]

The interaction has the same direction in

\[
10/10
\]

splits.

For high-damage MAE:

\[
\Delta_{\mathrm{int}}
=
-0.0053673,
\]

with paired 95% interval

\[
[-0.008574,-0.002160],
\]

and the direction is repeated in

\[
9/10
\]

splits.

For high-damage absolute bias, the interaction direction occurs in

\[
8/10
\]

splits and is therefore weaker than the overall- and high-MAE evidence.

### Core claim

Information availability and estimator flexibility show repeated-split-supported complementarity.

### Preferred wording

“The performance gain associated with additional information was larger under the nonlinear estimator, indicating complementarity between information availability and estimator flexibility within the present repeated-holdout simulation protocol.”

### Boundary

The strongest interaction evidence concerns:

1. overall MAE;
2. high-damage MAE.

Do not present all interaction metrics as equally strong.

---

# 3.2 Improved overall inference does not eliminate severity-dependent directional failure

## Severity-dependent error

Under the primary 78D representation, both estimators exhibit increasing error with damage severity.

For RBF-SVR, mean high-damage MAE remains

\[
0.0603374
\]

despite the substantial improvement relative to Ridge.

For Ridge:

\[
Bias_{\mathrm{high}}
=
-0.0828038.
\]

For RBF-SVR:

\[
Bias_{\mathrm{high}}
=
-0.0474324.
\]

Thus RBF-SVR reduces the magnitude of severe-damage error but retains systematic negative bias.

Reference: Fig. 4(a,b).

---

## Directional nature of severe-damage failure

The repeated-split severity analysis shows that prediction bias becomes increasingly negative as damage severity increases.

The high-damage underestimation ratio for the 78D RBF-SVR remains

\[
0.761061,
\]

meaning that severe-damage predictions are still predominantly below the true damage values.

### Core claim

Severe-damage failure is not merely an increase in error magnitude.

It is directionally asymmetric:

\[
\hat y < y
\]

occurs systematically more often in the severe-damage regime.

### Preferred wording

“Nonlinear inference reduced severe-damage error but did not eliminate the persistent directional tendency toward underprediction.”

### Do NOT claim

- all predictions underestimate severe damage;
- the bias is a universal property of RBF-SVR;
- the mechanism has been validated on physical structures.

---

# 3.3 Training-only asymmetric calibration reduces the in-distribution severe-damage bias

## Frozen repeated-split calibration result

The one-sided residual calibration is evaluated under the same ten repeated-split protocol.

High-damage MAE:

\[
0.0603374
\rightarrow
0.0556958,
\]

corresponding to

\[
7.63\%
\]

mean improvement.

This improvement occurs in

\[
10/10
\]

splits.

High-damage absolute bias:

\[
0.0474324
\rightarrow
0.0339341,
\]

corresponding to

\[
29.27\%
\]

mean improvement.

Again:

\[
10/10
\]

splits improve.

High-damage underestimation ratio:

\[
0.761061
\rightarrow
0.657355,
\]

an absolute reduction of approximately

\[
10.37
\]

percentage points.

This direction also occurs in

\[
10/10
\]

splits.

Reference: Fig. 4(c,d).

---

## Calibration trade-off

The calibration is not uniformly beneficial across severity levels.

Mean relative changes in severity-specific MAE are approximately:

\[
\text{zero}: -1.02\%,
\]

\[
\text{low}: -7.71\%,
\]

\[
\text{medium}: +0.59\%,
\]

\[
\text{high}: +7.63\%,
\]

where positive values denote improvement after calibration.

The low-damage deterioration has paired confidence support.

The medium-damage difference is small and its paired interval crosses zero.

### Core claim

The asymmetric correction redistributes error across damage severity rather than improving all regimes simultaneously.

### Preferred wording

“The training-only directional correction reduced severe-damage underprediction at the cost of modest deterioration in the low-damage regime.”

### Boundary

Calibration is a targeted reliability-oriented extension, not a universal accuracy improvement.

---

# 3.4 Controlled measurement noise induces a nonlinear robustness boundary

## Controlled clean-trained protocol

The matched-noise experiment uses the clean-trained 78D RBF-SVR.

The estimator and training scaler are not retrained under noisy conditions.

Standard RBF-SVR overall MAE:

\[
0\%:
0.0220471,
\]

\[
5\%:
0.0443594,
\]

\[
10\%:
0.1118056,
\]

\[
20\%:
0.2838931.
\]

Relative to the clean condition, mean overall MAE increases by approximately:

\[
101.2\%,
\]

\[
407.1\%,
\]

and

\[
1187.7\%
\]

at 5%, 10%, and 20% noise, respectively.

High-damage MAE:

\[
0.0367800
\rightarrow
0.0582184
\rightarrow
0.1140227
\rightarrow
0.2069806.
\]

Reference: Fig. 5(a), Table 3.

### Core claim

The clean-trained descriptor-based estimator exhibits a pronounced nonlinear measurement-noise robustness boundary.

### Preferred wording

“Prediction error increased strongly and nonlinearly as response-measurement noise increased.”

### Do NOT claim

- robustness to high measurement noise;
- graceful degradation;
- noise invariance.

---

# 3.5 Measurement noise changes the direction of the severe-damage failure mode

## Standard RBF-SVR signed bias

High-damage signed bias evolves as

\[
-0.0213373
\rightarrow
-0.0047895
\rightarrow
+0.0425749
\rightarrow
+0.1285342
\]

for

\[
0\%,5\%,10\%,20\%
\]

noise.

The standard estimator therefore changes sign between

\[
5\%
\quad\text{and}\quad
10\%
\]

noise.

---

## Calibrated RBF-SVR signed bias

The calibrated model evolves as

\[
-0.0028640
\rightarrow
+0.0128238
\rightarrow
+0.0570517
\rightarrow
+0.1323379.
\]

Its sign reversal occurs between

\[
0\%
\quad\text{and}\quad
5\%.
\]

Reference: Fig. 5(b).

---

## Core interpretation

The failure mode observed under clean/in-distribution conditions is therefore not directionally stable under measurement-noise shift.

The sequence changes from:

\[
\text{underprediction}
\]

toward:

\[
\text{approximately balanced error}
\]

and then:

\[
\text{overprediction}.
\]

### Core claim

Measurement noise changes not only error magnitude but the direction of the severe-damage prediction bias.

### Strong but safe wording

“Controlled response-measurement noise produced a clear reversal of the high-damage signed bias from underprediction toward overprediction.”

---

## Consequence for asymmetric calibration

Under the clean benchmark, the one-sided upward correction reduces negative severe-damage bias.

At every non-zero matched-noise level, however, calibration increases both overall and high-damage MAE relative to the uncalibrated standard estimator.

Overall MAE:

\[
5\%:
0.044359
\;(\mathrm{standard})
<
0.046729
\;(\mathrm{calibrated}),
\]

\[
10\%:
0.111806
<
0.119924,
\]

\[
20\%:
0.283893
<
0.292050.
\]

### Core interpretation

Once measurement noise drives the underlying bias toward positive values, the upward-only correction becomes directionally misaligned.

### Preferred wording

“The in-distribution benefit of one-sided calibration did not transfer under noise-induced distribution shift; after the bias reversed, the same upward correction became directionally misaligned.”

### Do NOT claim

- isotonic calibration is generally non-robust;
- calibration causes the original noise failure;
- this result applies to arbitrary noise distributions.

---

# 3.6 Noise-induced prediction inflation progresses into upper-bound saturation

## Upper clipping

For the standard RBF-SVR, overall upper-bound clipping ratio increases approximately as:

\[
0
\rightarrow
0.0036
\rightarrow
0.0319
\rightarrow
0.3226.
\]

For high-damage predictions, the 20% noise condition reaches:

\[
R_{\mathrm{clip,high}}
=
0.640876.
\]

At 20% noise, the median high-damage prediction is

\[
0.5,
\]

which equals the imposed upper prediction bound.

Reference: Fig. 5(c).

### Core claim

At high noise, the estimator enters a saturation regime rather than exhibiting only larger unsaturated prediction variance.

### Preferred wording

“At 20% response-measurement noise, prediction inflation progressed into substantial upper-bound saturation, particularly for high-damage cases.”

---

# 3.7 Descriptor-space shift accompanies the noise-induced failure transition

## Standardized descriptor displacement

The matched-noise analysis evaluates descriptor displacement relative to the clean training standardization.

The descriptor exhibiting the largest mean absolute standardized displacement at 20% noise is:

\[
\texttt{story\_1\_spectral\_centroid}.
\]

For this descriptor:

\[
E(|\Delta z|)
=
1.46105.
\]

The fraction of descriptor values satisfying

\[
|\Delta z|>1
\]

is approximately

\[
0.5058,
\]

and the fraction satisfying

\[
|\Delta z|>2
\]

is approximately

\[
0.2933.
\]

Other strongly shifted descriptors are concentrated among:

- spectral-centroid descriptors;
- relative-to-lower maximum-amplitude descriptors.

Reference: Fig. 5(d).

### Core claim

The largest standardized distribution shifts are concentrated in specific spectral and inter-storey response descriptor families.

### Safe wording

“The diagnostic evidence is consistent with a failure pathway in which response noise shifts the descriptor representation before the observed bias reversal and output saturation.”

### Critical boundary

No feature-ablation or causal intervention establishes that the most shifted descriptors individually cause the prediction collapse.

Do NOT write:

“spectral centroid causes the noise failure.”

---

# 3.8 Sensor sparsity causes substantial observability loss and severe-damage underprediction

## Exhaustive layout evaluation

All

\[
15
\]

non-empty structural sensor layouts are evaluated.

Mean overall MAE across layouts grouped by sensor count:

\[
1\text{ sensor}:
0.0733744,
\]

\[
2\text{ sensors}:
0.0581866,
\]

\[
3\text{ sensors}:
0.0400413,
\]

\[
4\text{ sensors}:
0.0220471.
\]

Corresponding mean high-damage MAE:

\[
0.2187051,
\]

\[
0.1455293,
\]

\[
0.0840897,
\]

\[
0.0367800.
\]

Reference: Fig. 6 and Table 4.

---

## Single-sensor severe-damage failure

High-damage underestimation ratios for the four single-sensor layouts are:

\[
S=\{1\}:
0.981752,
\]

\[
S=\{2\}:
1.000000,
\]

\[
S=\{3\}:
0.992701,
\]

\[
S=\{4\}:
0.996350.
\]

Thus extreme structural sensing sparsity is associated with nearly universal severe-damage underprediction in the present topology.

### Core claim

Sensor loss produces a failure mode qualitatively different from measurement-noise degradation.

Under sparse sensing, severe-damage bias remains strongly negative rather than reversing toward positive values.

### Preferred wording

“Severe sensor sparsity primarily reduced structural observability and intensified high-damage underprediction.”

---

# 3.9 Sensor placement matters in addition to sensor count

## Within-budget placement variation

Best overall-MAE layouts by sensor count:

\[
1:
\{1\},
\]

\[
2:
\{1,2\},
\]

\[
3:
\{1,2,3\},
\]

\[
4:
\{1,2,3,4\}.
\]

However, the best 2-sensor layout for high-damage MAE is

\[
\{1,3\},
\]

rather than

\[
\{1,2\}.
\]

Therefore the preferred layout depends on the monitoring objective.

---

## Observable dimension does not fully determine performance

Several layouts provide equal descriptor dimensionality but different prediction performance.

Examples:

\[
\{1,3\}
\quad\text{and}\quad
\{1,4\}
\]

both provide

\[
31D,
\]

yet have different overall MAE.

The paired comparison between these two layouts provides stable evidence that placement effects cannot be reduced solely to feature count.

Other equal-dimensional comparisons show weaker or uncertain differences.

### Core claim

Descriptor availability broadly tracks sensing coverage, but descriptor dimensionality alone does not fully explain placement-dependent inference performance.

### Boundary

Do NOT infer a universal lower-storey sensor-placement rule.

All placement conclusions are conditional on:

- the current four-storey topology;
- the current descriptor dependencies;
- the present simulated loading/damage population.

---

# 3.10 Every admissible one-sensor addition improves both overall and severe-damage MAE in the tested lattice

## Sensor-count step effects

Paired sensor-count improvements in overall MAE are:

### 1 → 2 sensors

\[
\Delta MAE
=
0.0151878,
\]

relative improvement:

\[
20.70\%,
\]

paired 95% interval:

\[
[0.0137090,0.0167358].
\]

### 2 → 3 sensors

\[
\Delta MAE
=
0.0181454,
\]

relative improvement:

\[
31.18\%,
\]

paired 95% interval:

\[
[0.0168152,0.0194642].
\]

### 3 → 4 sensors

\[
\Delta MAE
=
0.0179941,
\]

relative improvement:

\[
44.94\%,
\]

paired 95% interval:

\[
[0.0165347,0.0194298].
\]

All bootstrap positive fractions are

\[
1.0.
\]

---

## High-damage sensor-count effects

High-damage MAE relative improvements are:

\[
1\rightarrow2:
33.46\%,
\]

\[
2\rightarrow3:
42.22\%,
\]

\[
3\rightarrow4:
56.26\%.
\]

All paired 95% intervals remain entirely beneficial.

Reference: Fig. 7(a,b).

---

## Complete 28-edge subset lattice

The complete 15-layout non-empty subset lattice contains

\[
28
\]

admissible one-sensor additions:

\[
S
\rightarrow
S\cup\{j\}.
\]

For overall MAE:

\[
28/28
\]

edges have positive point improvement, and

\[
28/28
\]

paired 95% bootstrap intervals remain entirely above zero.

For high-damage MAE:

\[
28/28
\]

edges have positive point improvement, and

\[
28/28
\]

paired 95% intervals also remain entirely above zero.

All bootstrap positive fractions equal

\[
1.0.
\]

Reference: Fig. 7(c,d).

### Core claim

Within the complete tested four-sensor subset lattice, every admissible one-sensor addition improves both overall and high-damage MAE with paired case-bootstrap support.

### Preferred headline wording

“Across all 28 admissible one-sensor additions in the complete tested subset lattice, both overall and high-damage MAE decreased, with every paired 95% case-bootstrap interval remaining entirely on the beneficial side of zero.”

### Critical boundary

Do NOT write:

“Adding any sensor always improves structural health monitoring.”

The result is conditional on:

- the current structural system;
- the current descriptor architecture;
- the current simulated test population.

No family-wise multiplicity adjustment is applied.

Use:

“paired bootstrap support”

rather than:

“universal statistical proof”.

---

# 3.11 Relative sensor gains and absolute sensor gains must not be conflated

For overall MAE, absolute sensor-count improvements are:

\[
0.01519,
\quad
0.01815,
\quad
0.01799,
\]

for:

\[
1\rightarrow2,
\quad
2\rightarrow3,
\quad
3\rightarrow4.
\]

Thus absolute improvement does NOT increase monotonically.

However, relative improvements increase:

\[
20.7\%
\rightarrow
31.2\%
\rightarrow
44.9\%.
\]

For high-damage MAE, absolute improvements decrease:

\[
0.07318
\rightarrow
0.06144
\rightarrow
0.04731,
\]

while relative improvements increase:

\[
33.5\%
\rightarrow
42.2\%
\rightarrow
56.3\%.
\]

### Reporting rule

Do NOT state:

“the marginal absolute value of each additional sensor increases with sensor count.”

Preferred wording:

“Relative gains increased at higher sensing levels, whereas absolute marginal reductions remained metric- and context-dependent.”

---

# 3.12 Integrated result: sensing degradation follows two distinct failure pathways

## Measurement noise

Observed sequence:

\[
\text{response noise}
\rightarrow
\text{descriptor-space shift}
\rightarrow
\text{prediction inflation}
\rightarrow
\text{bias reversal}
\rightarrow
\text{saturation}.
\]

Dominant directional outcome at high noise:

\[
\text{overprediction / upper-bound saturation}.
\]

---

## Sensor sparsity

Observed sequence:

\[
\text{sensor removal}
\rightarrow
\text{descriptor unavailability}
\rightarrow
\text{reduced cross-storey observability}
\rightarrow
\text{severe underprediction}.
\]

Dominant directional outcome at extreme sparsity:

\[
\text{persistent negative bias / underestimation}.
\]

---

## Central scientific result

Measurement noise and sensor loss should not be collapsed into a generic statement that:

“less sensing quality causes larger error.”

They modify the inverse problem through different mechanisms.

### Core manuscript claim

\[
\boxed{
\text{Noise perturbs the available information;}
\quad
\text{sensor loss removes information.}
}
\]

Consequently:

- measurement noise can change the direction of inference error;
- sensor sparsity primarily reduces observability and intensifies underprediction.

### Preferred wording

“The two sensing-degradation modes produced qualitatively different failure mechanisms: measurement noise altered the descriptor distribution and eventually reversed the direction of severe-damage bias, whereas sensor loss reduced descriptor observability and intensified severe-damage underprediction.”

This is a central Discussion/Conclusion claim.

---

# Results reporting boundaries

The Results section supports the following strong claims:

1. The primary 78D signal-derived representation supports materially better nonlinear than linear inference.
2. Severe-damage underprediction persists despite improved overall accuracy.
3. Training-only asymmetric correction reduces this in-distribution directional failure but introduces a severity-dependent trade-off.
4. Controlled measurement noise causes strong nonlinear degradation, descriptor shift, high-damage bias reversal, and eventual prediction saturation.
5. Sparse sensing reduces descriptor observability and strongly intensifies severe-damage underprediction.
6. All 28 admissible one-sensor additions in the complete tested subset lattice improve both overall and high-damage MAE with paired bootstrap support.

The Results section does NOT establish:

- cross-structure generalization;
- physical laboratory validation;
- field deployment performance;
- universal sensor-placement laws;
- universal noise robustness;
- causal importance of individual descriptors;
- independent replication across physical systems.

---

# Protocol-separation rule

Do NOT numerically compare values across different experimental protocols as though they came from one common estimator evaluation.

In particular:

- repeated-split 78D RBF-SVR mean MAE:

\[
0.0314500
\]

belongs to the repeated 10-split factorial experiment;

- clean matched-noise fixed-split MAE:

\[
0.0220471
\]

belongs to the controlled clean-trained matched-noise/sensor-layout benchmark.

These values answer different questions and must remain clearly separated in the final text.
