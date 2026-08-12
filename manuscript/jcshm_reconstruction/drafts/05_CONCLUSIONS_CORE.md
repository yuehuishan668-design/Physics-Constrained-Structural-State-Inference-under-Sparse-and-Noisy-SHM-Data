# 5. Conclusions — Scientific Core

> STATUS: SCIENTIFIC CORE ONLY
>
> Purpose:
> freeze the final manuscript-level conclusions and their claim boundaries.
>
> Later editing may improve language, concision, paragraph flow, and journal
> style.
>
> Later editing MUST NOT:
>
> - introduce new numerical results;
> - strengthen benchmark-specific findings into universal claims;
> - describe the study as field validated;
> - describe the 92D representation as a physical upper bound;
> - claim universal noise robustness;
> - claim universal sensor-placement optimality;
> - reinterpret diagnostic descriptor shifts as causal proof.

---

# 5.1 Overall conclusion

This study examined storey-level structural damage inference from a
deployment-aware perspective in which prediction performance is evaluated
jointly with:

\[
\text{information provenance},
\]

\[
\text{sensing observability},
\]

\[
\text{estimator capability},
\]

and

\[
\text{failure direction}.
\]

The principal result is that damage-inference failure cannot be characterized
adequately by aggregate prediction error alone.

Within the controlled numerical benchmark, both the origin of the information
supplied to the estimator and the manner in which sensing quality is degraded
materially alter the resulting inference behavior.

---

# 5.2 Conclusion 1 — Information provenance materially affects apparent inference performance

The 92D simulation-privileged representation outperformed the primary 78D
signal-derived representation under both Ridge and RBF-SVR.

However, the additional information in the 92D representation includes
quantities derived from exact simulation-generator parameters.

Therefore:

\[
\boxed{
\text{lower simulation error does not necessarily imply greater deployable SHM capability}.
}
\]

The main methodological implication is that information provenance should be
reported explicitly when simulation-based damage-inference performance is
interpreted for deployment.

The 92D representation should remain a privileged-information reference rather
than a deployment benchmark or universal upper bound.

---

# 5.3 Conclusion 2 — Nonlinear inference improves accuracy but does not eliminate severe-damage underprediction

Under the primary 78D representation, RBF-SVR reduced mean overall MAE relative
to Ridge by approximately

\[
29.9\%.
\]

High-damage MAE decreased by approximately

\[
36.3\%.
\]

High-damage absolute bias was also substantially reduced.

Nevertheless, severe-damage underestimation remained pronounced.

Thus:

\[
\boxed{
\text{improved global accuracy}
\neq
\text{elimination of directional severe-damage failure}.
}
\]

The result supports severity-stratified error magnitude and signed-bias
diagnostics as necessary complements to aggregate MAE.

---

# 5.4 Conclusion 3 — Directional calibration is distribution-dependent

The training-only one-sided residual calibration reduced high-damage MAE,
absolute bias, and underestimation under the repeated in-distribution benchmark.

However, the same correction did not remain beneficial after measurement-noise
shift changed the direction of the prediction bias.

Therefore:

\[
\boxed{
\text{failure-targeted calibration is conditional on the stability of the
underlying error direction}.
}
\]

Calibration should not be interpreted as eliminating the underlying failure
mechanism.

---

# 5.5 Conclusion 4 — Measurement noise and sensor sparsity produce distinct failure mechanisms

Controlled response-measurement noise generated the sequence

\[
\text{descriptor shift}
\rightarrow
\text{prediction inflation}
\rightarrow
\text{high-damage bias reversal}
\rightarrow
\text{upper-bound saturation}.
\]

The high-damage signed bias changed from approximately

\[
-0.021
\]

under the clean condition to

\[
+0.129
\]

at 20% measurement noise.

By contrast, structural sensor removal produced

\[
\text{descriptor unavailability}
\rightarrow
\text{reduced observability}
\rightarrow
\text{increasing severe-damage underprediction}.
\]

The central conclusion is therefore:

\[
\boxed{
\text{measurement noise perturbs available information, whereas sensor loss
removes observable information}.
}
\]

These two forms of sensing degradation should not be collapsed into a single
generic notion of reduced robustness.

---

# 5.6 Conclusion 5 — Additional sensing has consistently beneficial value within the complete tested subset lattice

All 15 non-empty subsets of the four structural sensors were evaluated using
dependency-aware descriptor availability.

Across all

\[
28
\]

admissible one-sensor additions,

\[
28/28
\]

reduced overall MAE and

\[
28/28
\]

reduced high-damage MAE.

Every paired 95% case-bootstrap interval remained entirely on the beneficial
side of zero for both metrics.

This provides strong evidence within the present benchmark that additional
structural sensing improves damage-inference capability.

However:

\[
\boxed{
\text{this is a benchmark-specific empirical result, not a universal sensor
placement law}.
}
\]

Sensor placement also matters independently of sensor count, and the preferred
layout can depend on whether the objective is overall or severe-damage
performance.

---

# 5.7 Deployment-level implication

The study supports a deployment-oriented SHM workflow in which the effective
inference representation is determined before estimator performance is judged.

Conceptually:

\[
\boxed{
\mathcal F_{\mathrm{available}}
=
\left\{
f_j:
P_j\in\mathcal I_{\mathrm{deploy}}
\land
D_j\subseteq S
\right\}.
}
\]

Only descriptors whose information source is deployment-accessible and whose
required sensing channels are available should be treated as valid inputs to
the damage estimator.

Estimator evaluation should then include not only average prediction accuracy,
but also:

- severity-specific error;
- signed bias;
- underestimation behavior;
- measurement-distribution shift;
- sensor-layout dependence.

---

# 5.8 Final limitation boundary

The conclusions are conditional on the present controlled computational
benchmark.

The study uses:

- one four-storey, two-bay frame;
- linear-elastic structural response;
- prescribed storey-level column stiffness degradation;
- randomized sinusoidal base excitation;
- simulation-generated monitoring data;
- additive Gaussian response-measurement noise;
- a base-input sensing assumption for the primary 78D representation;
- in-distribution train/validation/test partitions.

The study does not establish:

- cross-structure generalization;
- laboratory validation;
- field-monitoring performance;
- real-earthquake validation;
- nonlinear damage or collapse behavior;
- universal optimal sensor layouts;
- universal noise robustness;
- causal importance of individual descriptors.

The paired bootstrap intervals quantify uncertainty conditional on the present
450-case simulated test population and do not represent cross-structure
uncertainty.

---

# 5.9 Final manuscript-level conclusion

Preferred final scientific statement:

> Within the present controlled computational benchmark, structural
> damage-inference capability is governed not only by estimator accuracy but by
> the provenance, availability, and degradation mode of the sensing
> information supplied to the estimator. Measurement noise and sensor loss were
> found to produce qualitatively different severe-damage failure mechanisms,
> while dependency-aware sensing analysis showed consistent benefit from every
> admissible one-sensor addition in the tested four-sensor subset lattice.

Shorter possible final sentence:

> Deployment-oriented SHM inference should therefore be evaluated through the
> combined lens of information provenance, sensing observability, estimator
> capability, and failure direction.

---

# Conclusion claim lock

The final Conclusions section may strongly state:

1. information provenance materially affects apparent inference performance;
2. nonlinear estimation improves but does not eliminate severe-damage
   underprediction;
3. one-sided calibration is distribution-dependent;
4. measurement noise and sensor sparsity produce distinct failure mechanisms;
5. all 28 admissible sensor additions improve both overall and high-damage MAE
   within the complete tested subset lattice.

The final Conclusions section must NOT state:

- that the proposed framework is universally robust;
- that RBF-SVR is an optimal SHM estimator;
- that 92D is the true upper performance limit;
- that every sensor addition is universally beneficial;
- that the current sensor layouts are optimal for other structures;
- that the method has been validated under field conditions.
