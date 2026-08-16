# Frozen numerical anchors

`scripts/verify_manuscript_results.py` reads these values from tracked processed evidence. Exact stored values are listed below; manuscript text may display rounded forms.

## P1 — repeated information × estimator factorial

| Quantity | Exact stored value | Typical manuscript display |
|---|---:|---:|
| 78D Ridge overall MAE | 0.04485829455702703 | 0.0448583 |
| 78D RBF-SVR overall MAE | 0.031450000610448225 | 0.0314500 |
| 92D Ridge overall MAE | 0.03954549435827245 | 0.0395455 |
| 92D RBF-SVR overall MAE | 0.02310484788409254 | 0.0231048 |
| 78D Ridge high-damage MAE | 0.09471653851150405 | 0.0947165 |
| 78D RBF-SVR high-damage MAE | 0.06033738471601641 | 0.0603374 |
| 78D Ridge high-damage absolute bias | 0.08280379022791305 | 0.0828038 |
| 78D RBF-SVR high-damage absolute bias | 0.047432396528132394 | 0.0474324 |
| 78D high-damage underestimation | 0.8742895725141325 → 0.7610611424846312 | 11.32 percentage-point reduction |
| Raw difference-in-differences | -0.0030323525276011133 | reduction-style interaction +0.003032 |

The stored factorial file uses `(92D_SVR-78D_SVR)-(92D_Ridge-78D_Ridge)`, which is negative when the SVR reduction is larger at 92D. The manuscript reports the same effect using a positive reduction convention.

## P2 — repeated asymmetric calibration

| Quantity | Exact stored value | Manuscript display |
|---|---:|---:|
| Overall MSE relative improvement | 2.3482523459606943% | 2.35% |
| High-damage MAE relative improvement | 7.629745964842028% | 7.63% |
| High-damage absolute-bias improvement | 29.272903284325132% | 29.27% |
| High-damage underestimation | 0.7610611424846312 → 0.6573552417933878 | 10.37 percentage points |

Zero- and low-damage MAE worsen; the medium-damage interval crosses zero. Calibration redistributes error and is conditional on residual-direction stability.

## P3 — clean-trained matched-noise stress test

| Noise | Overall MAE | High-damage MAE | High-damage signed bias |
|---:|---:|---:|---:|
| 0% | 0.022047140089029485 | 0.03677995557782435 | -0.02133730479685522 |
| 5% | 0.04435935226467652 | 0.05821843237920153 | -0.004789494185237728 |
| 10% | 0.11180562156655098 | 0.11402268443598788 | +0.042574920761648025 |
| 20% | 0.28389311833100594 | 0.2069806407517331 | +0.1285342351836434 |

- Bias reverses between 5% and 10% noise.
- At 20% noise, overall clipping at the imposed upper bound is 0.32255555555555554.
- At 20% noise, high-damage clipping is 0.6408759124087591 and the high-damage median prediction is 0.5.

This is imposed-bound clipping, not intrinsic SVR saturation.

## P4 — exhaustive sensor layouts

- Layout count: 15.
- Full layout 1234: 78 descriptors, overall MAE 0.022047140089029485.
- Layout 12: best overall two-sensor layout.
- Layout 13: best high-damage two-sensor layout.
- Layouts 13 and 14: both 31D, different paired performance.
- Single-sensor high-damage underestimation: approximately 98–100%.

## P5 — complete one-sensor-addition lattice

- Node count: 15.
- Edge count: 28.
- Overall-MAE beneficial edges: 28/28.
- High-damage-MAE beneficial edges: 28/28.
- Every paired 95% interval lies entirely on the beneficial side of zero.

The claim is conditional on the present topology, descriptor definitions, estimator, 450 simulated test cases, case-level paired resampling, edge-wise intervals, and absence of a family-wise multiplicity correction.

