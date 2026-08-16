# Descriptor Observability and Failure Mechanisms of Physics-Guided Structural Damage Inference under Sparse and Noisy Sensing

Yuehui Shan — Department of Civil and Environmental Engineering, The Hong Kong Polytechnic University
ORCID: [0009-0001-8184-3311](https://orcid.org/0009-0001-8184-3311)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21967188.svg)](https://doi.org/10.5281/zenodo.21967188)

This branch is the submission snapshot associated with the manuscript prepared for **Structural Control and Health Monitoring (SCHM)**. The scientific experiments are frozen. The repository packages the existing evidence and code for audit and reproduction; it does not retune models or revise reported results.

## Scientific scope

The study treats structural-damage inference as a chain:

```text
information provenance -> descriptor observability -> inference
-> failure mechanism -> deployment boundary
```

It distinguishes two sensing degradations:

- Measurement noise perturbs descriptors that remain computable, producing distribution shift, prediction inflation, high-damage bias reversal, and saturation at the imposed prediction bound.
- Sensor sparsity removes structurally required signals, so descriptors with unmet dependencies become unavailable and severe-damage inference deteriorates.

These pathways must not be collapsed into a generic robustness claim.

## What this repository reproduces

| Protocol | Frozen question | Main output |
|---|---|---|
| P1 | 78D/92D information representation × Ridge/RBF-SVR over 10 predefined splits | Fig. 3, Table 2 |
| P2 | Training-derived asymmetric calibration of severe-damage underprediction | Fig. 4 |
| P3 | Clean-trained 78D RBF-SVR under matched 0/5/10/20% response noise, without retraining | Fig. 5, Table 3 |
| P4 | Dependency-aware evaluation of all 15 nonempty structural-sensor layouts | Fig. 6, Table 4 |
| P5 | Paired case-bootstrap analysis of all 28 admissible one-sensor additions | Fig. 7 |

The real script and evidence paths are listed in [`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md).

## Repository status

- Submission branch: `paper0-schm-submission`
- Scientific source branch: `paper0-jcshm-manuscript`
- Scientific base commit: `c78af2c5`
- Archived release tag: `schm-submission-v1.0.0`
- Archived release commit: `e1c89dc9fb54fd7cf777b16c0746f15138324885`
- GitHub Release: `SCHM Submission Reproducibility Snapshot v1.0.0`
- Zenodo archive DOI: `10.5281/zenodo.21967188`

The immutable release tag and Zenodo archive remain fixed to the archived
submission snapshot. This branch may contain later metadata-only updates,
such as DOI links, without changing the archived scientific release.

## Quick verification

This reads the tracked frozen CSV evidence and does not train a model:

```bash
python scripts/verify_manuscript_results.py
```

Successful completion ends with:

```text
ALL FROZEN MANUSCRIPT ANCHORS PASSED
```

Important protocol distinction:

```text
P1 repeated heterogeneous-noise benchmark: 78D RBF-SVR MAE ≈ 0.03145
P3/P4 clean fixed-split benchmark:          78D RBF-SVR MAE ≈ 0.022047
```

The values answer different questions and are not conflicting estimates.

## Environment setup

The frozen environment used for submission verification was Python 3.13.6 with OpenSeesPy 3.8.0.0, NumPy 2.5.1, Pandas 3.0.5, SciPy 1.18.0, scikit-learn 1.9.0, and Matplotlib 3.11.1.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements-lock.txt` records the complete tested macOS environment. It is stricter and less portable than the minimal `requirements.txt`.

## Smoke test

```bash
python scripts/run_smoke_test.py
```

The smoke test generates 20 short OpenSeesPy cases, extracts descriptors, fits a small Ridge model, and computes an MAE. It checks execution only and does **not** reproduce or validate manuscript metrics.

## Reproduce manuscript outputs

From frozen processed evidence:

```bash
python scripts/build_manuscript_outputs.py
```

This wrapper regenerates manuscript tables and Figs. 3–7 through the existing deterministic scripts, then reruns the frozen-anchor verifier. It performs no training or model selection. Figures 1 and 2 are conceptual/vector figures and are retained as editable sources rather than regenerated from training.

## Full protocol reproduction

The raw 3000-case arrays and intermediate NPZ files are intentionally excluded from Git. The numbered command sequence, prerequisites, and stop conditions are documented in [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md). Do not start a full rerun merely to verify the manuscript; use the quick verifier first.

## Data policy

Tracked evidence includes split/provenance metadata, processed metric tables, noise summaries, layout results, bootstrap summaries, and figure-source data. Large response, descriptor, prediction, and bootstrap arrays are regenerated locally and remain ignored. See [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

The immutable submission release is archived on Zenodo at
https://doi.org/10.5281/zenodo.21967188. Large raw and intermediate
simulation arrays are not part of the Git/Zenodo snapshot and are regenerated
from the supplied OpenSeesPy scripts, frozen configurations, and fixed random
seeds.

## Expected numerical anchors

Exact stored values and their manuscript display precision are listed in [`docs/NUMERICAL_ANCHORS.md`](docs/NUMERICAL_ANCHORS.md).

## Figures and tables

- Frozen figure-source CSVs: `manuscript/jcshm_reconstruction/figures/data/`
- Vector figures: `manuscript/jcshm_reconstruction/figures/final/`
- Main tables: `manuscript/jcshm_reconstruction/tables/final/`
- Supplementary layout table: `manuscript/jcshm_reconstruction/supplementary/tables/`

The historical internal path `scripts/jcshm_manuscript/` is retained for provenance. It is the source of the deterministic manuscript-output scripts and does not indicate the current target journal.

## Known evidence boundaries

- One simulated four-storey, two-bay, linear-elastic frame.
- Controlled parametric excitation ensemble, not recorded earthquake ground motion.
- Simulation-only evidence; no cross-structure, laboratory, or field validation.
- Descriptor-shift results are diagnostic associations, not causal feature attribution.
- The 28/28 sensor-addition result is conditional on the current topology, descriptors, estimator, 450 simulated test cases, edge-wise intervals, and no family-wise multiplicity correction.
- The 92D representation is a privileged-information reference, not a theoretical upper bound or deployment-valid input set.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Article DOI, volume, issue, and page metadata remain unset because they do not yet exist.

Shan, Y. (2026). *Reproducibility package for Descriptor Observability and
Failure Mechanisms of Physics-Guided Structural Damage Inference under Sparse
and Noisy Sensing* (`schm-submission-v1.0.0`). Zenodo.
https://doi.org/10.5281/zenodo.21967188

## License

Code and repository documentation are released under the [MIT License](LICENSE). Third-party software and cited publications remain under their own licenses.
