# pc-ssi-sparse-noisy-shm: OpenSeesPy starter

This starter contains the first OpenSeesPy model for the paper project:
Physics-Constrained Structural State Inference under Sparse and Noisy Monitoring Data.

## Run from the repository root

```bash
python -m opensees_models.frame_base
python -m opensees_models.run_dynamic_analysis
```

## Expected output

- Modal periods of the healthy model.
- Modal periods of a damaged model.
- One `.npz` file saved under `data_raw/opensees_outputs/`.
- Two figures saved under `results/figures/`.
