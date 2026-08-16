# Repository structure

```text
.
├── README.md                         # SCHM reviewer entry point
├── CITATION.cff                      # citation metadata; no fabricated DOI
├── LICENSE                           # MIT license
├── requirements.txt                  # minimal pinned scientific environment
├── requirements-lock.txt             # complete tested macOS environment
├── environment.yml                   # optional Python 3.13.6 environment entry
├── configs/
│   └── sss_fast_revision/            # frozen hashes and descriptor manifests
├── data_processed/                   # tracked lightweight indices/names only
├── docs/                             # submission-facing reproducibility documentation
├── manuscript/
│   └── jcshm_reconstruction/         # historical internal provenance path; frozen tables/figures/evidence
├── opensees_models/                  # four-storey OpenSeesPy frame and dynamics
├── scripts/
│   ├── verify_manuscript_results.py  # Level A verifier; no training
│   ├── build_manuscript_outputs.py   # Level B deterministic output builder
│   ├── run_smoke_test.py             # small execution test; not a paper metric
│   └── sss_fast_revision/            # frozen scientific P1–P5 scripts
├── src/                              # data generation, preprocessing, training, evaluation
└── results/
    └── sss_fast_revision/            # tracked processed evidence and reports
```

## Publication-facing versus historical content

The source branch contains earlier development analyses and a historical internal directory named `jcshm_reconstruction`. These files are retained to preserve provenance and avoid destructive history rewriting. The current public entry point is this README and the `docs/` directory; they identify SCHM and make the P1–P5 path explicit.

## Large data policy

Raw response, descriptor, prediction, and model arrays (`*.npz`, `*.npy`, `*.pkl`, `*.pt`, and related formats) are ignored. Tracked CSV/JSON/SVG/PDF products are lightweight evidence for review and manuscript verification. Regeneration paths are documented in `REPRODUCTION_GUIDE.md`.

