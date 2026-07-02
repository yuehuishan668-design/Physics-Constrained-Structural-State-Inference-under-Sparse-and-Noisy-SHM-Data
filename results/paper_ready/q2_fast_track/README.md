# Q2 Fast-track Paper-ready Package

This folder collects the core tables, figures, captions, and text snippets for the Q2 fast-track manuscript.

## Experimental chain

1. 3000-case controlled simulation dataset
2. Main physics-informed feature ablation
3. Repeated-split robustness
4. Damage-stratified reliability diagnosis
5. Noise robustness
6. Sensor sparsity stress test

## Folder structure

- `tables/`: core paper tables and machine-readable summaries
- `figures/`: copied paper-ready figures from experimental outputs
- `captions/`: draft figure and table captions
- `text/`: draft Results and Discussion paragraphs

## Notes

Large generated `.npz` datasets are not included in version control. All data generation, preprocessing, feature extraction, and evaluation scripts are preserved for reproducibility.

The sensor sparsity experiment should be described as a zero-masking stress test, not as a missing-data imputation method.

Noise robustness datasets were independently generated for each fixed noise level; therefore, performance trends need not be strictly monotonic.

## Paper-ready figure groups

- Figure 1: Main physics-informed feature ablation
- Figure 2: Repeated-split robustness
- Figure 3: Damage-stratified reliability diagnosis
- Figure 4: Noise robustness
- Figure 5: Sensor sparsity stress test
