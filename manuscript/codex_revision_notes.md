# Codex Revision Notes for Q2 Fast-track Manuscript

## Task

Create a coherent Q2 fast-track manuscript draft for an SCI Q2 target with Q1 ambition, following the 06 ready-to-use Codex prompt and applying the `nature-writing` skill during manuscript construction.

## Files read or used

### Guidance files

- `/Users/jerry/Downloads/codex_guidance_q2_fast_track_updated-1/00_CODEX_MASTER_INSTRUCTIONS.md`
- `/Users/jerry/Downloads/codex_guidance_q2_fast_track_updated-1/01_ACTUAL_PROJECT_FILE_MAP.md`
- `/Users/jerry/Downloads/codex_guidance_q2_fast_track_updated-1/02_Q2_EXPERIMENT_WORKFLOW_GUIDE.md`
- `/Users/jerry/Downloads/codex_guidance_q2_fast_track_updated-1/03_NUMERICAL_AND_INTERPRETATION_LOCKS.md`
- `/Users/jerry/Downloads/codex_guidance_q2_fast_track_updated-1/04_MANUSCRIPT_ASSEMBLY_TASKS_FOR_CODEX.md`
- `/Users/jerry/Downloads/codex_guidance_q2_fast_track_updated-1/05_DO_NOT_DO_AND_QUALITY_CHECK.md`
- `/Users/jerry/Downloads/codex_guidance_q2_fast_track_updated-1/06_READY_TO_USE_CODEX_PROMPT.md`
- `/Users/jerry/Downloads/codex_guidance_q2_fast_track_updated-1/07_OPTIONAL_LOCAL_CLEANUP_NOTES.md`

### Manuscript component drafts

- `manuscript/abstract_conclusion_q2_fast_track.md`
- `manuscript/introduction_q2_fast_track.md`
- `manuscript/methodology_q2_fast_track.md`
- `manuscript/results_q2_fast_track.md`
- `manuscript/discussion_limitations_q2_fast_track.md`
- `manuscript/outline_q2_fast_track.md`

### Q2 paper-ready evidence package

- `results/paper_ready/q2_fast_track/tables/T00_dataset_quality_debug_plus_3000.json`
- `results/paper_ready/q2_fast_track/tables/T01_physics_feature_summary_debug_plus_3000.json`
- `results/paper_ready/q2_fast_track/tables/T02_main_ablation_3000.csv`
- `results/paper_ready/q2_fast_track/tables/T03_repeated_split_robustness_summary.csv`
- `results/paper_ready/q2_fast_track/tables/T04_repeated_split_per_seed_best.csv`
- `results/paper_ready/q2_fast_track/tables/T05_damage_stratified_config_summary.csv`
- `results/paper_ready/q2_fast_track/tables/T06_damage_stratified_runs.csv`
- `results/paper_ready/q2_fast_track/tables/T07_noise_best_by_noise.csv`
- `results/paper_ready/q2_fast_track/tables/T08_noise_robustness_summary.csv`
- `results/paper_ready/q2_fast_track/tables/T09_sensor_best_by_sensor_count.csv`
- `results/paper_ready/q2_fast_track/tables/T10_sensor_sparsity_summary.csv`
- `results/paper_ready/q2_fast_track/text/R01_repeated_split_robustness_report.md`
- `results/paper_ready/q2_fast_track/text/R02_damage_stratified_report.md`
- `results/paper_ready/q2_fast_track/text/R03_noise_robustness_report.md`
- `results/paper_ready/q2_fast_track/text/R04_sensor_sparsity_report.md`
- `results/paper_ready/q2_fast_track/captions/figure_captions.md`
- `results/paper_ready/q2_fast_track/captions/table_captions.md`

### Code and workflow files inspected for methodological accuracy

- `opensees_models/frame_base.py`
- `opensees_models/run_dynamic_analysis.py`
- `src/data_generation/generate_dataset.py`
- `src/preprocessing/create_split_normalized_dataset.py`
- `src/preprocessing/create_sensor_masked_dataset.py`
- `src/preprocessing/extract_physics_features.py`
- `src/experiments/run_physics_feature_ablation.py`
- `src/training/train_physics_sklearn.py`
- `src/evaluation/run_seed_robustness.py`
- `src/evaluation/run_damage_stratified_analysis.py`
- `scripts/run_q2_generate_debug_plus_3000.sh`
- `scripts/run_q2_3000_seed_robustness.sh`
- `scripts/run_q2_3000_damage_stratified.sh`
- `scripts/run_q2_noise_robustness_1000.sh`
- `scripts/run_q2_sensor_sparsity_3000.sh`
- `scripts/check_q2_numerical_consistency.py`

## Files created

- `manuscript/full_manuscript_q2_fast_track.md`
- `manuscript/codex_revision_notes.md`

## Writing mode and argument

Detected `nature-writing` axes:

- Paper type: research / method-and-diagnosis empirical manuscript
- Sections: title, abstract, introduction, methodology, results, discussion, limitations, conclusion
- Language: English manuscript built from mixed Chinese/English project notes
- Journal: generic SCI, with Q2 target and Q1 ambition

One-sentence argument:

In controlled sparse and noisy SHM-like structural response data, physics-informed descriptor construction supports stable average story-level damage inference, but damage-stratified diagnosis reveals systematic high-damage underestimation and sensor sparsity stress tests identify severe sensing reduction as a practical reliability boundary.

## Terminology choices

- Used `physics-informed descriptor construction` as the central method phrase.
- Used `SHM-like response data` instead of `field SHM data`.
- Used `controlled OpenSeesPy-based simulation` instead of `real bridge validation`.
- Used `full descriptor set` and `full + ridge` consistently.
- Used `sensor sparsity stress test` for the zero-masking experiment.

## Major assembly decisions

- Rebuilt the manuscript as a coherent draft rather than concatenating component files.
- Moved the argument toward a reliability-oriented SHM framing suitable for SCI review.
- Kept Results as the evidence core, following the `nature-writing` recommendation to draft from evidence outward.
- Added a compact Terminology Ledger to stabilize method and dataset language.
- Added Data Availability, Code Availability, Ethics, Funding, Competing Interests, Author Contributions, AI-use Disclosure, and References placeholders because these are common SCI submission elements.

## Numerical constraints preserved

- Main dataset: 3000 cases, response shape 3000 x 2000 x 4, damage target shape 3000 x 4, train/validation/test = 2100 / 450 / 450, 92 descriptors, dt = 0.01 s.
- Main fixed split: full + ridge test MAE = 0.0393; test RMSE = 0.0585; damaged-entry MAE = 0.0634.
- Repeated split: full + ridge mean test MAE = 0.0454 +/- 0.0009; mean test RMSE = 0.0621; best in all 10 splits.
- Damage stratification: overall MAE = 0.0407; overall RMSE = 0.0600; high-damage MAE = 0.0888; high-damage bias = -0.0717; high-damage underestimation ratio = 0.8214; mean true high damage = 0.2758; mean predicted high damage = 0.2041.
- Noise MAEs: 0.0345, 0.0361, 0.0375, 0.0394, 0.0431, 0.0420.
- Sensor sparsity best MAEs: 0.0393, 0.0612, 0.0764, 0.0801.

## Claims deliberately bounded

- The manuscript does not claim field validation.
- The manuscript does not claim deployment readiness.
- The manuscript does not claim full + ridge is best under all sensor settings.
- The manuscript does not claim the noise curve increases at every adjacent noise step.
- The manuscript does not claim high-damage underestimation is solved.
- The manuscript does not present paired healthy-baseline results as the Q2 main method.

## Duplicated content removed or compressed

- Repeated statements about controlled simulation scope were consolidated into the Introduction, Methodology, Discussion, and Limitations.
- Repeated results summaries across abstract, results, conclusion, and outline were compressed so each section has a different job.
- Discussion was rewritten to interpret the results rather than repeat the five result subsections figure by figure.

## Citation and reference status

The local `references/literature_matrix.xlsx` file is not usable as a reference source because it is effectively empty. The manuscript therefore uses citation-needed placeholders rather than fabricated references. The user has a Zotero `first paper` collection in prior memory, and formal citation insertion should use that verified Zotero/BibTeX route in a later step.

## Remaining TODOs for human review

- Replace all `[CITATION NEEDED: ...]` placeholders with verified references from Zotero or user-supplied BibTeX/PDF sources.
- Decide target journal and adjust abstract structure, word limits, headings, and reference style.
- Decide whether to include a workflow figure before the current Figure 1 group.
- Convert concise table references into journal-formatted tables if required.
- Fill Funding, Competing Interests, Author Contributions, and AI-use Disclosure.
- Decide the final data/code release wording.
- Review whether title should emphasize descriptor construction, reliability diagnosis, or sparse/noisy SHM depending on the target journal.

## Verification commands to run after file creation

```bash
python scripts/check_q2_numerical_consistency.py
grep -n "FAIL\|WARN\|CHECK" results/paper_ready/q2_fast_track/text/numerical_consistency_check.md
grep -Rni -C 2 -E "strictly[[:space:]]+monotonic|field[- ]ready|pr[o]ve" manuscript/*.md
```
