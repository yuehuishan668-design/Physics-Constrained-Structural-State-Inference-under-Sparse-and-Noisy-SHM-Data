# Figure Captions

**Fig. 1. Physics feature ablation results measured by test MAE.**  
This figure compares different physics-informed feature subsets and estimators. The main purpose is to verify whether the full physics-informed descriptor set improves damage inference under sparse and noisy SHM observations.

**Fig. 2. Physics feature ablation results measured by test RMSE.**  
This figure provides an error-scale robustness check complementary to MAE.

**Fig. 3. Repeated-split robustness distribution.**  
This figure evaluates whether the selected feature/model configuration is stable across different random train/validation/test splits.

**Fig. 4. Damage-stratified prediction bias.**  
This figure shows whether the model systematically overestimates or underestimates damage in zero, low, medium, and high damage regimes.

**Fig. 5. Damage-stratified MAE.**  
This figure reports the prediction error by damage severity level and is used to diagnose reliability limitations in high-damage cases.

**Fig. 6. Clean paired healthy-baseline control experiment.**  
This figure shows whether strict input-only healthy-baseline matching can separate zero and damaged cases.

**Fig. 7. Clean paired healthy-baseline regression diagnosis.**  
This figure evaluates whether clean paired-baseline normalization improves continuous damage prediction.

**Fig. 8. Case-level any-damage precision-recall curve.**  
This figure diagnoses the separability of zero and damaged cases at the case level.

**Fig. 9. Case-level any-damage ROC curve.**  
This figure provides an additional separability check for the zero-vs-damaged decision problem.
