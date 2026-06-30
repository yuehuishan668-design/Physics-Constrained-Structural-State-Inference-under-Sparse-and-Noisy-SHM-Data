#!/usr/bin/env bash
set -euo pipefail

echo "Step 0: check required 500-case datasets"
ls data_processed/debug_plus_500_dataset.npz
ls data_processed/debug_plus_500_split_normalized.npz
ls data_processed/debug_plus_500_physics_features_mlp.npz
ls data_processed/debug_plus_500_physics_feature_names.csv

echo ""
echo "Step 1: train feature-based MLP baselines on the same 500-case split"
echo "Note: your current src.training.train_mlp uses --features, not --data."
python -m src.training.train_mlp \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --output-mode linear \
  --output-dir results/tables/mlp_linear_500 \
  --figures-dir results/figures/mlp_linear_500

python -m src.training.train_mlp \
  --features data_processed/debug_plus_500_physics_features_mlp.npz \
  --output-mode sigmoid \
  --max-damage 0.5 \
  --output-dir results/tables/mlp_sigmoid_500 \
  --figures-dir results/figures/mlp_sigmoid_500

echo ""
echo "Step 2: train raw-response LSTM baselines on the same 500-case split"
python -m src.training.train_lstm \
  --data data_processed/debug_plus_500_split_normalized.npz \
  --condition-mode none \
  --sequence-stride 10 \
  --output-mode linear \
  --output-dir results/tables/lstm_response_500 \
  --figures-dir results/figures/lstm_response_500

python -m src.training.train_lstm \
  --data data_processed/debug_plus_500_split_normalized.npz \
  --condition-mode meta \
  --sequence-stride 10 \
  --output-mode sigmoid \
  --max-damage 0.5 \
  --output-dir results/tables/lstm_meta_sigmoid_500 \
  --figures-dir results/figures/lstm_meta_sigmoid_500

echo ""
echo "Step 3: train two-head LSTM baselines on the same 500-case split"
python -m src.training.train_lstm_two_head \
  --data data_processed/debug_plus_500_split_normalized.npz \
  --condition-mode none \
  --sequence-stride 10 \
  --final-mode threshold \
  --prob-threshold 0.5 \
  --max-damage 0.5 \
  --output-dir results/tables/lstm_two_head_response_500 \
  --figures-dir results/figures/lstm_two_head_response_500

python -m src.training.train_lstm_two_head \
  --data data_processed/debug_plus_500_split_normalized.npz \
  --condition-mode meta \
  --sequence-stride 10 \
  --final-mode threshold \
  --prob-threshold 0.5 \
  --max-damage 0.5 \
  --output-dir results/tables/lstm_two_head_meta_500 \
  --figures-dir results/figures/lstm_two_head_meta_500

echo ""
echo "Step 4: compare 500-case baselines with physics-feature ablation models"
python -m src.evaluation.compare_all_metrics \
  --experiment-dirs \
    results/tables/mlp_linear_500 \
    results/tables/mlp_sigmoid_500 \
    results/tables/lstm_response_500 \
    results/tables/lstm_meta_sigmoid_500 \
    results/tables/lstm_two_head_response_500 \
    results/tables/lstm_two_head_meta_500 \
    results/tables/physics_ablation/debug_plus_500/full/ridge \
    results/tables/physics_ablation/debug_plus_500/no_meta/ridge \
    results/tables/physics_ablation/debug_plus_500/response_basic_only/ridge \
    results/tables/physics_ablation/debug_plus_500/response_spatial/ridge \
    results/tables/physics_ablation/debug_plus_500/response_frequency/ridge \
    results/tables/physics_ablation/debug_plus_500/response_correlation/ridge \
  --output results/tables/fair_baseline_500_comparison.csv

echo ""
echo "Step 5: show sorted comparison"
cat results/tables/fair_baseline_500_comparison.csv

echo ""
echo "Step 6: list prediction CSV files for later diagnostics"
find results/tables \
  \( -name "*predictions_test.csv" -o -name "predictions_test.csv" -o -name "*prediction*.csv" \) \
  | sort

echo ""
echo "All 500-case fair baseline commands completed."
