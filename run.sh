#!/bin/bash

# Quick Start Script for Active Learning Project
# Executes full pipeline: data generation → model training → AL simulation → analysis

echo "======================================================================"
echo "Active Learning Multimodal Content Moderation - Full Pipeline"
echo "======================================================================"

# Configuration
NUM_SAMPLES=${1:-10000}
NUM_ROUNDS=${2:-5}
BUDGET_PER_ROUND=${3:-100}
DEVICE=${4:-cuda}

echo ""
echo "Configuration:"
echo "  Dataset size: $NUM_SAMPLES"
echo "  AL rounds: $NUM_ROUNDS"
echo "  Budget per round: $BUDGET_PER_ROUND"
echo "  Device: $DEVICE"
echo ""

# Create directories
mkdir -p data models results

# ====================================================================
# STEP 1: Generate Dataset
# ====================================================================
echo "======================================================================"
echo "STEP 1: Generating Dataset"
echo "======================================================================"
python data/synthetic_dataset.py \
    --num_samples $NUM_SAMPLES \
    --output_dir data \
    --seed 42

# ====================================================================
# STEP 2: Simulate Labels
# ====================================================================
echo ""
echo "======================================================================"
echo "STEP 2: Simulating Labels (Oracle)"
echo "======================================================================"
python data/labels_simulation.py \
    --dataset_path data/dataset.json \
    --output_path data/labels_oracle.json \
    --error_rate 0.05 \
    --seed 42

# ====================================================================
# STEP 3: Train Base Classifiers
# ====================================================================
echo ""
echo "======================================================================"
echo "STEP 3: Training Base Classifiers"
echo "======================================================================"

echo ""
echo "  3a: Training Text Classifier..."
python models/text_classifier.py \
    --dataset_path data/dataset.json \
    --split_file data/splits.json \
    --output_dir models/text \
    --epochs 3 \
    --batch_size 32 \
    --device $DEVICE

echo ""
echo "  3b: Training Image Classifier..."
python models/image_classifier.py \
    --dataset_path data/dataset.json \
    --split_file data/splits.json \
    --output_dir models/image \
    --epochs 3 \
    --batch_size 32 \
    --freeze_backbone false \
    --device $DEVICE

# ====================================================================
# STEP 4: Run Active Learning Simulation
# ====================================================================
echo ""
echo "======================================================================"
echo "STEP 4: Active Learning Simulation"
echo "======================================================================"
python experiments/run_al_simulation.py \
    --num_samples $NUM_SAMPLES \
    --num_rounds $NUM_ROUNDS \
    --budget_per_round $BUDGET_PER_ROUND \
    --device $DEVICE

# ====================================================================
# STEP 5: Benchmark Strategies
# ====================================================================
echo ""
echo "======================================================================"
echo "STEP 5: Benchmarking Strategies"
echo "======================================================================"
python experiments/benchmark.py \
    --results_dir results

# ====================================================================
# STEP 6: Cost Analysis
# ====================================================================
echo ""
echo "======================================================================"
echo "STEP 6: Cost-Benefit Analysis"
echo "======================================================================"
python experiments/cost_analysis.py \
    --results_dir results

# ====================================================================
# COMPLETION
# ====================================================================
echo ""
echo "======================================================================"
echo "✓ PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - Data: data/"
echo "  - Models: models/"
echo "  - Results: results/"
echo ""
echo "Next steps:"
echo "  1. View results: less results/summary.json"
echo "  2. Launch Streamlit app: streamlit run app.py"
echo "  3. Analyze plots: open results/*.png"
echo ""
