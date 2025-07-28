#!/bin/bash

# Event Flare Removal Project - Easy Run Script
# Usage: ./run_project.sh [train|evaluate|test]

set -e  # Exit on any error

echo "🚀 Event Flare Removal with Mamba Architecture"
echo "=============================================="

# Check if conda environment exists
if ! conda env list | grep -q "event_flare"; then
    echo "📦 Creating conda environment 'event_flare'..."
    conda create -n event_flare python=3.10 -y
    echo "✅ Environment created successfully!"
fi

# Activate environment
echo "🔧 Activating environment..."
source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare

# Install dependencies if needed
echo "📋 Checking dependencies..."
if ! python -c "import torch" 2>/dev/null; then
    echo "📦 Installing PyTorch (CPU version)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

if ! python -c "import numpy, yaml, tqdm, sklearn" 2>/dev/null; then
    echo "📦 Installing other dependencies..."
    pip install numpy pyyaml tqdm scikit-learn
fi

echo "✅ All dependencies installed!"

# Create sample data if not exists
if [ ! -f "data/simulated_events/train_data.txt" ]; then
    echo "📊 Creating sample data..."
    python create_simple_data.py
    echo "✅ Sample data created!"
fi

# Get mode from argument or default to train
MODE=${1:-train}

echo "🎯 Running in mode: $MODE"

# Set mode in config
if [ "$MODE" = "train" ]; then
    sed -i 's/mode: .*/mode: train/' configs/config.yaml
elif [ "$MODE" = "evaluate" ]; then
    sed -i 's/mode: .*/mode: evaluate/' configs/config.yaml
elif [ "$MODE" = "test" ]; then
    echo "🧪 Running pipeline tests..."
    python test_pipeline.py
    exit 0
else
    echo "❌ Invalid mode: $MODE. Use 'train', 'evaluate', or 'test'"
    exit 1
fi

# Run the main program
echo "🎬 Starting $MODE pipeline..."
python main.py --config configs/config.yaml

echo ""
echo "🎉 $MODE completed successfully!"
echo ""
echo "📝 Next steps:"
if [ "$MODE" = "train" ]; then
    echo "   - Run './run_project.sh evaluate' to evaluate the trained model"
    echo "   - Check './checkpoints/best_model.pth' for saved model"
else
    echo "   - Check evaluation results above"
    echo "   - Modify configs/config.yaml to adjust parameters"
fi
echo "   - See readme_new.md for detailed documentation"