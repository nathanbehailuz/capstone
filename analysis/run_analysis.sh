#!/bin/bash
# Helper script to run analysis with proper environment setup

# Set OpenBLAS thread limits to avoid resource issues
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Activate visualization environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/viz_env/bin/activate"

# Default dataset
DATASET="${1:-syn3}"

echo "===================================="
echo "Running Analysis for: $DATASET"
echo "===================================="
echo ""

# Step 1: Build JSON summaries
echo "Step 1: Building JSON summaries..."
python "$SCRIPT_DIR/build_json_summaries.py" --dataset "$DATASET"

if [ $? -ne 0 ]; then
    echo "Error: Failed to build JSON summaries"
    exit 1
fi

echo ""

# Step 2: Generate plots
echo "Step 2: Generating visualizations..."
python "$SCRIPT_DIR/plot_visualizations.py" --dataset "$DATASET"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate visualizations"
    exit 1
fi

echo ""
echo "===================================="
echo "✓ Analysis complete!"
echo "Results saved to: visualization/$DATASET/"
echo "===================================="

