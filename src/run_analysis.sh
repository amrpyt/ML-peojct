#!/bin/bash

# Change log:
# - Added support for hybrid models (CNN-LSTM, CNN-BiLSTM)
# - Integrated efficiency metrics tracking
# - Added model comparison functionality
# - Updated validation checks

echo "Starting Air Quality Analysis..."
echo "=============================="

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create and activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

# Run tests first
echo
echo "Running validation tests..."
echo "------------------------------"
python src/test_updates.py
if [ $? -ne 0 ]; then
    echo "Error: Tests failed"
    echo "Please fix the issues before proceeding"
    read -p "Press Enter to continue..."
    exit 1
fi

# Create required directories
mkdir -p models logs results preprocessed

echo
echo "Environment ready. Starting notebook validation..."
echo "------------------------------"

# Validate notebooks
python src/validate_notebooks.py
if [ $? -ne 0 ]; then
    echo "Error: Notebook validation failed"
    read -p "Press Enter to continue..."
    exit 1
fi

echo
echo "Combining notebooks..."
echo "------------------------------"

# Combine notebooks
python src/combine_notebooks.py
if [ $? -ne 0 ]; then
    echo "Error: Notebook combination failed"
    read -p "Press Enter to continue..."
    exit 1
fi

echo
echo "Analysis Pipeline:"
echo "1. Data Preprocessing"
echo "2. Model Definitions [including CNN-LSTM and CNN-BiLSTM hybrids]"
echo "3. Model Training"
echo "4. Performance Analysis [with efficiency metrics]"
echo "5. Model Optimization"
echo "6. Temperature and Humidity Prediction"
echo "7. Visualization"

echo
echo "To run the analysis:"
echo "1. Start Jupyter Notebook:"
echo "   jupyter notebook src/final_air_quality_analysis.ipynb"
echo
echo "2. Run all cells in order"
echo
echo "3. Check results in:"
echo "   - models/ for saved models"
echo "   - results/ for metrics and comparisons"
echo "   - logs/ for training logs"
echo
echo "Note: Look for efficiency comparisons between"
echo "standard and hybrid models in the results."

# Check if tmux is available for memory monitoring
if command -v tmux &> /dev/null; then
    echo
    echo "Memory monitoring available:"
    echo "Run 'tmux new-session \"top -b -n 1\"' in another terminal"
    echo "to monitor resource usage during training"
fi

# Keep terminal open
read -p "Press Enter to exit..."