#!/bin/bash

echo "Creating Python virtual environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if venv exists and remove if it does
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create new virtual environment
echo "Creating new virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install base requirements first
echo "Installing base requirements..."
pip install "numpy>=1.20.0" "pandas>=1.3.0" "scikit-learn>=1.0.0" "matplotlib>=3.4.0" "seaborn>=0.11.0"

# Try installing TensorFlow
echo "Attempting to install TensorFlow..."
if ! pip install "tensorflow>=2.16.0"; then
    echo "TensorFlow install failed, trying CPU version..."
    pip install "tensorflow-cpu>=2.16.0"
fi

# Install remaining requirements
echo "Installing remaining requirements..."
pip install "imbalanced-learn>=0.8.0" "jupyter>=1.0.0" "h5py>=3.10.0"

# Run setup script
echo "Running Python setup script..."
python src/setup.py

echo
echo "Environment setup complete! To activate the environment:"
echo "source .venv/bin/activate"

# Make the virtual environment relocatable
python -m venv --relocatable .venv