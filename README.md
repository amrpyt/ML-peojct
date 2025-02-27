# Air Quality Analysis Project

This project implements comprehensive air quality analysis using deep learning models. It includes data preprocessing, model training, optimization, and temperature/humidity prediction.

## Quick Start

### Windows:
```batch
# Setup environment
src\setup_env.bat

# Activate environment
.venv\Scripts\activate.bat
```

### Linux/macOS:
```bash
# Make script executable
chmod +x src/setup_env.sh

# Setup environment
./src/setup_env.sh

# Activate environment
source .venv/bin/activate
```

## Requirements

### Python Version
- Python 3.8 or higher required
- Python 3.9+ recommended for best TensorFlow compatibility

### Hardware Requirements
- CPU: Any x86-64 processor
- RAM: 8GB minimum, 16GB recommended
- GPU: NVIDIA GPU (optional but recommended)
  - CUDA compatible for GPU acceleration
  - 4GB VRAM minimum

### Software Dependencies
Core packages (installed automatically):
```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.16.0  # CPU or GPU version based on hardware
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
jupyter>=1.0.0
h5py>=3.10.0
```

## Project Structure

```
├── src/
│   ├── air_quality_analysis.ipynb    # Main analysis notebook
│   ├── temperature_humidity_prediction.ipynb  # Prediction models
│   ├── visualization_final.ipynb     # Visualizations
│   ├── combine_notebooks.py         # Notebook combination script
│   ├── setup.py                    # Environment setup script
│   ├── setup_env.bat              # Windows environment setup
│   ├── setup_env.sh               # Unix environment setup
│   └── data.csv                    # Dataset
├── models/                         # Saved models
├── logs/                          # Training logs
└── results/                       # Analysis results
```

## Features

1. Data Preprocessing:
   - Outlier detection and handling
   - Feature normalization
   - Class balancing using SMOTE

2. Deep Learning Models:
   - 1D Convolutional Neural Network (1DCNN)
   - Recurrent Neural Network (RNN)
   - Deep Neural Network (DNN)
   - Long Short-Term Memory (LSTM)
   - Bidirectional LSTM (BiLSTM)

3. Model Optimization:
   - TensorFlow Lite conversion
   - Float16 quantization
   - Size optimization

4. Additional Analysis:
   - Temperature prediction
   - Humidity prediction
   - Performance metrics (MSE, RMSE, R², MAE)

## Troubleshooting

### TensorFlow Installation Issues

1. "No matching distribution found for tensorflow":
   ```bash
   # Try installing CPU-only version
   pip install tensorflow-cpu>=2.16.0
   ```

2. GPU-related errors:
   ```bash
   # Check GPU visibility
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

3. Memory errors:
   - Reduce batch size in notebooks
   - Enable memory growth (automatic in setup)
   - Clear session between models (implemented)

### Environment Setup Issues

1. Virtual Environment Errors:
   ```bash
   # Windows
   python -m venv --clear .venv
   
   # Linux/macOS
   python3 -m venv --clear .venv
   ```

2. Package Conflicts:
   ```bash
   # Clean installation
   pip uninstall tensorflow tensorflow-cpu -y
   pip install tensorflow>=2.16.0
   ```

3. Jupyter Notebook Issues:
   ```bash
   # Install IPython kernel
   python -m ipykernel install --user --name=air_quality_env
   ```

## Running the Analysis

1. Setup environment (if not done):
   ```bash
   # Windows
   src\setup_env.bat
   
   # Linux/macOS
   ./src/setup_env.sh
   ```

2. Generate final notebook:
   ```bash
   python src/combine_notebooks.py
   ```

3. Start Jupyter:
   ```bash
   jupyter notebook src/final_air_quality_analysis.ipynb
   ```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
