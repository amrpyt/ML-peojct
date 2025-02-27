# Air Quality Analysis Project

## Latest Updates
- Added hybrid CNN-LSTM and CNN-BiLSTM models for improved efficiency
- Implemented comprehensive model efficiency metrics
- Added performance comparison and optimization tools
- Enhanced model size reduction techniques

## Overview

This project implements a comprehensive air quality analysis system using deep learning models, with a focus on efficient hybrid architectures and optimized performance.

## Key Features

### Models
- Standard architectures (1DCNN, RNN, LSTM, BiLSTM)
- Hybrid architectures:
  - CNN-LSTM (lightweight feature extraction)
  - CNN-BiLSTM (enhanced temporal processing)

### Efficiency Metrics
- Memory usage tracking
- Inference time measurement
- Model size optimization
- Performance/resource trade-offs

## Project Structure

```
├── src/                          # Source code
│   ├── 01_data_preprocessing.ipynb    # Data preparation
│   ├── 02_model_definitions.ipynb     # Model architectures
│   ├── 03_model_training.ipynb        # Training pipeline
│   ├── model_metrics.ipynb            # Efficiency analysis
│   ├── 04_model_optimization.ipynb    # Model optimization
│   ├── 05_temp_hum_prediction.ipynb   # Additional predictions
│   ├── 06_visualization.ipynb         # Results visualization
│   ├── utils.py                       # Utility functions
│   ├── test_updates.py               # Validation tests
│   └── README.md                     # Detailed documentation
├── models/                       # Saved models
├── logs/                        # Training logs
├── results/                     # Analysis results
└── preprocessed/                # Processed data
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- TensorFlow >= 2.16.0
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- Psutil >= 5.8.0 (for memory tracking)

## Quick Start

### Windows:
```batch
src\run_analysis.bat
```

### Linux/macOS:
```bash
chmod +x src/run_analysis.sh
./src/run_analysis.sh
```

The script will:
1. Set up the environment
2. Run validation tests
3. Process notebooks
4. Generate final analysis

## Model Performance

Hybrid models are designed for efficiency:

1. CNN-LSTM:
   - Reduced parameter count
   - Efficient feature extraction
   - Balanced memory usage

2. CNN-BiLSTM:
   - Enhanced feature capture
   - Optimized architecture
   - Resource-efficient processing

## Efficiency Metrics

The project now tracks:
- Memory usage per model
- Inference time
- Model size
- Efficiency scores

## Results Analysis

Check the following files:
- `results/model_metrics.pkl`: Detailed performance data
- `results/model_comparison.csv`: Model comparisons
- `logs/`: Training history and resource usage

## Optimization Features

1. Model Size Reduction:
   - Architecture optimization
   - Weight quantization
   - Tensor compression

2. Memory Optimization:
   - Efficient data loading
   - Memory monitoring
   - Resource cleanup

3. Speed Optimization:
   - Batch size tuning
   - GPU utilization
   - Inference optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests
4. Submit pull request

## Testing

Run the test suite:
```bash
python src/test_updates.py
```

## Troubleshooting

1. Memory Issues:
   ```bash
   # Check GPU memory
   nvidia-smi
   
   # Monitor process memory
   top -p $(pgrep python)
   ```

2. Performance Issues:
   - Adjust batch size in notebooks
   - Enable memory growth
   - Monitor resource usage

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for optimization tools
- scikit-learn for preprocessing utilities
- NVIDIA for GPU support
