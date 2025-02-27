# Air Quality Analysis Source Code

This directory contains the complete implementation of the air quality analysis project, including hybrid models and efficiency metrics.

## Project Structure

```
src/
├── 01_data_preprocessing.ipynb    # Data preparation and feature engineering
├── 02_model_definitions.ipynb     # Model architectures including hybrids
├── 03_model_training.ipynb        # Model training and evaluation
├── model_metrics.ipynb           # Performance and efficiency analysis
├── 04_model_optimization.ipynb    # Model optimization and compression
├── 05_temp_hum_prediction.ipynb   # Temperature/humidity prediction
├── 06_visualization.ipynb         # Results visualization
├── utils.py                      # Utility functions for metrics
├── test_updates.py              # Validation tests
├── validate_notebooks.py        # Notebook structure validation
├── combine_notebooks.py         # Notebook combination
└── run_analysis.bat            # Main execution script
```

## New Features

### 1. Hybrid Models
- **CNN-LSTM**: Combines CNN for feature extraction with LSTM for temporal patterns
  - Lightweight architecture with shared features
  - Optimized for efficiency
- **CNN-BiLSTM**: Enhanced version with bidirectional processing
  - Better feature capture with similar efficiency
  - Balanced accuracy vs. resource usage

### 2. Efficiency Metrics
- Memory Usage Tracking
- Inference Time Measurement
- Model Size Analysis
- Efficiency Scoring System

### 3. Model Optimization
- Architecture Optimization
- Weight Quantization
- Tensor Compression
- Memory Usage Reduction

## Running the Analysis

1. **Setup Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Tests**:
   ```bash
   python src/test_updates.py
   ```

3. **Execute Analysis**:
   ```bash
   src/run_analysis.bat
   ```

## Key Components

### Data Preprocessing
- Feature normalization
- Outlier detection
- Class balancing
- Data reshaping for hybrid models

### Model Training
- Balanced mini-batch training
- Early stopping
- Learning rate scheduling
- Memory-efficient training

### Performance Analysis
- Comprehensive metrics tracking
- Efficiency comparisons
- Resource usage monitoring
- Performance vs. size trade-offs

### Model Optimization
- Post-training quantization
- Architecture pruning
- Weight compression
- Memory footprint reduction

## Efficiency Metrics

1. **Memory Usage**
   - Runtime memory allocation
   - Peak memory usage
   - Memory per parameter

2. **Inference Time**
   - Batch processing speed
   - Single sample inference
   - Warm-up compensation

3. **Model Size**
   - Parameter count
   - Stored model size
   - Compressed size

4. **Efficiency Score**
   - Accuracy/resource ratio
   - Size-adjusted performance
   - Speed-weighted metrics

## Results Output

The analysis generates several output files:

```
results/
├── model_metrics.pkl           # Detailed performance metrics
├── model_comparison.csv        # Model comparison table
├── optimization_results.pkl    # Optimization results
└── final_summary.pkl          # Overall analysis summary
```

## Troubleshooting

1. **Memory Issues**
   - Adjust batch size in training
   - Enable memory growth
   - Use model checkpointing

2. **Performance Issues**
   - Check GPU utilization
   - Monitor memory leaks
   - Optimize batch sizes

3. **Optimization Issues**
   - Verify quantization compatibility
   - Check compression ratios
   - Monitor accuracy degradation

## Contributing

When adding new features:
1. Update test_updates.py
2. Add efficiency metrics
3. Document resource usage
4. Validate with hybrid models

## Notes

- Monitor GPU memory for hybrid models
- Check efficiency metrics regularly
- Compare with baseline models
- Document resource requirements