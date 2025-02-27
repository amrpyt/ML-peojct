# Air Quality Classification and Prediction Project

This project implements multiple deep learning models to classify air quality and predict temperature and humidity based on sensor data.

## Project Structure

```
/ML peojct/
├── src/                      # Source code
│   ├── main.py               # Main execution file
│   ├── config.py             # Configuration parameters
│   ├── preprocess.py         # Data preprocessing functions
│   ├── models.py             # Deep learning model implementations
│   ├── evaluate.py           # Model evaluation functions
│   ├── optimize.py           # Model optimization techniques
│   ├── predict.py            # Functions for making predictions
│   ├── utils.py              # Utility functions
│   ├── data.csv              # Dataset
│   └── see_data.ipynb        # Jupyter notebook to explore data
├── models/                   # Saved models
├── results/                  # Evaluation results and plots
├── logs/                     # Training logs
└── requirements.txt          # Project dependencies
```

## Features

1. **Data Preprocessing**:
   - Handling missing values and outliers
   - Data normalization
   - Class balancing using SMOTE
   - Feature extraction from timestamps

2. **Deep Learning Models**:
   - Deep Neural Network (DNN)
   - 1D Convolutional Neural Network (1D CNN)
   - Recurrent Neural Network (RNN)
   - Long Short-Term Memory Network (LSTM)
   - Bidirectional LSTM (BiLSTM)

3. **Model Evaluation**:
   - Classification metrics (accuracy, precision, recall, F1-score)
   - Confusion matrices
   - Regression metrics (MSE, RMSE, MAE, R²)

4. **Model Optimization Techniques**:
   - Pruning (weight sparsity)
   - Quantization
   - Clustering
   - Weight clipping
   - Knowledge distillation

5. **Air Quality Classification**:
   - Good (0-50)
   - Moderate (51-100)
   - Unhealthy for Sensitive Groups (101-150)
   - Unhealthy (151-200)
   - Hazardous (201-400)

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python src/main.py
   ```

## Dataset

The dataset contains the following columns:
- `CO2`: Carbon dioxide level
- `TVOC`: Total volatile organic compounds
- `PM10`: Particulate matter (10 micrometers)
- `PM2.5`: Particulate matter (2.5 micrometers)
- `CO`: Carbon monoxide level
- `Air Quality`: Air quality index (target for classification)
- `LDR`: Light dependent resistor value
- `O3`: Ozone level
- `Temp`: Temperature (target for regression)
- `Hum`: Humidity (target for regression)
- `ts`: Timestamp

## Project Workflow

1. **Data Preprocessing**:
   - Handle missing values using median/mode imputation
   - Replace outliers with median values
   - Apply normalization to all numeric features
   - Balance classes using SMOTE

2. **Model Training**:
   - Train 5 different deep learning models
   - Evaluate each model using classification metrics
   - Select the best model based on F1 score

3. **Model Optimization**:
   - Apply various optimization techniques to the best model
   - Evaluate the optimized model's performance
   - Compare original and optimized models

4. **Temperature and Humidity Prediction**:
   - Use the optimized model as a base for regression
   - Add regression heads for temperature and humidity prediction
   - Evaluate regression performance using MSE, RMSE, MAE, and R²
