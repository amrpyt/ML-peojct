"""
Configuration parameters for the improved hybrid model
"""

# Data configuration
WINDOW_SIZE = 60  # Sliding window size for time series
FEATURES = ['CO2', 'TVOC', 'PM10', 'PM2.5', 'CO', 'LDR', 'O3']
DATA_PATH = 'data/air_quality.csv'

# Air quality categories
AQ_CATEGORIES = {
    'Good': (0, 50),
    'Moderate': (51, 100),
    'Unhealthy for Sensitive Groups': (101, 150),
    'Unhealthy': (151, 200),
    'Very Unhealthy': (201, 300),
    'Hazardous': (301, 500)
}

# Model parameters
MODEL_CONFIG = {
    'cnn_filters': [32, 64],  # Reduced from [64, 128, 256]
    'kernel_size': 3,
    'lstm_units': 32,  # Reduced from 128
    'dense_units': [32],  # Reduced from [128, 64]
    'dropout_rate': 0.3
}

# Training parameters
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
    'learning_rate': 0.001
}

# Optimization parameters
OPTIMIZATION_CONFIG = {
    'pruning_sparsity': 0.5,
    'weight_clip_value': 0.01,
    'min_learning_rate': 0.00001,
    'lr_reduction_factor': 0.1,
    'early_stopping_patience': 10,
    'lr_patience': 5
}

# Loss weights for multi-task learning
LOSS_WEIGHTS = {
    'classification': 1.0,
    'temperature': 0.5,
    'humidity': 0.5
}

# Model saving paths
PATHS = {
    'models': 'models',
    'results': 'results',
    'logs': 'logs'
}

# Visualization settings
VIS_CONFIG = {
    'figsize': (15, 10),
    'style': 'seaborn',
    'palette': 'husl'
}
