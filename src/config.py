# Configuration parameters for the project

# Data paths
DATA_PATH = r"d:\Work\Client Projects\Paid projects\ML peojct\src\data.csv"
MODEL_SAVE_PATH = r"d:\Work\Client Projects\Paid projects\ML peojct\models"
RESULTS_PATH = r"d:\Work\Client Projects\Paid projects\ML peojct\results"

# Model parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Features configuration
FEATURES = ['CO2', 'TVOC', 'PM10', 'PM2.5', 'CO', 'LDR', 'O3']
TARGET_CLASS = 'Air Quality'
REGRESSION_TARGETS = ['Temp', 'Hum']
TIMESTAMP_COL = 'ts'

# Air quality categories
AQ_CATEGORIES = {
    'Good': (0, 50),
    'Moderate': (51, 100),
    'Unhealthy for Sensitive': (101, 150),
    'Unhealthy': (151, 200),
    'Hazardous': (201, 400)
}

# Optimization parameters
PRUNING_PARAMS = {
    'initial_sparsity': 0.0,
    'final_sparsity': 0.5,
    'begin_step': 0,
    'end_step': 100
}

QUANTIZATION_PARAMS = {
    'quantize_input': True,
    'quantize_output': True
}
