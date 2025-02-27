import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import time
from config import TEST_SIZE, RANDOM_STATE, AQ_CATEGORIES, FEATURES, TARGET_CLASS

def preprocess_data(df, features, target_class, regression_targets=None, timestamp_col=None):
    """
    Preprocess the data with progress updates to avoid appearing frozen
    """
    print("   - Checking for missing values...")
    df = handle_missing_values(df)
    
    print("   - Handling outliers...")
    df = handle_outliers(df, features + [target_class] + (regression_targets or []))
    
    # Process timestamp if provided
    if timestamp_col and timestamp_col in df.columns:
        print("   - Extracting time features...")
        df = extract_time_features(df, timestamp_col)
    
    # Preprocess for classification
    print("   - Preparing classification data...")
    class_data = preprocess_classification(df, features, target_class)
    
    # Preprocess for regression if targets are provided
    reg_data = None
    if regression_targets:
        print("   - Preparing regression data...")
        reg_data = preprocess_regression(df, features, regression_targets)
    
    return {'classification': class_data, 'regression': reg_data}

def handle_missing_values(df):
    """Handle missing values in the dataframe"""
    na_count = df.isna().sum()
    if na_count.sum() > 0:
        print(f"      Found {na_count.sum()} missing values")
    
    # For numeric columns, fill with median
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with mode
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    return df

def handle_outliers(df, columns):
    """Handle outliers using IQR method"""
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with median instead of removing
            median_val = df[col].median()
            df.loc[df[col] < lower_bound, col] = median_val
            df.loc[df[col] > upper_bound, col] = median_val
            
    return df

def extract_time_features(df, timestamp_col):
    """Extract features from timestamp column"""
    try:
        df['datetime'] = pd.to_datetime(df[timestamp_col])
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['dayofweek'] = df['datetime'].dt.dayofweek
        
        # Drop original datetime column
        df = df.drop(['datetime'], axis=1)
        
    except Exception as e:
        print(f"      Warning: Error processing timestamp: {e}")
        
    return df

def preprocess_classification(df, features, target):
    """Preprocess data for classification task with progress updates"""
    print("      Encoding target variable...")
    le = LabelEncoder()
    y = le.fit_transform(df[target])
    
    print("      Normalizing features...")
    X = df[features].copy()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("      Splitting data...")
    
    # Check class distribution before splitting
    class_counts = np.bincount(y)
    min_class_count = class_counts.min()
    
    if min_class_count < 2:
        # If any class has fewer than 2 samples, don't use stratify
        print(f"      Warning: Found class with only {min_class_count} sample(s). Disabling stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    else:
        # Use stratification for balanced classes
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    
    print(f"      Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Balance classes using SMOTE - this can be slow for large datasets
    try:
        print("      Balancing classes with SMOTE (this may take a moment)...")
        start_time = time.time()
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"      SMOTE completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"      Warning: SMOTE failed, using original data: {e}")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    print("      One-hot encoding targets...")
    encoder = OneHotEncoder(sparse_output=False)
    y_train_cat = encoder.fit_transform(y_train_balanced.reshape(-1, 1))
    y_test_cat = encoder.transform(y_test.reshape(-1, 1))
    
    print("      Creating sequences for CNN/RNN models...")
    window_size = 3  # Smaller window to avoid memory issues
    X_train_seq = create_sequences(X_train_balanced, window_size)
    X_test_seq = create_sequences(X_test, window_size)
    
    print("      Classification preprocessing complete.")
    return {
        'X_train': X_train_balanced,
        'X_test': X_test,
        'y_train': y_train_balanced,
        'y_test': y_test,
        'y_train_cat': y_train_cat,
        'y_test_cat': y_test_cat,
        'X_train_seq': X_train_seq,
        'X_test_seq': X_test_seq,
        'scaler': scaler,
        'encoder': encoder,
        'label_encoder': le,
        'n_classes': len(np.unique(y)),
        'class_names': le.classes_
    }

def preprocess_regression(df, features, targets):
    """Preprocess data for regression task"""
    X = df[features].copy()
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    y_temp = df[targets[0]].values.reshape(-1, 1)
    y_hum = df[targets[1]].values.reshape(-1, 1)
    
    scaler_temp = MinMaxScaler()
    scaler_hum = MinMaxScaler()
    y_temp_scaled = scaler_temp.fit_transform(y_temp).flatten()
    y_hum_scaled = scaler_hum.fit_transform(y_hum).flatten()
    
    X_train, X_test, y_train_temp, y_test_temp, y_train_hum, y_test_hum = train_test_split(
        X_scaled, y_temp_scaled, y_hum_scaled, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print("      Regression preprocessing complete.")
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train_temp': y_train_temp,
        'y_test_temp': y_test_temp,
        'y_train_hum': y_train_hum,
        'y_test_hum': y_test_hum,
        'scaler_X': scaler_X,
        'scaler_temp': scaler_temp,
        'scaler_hum': scaler_hum
    }

def create_sequences(data, window_size):
    """Create sequences for sequential models"""
    # If data is too small, use simpler approach
    if len(data) < 1000:
        # Reshape data to [samples, timesteps=1, features]
        return np.expand_dims(data, axis=1)
    
    # For larger datasets, use a memory-efficient approach
    sequences = []
    step = max(1, len(data) // 1000)  # Take fewer sequences for very large datasets
    
    for i in range(0, len(data) - window_size + 1, step):
        sequences.append(data[i:i+window_size])
    
    return np.array(sequences)
