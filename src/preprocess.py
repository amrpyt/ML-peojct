import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from config import AQ_CATEGORIES, FEATURES, TARGET_CLASS, REGRESSION_TARGETS, TIMESTAMP_COL
from utils import categorize_air_quality
import warnings

def load_data(filepath):
    """Load data from CSV file"""
    df = pd.read_csv(filepath)
    return df

def handle_missing_values(df):
    """Handle missing values in the dataframe"""
    # Check for missing values
    missing_values = df.isnull().sum()
    
    # For numeric columns, fill missing values with the median
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # For non-numeric columns, fill missing values with the mode
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns
    for col in non_numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def handle_outliers(df):
    """Handle outliers using IQR method and replace with median/mean"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    for col in numeric_cols:
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with median
        df[col] = df[col].apply(lambda x: df[col].median() if (x < lower_bound or x > upper_bound) else x)
    
    return df

def normalize_data(df, method='minmax'):
    """Normalize the data using specified method"""
    # Create a copy to avoid modifying the original dataframe
    df_normalized = df.copy()
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Method should be either 'minmax' or 'standard'")
    
    # Normalize numeric columns
    df_normalized[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_normalized, scaler

def process_timestamps(df):
    """Process timestamp column to extract features"""
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors='coerce')
        df['hour'] = df[TIMESTAMP_COL].dt.hour
        df['day'] = df[TIMESTAMP_COL].dt.day
        df['month'] = df[TIMESTAMP_COL].dt.month
        df['day_of_week'] = df[TIMESTAMP_COL].dt.dayofweek
    
    return df

def encode_target(df):
    """Encode the target variable for classification"""
    # Apply categorization function to Air Quality column
    df['AQ_Category'] = df[TARGET_CLASS].apply(categorize_air_quality)
    
    # Encode categories
    label_encoder = LabelEncoder()
    df['AQ_Category_Encoded'] = label_encoder.fit_transform(df['AQ_Category'])
    
    return df, label_encoder

def prepare_data_for_deep_learning(df, test_size=0.2, random_state=42, use_smote=True):
    """Prepare data for deep learning models"""
    # Select features and target
    X = df[FEATURES].values
    y_class = df['AQ_Category_Encoded'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=test_size, random_state=random_state, stratify=y_class
    )
    
    # Apply SMOTE for balancing classes
    if use_smote:
        smote = SMOTE(random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Prepare data for regression tasks
    X_reg = df[FEATURES].values
    y_temp = df['Temp'].values
    y_hum = df['Hum'].values
    
    X_train_reg, X_test_reg, y_temp_train, y_temp_test, y_hum_train, y_hum_test = train_test_split(
        X_reg, y_temp, y_hum, test_size=test_size, random_state=random_state
    )
    
    # Reshape for sequence models (RNN, LSTM)
    X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    X_train_reg_seq = X_train_reg.reshape(X_train_reg.shape[0], 1, X_train_reg.shape[1])
    X_test_reg_seq = X_test_reg.reshape(X_test_reg.shape[0], 1, X_test_reg.shape[1])
    
    # Get number of classes
    n_classes = len(np.unique(y_train))
    
    # Convert targets to categorical for classification
    y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, n_classes)
    
    class_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_cat': y_train_cat,
        'y_test_cat': y_test_cat,
        'X_train_seq': X_train_seq,
        'X_test_seq': X_test_seq,
        'n_classes': n_classes
    }
    
    reg_data = {
        'X_train_reg': X_train_reg,
        'X_test_reg': X_test_reg,
        'y_temp_train': y_temp_train,
        'y_temp_test': y_temp_test,
        'y_hum_train': y_hum_train,
        'y_hum_test': y_hum_test,
        'X_train_reg_seq': X_train_reg_seq,
        'X_test_reg_seq': X_test_reg_seq
    }
    
    return class_data, reg_data

def preprocess_pipeline(filepath):
    """Complete preprocessing pipeline"""
    # Load data
    df = load_data(filepath)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Process timestamps
    df = process_timestamps(df)
    
    # Encode target
    df, label_encoder = encode_target(df)
    
    # Normalize data
    df_normalized, scaler = normalize_data(df)
    
    # Prepare data for deep learning
    class_data, reg_data = prepare_data_for_deep_learning(df_normalized)
    
    return {
        'df': df,
        'df_normalized': df_normalized,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'class_data': class_data,
        'reg_data': reg_data
    }
