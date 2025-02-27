"""
Data preparation script for improved model
Changes:
- Added data verification
- Creates preprocessed directory if missing
- Handles data preprocessing if needed
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def prepare_data():
    """Prepare and save preprocessed data."""
    print("Starting data preparation...")
    
    # Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Create preprocessed directory
    preprocessed_dir = os.path.join(current_dir, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Check if preprocessed files already exist
    required_files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']
    files_exist = all(
        os.path.exists(os.path.join(preprocessed_dir, f))
        for f in required_files
    )
    
    if files_exist:
        print("Preprocessed files already exist.")
        return True
    
    print("Preprocessing data...")
    
    try:
        # Load raw data
        data_path = os.path.join(current_dir, 'data.csv')
        if not os.path.exists(data_path):
            print(f"Error: data.csv not found at {data_path}")
            return False
            
        df = pd.read_csv(data_path)
        print("Data loaded successfully")
        
        # Define features and target
        features = ["CO2", "TVOC", "PM10", "PM2.5", "CO", "LDR", "O3", "Temp", "Hum"]
        
        # Air quality categorization
        def categorize_air_quality(value):
            if 0 <= value <= 50:
                return 0  # "Good"
            elif 51 <= value <= 100:
                return 1  # "Moderate"
            elif 101 <= value <= 150:
                return 2  # "Unhealthy for Sensitive"
            elif 151 <= value <= 200:
                return 3  # "Unhealthy"
            elif 201 <= value <= 400:
                return 4  # "Hazardous"
            else:
                return 2  # Default to middle category
        
        # Create categorical labels
        df['air_quality_category'] = df['Air Quality'].apply(categorize_air_quality)
        
        # Prepare features and target
        X = df[features]
        y = df['air_quality_category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save preprocessed data
        print("\nSaving preprocessed data...")
        np.save(os.path.join(preprocessed_dir, 'X_train.npy'), X_train_scaled)
        np.save(os.path.join(preprocessed_dir, 'X_test.npy'), X_test_scaled)
        np.save(os.path.join(preprocessed_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(preprocessed_dir, 'y_test.npy'), y_test)
        
        print("Data preprocessing completed successfully!")
        print(f"Files saved in: {preprocessed_dir}")
        
        # Print shapes
        print("\nData shapes:")
        print(f"X_train: {X_train_scaled.shape}")
        print(f"X_test: {X_test_scaled.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        return False

if __name__ == "__main__":
    success = prepare_data()
    if success:
        print("\nData preparation completed successfully")
    else:
        print("\nData preparation failed")
        exit(1)