"""
Data preprocessing module for air quality analysis
Changes:
- Updated air quality categorization
- Fixed SMOTE implementation
- Added sliding window preparation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self, window_size=60):
        self.scaler = StandardScaler()
        self.window_size = window_size
        
    def categorize_air_quality(self, value):
        """Updated air quality categorization"""
        if value <= 50:
            return 'Good'
        elif 51 <= value <= 100:
            return 'Moderate'
        elif 101 <= value <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif 151 <= value <= 200:
            return 'Unhealthy'
        elif 201 <= value <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'

    def create_sequences(self, data, target):
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:(i + self.window_size)])
            y.append(target[i + self.window_size])
        return np.array(X), np.array(y)

    def prepare_data(self, df, features, target_class):
        """Prepare data with proper balancing"""
        # Convert categories
        df['air_quality_category'] = df[target_class].apply(self.categorize_air_quality)
        
        # Create sequences for temporal data
        X_seq, y_seq = self.create_sequences(df[features].values, 
                                           df[['Temp', 'Hum']].values)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], df['air_quality_category'],
            test_size=0.2, random_state=42, stratify=df['air_quality_category']
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE with proper strategy
        smote = SMOTE(random_state=42, sampling_strategy='not majority')
        X_balanced, y_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        return {
            'X_train': X_balanced,
            'X_test': X_test_scaled,
            'y_train': y_balanced,
            'y_test': y_test,
            'X_seq': X_seq,
            'y_seq': y_seq
        }