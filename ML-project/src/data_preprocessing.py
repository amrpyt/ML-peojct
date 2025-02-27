import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def detect_replace_outliers(self, df, columns, method='zscore', threshold=3):
        """
        Detect and replace outliers in specified columns
        method: 'zscore' or 'iqr'
        """
        df_copy = df.copy()
        
        for column in columns:
            if method == 'zscore':
                z_scores = np.abs((df_copy[column] - df_copy[column].mean()) / df_copy[column].std())
                outliers = z_scores > threshold
                df_copy.loc[outliers, column] = df_copy[column].median()
            
            elif method == 'iqr':
                Q1 = df_copy[column].quantile(0.25)
                Q3 = df_copy[column].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (df_copy[column] < (Q1 - 1.5 * IQR)) | (df_copy[column] > (Q3 + 1.5 * IQR))
                df_copy.loc[outliers, column] = df_copy[column].median()
        
        return df_copy

    def normalize_features(self, X_train, X_test):
        """
        Normalize numerical features using StandardScaler
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def balance_classes(self, X, y):
        """
        Balance classes using SMOTE
        """
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        return X_resampled, y_resampled

    def categorize_air_quality(self, value):
        """
        Assign labels based on air quality ranges
        """
        if value <= 50:
            return 0  # Good
        elif value <= 100:
            return 1  # Moderate
        elif value <= 150:
            return 2  # Unhealthy for Sensitive Groups
        elif value <= 200:
            return 3  # Unhealthy
        elif value <= 300:
            return 4  # Very Unhealthy
        else:
            return 5  # Hazardous

    def prepare_data(self, df):
        # Dummy implementation: split 80/20
        train_size = int(len(df) * 0.8)
        X_train = df.iloc[:train_size, :-1].values
        y_train = df.iloc[:train_size, -1].values
        X_test = df.iloc[train_size:, :-1].values
        y_test = df.iloc[train_size:, -1].values
        return X_train, X_test, y_train, y_test