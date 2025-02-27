import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_confusion_matrix, plot_training_history
import math
import os
from config import RESULTS_PATH

def evaluate_classification_model(model, X_test, y_test, y_test_cat, label_encoder, model_name):
    """Evaluate a classification model and return metrics"""
    # Predict using the model
    if len(X_test.shape) == 2 and model_name in ['1dcnn', 'rnn', 'lstm', 'bilstm']:
        # Reshape for sequence models if needed
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Get class labels for better visualization
    class_names = label_encoder.classes_
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names, model_name)
    
    # Format results
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }
    
    return metrics, y_pred

def evaluate_regression_model(model, X_test, y_temp_test, y_hum_test, model_name):
    """Evaluate a regression model for temperature and humidity prediction"""
    # Predict
    y_pred = model.predict(X_test)
    y_temp_pred, y_hum_pred = y_pred[0], y_pred[1]
    
    # Calculate metrics for temperature
    temp_mse = mean_squared_error(y_temp_test, y_temp_pred)
    temp_rmse = math.sqrt(temp_mse)
    temp_mae = mean_absolute_error(y_temp_test, y_temp_pred)
    temp_r2 = r2_score(y_temp_test, y_temp_pred)
    
    # Calculate metrics for humidity
    hum_mse = mean_squared_error(y_hum_test, y_hum_pred)
    hum_rmse = math.sqrt(hum_mse)
    hum_mae = mean_absolute_error(y_hum_test, y_hum_pred)
    hum_r2 = r2_score(y_hum_test, y_hum_pred)
    
    # Format results
    metrics = {
        'model_name': model_name,
        'temp_mse': temp_mse,
        'temp_rmse': temp_rmse,
        'temp_mae': temp_mae,
        'temp_r2': temp_r2,
        'hum_mse': hum_mse,
        'hum_rmse': hum_rmse,
        'hum_mae': hum_mae,
        'hum_r2': hum_r2
    }
    
    return metrics, (y_temp_pred, y_hum_pred)

def evaluate_all_models(models, class_data, label_encoder):
    """Evaluate all models and return the best one"""
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    
    results = []
    
    for model_name, model in models.items():
        # Prepare test data based on model type
        if model_name in ['1dcnn', 'rnn', 'lstm', 'bilstm']:
            X_test = class_data['X_test_seq']
        else:
            X_test = class_data['X_test']
        
        metrics, _ = evaluate_classification_model(
            model, 
            X_test, 
            class_data['y_test'], 
            class_data['y_test_cat'], 
            label_encoder, 
            model_name
        )
        
        results.append(metrics)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(os.path.join(RESULTS_PATH, 'classification_results.csv'), index=False)
    
    # Find best model based on F1 score (weighted)
    best_idx = results_df['f1_weighted'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model_name']
    best_model = models[best_model_name]
    
    print(f"Best model: {best_model_name} with F1 score: {results_df.loc[best_idx, 'f1_weighted']:.4f}")
    
    return best_model_name, best_model, results_df