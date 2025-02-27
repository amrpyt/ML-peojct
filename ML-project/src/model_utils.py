"""
Utility functions for improved model training.
Changes:
- Added plotting functions for multi-task learning
- Added metrics calculation helpers
- Added model comparison utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

def plot_training_history(history, save_path=None):
    """Plot training metrics for all tasks."""
    plt.figure(figsize=(15, 5))
    
    # Classification accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['classification_accuracy'], label='Training')
    plt.plot(history.history['val_classification_accuracy'], label='Validation')
    plt.title('Classification Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Temperature MSE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['temperature_mse'], label='Training')
    plt.plot(history.history['val_temperature_mse'], label='Validation')
    plt.title('Temperature MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    # Humidity MSE
    plt.subplot(1, 3, 3)
    plt.plot(history.history['humidity_mse'], label='Training')
    plt.plot(history.history['val_humidity_mse'], label='Validation')
    plt.title('Humidity MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_regression_results(actual, predicted, title, save_path=None):
    """Plot regression predictions vs actual values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # Add R² value
    r2 = np.corrcoef(actual.flatten(), predicted.flatten())[0, 1]**2
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix with improved visualization."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def calculate_metrics(y_true, y_pred, task_type='classification'):
    """Calculate and return metrics based on task type."""
    if task_type == 'classification':
        return {
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'report': classification_report(y_true, y_pred, output_dict=True)
        }
    else:  # regression
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

def compare_models(original_metrics, improved_metrics):
    """Compare and display metrics between original and improved models."""
    print("Model Comparison:")
    print("-" * 50)
    
    # Classification comparison
    print("\nClassification Metrics:")
    print("Original Accuracy:", original_metrics['classification']['accuracy'])
    print("Improved Accuracy:", improved_metrics['classification']['accuracy'])
    
    # Temperature prediction comparison
    print("\nTemperature Prediction:")
    print(f"Original MSE: {original_metrics['temperature']['mse']:.4f}")
    print(f"Improved MSE: {improved_metrics['temperature']['mse']:.4f}")
    print(f"Improvement: {((original_metrics['temperature']['mse'] - improved_metrics['temperature']['mse']) / original_metrics['temperature']['mse'] * 100):.2f}%")
    
    # Humidity prediction comparison
    print("\nHumidity Prediction:")
    print(f"Original MSE: {original_metrics['humidity']['mse']:.4f}")
    print(f"Improved MSE: {improved_metrics['humidity']['mse']:.4f}")
    print(f"Improvement: {((original_metrics['humidity']['mse'] - improved_metrics['humidity']['mse']) / original_metrics['humidity']['mse'] * 100):.2f}%")

def get_gpu_memory_usage():
    """Monitor GPU memory usage if available."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            return {
                'current': memory_info['current'] / 1024**2,  # Convert to MB
                'peak': memory_info['peak'] / 1024**2
            }
    except:
        pass
    return None