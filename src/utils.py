import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

def create_directories():
    """Create necessary directories for the project"""
    dirs = [
        r"d:\Work\Client Projects\Paid projects\ML peojct\models",
        r"d:\Work\Client Projects\Paid projects\ML peojct\results",
        r"d:\Work\Client Projects\Paid projects\ML peojct\logs"
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def categorize_air_quality(value):
    """Categorize air quality values into classes"""
    if 0 <= value <= 50:
        return 'Good'
    elif 51 <= value <= 100:
        return 'Moderate'
    elif 101 <= value <= 150:
        return 'Unhealthy for Sensitive'
    elif 151 <= value <= 200:
        return 'Unhealthy'
    elif 201 <= value <= 400:
        return 'Hazardous'
    else:
        return 'Unknown'

def plot_training_history(history, model_name):
    """Plot training and validation loss and accuracy"""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"d:\\Work\\Client Projects\\Paid projects\\ML peojct\\results\\{model_name}_training_history.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"d:\\Work\\Client Projects\\Paid projects\\ML peojct\\results\\{model_name}_confusion_matrix.png")
    plt.close()

def plot_regression_results(y_true, y_pred, target_name, model_name):
    """Plot regression results"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.title(f'{model_name} - {target_name} Prediction')
    plt.xlabel(f'Actual {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.tight_layout()
    plt.savefig(f"d:\\Work\\Client Projects\\Paid projects\\ML peojct\\results\\{model_name}_{target_name}_prediction.png")
    plt.close()

def save_model_summary(model, model_name):
    """Save model summary to a text file"""
    with open(f"d:\\Work\\Client Projects\\Paid projects\\ML peojct\\results\\{model_name}_summary.txt", 'w') as f:
        # Save a string representation of the model
        model.summary(print_fn=lambda x: f.write(x + '\n'))
