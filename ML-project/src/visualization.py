import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

class Visualizer:
    @staticmethod
    def plot_training_history(history, title='Training History'):
        """
        Plot training and validation metrics
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt.gcf()

    @staticmethod
    def plot_class_distribution(y, title='Class Distribution'):
        """
        Plot class distribution
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=y)
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Count')
        return plt.gcf()

    @staticmethod
    def plot_feature_importance(model, feature_names):
        """
        Plot feature importance for interpretable models
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importances')
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            return plt.gcf()
        return None

    @staticmethod
    def plot_regression_results(y_true, y_pred, title='Regression Results'):
        """
        Plot actual vs predicted values for regression
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        return plt.gcf()

    @staticmethod
    def plot_optimization_comparison(original_metrics, optimized_metrics, metric_names):
        """
        Compare metrics before and after optimization
        """
        x = np.arange(len(metric_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, original_metrics, width, label='Original')
        ax.bar(x + width/2, optimized_metrics, width, label='Optimized')
        
        ax.set_ylabel('Metric Value')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        
        plt.tight_layout()
        return  fig