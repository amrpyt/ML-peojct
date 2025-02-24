import pandas as pd
from data_preprocessing import DataPreprocessor
from models import ModelBuilder
from model_optimizer import ModelOptimizer
from visualization import Visualizer
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def train_evaluate_model(model, X_train, X_test, y_train, y_test, epochs=50):
    """
    Train and evaluate a model
    """
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=0.2,
        batch_size=32,
        verbose=1
    )

    y_pred = model.predict(X_test)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    return model, history, metrics

def main():
    # Load data
    df = pd.read_csv('data.csv')

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)

    # Get indices for regression data
    train_size = int(len(df) * 0.8)
    train_indices = np.arange(train_size)
    test_indices = np.arange(train_size, len(df))

    # Initialize model builder
    model_builder = ModelBuilder()

    # Train different models
    models = {
        'DNN': model_builder.build_dnn(X_train.shape[1:], 6),
        'CNN': model_builder.build_cnn((X_train.shape[1], 1), 6),
        'LSTM': model_builder.build_lstm((X_train.shape[1], 1), 6),
        'BiLSTM': model_builder.build_bilstm((X_train.shape[1], 1), 6)
    }

    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model, history, metrics = train_evaluate_model(
            model, X_train, X_test, y_train, y_test
        )
        results[name] = {
            'model': model,
            'history': history,
            'metrics': metrics
        }
        
        # Visualize training
        viz = Visualizer()
        viz.plot_training_history(history, title=f'{name} Training History')
        
    # Select best model based on accuracy
    best_model_name = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])[0]
    best_model = results[best_model_name]['model']

    # Optimize best model
    optimizer = ModelOptimizer(best_model)

    # Apply optimization techniques
    print("\nOptimizing best model...")
    pruned_model = optimizer.apply_pruning(
        (X_train, y_train),
        (X_test, y_test)
    )

    quantized_model = optimizer.apply_quantization()

    clustered_model = optimizer.apply_clustering(
        (X_train, y_train),
        (X_test, y_test)
    )

    # Evaluate optimized model
    _, _, optimized_metrics = train_evaluate_model(
        clustered_model, X_train, X_test, y_train, y_test
    )

    # Visualize optimization results
    viz.plot_optimization_comparison(
        [results[best_model_name]['metrics'][m] for m in ['accuracy', 'precision', 'recall', 'f1']],
        [optimized_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1']],
        ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    )

    # Train regression model for temp and hum
    regression_model = model_builder.build_regression_model(X_train.shape[1:], 2)
    regression_history = regression_model.fit(
        X_train, df[['temp', 'hum']].values[train_indices],
        epochs=50,
        validation_split=0.2,
        batch_size=32,
        verbose=1
    )

    # Evaluate regression results
    reg_predictions = regression_model.predict(X_test)
    regression_metrics = {
        'temp_mse': mean_squared_error(df[['temp']].values[test_indices], reg_predictions[:, 0]),
        'temp_rmse': np.sqrt(mean_squared_error(df[['temp']].values[test_indices], reg_predictions[:, 0])),
        'temp_r2': r2_score(df[['temp']].values[test_indices], reg_predictions[:, 0]),
        'temp_mae': mean_absolute_error(df[['temp']].values[test_indices], reg_predictions[:, 0]),
        'hum_mse': mean_squared_error(df[['hum']].values[test_indices], reg_predictions[:, 1]),
        'hum_rmse': np.sqrt(mean_squared_error(df[['hum']].values[test_indices], reg_predictions[:, 1])),
        'hum_r2': r2_score(df[['hum']].values[test_indices], reg_predictions[:, 1]),
        'hum_mae': mean_absolute_error(df[['hum']].values[test_indices], reg_predictions[:, 1])
    }

    print("\nRegression Metrics:")
    for metric, value in regression_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()