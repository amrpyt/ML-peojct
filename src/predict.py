import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import os
from utils import categorize_air_quality, plot_regression_results
from evaluate import evaluate_classification_model, evaluate_regression_model
from config import MODEL_SAVE_PATH, FEATURES, TARGET_CLASS, REGRESSION_TARGETS

def load_optimized_model(model_path):
    """Load a trained model from disk"""
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_input_data(df, scaler, input_features=FEATURES):
    """Prepare input data for model prediction"""
    # Extract features
    X = df[input_features].values
    
    # Scale features if a scaler is provided
    if scaler is not None:
        X = scaler.transform(X)
    
    return X

def classify_air_quality(model, X, label_encoder, is_sequence_model=False):
    """Classify air quality using the trained model"""
    # Reshape input for sequence models if needed
    if is_sequence_model:
        X = X.reshape(X.shape[0], 1, X.shape[1])
    
    # Make predictions
    y_pred_prob = model.predict(X)
    y_pred_class = np.argmax(y_pred_prob, axis=1)
    
    # Convert to human-readable categories
    if label_encoder is not None:
        y_pred_categories = label_encoder.inverse_transform(y_pred_class)
    else:
        y_pred_categories = [categorize_air_quality(val) for val in y_pred_class]
    
    return y_pred_prob, y_pred_class, y_pred_categories

def predict_temp_hum(model, X):
    """Predict temperature and humidity using the trained model"""
    # Make predictions
    predictions = model.predict(X)
    
    # Extract temperature and humidity predictions
    if isinstance(predictions, list):
        temp_pred, hum_pred = predictions
    else:
        temp_pred = predictions[:, 0]
        hum_pred = predictions[:, 1]
    
    return temp_pred, hum_pred

def evaluate_optimized_model(optimized_model, class_data, label_encoder, best_model_name):
    """Evaluate the performance of the optimized model"""
    print("Evaluating optimized model...")
    
    # Prepare test data based on model type
    if best_model_name in ['1dcnn', 'rnn', 'lstm', 'bilstm']:
        X_test = class_data['X_test_seq']
        is_sequence_model = True
    else:
        X_test = class_data['X_test']
        is_sequence_model = False
    
    # Evaluate classification performance
    metrics, y_pred = evaluate_classification_model(
        optimized_model,
        X_test,
        class_data['y_test'],
        class_data['y_test_cat'],
        label_encoder,
        f"{best_model_name}_optimized"
    )
    
    print("Optimized Model Classification Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return metrics

def predict_with_best_model(optimized_model, reg_data, best_model_name):
    """Use the best model to predict temperature and humidity"""
    print("Predicting temperature and humidity with optimized model...")
    
    # Create a regression head for the base model
    base_model = tf.keras.Model(
        inputs=optimized_model.input,
        outputs=optimized_model.layers[-2].output  # Get the layer before softmax
    )
    
    # Add regression heads
    inputs = base_model.input
    features = base_model.output
    
    temp_output = tf.keras.layers.Dense(1, name='temperature')(features)
    hum_output = tf.keras.layers.Dense(1, name='humidity')(features)
    
    regression_model = tf.keras.Model(inputs=inputs, outputs=[temp_output, hum_output])
    
    # Compile the model
    regression_model.compile(
        optimizer='adam',
        loss={'temperature': 'mse', 'humidity': 'mse'},
        metrics={'temperature': ['mae', 'mse'], 'humidity': ['mae', 'mse']}
    )
    
    # Prepare data based on model type
    if best_model_name in ['1dcnn', 'rnn', 'lstm', 'bilstm']:
        X_train = reg_data['X_train_reg_seq']
        X_test = reg_data['X_test_reg_seq']
    else:
        X_train = reg_data['X_train_reg']
        X_test = reg_data['X_test_reg']
    
    # Train the regression model
    history = regression_model.fit(
        X_train, 
        [reg_data['y_temp_train'], reg_data['y_hum_train']],
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluate the regression model
    metrics, (temp_pred, hum_pred) = evaluate_regression_model(
        regression_model,
        X_test,
        reg_data['y_temp_test'],
        reg_data['y_hum_test'],
        f"{best_model_name}_regression"
    )
    
    print("Regression Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Plot results
    plot_regression_results(
        reg_data['y_temp_test'],
        temp_pred,
        'Temperature',
        f"{best_model_name}_regression"
    )
    
    plot_regression_results(
        reg_data['y_hum_test'],
        hum_pred,
        'Humidity',
        f"{best_model_name}_regression"
    )
    
    # Save the regression model
    regression_model.save(os.path.join(MODEL_SAVE_PATH, f"{best_model_name}_regression.h5"))
    
    return regression_model, metrics
