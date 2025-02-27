import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Import project modules
from config import DATA_PATH, MODEL_SAVE_PATH, RESULTS_PATH
from utils import create_directories, plot_training_history, save_model_summary
from preprocess import preprocess_pipeline
from models import build_and_train_all_models
from evaluate import evaluate_all_models
from optimize import optimize_model
from predict import evaluate_optimized_model, predict_with_best_model

def main():
    """Main execution function"""
    print("Starting Air Quality Analysis Project")
    
    # Create necessary directories
    create_directories()
    
    # Step 1: Data preprocessing
    print("\n1. Preprocessing data...")
    data_dict = preprocess_pipeline(DATA_PATH)
    df = data_dict['df']
    df_normalized = data_dict['df_normalized']
    scaler = data_dict['scaler']
    label_encoder = data_dict['label_encoder']
    class_data = data_dict['class_data']
    reg_data = data_dict['reg_data']
    
    # Print class distribution
    print("\nClass distribution before balancing:")
    class_dist = df['AQ_Category'].value_counts()
    print(class_dist)
    
    # Step 2: Build and train all models
    print("\n2. Training models...")
    models, histories = build_and_train_all_models(class_data)
    
    # Plot training histories
    for model_name, history in histories.items():
        plot_training_history(history, model_name)
        save_model_summary(models[model_name], model_name)
    
    # Step 3: Evaluate all models and find the best one
    print("\n3. Evaluating models...")
    best_model_name, best_model, results_df = evaluate_all_models(models, class_data, label_encoder)
    
    # Print evaluation results
    print("\nModel Evaluation Results:")
    print(results_df)
    
    # Step 4: Optimize the best model
    print(f"\n4. Optimizing best model: {best_model_name}...")
    optimized_model = optimize_model(best_model, best_model_name, class_data)
    
    # Step 5: Evaluate optimized model
    print("\n5. Evaluating optimized model...")
    opt_metrics = evaluate_optimized_model(optimized_model, class_data, label_encoder, best_model_name)
    
    # Compare original vs optimized model
    print("\nComparison of Original vs Optimized Model:")
    original_metrics = results_df[results_df['model_name'] == best_model_name].iloc[0].to_dict()
    
    comparison = pd.DataFrame({
        'Metric': list(original_metrics.keys()),
        'Original': list(original_metrics.values()),
        'Optimized': [opt_metrics[k] for k in original_metrics.keys()]
    })
    print(comparison)
    
    # Save comparison
    comparison.to_csv(os.path.join(RESULTS_PATH, 'model_comparison.csv'), index=False)
    
    # Step 6: Use the optimized model for regression task
    print("\n6. Training regression model for temperature and humidity prediction...")
    regression_model, reg_metrics = predict_with_best_model(
        optimized_model, reg_data, best_model_name
    )
    
    print("\nProject completed successfully!")

if __name__ == "__main__":
    main()