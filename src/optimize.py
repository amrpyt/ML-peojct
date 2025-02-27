import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import tempfile
import os
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Dense
from config import MODEL_SAVE_PATH, PRUNING_PARAMS, QUANTIZATION_PARAMS

def apply_pruning(model, X_train, y_train, X_val=None, y_val=None, model_name='pruned_model', epochs=50, batch_size=32):
    """Apply weight pruning to reduce model size"""
    print("Applying pruning...")
    
    # Clone the model to avoid modifying the original
    pruning_model = clone_model(model)
    pruning_model.set_weights(model.get_weights())
    
    # Compile the model with the same optimizer and loss
    pruning_model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=['accuracy']
    )
    
    # Define pruning schedule
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=PRUNING_PARAMS['initial_sparsity'],
            final_sparsity=PRUNING_PARAMS['final_sparsity'],
            begin_step=PRUNING_PARAMS['begin_step'],
            end_step=PRUNING_PARAMS['end_step']
        )
    }
    
    # Apply pruning to the model
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(pruning_model, **pruning_params)
    
    # Compile the pruned model
    pruned_model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=['accuracy']
    )
    
    # Create pruning callbacks
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=f"{MODEL_SAVE_PATH}/pruning_logs"),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    # Train the pruned model
    if X_val is None or y_val is None:
        history = pruned_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks
        )
    else:
        history = pruned_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
    
    # Remove pruning wrappers for deployment
    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    # Save the model
    final_model.save(os.path.join(MODEL_SAVE_PATH, f"{model_name}.h5"))
    
    return final_model, history

def apply_quantization(model, model_name='quantized_model', is_post_training=True):
    """Apply quantization to reduce model size"""
    print("Applying quantization...")
    
    if is_post_training:
        # Apply post-training quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Apply additional optimizations
        if QUANTIZATION_PARAMS['quantize_input']:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.representative_dataset = _get_representative_dataset
            if QUANTIZATION_PARAMS['quantize_output']:
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        
        # Convert the model
        quantized_model = converter.convert()
        
        # Save the model
        with open(os.path.join(MODEL_SAVE_PATH, f"{model_name}.tflite"), 'wb') as f:
            f.write(quantized_model)
        
        return quantized_model
    else:
        # Apply quantization-aware training
        # Clone the model
        quantized_model = clone_model(model)
        quantized_model.set_weights(model.get_weights())
        
        # Apply quantization-aware training
        quant_aware_model = tfmot.quantization.keras.quantize_model(quantized_model)
        
        # Compile the model
        quant_aware_model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=['accuracy']
        )
        
        return quant_aware_model

def _get_representative_dataset():
    """Generate representative dataset for quantization"""
    # This is a placeholder function
    # In a real implementation, you would provide real data
    def generator():
        for _ in range(100):
            # Generate random data that matches your model's input shape
            yield [np.random.random((1, 7)).astype(np.float32)]
    return generator

def apply_clustering(model, X_train, y_train, X_val=None, y_val=None, model_name='clustered_model', epochs=50, batch_size=32):
    """Apply weight clustering to reduce model size"""
    print("Applying clustering...")
    
    # Define clustering parameters
    clustering_params = {
        'number_of_clusters': 16,
        'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS
    }
    
    # Apply clustering to the model
    clustered_model = tfmot.clustering.keras.cluster_weights(
        clone_model(model),
        **clustering_params
    )
    
    # Copy original model weights to the clustered model
    clustered_model.set_weights(model.get_weights())
    
    # Compile the model
    clustered_model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=['accuracy']
    )
    
    # Create clustering callbacks
    callbacks = [
        tfmot.clustering.keras.ClusteringSummaries(log_dir=f"{MODEL_SAVE_PATH}/clustering_logs"),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    # Train the clustered model
    if X_val is None or y_val is None:
        history = clustered_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks
        )
    else:
        history = clustered_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
    
    # Strip clustering for deployment
    final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
    
    # Save the model
    final_model.save(os.path.join(MODEL_SAVE_PATH, f"{model_name}.h5"))
    
    return final_model, history

def apply_weight_clipping(model, clip_value=0.01, model_name='weight_clipped_model'):
    """Apply weight clipping to improve model stability"""
    print("Applying weight clipping...")
    
    # Clone the model
    clipped_model = clone_model(model)
    clipped_model.set_weights(model.get_weights())
    
    # Get and clip weights
    weights = clipped_model.get_weights()
    clipped_weights = []
    
    for w in weights:
        clipped_w = np.clip(w, -clip_value, clip_value)
        clipped_weights.append(clipped_w)
    
    # Set clipped weights
    clipped_model.set_weights(clipped_weights)
    
    # Compile the model
    clipped_model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=['accuracy']
    )
    
    # Save the model
    clipped_model.save(os.path.join(MODEL_SAVE_PATH, f"{model_name}.h5"))
    
    return clipped_model

def apply_knowledge_distillation(teacher_model, X_train, y_train, X_val=None, y_val=None, model_name='distilled_model', 
                               temperature=5.0, alpha=0.5, epochs=50, batch_size=32):
    """Apply knowledge distillation to create a smaller student model"""
    print("Applying knowledge distillation...")
    
    # Define a smaller student model
    # This is a simplified version of the teacher model
    input_shape = teacher_model.input_shape[1:]
    output_shape = teacher_model.output_shape[1]
    
    student_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    
    # Custom loss function for knowledge distillation
    def knowledge_distillation_loss(y_true, y_pred):
        # Get soft targets from teacher model
        teacher_preds = teacher_model(student_model.input, training=False)
        
        # Apply temperature scaling
        teacher_preds = tf.nn.softmax(teacher_preds / temperature)
        y_pred_soft = tf.nn.softmax(y_pred / temperature)
        
        # Calculate distillation loss and student loss
        distillation_loss = tf.keras.losses.categorical_crossentropy(teacher_preds, y_pred_soft)
        student_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Combine losses
        return alpha * student_loss + (1 - alpha) * distillation_loss * (temperature ** 2)
    
    # Compile student model with custom loss
    student_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=knowledge_distillation_loss,
        metrics=['accuracy']
    )
    
    # Train student model
    if X_val is None or y_val is None:
        history = student_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
        )
    else:
        history = student_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
        )
    
    # Save the student model
    student_model.save(os.path.join(MODEL_SAVE_PATH, f"{model_name}.h5"))
    
    return student_model, history

def optimize_model(best_model, best_model_name, class_data):
    """Apply multiple optimization techniques to the best model"""
    print(f"Optimizing model: {best_model_name}")
    
    # Prepare data based on model type
    if best_model_name in ['1dcnn', 'rnn', 'lstm', 'bilstm']:
        X_train = class_data['X_train_seq']
        X_test = class_data['X_test_seq']
    else:
        X_train = class_data['X_train']
        X_test = class_data['X_test']
    
    y_train = class_data['y_train_cat']
    y_test = class_data['y_test_cat']
    
    # First, apply pruning to reduce model size
    pruned_model, _ = apply_pruning(
        best_model, 
        X_train, 
        y_train,
        model_name=f"{best_model_name}_pruned"
    )
    
    # Next, apply clustering for further compression
    clustered_model, _ = apply_clustering(
        pruned_model,
        X_train,
        y_train,
        model_name=f"{best_model_name}_pruned_clustered"
    )
    
    # Apply weight clipping for stability
    clipped_model = apply_weight_clipping(
        clustered_model,
        model_name=f"{best_model_name}_pruned_clustered_clipped"
    )
    
    # Apply quantization as the final step
    quantized_model = apply_quantization(
        clipped_model,
        model_name=f"{best_model_name}_pruned_clustered_clipped_quantized"
    )
    
    # Optional: Apply knowledge distillation
    # This creates a smaller model guided by the optimized model
    distilled_model, _ = apply_knowledge_distillation(
        clipped_model,  # Use pre-quantized model as teacher
        X_train,
        y_train,
        model_name=f"{best_model_name}_distilled"
    )
    
    # Return the optimized model
    return clipped_model  # Return the pre-quantized model for evaluation
