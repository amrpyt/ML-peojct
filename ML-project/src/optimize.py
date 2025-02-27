import tensorflow as tf
import numpy as np
import tempfile
import os
from tensorflow.python.keras.models import load_model, clone_model
from tensorflow.python.keras.layers import Dense
from config import MODEL_SAVE_PATH, PRUNING_PARAMS, QUANTIZATION_PARAMS

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
    """Apply optimization techniques to the best model"""
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
    
    # Apply weight clipping for stability - this doesn't require TFMOT
    clipped_model = apply_weight_clipping(
        best_model,
        model_name=f"{best_model_name}_clipped"
    )
    
    # Optional: Apply knowledge distillation - this doesn't require TFMOT
    distilled_model, _ = apply_knowledge_distillation(
        clipped_model,
        X_train,
        y_train,
        model_name=f"{best_model_name}_distilled"
    )
    
    # Return the optimized model
    return clipped_model
