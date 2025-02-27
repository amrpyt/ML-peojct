"""
Model definitions with improved hybrid architecture
Changes:
- Reduced parameters through efficient architecture
- Added pruning and weight clipping
- Support for temporal predictions
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Bidirectional
from tensorflow.keras.layers import MaxPooling1D, Dropout, BatchNormalization
import numpy as np

def create_efficient_cnn_bilstm(input_shape, num_classes):
    """Create efficient CNN-BiLSTM with reduced parameters"""
    model = Sequential([
        # Efficient CNN layers
        Conv1D(32, kernel_size=3, padding='same', activation='relu', 
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding='same'),
        
        Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, padding='same'),
        
        # Single efficient BiLSTM layer
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.3),
        
        # Reduced dense layers
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    return model

def apply_pruning(model, pruning_params):
    """Apply pruning to reduce model size"""
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    
    pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(
        target_sparsity=pruning_params['sparsity'],
        begin_step=0,
        frequency=100
    )
    
    model_for_pruning = prune_low_magnitude(
        model, pruning_schedule=pruning_schedule
    )
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_for_pruning.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model_for_pruning

def apply_weight_clipping(model, clip_value=0.01):
    """Apply weight clipping for stability"""
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            weights = layer.get_weights()
            clipped_weights = [np.clip(w, -clip_value, clip_value) 
                             for w in weights]
            layer.set_weights(clipped_weights)
    return model

class HybridTimeSeriesModel:
    """Hybrid model for time series prediction"""
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        # CNN for feature extraction
        cnn_input = tf.keras.Input(shape=self.input_shape)
        x = Conv1D(32, 3, padding='same', activation='relu')(cnn_input)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2, padding='same')(x)
        
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # BiLSTM for temporal processing
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(16))(x)
        
        # Classification output
        class_output = Dense(self.num_classes, activation='softmax', 
                           name='classification')(x)
        
        # Regression outputs for temp/humidity
        temp_output = Dense(1, name='temperature')(x)
        hum_output = Dense(1, name='humidity')(x)
        
        model = tf.keras.Model(
            inputs=cnn_input,
            outputs=[class_output, temp_output, hum_output]
        )
        
        model.compile(
            optimizer='adam',
            loss={
                'classification': 'sparse_categorical_crossentropy',
                'temperature': 'mse',
                'humidity': 'mse'
            },
            loss_weights={
                'classification': 1.0,
                'temperature': 0.5,
                'humidity': 0.5
            },
            metrics={
                'classification': 'accuracy',
                'temperature': 'mse',
                'humidity': 'mse'
            }
        )
        
        return model
    
    def train(self, X, y_class, y_temp, y_hum, validation_split=0.2, 
              epochs=100, batch_size=32):
        """Train model with all outputs"""
        return self.model.fit(
            X,
            {
                'classification': y_class,
                'temperature': y_temp,
                'humidity': y_hum
            },
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_classification_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_classification_accuracy',
                    factor=0.1,
                    patience=5,
                    min_lr=0.00001
                )
            ]
        )
    
    def predict(self, X):
        """Get predictions for all outputs"""
        return self.model.predict(X)