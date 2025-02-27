import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from config import MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, VALIDATION_SPLIT

def create_dnn_model(input_dim, n_classes):
    """Create a Deep Neural Network model"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_1d_cnn_model(input_shape, n_classes):
    """Create a 1D CNN model"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_rnn_model(input_shape, n_classes):
    """Create a simple RNN model"""
    model = Sequential([
        tf.keras.layers.SimpleRNN(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.SimpleRNN(32),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lstm_model(input_shape, n_classes):
    """Create an LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_bilstm_model(input_shape, n_classes):
    """Create a Bidirectional LSTM model"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32)),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_regression_model(input_dim):
    """Create a regression model for temperature and humidity prediction"""
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    
    # Two outputs: temperature and humidity
    temp_output = Dense(1, name='temperature')(x)
    humid_output = Dense(1, name='humidity')(x)
    
    model = Model(inputs=inputs, outputs=[temp_output, humid_output])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={'temperature': 'mse', 'humidity': 'mse'},
        metrics={'temperature': ['mae', 'mse'], 'humidity': ['mae', 'mse']}
    )
    
    return model

def train_model(model, X_train, y_train, X_val=None, y_val=None, model_name='model', callbacks=None):
    """Train a model with early stopping and checkpointing"""
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    
    # Default callbacks if none provided
    if callbacks is None:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(MODEL_SAVE_PATH, f"{model_name}.h5"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
    
    # Use validation_split if validation data not provided
    if X_val is None or y_val is None:
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    return history

def build_and_train_all_models(class_data):
    """Build and train all classification models"""
    input_dim = class_data['X_train'].shape[1]
    seq_input_shape = class_data['X_train_seq'].shape[1:]
    n_classes = class_data['n_classes']
    
    models = {}
    histories = {}
    
    # Train DNN
    dnn_model = create_dnn_model(input_dim, n_classes)
    dnn_history = train_model(
        dnn_model, 
        class_data['X_train'], 
        class_data['y_train_cat'],
        model_name='dnn'
    )
    models['dnn'] = dnn_model
    histories['dnn'] = dnn_history
    
    # Train 1D CNN
    cnn_model = create_1d_cnn_model(seq_input_shape, n_classes)
    cnn_history = train_model(
        cnn_model, 
        class_data['X_train_seq'], 
        class_data['y_train_cat'],
        model_name='1dcnn'
    )
    models['1dcnn'] = cnn_model
    histories['1dcnn'] = cnn_history
    
    # Train RNN
    rnn_model = create_rnn_model(seq_input_shape, n_classes)
    rnn_history = train_model(
        rnn_model, 
        class_data['X_train_seq'], 
        class_data['y_train_cat'],
        model_name='rnn'
    )
    models['rnn'] = rnn_model
    histories['rnn'] = rnn_history
    
    # Train LSTM
    lstm_model = create_lstm_model(seq_input_shape, n_classes)
    lstm_history = train_model(
        lstm_model, 
        class_data['X_train_seq'], 
        class_data['y_train_cat'],
        model_name='lstm'
    )
    models['lstm'] = lstm_model
    histories['lstm'] = lstm_history
    
    # Train BiLSTM
    bilstm_model = create_bilstm_model(seq_input_shape, n_classes)
    bilstm_history = train_model(
        bilstm_model, 
        class_data['X_train_seq'], 
        class_data['y_train_cat'],
        model_name='bilstm'
    )
    models['bilstm'] = bilstm_model
    histories['bilstm'] = bilstm_history
    
    return models, histories