import tensorflow as tf
from tensorflow.keras import layers, models

class ModelBuilder:
    def build_dnn(self, input_shape, num_classes):
        model = models.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_cnn(self, input_shape, num_classes):
        model = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_lstm(self, input_shape, num_classes):
        model = models.Sequential([
            layers.LSTM(64, input_shape=input_shape),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_bilstm(self, input_shape, num_classes):
        model = models.Sequential([
            layers.Bidirectional(layers.LSTM(64, input_shape=input_shape)),
            layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_regression_model(self, input_shape, output_units):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=input_shape),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_units)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model