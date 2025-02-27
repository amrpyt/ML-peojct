import tensorflow as tf
from tensorflow import keras
import numpy as np

class ModelOptimizer:
    def __init__(self, model):
        self.model = model

    def apply_pruning(self, train_data, test_data):
        """
        Apply pruning to reduce model size
        """
        import tensorflow_model_optimization as tfmot
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        
        # Define pruning parameters
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=0.5,
                begin_step=0,
                frequency=100
            )
        }
        
        model_for_pruning = prune_low_magnitude(self.model, **pruning_params)
        model_for_pruning.compile(
            optimizer='adam',
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        # Train the pruned model
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        model_for_pruning.fit(
            train_data[0], train_data[1],
            epochs=10,
            validation_data=test_data,
            callbacks=callbacks
        )
        
        # Strip pruning wrappers
        self.model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        print("Pruning applied")
        return self.model

    def apply_quantization(self):
        """
        Apply post-training quantization
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        quantized_tflite_model = converter.convert()
        print("Quantization applied")
        return quantized_tflite_model

    def apply_clustering(self, train_data, test_data, num_clusters=8, epochs=10):
        """
        Apply weight clustering
        """
        import tensorflow_model_optimization as tfmot
        cluster_weights = tfmot.clustering.keras.cluster_weights
        clustering_params = {
            'number_of_clusters': num_clusters,
            'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.LINEAR
        }
        
        clustered_model = cluster_weights(self.model, **clustering_params)
        clustered_model.compile(
            optimizer='adam',
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        # Train the clustered model
        clustered_model.fit(
            train_data[0], train_data[1],
            epochs=epochs,
            validation_data=test_data
        )
        
        self.model = clustered_model
        print("Clustering applied")
        return self.model

    def apply_weight_clipping(self, clip_value=0.01):
        """
        Apply weight clipping to prevent extreme values
        """
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.get_weights()
                clipped_weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                layer.set_weights(clipped_weights)
        
        return self.model