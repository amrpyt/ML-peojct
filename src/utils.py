import os
import time
import psutil
import numpy as np
import tensorflow as tf

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def measure_model_metrics(model, X_test, batch_size=32, num_runs=100):
    """Measure model performance metrics including memory and speed."""
    metrics = {}
    
    # Memory usage
    base_memory = get_memory_usage()
    model.predict(X_test[:1], verbose=0)  # Warm up
    peak_memory = get_memory_usage()
    metrics['memory_usage'] = peak_memory - base_memory
    
    # Inference time
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        model.predict(X_test[:batch_size], verbose=0)
        times.append(time.time() - start_time)
    
    metrics['inference_time'] = {
        'mean': np.mean(times) * 1000,  # Convert to ms
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000
    }
    
    # Model size
    tf.keras.backend.clear_session()
    temp_path = 'temp_model.h5'
    model.save(temp_path)
    metrics['model_size'] = os.path.getsize(temp_path) / 1024  # KB
    os.remove(temp_path)
    
    return metrics

def calculate_efficiency_score(accuracy, size_kb, inference_time):
    """Calculate model efficiency score.
    
    Higher score means better efficiency (high accuracy, low size, fast inference).
    """
    # Normalize factors
    size_factor = np.log(size_kb + 1)  # Log scale for size
    time_factor = np.log(inference_time + 1)  # Log scale for time
    
    # Efficiency = accuracy / (size * time)
    # Higher accuracy and lower size/time = better efficiency
    return (accuracy * 1000) / (size_factor * time_factor)

def format_metrics(metrics):
    """Format metrics for display."""
    return {
        'Memory Usage (MB)': f"{metrics['memory_usage']:.2f}",
        'Inference Time (ms)': f"{metrics['inference_time']['mean']:.2f} ± {metrics['inference_time']['std']:.2f}",
        'Model Size (KB)': f"{metrics['model_size']:.2f}"
    }

def optimize_batch_size(model, X_test, start_size=32, max_size=512):
    """Find optimal batch size for inference."""
    batch_sizes = []
    times = []
    memory_usage = []
    
    for batch_size in range(start_size, max_size + 1, 32):
        # Measure time
        start = time.time()
        model.predict(X_test[:batch_size], verbose=0)
        end = time.time()
        inference_time = (end - start) * 1000  # ms
        
        # Measure memory
        mem = get_memory_usage()
        
        batch_sizes.append(batch_size)
        times.append(inference_time)
        memory_usage.append(mem)
        
        # Stop if memory usage increases significantly
        if len(memory_usage) > 1:
            if memory_usage[-1] > memory_usage[-2] * 1.5:
                break
    
    # Find optimal point (minimum time per sample)
    time_per_sample = [t/b for t, b in zip(times, batch_sizes)]
    optimal_idx = np.argmin(time_per_sample)
    
    return {
        'optimal_batch_size': batch_sizes[optimal_idx],
        'inference_time': times[optimal_idx],
        'memory_usage': memory_usage[optimal_idx]
    }

def get_model_summary(model, metrics):
    """Generate comprehensive model summary."""
    summary = {
        'parameters': model.count_params(),
        'layers': len(model.layers),
        'memory_usage': metrics['memory_usage'],
        'inference_time': metrics['inference_time']['mean'],
        'model_size': metrics['model_size']
    }
    
    # Calculate memory per parameter
    summary['memory_per_param'] = metrics['memory_usage'] * 1024 * 1024 / summary['parameters']  # bytes per parameter
    
    # Layer-wise analysis
    layer_info = []
    total_params = 0
    for layer in model.layers:
        params = layer.count_params()
        total_params += params
        layer_info.append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'parameters': params,
            'percentage': params/summary['parameters'] * 100 if summary['parameters'] > 0 else 0
        })
    
    summary['layer_info'] = layer_info
    return summary

def save_model_metrics(metrics, model_name, save_dir='results'):
    """Save model metrics to file."""
    os.makedirs(save_dir, exist_ok=True)
    
    import json
    with open(f'{save_dir}/{model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

def load_model_metrics(model_name, save_dir='results'):
    """Load saved model metrics."""
    import json
    metrics_path = f'{save_dir}/{model_name}_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None
