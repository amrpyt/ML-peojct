"""
Test script to verify all updates and new functionalities.
"""
import os
import sys
import tensorflow as tf
import numpy as np
from utils import measure_model_metrics, calculate_efficiency_score

def test_environment():
    """Test environment setup and dependencies."""
    print("Testing environment setup...")
    
    required_packages = [
        'tensorflow',
        'numpy',
        'pandas',
        'psutil',
        'h5py',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} missing")
    
    if missing:
        print("\nMissing packages. Please install:")
        print("pip install " + " ".join(missing))
        return False
    return True

def test_directory_structure():
    """Test required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = ['models', 'logs', 'results', 'preprocessed']
    missing = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing.append(dir_name)
            print(f"✗ {dir_name}/ missing")
        else:
            print(f"✓ {dir_name}/ exists")
    
    if missing:
        print("\nCreating missing directories...")
        for dir_name in missing:
            os.makedirs(dir_name)
            print(f"Created {dir_name}/")
    
    return True

def test_notebooks():
    """Test required notebooks exist and have correct structure."""
    print("\nTesting notebook files...")
    
    required_notebooks = [
        'src/01_data_preprocessing.ipynb',
        'src/02_model_definitions.ipynb',
        'src/03_model_training.ipynb',
        'src/model_metrics.ipynb',
        'src/04_model_optimization.ipynb',
        'src/05_temp_hum_prediction.ipynb',
        'src/06_visualization.ipynb'
    ]
    
    missing = []
    for notebook in required_notebooks:
        if not os.path.exists(notebook):
            missing.append(notebook)
            print(f"✗ {notebook} missing")
        else:
            print(f"✓ {notebook} exists")
    
    if missing:
        print("\nMissing notebooks:")
        for notebook in missing:
            print(f"- {notebook}")
        return False
    return True

def test_hybrid_models():
    """Test hybrid model creation and metrics."""
    print("\nTesting hybrid model creation...")
    
    # Create small test dataset
    X = np.random.random((100, 10, 1))
    y = np.random.randint(0, 2, 100)
    
    # Test CNN-LSTM
    try:
        cnn_lstm = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 2, activation='relu', input_shape=(10, 1)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        cnn_lstm.compile(optimizer='adam', loss='binary_crossentropy')
        print("✓ CNN-LSTM model created")
        
        # Test metrics
        metrics = measure_model_metrics(cnn_lstm, X)
        print("✓ Metrics calculation successful")
        print(f"  Memory usage: {metrics['memory_usage']:.2f} MB")
        print(f"  Inference time: {metrics['inference_time']['mean']:.2f} ms")
        print(f"  Model size: {metrics['model_size']:.2f} KB")
        
    except Exception as e:
        print(f"✗ Error testing CNN-LSTM: {str(e)}")
        return False
    
    # Test CNN-BiLSTM
    try:
        cnn_bilstm = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 2, activation='relu', input_shape=(10, 1)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        cnn_bilstm.compile(optimizer='adam', loss='binary_crossentropy')
        print("✓ CNN-BiLSTM model created")
        
        # Test metrics
        metrics = measure_model_metrics(cnn_bilstm, X)
        print("✓ Metrics calculation successful")
        print(f"  Memory usage: {metrics['memory_usage']:.2f} MB")
        print(f"  Inference time: {metrics['inference_time']['mean']:.2f} ms")
        print(f"  Model size: {metrics['model_size']:.2f} KB")
        
    except Exception as e:
        print(f"✗ Error testing CNN-BiLSTM: {str(e)}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running validation tests...")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("Directory Structure", test_directory_structure),
        ("Notebooks", test_notebooks),
        ("Hybrid Models", test_hybrid_models)
    ]
    
    success = True
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        print("-" * 50)
        try:
            if not test_func():
                print(f"\n✗ {test_name} tests failed!")
                success = False
            else:
                print(f"\n✓ {test_name} tests passed!")
        except Exception as e:
            print(f"\n✗ Error in {test_name} tests: {str(e)}")
            success = False
    
    print("\nTest Summary")
    print("=" * 50)
    if success:
        print("All tests passed successfully!")
    else:
        print("Some tests failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()