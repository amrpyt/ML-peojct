import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"Python version {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install required packages with fallback options."""
    base_requirements = [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "imbalanced-learn>=0.8.0",
        "jupyter>=1.0.0",
        "h5py>=3.10.0"
    ]
    
    print("\nInstalling base requirements...")
    for req in base_requirements:
        try:
            print(f"Installing {req}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {req}: {str(e)}")
            return False

    # Try installing TensorFlow with GPU support first
    print("\nAttempting to install TensorFlow with GPU support...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "tensorflow>=2.16.0"
        ])
        print("TensorFlow with GPU support installed successfully")
    except subprocess.CalledProcessError as e:
        print("GPU installation failed, falling back to CPU version...")
        try:
            # Try CPU-only version
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "tensorflow-cpu>=2.16.0"
            ])
            print("TensorFlow CPU version installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing TensorFlow: {str(e)}")
            return False
    
    return True

def verify_installation():
    """Verify that all required packages are installed correctly."""
    try:
        # Import and check versions
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        
        print("\nPackage versions:")
        print(f"NumPy: {np.__version__}")
        print(f"Pandas: {pd.__version__}")
        print(f"TensorFlow: {tf.__version__}")
        
        # Test TensorFlow
        print("\nTesting TensorFlow...")
        # Simple matrix multiplication test
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
        c = tf.matmul(a, b)
        print("TensorFlow basic operations test: Passed")
        
        # Check GPU availability
        print("\nGPU Information:")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"- {gpu.device_type}: {gpu.name}")
            
            # Enable memory growth
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Memory growth enabled for {gpu.name}")
                except RuntimeError as e:
                    print(f"Error setting memory growth: {str(e)}")
        else:
            print("No GPU found. Running on CPU mode")
        
        return True
        
    except ImportError as e:
        print(f"\nError during verification: {str(e)}")
        return False

def create_directories():
    """Create necessary project directories."""
    directories = ['models', 'logs', 'results']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def check_system():
    """Check system compatibility."""
    print("\nSystem Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    if platform.system() == 'Windows':
        # Check if running in virtual environment
        if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
            print("\nWarning: It's recommended to run this setup in a virtual environment")

def main():
    print("Setting up Air Quality Analysis project...")
    print("=" * 50)
    
    # Check system compatibility
    check_system()
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    if not install_requirements():
        print("\nError: Failed to install requirements")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\nError: Installation verification failed")
        sys.exit(1)
    
    # Create project directories
    create_directories()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. python src/combine_notebooks.py")
    print("2. jupyter notebook src/final_air_quality_analysis.ipynb")

if __name__ == "__main__":
    main()