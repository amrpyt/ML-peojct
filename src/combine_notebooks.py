import json
import os
import shutil
from validate_notebooks import validate_notebooks

def read_notebook(filename):
    """Read a Jupyter notebook file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def combine_notebooks():
    """Combine notebook components into a final notebook."""
    # Validate notebooks first
    print("Validating notebooks...")
    if not validate_notebooks():
        print("\nNotebook validation failed. Please fix the issues before combining.")
        return False
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('preprocessed', exist_ok=True)
    
    # Initialize final notebook
    final_notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Ordered list of notebooks to combine
    notebooks = [
        ('src/01_data_preprocessing.ipynb', 'Data Loading and Preprocessing'),
        ('src/02_model_definitions.ipynb', 'Model Definitions'),
        ('src/03_model_training.ipynb', 'Model Training'),
        ('src/model_metrics.ipynb', 'Model Performance Analysis'),
        ('src/04_model_optimization.ipynb', 'Model Optimization'),
        ('src/05_temp_hum_prediction.ipynb', 'Temperature and Humidity Prediction'),
        ('src/06_visualization.ipynb', 'Visualization and Comparison')
    ]

    # Add title cell
    final_notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Air Quality Analysis with Deep Learning\n",
                  "\n",
                  "This notebook combines all analysis steps into a single coherent workflow."]
    })

    # Add imports cell
    final_notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import tensorflow as tf\n",
            "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
            "from imblearn.over_sampling import SMOTE\n",
            "from sklearn.model_selection import train_test_split\n",
            "from tensorflow.keras.models import Sequential\n",
            "from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Bidirectional\n",
            "from tensorflow.keras.optimizers import Adam\n",
            "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix\n",
            "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n",
            "import psutil\n",
            "\n",
            "# Create necessary directories\n",
            "for directory in ['models', 'logs', 'results', 'preprocessed']:\n",
            "    os.makedirs(directory, exist_ok=True)\n",
            "\n",
            "# Set plotting style\n",
            "plt.style.use('seaborn')\n",
            "sns.set_palette('husl')"
        ]
    })

    # Process each notebook
    for filepath, title in notebooks:
        try:
            print(f"\nProcessing {filepath}...")
            notebook = read_notebook(filepath)
            
            # Add section title
            final_notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"\n## {title}"]
            })
            
            # Add cells, skipping imports and title cells
            for cell in notebook["cells"]:
                if cell["cell_type"] == "markdown":
                    source = "".join(cell["source"]).lower()
                    if source.startswith("# ") and title.lower() in source:
                        continue
                elif cell["cell_type"] == "code":
                    source = "".join(cell["source"]).lower()
                    if "import" in source and ("numpy" in source or "pandas" in source):
                        continue
                
                final_notebook["cells"].append(cell)
            
            print(f"Added {title} section")
            
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            return False

    # Save final notebook
    output_path = 'src/final_air_quality_analysis.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_notebook, f, indent=2)
    
    print(f"\nFinal notebook saved to: {output_path}")
    return True

if __name__ == "__main__":
    # Clean up existing files
    if os.path.exists('src/final_air_quality_analysis.ipynb'):
        os.remove('src/final_air_quality_analysis.ipynb')
    
    print("Creating final notebook...")
    print("-" * 50)
    
    if combine_notebooks():
        print("\nNotebook creation successful!")
    else:
        print("\nError: Notebook creation failed!")
        exit(1)