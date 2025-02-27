import os
import shutil
from combine_notebooks import combine_notebooks
from validate_notebooks import validate_notebook_sections

def clean_workspace():
    """Remove any existing combined notebooks and recreate directories."""
    if os.path.exists('src/final_air_quality_analysis.ipynb'):
        print("Removing existing final notebook...")
        os.remove('src/final_air_quality_analysis.ipynb')
    
    for dir_name in ['models', 'logs']:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name} directory...")
            shutil.rmtree(dir_name)
        os.makedirs(dir_name)
    
    print("Workspace cleaned.")

def verify_notebook():
    """Verify the final notebook structure and content."""
    required_sections = [
        "Data Loading and Preprocessing",
        "Model Definitions",
        "Model Training",
        "Model Optimization",
        "Temperature and Humidity Prediction",
        "Visualization and Comparison"
    ]
    
    print("\nVerifying notebook sections...")
    return validate_notebook_sections()

def main():
    """Create and verify the final notebook."""
    print("Creating final notebook...")
    print("-" * 50)
    
    # Clean workspace
    clean_workspace()
    
    # Validate source notebooks
    print("\nValidating source notebooks...")
    if not verify_notebook():
        print("\nWarning: Source notebooks validation failed!")
        user_input = input("Do you want to continue anyway? (y/N): ")
        if user_input.lower() != 'y':
            print("Aborting notebook creation.")
            return False
    
    # Combine notebooks
    try:
        combine_notebooks()
        print("\nNotebooks combined successfully.")
    except Exception as e:
        print(f"\nError combining notebooks: {str(e)}")
        return False
    
    # Verify final notebook
    print("\nVerifying final notebook...")
    if verify_notebook():
        print("\nFinal notebook created successfully!")
        print(f"Output: src/final_air_quality_analysis.ipynb")
        print("\nNext steps:")
        print("1. Start Jupyter Notebook server")
        print("2. Open src/final_air_quality_analysis.ipynb")
        print("3. Run all cells in order")
        return True
    else:
        print("\nWarning: Final notebook validation failed.")
        print("Please check the notebooks manually for missing sections.")
        return False

def print_section_guide():
    """Print guide for required notebook sections."""
    print("\nRequired Notebook Sections:")
    print("-" * 50)
    print("Each source notebook should contain these sections marked with appropriate headers:")
    print("1. Data Loading and Preprocessing")
    print("   - Data import")
    print("   - Outlier handling")
    print("   - Feature normalization")
    print("   - SMOTE balancing")
    
    print("\n2. Model Definitions")
    print("   - 1DCNN, RNN, DNN")
    print("   - LSTM, BiLSTM")
    
    print("\n3. Model Training")
    print("   - Training process")
    print("   - Model evaluation")
    
    print("\n4. Model Optimization")
    print("   - Model conversion")
    print("   - Size optimization")
    
    print("\n5. Temperature and Humidity Prediction")
    print("   - Regression models")
    print("   - Performance metrics")
    
    print("\n6. Visualization and Comparison")
    print("   - Performance plots")
    print("   - Comparison metrics")
    
    print("\nSection headers should be marked with '#' for main sections")
    print("Example: '# Data Loading and Preprocessing'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create final analysis notebook')
    parser.add_argument('--guide', action='store_true', help='Print section guide')
    args = parser.parse_args()
    
    if args.guide:
        print_section_guide()
    else:
        main()