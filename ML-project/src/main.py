import os
import sys
import time
import traceback
import numpy as np
import pandas as pd

# Import project modules
try:
    from config import RANDOM_STATE, DATA_PATH, FEATURES, TARGET_CLASS, REGRESSION_TARGETS, TIMESTAMP_COL
    import preprocess
    import models
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    print("Starting Air Quality Analysis Project\n")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_STATE)
    
    try:
        # 1. Load and preprocess data
        print("1. Preprocessing data...")
        
        # Check if data file exists
        if not os.path.exists(DATA_PATH):
            print(f"Error: Data file not found at {DATA_PATH}")
            sys.exit(1)
            
        # Load data with error handling
        try:
            print("   - Loading data...")
            df = pd.read_csv(DATA_PATH)
            print(f"   - Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        # Process data with timeout handling
        try:
            processed_data = preprocess.preprocess_data(df, 
                                                       features=FEATURES, 
                                                       target_class=TARGET_CLASS,
                                                       regression_targets=REGRESSION_TARGETS,
                                                       timestamp_col=TIMESTAMP_COL)
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        print("Preprocessing completed successfully!")
        # Continue with rest of pipeline...
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()