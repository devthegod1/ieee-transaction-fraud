# src/eda.py
import os
import sys
import pandas as pd

# STEP 1: Modify sys.path BEFORE importing local modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# STEP 2: Import from src
from src.data_loader import load_and_merge_data

def main():
    print("Starting EDA Pipeline...")
    
    # 3. Load the data
    data_path = os.path.join(project_root, "data", "raw")
    df = load_and_merge_data(data_dir=data_path)
    
    print(f"\nOriginal dataframe shape: {df.shape}")
    
    # 4. Identify and drop columns with > 50% missing values
    threshold = 0.50
    missing_pct = df.isnull().mean()  
    
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    print(f"Found {len(cols_to_drop)} columns with more than {threshold*100}% missing values.")
    
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Filtered dataframe shape: {df.shape}")
    
    # 5. Handle remaining null values based on data type
    print("\nImputing remaining missing values...")
    
    # Columns to exclude from numerical mean imputation
    exclude_cols = {'TransactionID', 'TransactionDT'}
    
    for col in df.columns:
        # Only process columns that actually have missing values remaining
        if df[col].isnull().any():
            
            # Scenario A: Numerical Columns (excluding ID/DT)
            if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols:
                col_mean = df[col].mean()
                df[col].fillna(col_mean, inplace=True)
                
            # Scenario B: Categorical or Object Columns
            elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                # If it's a formal Pandas 'category' dtype, we must add 'unknown' to its categories first
                if isinstance(df[col].dtype, pd.CategoricalDtype):
                    if 'unknown' not in df[col].cat.categories:
                        df[col] = df[col].cat.add_categories('unknown')
                
                df[col].fillna('unknown', inplace=True)

    print("Missing value imputation complete! (Remaining missing values:", df.isnull().sum().sum(), ")")
    
    # Optional: Preview the remaining columns
    print("\nRemaining columns preview:")
    print(df.head())
    return df

if __name__ == "__main__":
    main()