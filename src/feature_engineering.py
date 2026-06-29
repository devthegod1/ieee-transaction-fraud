# src/feature_engineering.py
import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# STEP 1: Modify sys.path BEFORE importing local modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# STEP 2: Import the main execution function from eda.py
from src.eda import main as run_eda_pipeline

def perform_feature_engineering(df):
    print("\n--- Starting Feature Engineering Pipeline ---")
    
    # 1. Separate target and identifiers to keep them completely safe
    protected_cols = ['TransactionID', 'TransactionDT', 'isFraud']
    # Mix of existing protected columns that actually exist in the current dataframe
    available_protected = [col for col in protected_cols if col in df.columns]
    
    df_protected = df[available_protected].copy()
    df_features = df.drop(columns=available_protected)
    
    # 2. Identify Categorical Columns
    categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Total categorical columns found: {len(categorical_cols)}")
    
    ohe_cols = []
    le_cols = []
    
    for col in categorical_cols:
        # Check cardinality (number of unique values)
        unique_count = df_features[col].nunique()
        if unique_count < 5:
            ohe_cols.append(col)
        else:
            le_cols.append(col)
            
    print(f"-> {len(ohe_cols)} columns selected for One-Hot Encoding (< 5 unique values)")
    print(f"-> {len(le_cols)} columns selected for Label Encoding (>= 5 unique values)")
    
    # 3. Apply One-Hot Encoding
    if ohe_cols:
        df_features = pd.get_dummies(df_features, columns=ohe_cols, drop_first=True)
        
    # 4. Apply Label Encoding
    if le_cols:
        for col in le_cols:
            le = LabelEncoder()
            # Convert to string to safely bypass any residual type anomalies
            df_features[col] = le.fit_transform(df_features[col].astype(str))
            
    # 5. Apply Standard Scaler to all features
    print("Applying StandardScaler to feature columns...")
    scaler = StandardScaler()
    
    # Scaler returns a numpy array, so we wrap it back into a DataFrame
    scaled_matrix = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled_matrix, columns=df_features.columns, index=df_features.index)
    
    # 6. Recombine the protected columns with the scaled feature columns
    final_df = pd.concat([df_protected, df_scaled], axis=1)
    
    print(f"Feature Engineering complete! Final dataframe shape: {final_df.shape}")
    return final_df

if __name__ == "__main__":
    # This automatically triggers data loading -> memory reduction -> cleaning -> imputation
    cleaned_df = run_eda_pipeline()
    
    # Transform the cleaned data
    final_processed_df = perform_feature_engineering(cleaned_df)
    
    # Quick sanity check print
    print("\nProcessed Data Head:")
    print(final_processed_df.head())