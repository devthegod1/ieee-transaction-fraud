import pandas as pd
import numpy as np
import os

def reduce_mem_usage(df):
    """
    Iterates through all the columns of a dataframe and modifies the data type
    to reduce memory usage without losing information. Fully compatible with
    modern Pandas 3.x and NumPy 2.x data types.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Initial dataframe memory usage: {start_mem:.2f} MB')
    
    for col in df.columns:
        # Use Pandas api checking instead of NumPy's strict type system
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Check if it's an integer type
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            # Otherwise, it must be a float type
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Safely converts modern StringDtype or objects to category
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Optimized memory usage: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

def load_and_merge_data(data_dir="data/raw"):
    """
    Loads train_transaction and train_identity, optimizes memory usage, 
    and merges them seamlessly using a Left Join on TransactionID.
    """
    transaction_path = os.path.join(data_dir, "train_transaction.csv")
    identity_path = os.path.join(data_dir, "train_identity.csv")
    
    if not os.path.exists(transaction_path) or not os.path.exists(identity_path):
        raise FileNotFoundError(f"Missing CSV files in {data_dir}. Check your path structure.")
    
    # Define standard text representations of missing data to catch them early
    custom_na_values = ['nan', 'NaN', 'NAN', 'null', 'NULL', '', ' ', 'none', 'None']
        
    print("Reading transaction data...")
    train_transaction = pd.read_csv(transaction_path)
    train_transaction = reduce_mem_usage(train_transaction)
    
    print("\nReading identity data...")
    train_identity = pd.read_csv(identity_path)
    train_identity = reduce_mem_usage(train_identity)
    
    print("\nMerging datasets on TransactionID...")
    merged_df = pd.merge(train_transaction, train_identity, on="TransactionID", how="left")
    
    print(f"Final Merged Dataset Shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    df = load_and_merge_data()