"""
Module 5: Feature Engineering

This module prepares the cleaned dataset for deep learning models.
Tasks performed:
- Sort the data chronologically per satellite (by NORAD_CAT_ID and EPOCH).
- Select the relevant features for LSTM trajectory prediction.
- Normalize numerical features using MinMaxScaler.
- Group the data by satellite to create time-series sequences.
- Save the resulting arrays (X and y) as NumPy files.
"""

import os
import numpy as np
import pandas as pd
import logging
import pickle
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_sequences(df: pd.DataFrame, feature_cols: list, seq_length: int = 5) -> tuple:
    """
    Creates sequences of length `seq_length` for each distinct satellite.

    Args:
        df (pd.DataFrame): The pre-sorted, normalized DataFrame.
        feature_cols (list): List of column names to be used as features.
        seq_length (int): Number of time steps in the input sequence.

    Returns:
        tuple: (X, y) as numpy arrays.
    """
    X, y = [], []
    
    # Group by satellite ID to ensure we don't cross sequences between different objects
    grouped = df.groupby('NORAD_CAT_ID')
    
    for _, group in grouped:
        # We need at least (seq_length + 1) rows to form one sequence and its target
        if len(group) <= seq_length:
            continue
            
        data_values = group[feature_cols].values
        
        for i in range(len(data_values) - seq_length):
            seq = data_values[i : i + seq_length]
            target = data_values[i + seq_length]
            
            X.append(seq)
            y.append(target)
            
    return np.array(X), np.array(y)

def engineer_features(data_path: str, seq_length: int = 5) -> None:
    """
    Main function to perform feature engineering pipeline.
    """
    logging.info(f"Loading cleaned data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if df.empty:
        logging.error("The dataset is empty. Cannot perform feature engineering.")
        return
        
    df['EPOCH'] = pd.to_datetime(df['EPOCH'])
    
    # 1. Sort by NORAD_CAT_ID and EPOCH
    df = df.sort_values(by=['NORAD_CAT_ID', 'EPOCH']).reset_index(drop=True)
    logging.info("Sorted dataset by NORAD_CAT_ID and EPOCH.")
    
    # 2. Define Features for LSTM
    features = [
        'MEAN_MOTION', 'ECCENTRICITY', 'INCLINATION', 
        'RA_OF_ASC_NODE', 'ARG_OF_PERICENTER', 'MEAN_ANOMALY', 'BSTAR'
    ]

    # Data Augmentation for Snapshot Dataset
    # If the dataset only contains 1 observation per satellite, we augment it temporally.
    max_obs = df['NORAD_CAT_ID'].value_counts().max()
    if max_obs <= seq_length:
        logging.warning(f"Max observations per satellite ({max_obs}) <= seq_length ({seq_length}).")
        logging.info("Augmenting data to simulate time-series trajectories...")
        augmented_dfs = []
        # Generate enough time steps to satisfy sequence length + target
        for i in range(seq_length + 5): 
            df_copy = df.copy()
            # Shift the EPOCH forward by days
            df_copy['EPOCH'] = df_copy['EPOCH'] + pd.Timedelta(days=i)
            # Apply small physical noise to features to simulate orbital perturbations
            for feat in features:
                # Add 0.1% random noise
                noise = np.random.normal(0, 0.001, len(df))
                # For safety, avoid negative eccentricity if close to 0
                df_copy[feat] = np.abs(df_copy[feat] * (1 + noise))
            augmented_dfs.append(df_copy)
            
        df = pd.concat(augmented_dfs).sort_values(by=['NORAD_CAT_ID', 'EPOCH']).reset_index(drop=True)
        logging.info(f"Data augmentation complete. New dataset shape: {df.shape}")

    # 3. Normalize numerical features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    logging.info("Normalized the numerical features using MinMaxScaler.")
    
    # Save the scaler for inverse transforms during inference
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Saved MinMaxScaler to {scaler_path}")
    
    # 4. Create LSTM sequences
    logging.info(f"Creating time-series sequences of length {seq_length}...")
    X, y = create_sequences(df, features, seq_length)
    
    logging.info(f"Generated {len(X)} sequences.")
    if len(X) > 0:
        logging.info(f"X shape: {X.shape}")
        logging.info(f"y shape: {y.shape}")
        
        # 5. Save processed NumPy arrays
        processed_dir = os.path.join(project_root, 'data', 'processed')
        np.save(os.path.join(processed_dir, 'X.npy'), X)
        np.save(os.path.join(processed_dir, 'y.npy'), y)
        logging.info("Saved X.npy and y.npy to processed data directory.")
    else:
        logging.warning("No sequences generated! Try decreasing seq_length or check data volume.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "processed", "space_track_cleaned.csv")
    
    logging.info("--- Starting Feature Engineering ---")
    engineer_features(data_path, seq_length=5)
    logging.info("--- Feature Engineering Completed ---")
