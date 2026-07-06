"""
Module 3: Data Cleaning

This module is responsible for cleaning the raw Space-Track data.
Tasks performed:
- Remove duplicate rows.
- Filter the dataset to include only essential orbital parameters.
- Handle missing values.
- Convert the EPOCH column to datetime objects.
- Save the cleaned dataset to the processed data directory.
"""

import os
import pandas as pd
import logging
from load_data import load_space_track_data

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Space-Track orbital DataFrame.

    Args:
        df (pd.DataFrame): The raw DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    if df.empty:
        logging.warning("Input DataFrame is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    logging.info(f"Original dataset shape: {df.shape}")

    # 1. Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    logging.info(f"Removed {duplicates_removed} duplicate rows.")

    # 2. Select important orbital parameters
    important_features = [
        "OBJECT_NAME",
        "NORAD_CAT_ID",
        "EPOCH",
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "BSTAR",
        "SEMIMAJOR_AXIS",
        "PERIOD",
        "APOAPSIS",
        "PERIAPSIS",
    ]

    # Check if all important features exist in the dataframe
    missing_features = [feat for feat in important_features if feat not in df.columns]
    if missing_features:
        logging.error(f"Missing essential columns in the data: {missing_features}")
        raise KeyError(f"Missing essential columns: {missing_features}")

    df = df[important_features].copy()
    logging.info(f"Selected {len(important_features)} essential orbital features.")

    # 3. Handle missing values
    initial_rows = len(df)
    df = df.dropna()
    missing_removed = initial_rows - len(df)
    logging.info(
        f"Dropped {missing_removed} rows with missing values "
        "in the selected features."
    )

    # 4. Convert EPOCH to datetime
    try:
        # Use errors='coerce' to handle unparseable dates gracefully.
        # Space-Track EPOCH is typically ISO8601 format.
        df["EPOCH"] = pd.to_datetime(df["EPOCH"], errors="coerce")

        # Drop rows where datetime conversion failed (NaT)
        nat_count = df["EPOCH"].isna().sum()
        if nat_count > 0:
            df = df.dropna(subset=["EPOCH"])
            logging.info(
                f"Dropped {nat_count} rows due to invalid EPOCH datetime parsing."
            )

        logging.info("Successfully converted EPOCH column to datetime objects.")
    except Exception as e:
        logging.error(f"Failed to convert EPOCH to datetime: {e}")

    logging.info(f"Final cleaned dataset shape: {df.shape}")
    return df


def save_cleaned_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the cleaned DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The cleaned DataFrame.
        filepath (str): Destination path for the CSV file.
    """
    if df.empty:
        logging.warning("Cannot save an empty DataFrame.")
        return

    try:
        df.to_csv(filepath, index=False)
        logging.info(f"Cleaned dataset successfully saved to: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save cleaned data: {e}")


if __name__ == "__main__":
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    raw_data_path = os.path.join(project_root, "data", "raw", "space_track_raw.csv")
    processed_data_path = os.path.join(
        project_root, "data", "processed", "space_track_cleaned.csv"
    )

    # Load raw data
    logging.info("--- Starting Data Cleaning Process ---")
    raw_df = load_space_track_data(raw_data_path)

    # Clean data
    if not raw_df.empty:
        cleaned_df = clean_data(raw_df)

        # Save cleaned data
        save_cleaned_data(cleaned_df, processed_data_path)
        logging.info("--- Data Cleaning Process Completed ---")
    else:
        logging.error("Data cleaning aborted due to empty raw data.")
