"""
Module 2: Data Loading

This module safely loads raw Space-Track orbital data from a CSV file
and performs initial inspection (shape, head, columns, missing values).
"""

import os
import pandas as pd
import logging

# Set up logging for basic output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_space_track_data(filepath: str) -> pd.DataFrame:
    """
    Loads Space-Track dataset from a specified CSV file.

    Args:
        filepath (str): The relative or absolute path to the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
            Returns an empty DataFrame if the file is not found
            or cannot be read.
    """
    try:
        logging.info(f"Attempting to load data from {filepath}...")
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {filepath}. Please check the path.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logging.error("The file is empty.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the data: {e}")
        return pd.DataFrame()


def inspect_data(df: pd.DataFrame) -> None:
    """
    Prints basic information about the dataset including shape, column names,
    data types, missing values, and the first five rows.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
    """
    if df.empty:
        logging.warning("DataFrame is empty. Cannot inspect data.")
        return

    print("=" * 50)
    print("DATASET SHAPE")
    print("=" * 50)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    print("=" * 50)
    print("FIRST FIVE ROWS")
    print("=" * 50)
    print(df.head(), "\n")

    print("=" * 50)
    print("COLUMN NAMES")
    print("=" * 50)
    print(list(df.columns), "\n")

    print("=" * 50)
    print("DATA TYPES")
    print("=" * 50)
    print(df.dtypes, "\n")

    print("=" * 50)
    print("MISSING VALUES")
    print("=" * 50)
    print(df.isnull().sum(), "\n")


if __name__ == "__main__":
    # Define the path to the raw data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "raw", "space_track_raw.csv")

    # Load the data
    orbit_df = load_space_track_data(data_path)

    # Inspect the data
    inspect_data(orbit_df)
