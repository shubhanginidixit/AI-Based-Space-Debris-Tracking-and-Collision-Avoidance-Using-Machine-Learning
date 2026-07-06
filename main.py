"""
main.py — Pipeline Orchestrator

Runs the complete Space Debris Tracking pipeline end-to-end:
  1. Load raw data
  2. Clean data
  3. Feature engineering (augmentation + sequences)
  4. Train LSTM
  5. Run conjunction detection
  6. Train collision probability models

Usage:
    python main.py
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Add scripts directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))


def run_pipeline() -> None:
    """Execute the full ML pipeline in sequence."""
    from load_data import load_space_track_data
    from clean_data import clean_data, save_cleaned_data
    from feature_engineering import engineer_features
    from train_lstm import train_and_evaluate
    from conjunction_detection import detect_conjunctions
    from collision_prediction import run_pipeline as train_collision_models

    raw_path = os.path.join(BASE_DIR, "data", "raw", "space_track_raw.csv")
    cleaned_path = os.path.join(BASE_DIR, "data", "processed", "space_track_cleaned.csv")

    # ── Step 1: Load ──────────────────────────────────────────────
    logging.info("=" * 55)
    logging.info("STEP 1 — Loading raw data")
    logging.info("=" * 55)
    df = load_space_track_data(raw_path)
    if df.empty:
        logging.error("No data loaded. Aborting pipeline.")
        return

    # ── Step 2: Clean ─────────────────────────────────────────────
    logging.info("=" * 55)
    logging.info("STEP 2 — Cleaning data")
    logging.info("=" * 55)
    cleaned = clean_data(df)
    save_cleaned_data(cleaned, cleaned_path)

    # ── Step 3: Feature Engineering ───────────────────────────────
    logging.info("=" * 55)
    logging.info("STEP 3 — Feature engineering")
    logging.info("=" * 55)
    engineer_features(cleaned_path, seq_length=5)

    # ── Step 4: Train LSTM ────────────────────────────────────────
    logging.info("=" * 55)
    logging.info("STEP 4 — Training LSTM orbit prediction model")
    logging.info("=" * 55)
    train_and_evaluate()

    # ── Step 5: Conjunction Detection ────────────────────────────
    logging.info("=" * 55)
    logging.info("STEP 5 — Running conjunction detection")
    logging.info("=" * 55)
    import pandas as pd
    snap = (
        pd.read_csv(cleaned_path)
        .sort_values("EPOCH")
        .groupby("NORAD_CAT_ID")
        .last()
        .reset_index()
    )
    results = detect_conjunctions(snap, threshold_km=200.0)
    logging.info(f"Found {len(results)} conjunction candidates.")

    # ── Step 6: Collision Models ──────────────────────────────────
    logging.info("=" * 55)
    logging.info("STEP 6 — Training collision probability models")
    logging.info("=" * 55)
    train_collision_models()

    logging.info("=" * 55)
    logging.info("✅  Pipeline complete. Launch the API with:")
    logging.info("    uvicorn backend.main:app --reload")
    logging.info("    Then open frontend/index.html in your browser.")
    logging.info("=" * 55)


if __name__ == "__main__":
    run_pipeline()
