"""
Module 8: Collision Probability Prediction

This module trains ML classifiers (Random Forest & XGBoost) to estimate
the probability of collision between two objects flagged as conjunction
candidates by the KD-Tree module.

Feature Inputs:
  - relative_position_km   : Euclidean distance between two objects (km)
  - relative_velocity_km_s : Magnitude of relative velocity vector (km/s)
  - time_to_closest_approach_s : TCA estimate (seconds)

Target Output:
  - collision_probability  : Float in [0, 1]

The synthetic training dataset is generated using physics-grounded rules:
  - High probability -> objects are close, fast-moving, and TCA is near.
  - Low probability  -> objects are far apart, slow, TCA is distant.

Production Usage:
  In production, this module would ingest real conjunction data messages
  (CDMs) from Space-Track.org or equivalent sources.
"""

import os
import logging
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_synthetic_conjunction_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generates a synthetic dataset of conjunction events with physics-based rules.

    Args:
        n_samples (int): Number of synthetic conjunction events to generate.

    Returns:
        pd.DataFrame: Dataset with features and collision probability label.
    """
    np.random.seed(42)

    # Feature: relative position (km) — closer means higher risk
    rel_pos = np.random.uniform(0.1, 500.0, n_samples)

    # Feature: relative velocity (km/s) — higher speed = higher kinetic impact
    rel_vel = np.random.uniform(0.01, 15.0, n_samples)

    # Feature: time to closest approach (seconds) — lower TCA = imminent risk
    tca = np.random.uniform(0, 86400.0, n_samples)  # 0 to 24 hours

    # Physics-based probability formula:
    # P ~ exp(-dist/scale) * (vel/max_vel) * exp(-tca/tca_scale)
    dist_factor = np.exp(-rel_pos / 50.0)
    vel_factor = rel_vel / 15.0
    tca_factor = np.exp(-tca / 3600.0)
    raw_prob = dist_factor * vel_factor * tca_factor

    # Normalise to [0, 1]
    collision_prob = raw_prob / raw_prob.max()

    df = pd.DataFrame(
        {
            "relative_position_km": rel_pos,
            "relative_velocity_km_s": rel_vel,
            "time_to_closest_approach_s": tca,
            "collision_probability": collision_prob,
        }
    )
    return df


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Trains a Random Forest Regressor to estimate collision probability.

    Args:
        X_train, y_train: Training features and labels.
        X_test, y_test: Test features and labels.

    Returns:
        sklearn.ensemble.RandomForestRegressor: Trained model.
    """
    logging.info("Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    logging.info(f"Random Forest  |  Test MAE: {mae:.4f}")
    return rf


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Trains an XGBoost Regressor to estimate collision probability.

    Args:
        X_train, y_train: Training features and labels.
        X_test, y_test: Test features and labels.

    Returns:
        xgboost.XGBRegressor: Trained model.
    """
    logging.info("Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    preds = xgb_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    logging.info(f"XGBoost        |  Test MAE: {mae:.4f}")
    return xgb_model


def plot_feature_importance(model, feature_names: list, save_path: str) -> None:
    """
    Plots and saves feature importance from a trained tree-based model.

    Args:
        model: A trained sklearn or XGBoost model with feature_importances_.
        feature_names (list): Names of input features.
        save_path (str): Path to save the figure.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.title("Feature Importance — Collision Probability")
    plt.bar(
        range(len(feature_names)),
        importances[indices],
        color="steelblue",
    )
    plt.xticks(
        range(len(feature_names)),
        [feature_names[i] for i in indices],
        rotation=20,
        ha="right",
    )
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Feature importance plot saved to {save_path}")


def predict_collision_probability(
    model,
    relative_position_km: float,
    relative_velocity_km_s: float,
    time_to_closest_approach_s: float,
) -> float:
    """
    Predicts collision probability for a single conjunction event.

    Args:
        model: Trained regression model.
        relative_position_km (float): Distance between objects (km).
        relative_velocity_km_s (float): Relative speed (km/s).
        time_to_closest_approach_s (float): TCA in seconds.

    Returns:
        float: Estimated collision probability in [0, 1].
    """
    features = np.array(
        [[relative_position_km, relative_velocity_km_s, time_to_closest_approach_s]]
    )
    prob = float(np.clip(model.predict(features)[0], 0.0, 1.0))
    return prob


def run_pipeline() -> None:
    """End-to-end pipeline: generate data, train models, save artifacts."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, "models")
    vis_dir = os.path.join(project_root, "visualization")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Generate synthetic data
    logging.info("Generating synthetic conjunction dataset...")
    df = generate_synthetic_conjunction_data(n_samples=2000)
    logging.info(f"Dataset shape: {df.shape}")

    feature_cols = [
        "relative_position_km",
        "relative_velocity_km_s",
        "time_to_closest_approach_s",
    ]
    X = df[feature_cols].values
    y = df["collision_probability"].values

    # 2. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train both models
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

    # 4. Save models
    rf_path = os.path.join(models_dir, "collision_rf_model.pkl")
    xgb_path = os.path.join(models_dir, "collision_xgb_model.pkl")

    with open(rf_path, "wb") as f:
        pickle.dump(rf_model, f)
    logging.info(f"Random Forest model saved to {rf_path}")

    with open(xgb_path, "wb") as f:
        pickle.dump(xgb_model, f)
    logging.info(f"XGBoost model saved to {xgb_path}")

    # 5. Plot feature importance (using XGBoost)
    plot_feature_importance(
        xgb_model,
        feature_cols,
        os.path.join(vis_dir, "collision_feature_importance.png"),
    )

    # 6. Example inference
    logging.info("\n--- Example Collision Probability Predictions ---")
    examples = [
        (0.5, 14.0, 30.0, "HIGH RISK"),
        (50.0, 2.0, 3600.0, "MEDIUM RISK"),
        (300.0, 0.5, 80000.0, "LOW RISK"),
    ]
    for pos, vel, tca, label in examples:
        prob = predict_collision_probability(xgb_model, pos, vel, tca)
        logging.info(
            f"[{label}] pos={pos} km, vel={vel} km/s, "
            f"tca={tca}s  =>  P_collision = {prob:.4f}"
        )


if __name__ == "__main__":
    logging.info("--- Starting Collision Probability Module ---")
    run_pipeline()
    logging.info("--- Module Completed ---")
