"""
Module 9: FastAPI Backend

Provides a REST API for the Space Debris Tracking system.
All ML models are loaded once at startup for fast inference.

Endpoints:
    GET  /         : Welcome message
    GET  /health   : Health check
    GET  /satellites: Return all satellites from the cleaned dataset
    POST /predict  : Predict collision probability for a conjunction event
"""

import os
import sys
import pickle
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# ─── Load ML Models at startup ─────────────────────────────────────────────────

def _load_model(filename: str):
    """Load a pickled model from the models directory."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        logging.warning(f"Model not found: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


xgb_model = _load_model("collision_xgb_model.pkl")
rf_model = _load_model("collision_rf_model.pkl")

# ─── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Space Debris Tracking API",
    description=(
        "REST API for the AI-Based Space Debris Tracking "
        "and Collision Avoidance system."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Schemas ───────────────────────────────────────────────────────────
class ConjunctionRequest(BaseModel):
    """Input schema for the /predict endpoint."""

    relative_position_km: float = Field(
        ..., gt=0, description="Distance between two objects in km"
    )
    relative_velocity_km_s: float = Field(
        ..., gt=0, description="Relative speed between objects in km/s"
    )
    time_to_closest_approach_s: float = Field(
        ..., ge=0, description="Time to closest approach in seconds"
    )


class CollisionResponse(BaseModel):
    """Output schema for the /predict endpoint."""

    relative_position_km: float
    relative_velocity_km_s: float
    time_to_closest_approach_s: float
    collision_probability_rf: float
    collision_probability_xgb: float
    risk_level: str


# ─── Helper ─────────────────────────────────────────────────────────────────────
def _risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.3:
        return "MEDIUM"
    return "LOW"


# ─── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    """Welcome message."""
    return {
        "message": "Welcome to the Space Debris Tracking API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["General"])
def health():
    """Health check — confirms models are loaded."""
    return {
        "status": "ok",
        "xgb_model_loaded": xgb_model is not None,
        "rf_model_loaded": rf_model is not None,
    }


@app.get("/satellites", tags=["Data"])
def get_satellites(limit: int = 50):
    """
    Returns satellite records from the cleaned dataset.

    Args:
        limit (int): Number of records to return (default 50).
    """
    csv_path = os.path.join(DATA_DIR, "space_track_cleaned.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail="Cleaned dataset not found. Run clean_data.py first.",
        )
    df = pd.read_csv(csv_path).head(limit)
    return {"count": len(df), "satellites": df.to_dict(orient="records")}


@app.post("/predict", response_model=CollisionResponse, tags=["Prediction"])
def predict_collision(req: ConjunctionRequest):
    """
    Predicts collision probability for a given conjunction event.

    Uses both Random Forest and XGBoost models to return probability estimates.
    """
    if xgb_model is None or rf_model is None:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Run collision_prediction.py first.",
        )

    features = np.array(
        [
            [
                req.relative_position_km,
                req.relative_velocity_km_s,
                req.time_to_closest_approach_s,
            ]
        ]
    )

    prob_rf = float(np.clip(rf_model.predict(features)[0], 0.0, 1.0))
    prob_xgb = float(np.clip(xgb_model.predict(features)[0], 0.0, 1.0))
    avg_prob = (prob_rf + prob_xgb) / 2.0

    return CollisionResponse(
        relative_position_km=req.relative_position_km,
        relative_velocity_km_s=req.relative_velocity_km_s,
        time_to_closest_approach_s=req.time_to_closest_approach_s,
        collision_probability_rf=round(prob_rf, 4),
        collision_probability_xgb=round(prob_xgb, 4),
        risk_level=_risk_label(avg_prob),
    )
