# AI-Based Space Debris Tracking and Collision Avoidance Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.x-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Final Year Engineering Project** — An intelligent Space Situational Awareness (SSA) platform that loads real Space-Track GP orbital data, cleans it, performs EDA, engineers LSTM time-series features, predicts future satellite trajectories, detects conjunction candidates using a KD-Tree, estimates collision probability with Random Forest and XGBoost, serves predictions via a FastAPI REST API, and visualises everything in an animated interactive dashboard.

---

## 📐 Architecture

```
Raw CSV (Space-Track)
      │
      ▼
 load_data.py  ──►  clean_data.py  ──►  feature_engineering.py
                                               │
                                               ▼
                                        train_lstm.py (PyTorch LSTM)
                                               │
                                               ▼
                                  conjunction_detection.py (KD-Tree)
                                               │
                                               ▼
                                  collision_prediction.py (RF + XGBoost)
                                               │
                                               ▼
                                     backend/main.py (FastAPI)
                                               │
                                               ▼
                                    frontend/index.html (Dashboard)
```

---

## 📁 Project Structure

```
AI-Based-Space-Debris-Tracking/
├── data/
│   ├── raw/                        # Original Space-Track CSV
│   └── processed/                  # Cleaned CSV, X.npy, y.npy
├── notebooks/
│   └── 01_EDA.ipynb                # Exploratory Data Analysis
├── scripts/
│   ├── load_data.py                # Module 2: Data Loading
│   ├── clean_data.py               # Module 3: Data Cleaning
│   ├── feature_engineering.py      # Module 5: Feature Engineering
│   ├── train_lstm.py               # Module 6: LSTM Orbit Prediction
│   ├── conjunction_detection.py    # Module 7: KD-Tree Conjunction Detection
│   └── collision_prediction.py     # Module 8: Collision Probability (RF + XGBoost)
├── models/
│   ├── lstm_orbit_model.pth        # Trained PyTorch LSTM
│   ├── scaler.pkl                  # MinMaxScaler for features
│   ├── collision_rf_model.pkl      # Random Forest model
│   └── collision_xgb_model.pkl     # XGBoost model
├── backend/
│   └── main.py                     # Module 9: FastAPI REST API
├── frontend/
│   └── index.html                  # Module 10: Interactive Dashboard
├── visualization/
│   ├── lstm_training_history.png
│   └── collision_feature_importance.png
├── tests/
├── setup.cfg                       # Flake8 configuration
├── requirements.txt
├── main.py                         # Pipeline orchestrator
└── README.md
```

---

## 🧠 Modules

| # | Module | File | Description |
|---|--------|------|-------------|
| 1 | Project Initialisation | — | Folder structure, requirements |
| 2 | Data Loading | `load_data.py` | Load raw CSV, inspect dataset |
| 3 | Data Cleaning | `clean_data.py` | Deduplicate, select 14 features, parse EPOCH |
| 4 | EDA | `01_EDA.ipynb` | Histograms, scatter plots, correlation matrix |
| 5 | Feature Engineering | `feature_engineering.py` | Temporal augmentation, normalisation, LSTM sequences |
| 6 | LSTM Training | `train_lstm.py` | 2-layer PyTorch LSTM, MAE 0.0002 |
| 7 | Conjunction Detection | `conjunction_detection.py` | Keplerian → Cartesian, KD-Tree nearest-neighbour |
| 8 | Collision Probability | `collision_prediction.py` | Random Forest + XGBoost, physics-grounded labels |
| 9 | FastAPI Backend | `backend/main.py` | REST API: `/`, `/health`, `/satellites`, `/predict` |
| 10 | Dashboard | `frontend/index.html` | Animated solar system + satellite orbits + alerts |
| 11 | Documentation | `README.md` | This file |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place raw data

Copy your Space-Track CSV into:
```
data/raw/space_track_raw.csv
```

### 3. Run the full pipeline

```bash
python main.py
```

Or run each module individually:

```bash
cd scripts
python load_data.py
python clean_data.py
python feature_engineering.py
python train_lstm.py
python conjunction_detection.py
python collision_prediction.py
```

### 4. Launch the API

```bash
uvicorn backend.main:app --reload
```

API docs available at: `http://localhost:8000/docs`

### 5. Open the Dashboard

Open `frontend/index.html` in any modern browser.

---

## 🔌 API Reference

### `GET /`
Welcome message.

### `GET /health`
Returns model load status.

### `GET /satellites?limit=50`
Returns satellite records from the cleaned dataset.

### `POST /predict`
```json
{
  "relative_position_km": 5.0,
  "relative_velocity_km_s": 12.0,
  "time_to_closest_approach_s": 120
}
```
Returns:
```json
{
  "collision_probability_rf": 0.9812,
  "collision_probability_xgb": 0.9740,
  "risk_level": "HIGH"
}
```

---

## 📊 Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| LSTM (PyTorch) | Test MAE | 0.0002 |
| Random Forest | Test MAE | 0.0019 |
| XGBoost | Test MAE | 0.0020 |

---

## 🛠️ Tech Stack

- **Data**: pandas, numpy, scikit-learn
- **Deep Learning**: PyTorch (LSTM)
- **Machine Learning**: scikit-learn (Random Forest), XGBoost
- **Orbital Mechanics**: scipy (KD-Tree), numpy (Keplerian transforms)
- **API**: FastAPI, Uvicorn, Pydantic
- **Visualisation**: matplotlib, seaborn, HTML5 Canvas (dashboard)
- **Code Quality**: black, flake8

---

## 📋 Coding Standards

- PEP 8 compliant (formatted with `black`, linted with `flake8`)
- Google-style docstrings throughout
- Full type hints on all public functions
- Modular architecture — each script is independently runnable
- Exception handling with informative logging

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
