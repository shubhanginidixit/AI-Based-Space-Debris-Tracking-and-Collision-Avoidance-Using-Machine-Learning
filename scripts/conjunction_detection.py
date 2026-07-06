"""
Module 7: Conjunction Detection

This module detects potential collisions (conjunctions) between space objects.
It converts Keplerian orbital elements into Cartesian coordinates (X, Y, Z)
and uses a KD-Tree for highly efficient nearest-neighbor searches in 3D space.

MATHEMATICS EXPLANATION:
------------------------
To calculate distances between satellites, we must convert
Orbital Elements to Cartesian (X, Y, Z):
1. Solve Kepler's Equation for Eccentric Anomaly (E):
   M = E - e * sin(E)  (where M = Mean Anomaly, e = Eccentricity)
   *Solved iteratively via Newton-Raphson method.*

2. Calculate True Anomaly (v) and distance to central body (r):
   r = a * (1 - e * cos(E))  (where a = semi-major axis)
   x_orb = r * cos(E) - a * e
   y_orb = a * sqrt(1 - e^2) * sin(E)

3. Rotate from the orbital plane to the Geocentric Equatorial frame (ECI):
   Using rotation matrices involving:
   Ω (RA_OF_ASC_NODE)
   ω (ARG_OF_PERICENTER)
   i (INCLINATION)

   X = x_orb*(cos(Ω)cos(ω) - sin(Ω)sin(ω)cos(i))
       - y_orb*(cos(Ω)sin(ω) + sin(Ω)cos(ω)cos(i))
   Y = x_orb*(sin(Ω)cos(ω) + cos(Ω)sin(ω)cos(i))
       - y_orb*(sin(Ω)sin(ω) - cos(Ω)cos(ω)cos(i))
   Z = x_orb*(sin(ω)sin(i)) + y_orb*(cos(ω)sin(i))

4. KD-Tree (K-Dimensional Tree):
   a KD-Tree recursively partitions 3D space. Finding neighbors
   within a distance D becomes O(N log N), crucial for real-time
   collision detection.
"""

import os
import numpy as np
import pandas as pd
import logging
from scipy.spatial import KDTree

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def solve_kepler(M: np.ndarray, e: np.ndarray, tol=1e-6, max_iter=10) -> np.ndarray:
    """Solves Kepler's Equation M = E - e*sin(E) using Newton-Raphson."""
    E = M.copy()
    for _ in range(max_iter):
        delta_E = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta_E
        if np.max(np.abs(delta_E)) < tol:
            break
    return E


def keplerian_to_cartesian(df: pd.DataFrame) -> np.ndarray:
    """
    Converts a DataFrame of Keplerian elements to Cartesian (X, Y, Z) coordinates.
    Expects columns: SEMIMAJOR_AXIS, ECCENTRICITY, INCLINATION, RA_OF_ASC_NODE,
                     ARG_OF_PERICENTER, MEAN_ANOMALY.
    Returns: Nx3 numpy array of [X, Y, Z] coordinates in kilometers.
    """
    # Convert degrees to radians for trigonometric functions
    deg2rad = np.pi / 180.0

    a = df["SEMIMAJOR_AXIS"].values
    e = df["ECCENTRICITY"].values
    i = df["INCLINATION"].values * deg2rad
    omega_node = df["RA_OF_ASC_NODE"].values * deg2rad
    omega_peri = df["ARG_OF_PERICENTER"].values * deg2rad
    M = df["MEAN_ANOMALY"].values * deg2rad

    # 1. Eccentric Anomaly
    E = solve_kepler(M, e)

    # 2. Orbital plane coordinates
    x_orb = a * (np.cos(E) - e)
    y_orb = a * (np.sqrt(1 - e**2) * np.sin(E))

    # 3. Rotate to 3D Cartesian (ECI)
    X = x_orb * (
        np.cos(omega_node) * np.cos(omega_peri)
        - np.sin(omega_node) * np.sin(omega_peri) * np.cos(i)
    ) - y_orb * (
        np.cos(omega_node) * np.sin(omega_peri)
        + np.sin(omega_node) * np.cos(omega_peri) * np.cos(i)
    )

    Y = x_orb * (
        np.sin(omega_node) * np.cos(omega_peri)
        + np.cos(omega_node) * np.sin(omega_peri) * np.cos(i)
    ) - y_orb * (
        np.sin(omega_node) * np.sin(omega_peri)
        - np.cos(omega_node) * np.cos(omega_peri) * np.cos(i)
    )

    Z = x_orb * (np.sin(omega_peri) * np.sin(i)) + y_orb * (
        np.cos(omega_peri) * np.sin(i)
    )

    return np.column_stack((X, Y, Z))


def detect_conjunctions(df: pd.DataFrame, threshold_km: float = 10.0) -> list:
    """
    Builds a KD-Tree of satellite positions and finds all pairs
    within the threshold distance.
    """
    logging.info("Converting orbital elements to Cartesian coordinates...")
    coords = keplerian_to_cartesian(df)

    logging.info(f"Building KD-Tree for {len(coords)} objects...")
    tree = KDTree(coords)

    logging.info(f"Querying KD-Tree for conjunctions within {threshold_km} km...")
    # query_pairs returns a set of index tuples (i, j)
    pairs = tree.query_pairs(r=threshold_km)

    conjunctions = []
    for i, j in pairs:
        dist = np.linalg.norm(coords[i] - coords[j])
        conjunctions.append(
            {
                "object_1_id": df.iloc[i]["NORAD_CAT_ID"],
                "object_1_name": df.iloc[i]["OBJECT_NAME"],
                "object_2_id": df.iloc[j]["NORAD_CAT_ID"],
                "object_2_name": df.iloc[j]["OBJECT_NAME"],
                "distance_km": dist,
            }
        )

    logging.info(f"Detected {len(conjunctions)} potential conjunction candidates.")
    return conjunctions


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(
        project_root, "data", "processed", "space_track_cleaned.csv"
    )

    logging.info("--- Starting Conjunction Detection ---")
    df = pd.read_csv(data_path)

    # We only want to test detection on a single snapshot in time.
    # Let's take the latest epoch for each satellite.
    df_snapshot = df.sort_values("EPOCH").groupby("NORAD_CAT_ID").last().reset_index()

    # Run detection. Using a generous threshold for demonstration
    # so we find pairs in a sparse 1000-object snapshot dataset.
    results = detect_conjunctions(df_snapshot, threshold_km=200.0)

    if results:
        print("\nTop 5 Conjunction Candidates:")
        # Sort by distance
        results = sorted(results, key=lambda x: x["distance_km"])
        for r in results[:5]:
            obj1 = f"{r['object_1_name']} (ID:{r['object_1_id']})"
            obj2 = f"{r['object_2_name']} (ID:{r['object_2_id']})"
            print(f"[{r['distance_km']:.2f} km] {obj1} <--> {obj2}")
    else:
        print("\nNo conjunctions found within the threshold.")

    logging.info("--- Conjunction Detection Completed ---")
