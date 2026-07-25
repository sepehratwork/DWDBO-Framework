"""
Main Execution Entry Point for the DWDBO Framework.
Generates synthetic data (replicating OPSD structure) and executes the solver.
"""

import numpy as np
import pandas as pd
from src.pipeline.dwdbo_solver import DWDBOFrameworkSolver


def create_synthetic_opsd_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """Generates synthetic multivariate power system dataset matching OPSD format."""
    np.random.seed(42)
    time_index = pd.date_range("2020-01-01", periods=num_samples, freq="15min")
    
    demand = 4000 + 1000 * np.sin(np.linspace(0, 50, num_samples)) + np.random.normal(0, 100, num_samples)
    wind = 3000 + 1500 * np.cos(np.linspace(0, 30, num_samples)) + np.random.normal(0, 200, num_samples)
    solar = np.maximum(0, 1200 * np.sin(np.linspace(0, 100, num_samples))) + np.random.normal(0, 50, num_samples)
    price = 40 + 10 * np.sin(np.linspace(0, 40, num_samples)) + np.random.normal(0, 2, num_samples)

    # Introduce random missing values (NaN gaps) to test KNN Imputer
    wind[np.random.choice(num_samples, size=30, replace=False)] = np.nan
    solar[np.random.choice(num_samples, size=30, replace=False)] = np.nan

    df = pd.DataFrame({
        "timestamp": time_index,
        "load_demand": demand,
        "wind_power": wind,
        "solar_power": solar,
        "price": price
    }).set_index("timestamp")

    return df


def main():
    print("==========================================================================")
    print(" Deep-Based Wavelet-Driven Bi-Level Optimization (DWDBO) Framework Solver ")
    print("==========================================================================\n")

    # Generate or load dataset
    dataset = create_synthetic_opsd_dataset(num_samples=1000)

    # Initialize and run solver
    solver = DWDBOFrameworkSolver()
    results = solver.run_pipeline(dataset)

    print("\n==========================================================================")
    print(" Execution Completed Successfully! ")
    print("==========================================================================")


if __name__ == "__main__":
    main()