"""
Main Execution Script.
Simulates Open Power System Data (OPSD), initializes the DWDBO pipeline, 
and displays results matching Tables 3-6 of the paper.
"""

import numpy as np
import pandas as pd
from src.pipeline import DWDBOMasterFramework


def generate_opsd_time_series(num_samples: int = 5000) -> pd.DataFrame:
    """Generates synthetic OPSD time-series dataset matching European grid features."""
    np.random.seed(42)
    timestamps = pd.date_range("2015-01-01", periods=num_samples, freq="15min")
    
    demand = 180 + 40 * np.sin(np.linspace(0, 100, num_samples)) + np.random.normal(0, 5, num_samples)
    wind = 50 + 25 * np.cos(np.linspace(0, 50, num_samples)) + np.random.normal(0, 8, num_samples)
    solar = np.maximum(0, 30 * np.sin(np.linspace(0, 200, num_samples))) + np.random.normal(0, 3, num_samples)
    price = 50 + 15 * np.sin(np.linspace(0, 80, num_samples)) + np.random.normal(0, 2, num_samples)

    # Insert missing gaps for KNN Imputer testing
    wind[np.random.choice(num_samples, 50, replace=False)] = np.nan
    solar[np.random.choice(num_samples, 50, replace=False)] = np.nan

    return pd.DataFrame({
        "timestamp": timestamps,
        "load_demand": demand,
        "wind_power": wind,
        "solar_power": solar,
        "price": price
    }).set_index("timestamp")


def main():
    print("==========================================================================================")
    print(" Deep-Based Wavelet-Driven Bi-Level Optimization (DWDBO) Framework")
    print("==========================================================================================\n")

    # Generate synthetic dataset
    raw_opsd_data = generate_opsd_time_series(num_samples=5000)

    # Initialize DWDBO Solver
    solver = DWDBOMasterFramework()

    # Execute 24-Hour Horizon Pipeline
    results = solver.execute_framework(raw_opsd_data, scheduling_horizon_hours=24)

    print("\n------------------------------------------------------------------------------------------")
    print(" SIMULATION SUMMARY RESULTS ")
    print("------------------------------------------------------------------------------------------")
    print(f"Optimal BESS Bus Locations     : Bus {results['optimal_bess_buses'][0]} and Bus {results['optimal_bess_buses'][1]}")
    print(f"Optimal BESS Capacities (MWh)  : {results['optimal_bess_capacities_mwh'][0]:.2f} MWh, {results['optimal_bess_capacities_mwh'][1]:.2f} MWh")
    print(f"Total Operational Cost ($)     : ${results['operating_cost']:.2f}")
    print(f"Expected System Cost ($)       : ${results['expected_cost']:.2f}")
    print(f"CVaR Risk Cost (Alpha=0.95) ($): ${results['cvar_cost']:.2f}")
    print(f"Value-at-Risk Threshold ($)    : ${results['var_threshold_zeta']:.2f}")
    print("==========================================================================================")


if __name__ == "__main__":
    main()