"""
Main Execution Script.
Simulates Open Power System Data (OPSD), initializes the DWDBO pipeline,
executes multi-core optimization, saves checkpoints, and outputs all
figures and tables matching the paper.
"""

import time
import numpy as np
import pandas as pd

from config import ParallelConfig, CacheConfig, OutputConfig
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
    print(" Deep-Learning-Based Wavelet-Driven Bi-Level Optimization (DWDBO) Framework")
    print("==========================================================================================\n")

    # Global Parallel & System Settings
    parallel_cfg = ParallelConfig(num_workers=-1)
    cache_cfg = CacheConfig(enable_cache=True, cache_dir="cache")
    output_cfg = OutputConfig(results_dir="results", export_figures=True, export_tables=True)

    effective_workers = parallel_cfg.get_effective_workers()
    print(f"[System Config] Parallelization Workers Active: {effective_workers} Cores")
    print(f"[System Config] Step Caching Enabled: {cache_cfg.enable_cache} ('{cache_cfg.cache_dir}/')")
    print(f"[System Config] Output Results Directory: '{output_cfg.results_dir}/'\n")

    start_time = time.perf_counter()

    # Generate or load dataset
    raw_opsd_data = generate_opsd_time_series(num_samples=5000)

    # Initialize DWDBO Solver Engine
    solver = DWDBOMasterFramework(
        parallel_cfg=parallel_cfg,
        cache_cfg=cache_cfg,
        output_cfg=output_cfg
    )

    # Execute Complete 24-Hour Pipeline
    results = solver.execute_framework(raw_opsd_data, scheduling_horizon_hours=24)

    # total_time = time.perf_counter() - start_time

    # print("\n------------------------------------------------------------------------------------------")
    # print(" SIMULATION SUMMARY RESULTS ")
    # print("------------------------------------------------------------------------------------------")
    # print(f"Optimal BESS Bus Locations     : Bus {results['optimal_bess_buses'][0]} and Bus {results['optimal_bess_buses'][1]}")
    # print(f"Optimal BESS Capacities (MWh)  : {results['optimal_bess_capacities_mwh'][0]:.2f} MWh, {results['optimal_bess_capacities_mwh'][1]:.2f} MWh")
    # print(f"Total Operational Cost ($)     : ${results['operating_cost']:.2f}")
    # print(f"Expected System Cost ($)       : ${results['expected_cost']:.2f}")
    # print(f"CVaR Risk Cost (Alpha=0.95) ($): ${results['cvar_cost']:.2f}")
    # print(f"Value-at-Risk Threshold ($)    : ${results['var_threshold_zeta']:.2f}")
    # print(f"Total Pipeline Execution Time  : {total_time:.2f} seconds")
    # print("==========================================================================================")


if __name__ == "__main__":
    main()