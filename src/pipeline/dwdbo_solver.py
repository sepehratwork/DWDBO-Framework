"""
Integrated DWDBO Master Pipeline Coordinator.
Implements Algorithm 1 bi-level iteration logic, convergence checking (Eq. 23),
and executes complete end-to-end framework execution using clean package imports.
"""

from typing import Dict, Any
import numpy as np

from config import (
    DatasetConfig, TFTConfig, BESSConfig, AOAConfig, CVaRConfig, ConvergenceConfig
)
from src.data_processing import TimeSeriesKNNImputer, DiscreteWaveletDecomposer
from src.models import TFTTrainerEngine
from src.power_system import IEEE30BusData, MultiPeriodOPFSolver
from src.optimization import AdaptiveAOASolver, CVaRRealTimeOptimizer


class DWDBOMasterFramework:
    """Master controller executing integrated bi-level optimization with CVaR risk control."""

    def __init__(self):
        self.dataset_cfg = DatasetConfig()
        self.tft_cfg = TFTConfig()
        self.bess_cfg = BESSConfig()
        self.aoa_cfg = AOAConfig()
        self.cvar_cfg = CVaRConfig()
        self.conv_cfg = ConvergenceConfig()

        self.imputer = TimeSeriesKNNImputer()
        self.decomposer = DiscreteWaveletDecomposer()
        self.tft_engine = TFTTrainerEngine(self.tft_cfg)
        self.sys_data = IEEE30BusData()
        self.opf_solver = MultiPeriodOPFSolver(self.sys_data, self.bess_cfg)
        self.aoa_solver = AdaptiveAOASolver(self.aoa_cfg, self.bess_cfg)
        self.cvar_optimizer = CVaRRealTimeOptimizer(self.cvar_cfg)

    def execute_framework(self, df_raw: Any, scheduling_horizon_hours: int = 24) -> Dict[str, Any]:
        """
        Executes complete paper flow:
        1. Imputation -> 2. DWT -> 3. TFT -> 4. Upper Adaptive AOA -> 5. Lower CVaR -> 6. Convergence.
        """
        print(f"--- Executing DWDBO Pipeline ({scheduling_horizon_hours}-Hour Horizon) ---")

        # Step 1: Missing Data Imputation
        df_clean = self.imputer.impute_missing_data(df_raw)
        res_signal = df_clean["wind_power"].values

        # Step 2: DWT Signal Decomposition
        p_long, p_short, depth_J = self.decomposer.decompose_signal(res_signal)
        print(f"DWT Decomposition Completed (Level J = {depth_J})")

        # Step 3: Dual-Path TFT Forecasting
        metrics, pred_long, pred_short = self.tft_engine.train_and_forecast(p_long, p_short)
        print(f"TFT Forecasting Accuracy -> R2: {metrics['R2']:.4f}, MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")

        # Horizon vectors
        T = scheduling_horizon_hours
        demand_h = np.full(T, self.sys_data.base_demand)
        p_long_h = pred_long[:T]
        p_short_h = pred_short[:T]

        # Step 4: Upper-Level Adaptive AOA Optimization
        def multi_objective_fitness(X: np.ndarray) -> float:
            buses = X[: self.bess_cfg.num_units].astype(int)
            capacities = X[self.bess_cfg.num_units :]
            
            c_op, c_bess, v_dev, l_loss = self.opf_solver.solve_multi_period_dispatch(
                T, demand_h, p_long_h, buses, capacities
            )
            c_inv = np.sum(capacities) * 15.0  # Investment cost proxy
            
            w = self.aoa_cfg.weights
            fitness = w[0] * c_op + w[1] * c_inv + w[2] * v_dev * 100.0 + w[3] * l_loss * 100.0
            return fitness

        # Bi-level iterative convergence loop (Eq. 23)
        prev_fitness = float("inf")
        best_X = None
        best_fitness = float("inf")
        conv_history = []

        for outer_iter in range(1, self.conv_cfg.max_outer_iterations + 1):
            best_X, best_fitness, curve = self.aoa_solver.optimize(multi_objective_fitness)
            conv_history.extend(curve)

            # Stopping condition check (Eq. 23)
            rel_improvement = abs(best_fitness - prev_fitness) / (prev_fitness + self.conv_cfg.epsilon_stabilizer)
            if rel_improvement < self.conv_cfg.tau_tolerance:
                print(f"Bi-Level Framework Converged at Outer Iteration {outer_iter} (Ratio < 1e-6)")
                break
            prev_fitness = best_fitness

        opt_buses = best_X[: self.bess_cfg.num_units].astype(int)
        opt_capacities = best_X[self.bess_cfg.num_units :]

        # Step 5: Lower-Level CVaR Risk Optimization
        c_op_opt, _, _, _ = self.opf_solver.solve_multi_period_dispatch(
            T, demand_h, p_long_h, opt_buses, opt_capacities
        )
        error_scenarios = self.cvar_optimizer.sample_forecast_error_scenarios(np.mean(p_short_h))
        cvar_cost, exp_cost, zeta = self.cvar_optimizer.optimize_cvar_risk(c_op_opt, error_scenarios)

        return {
            "forecast_metrics": metrics,
            "optimal_bess_buses": opt_buses,
            "optimal_bess_capacities_mwh": opt_capacities,
            "operating_cost": c_op_opt,
            "expected_cost": exp_cost,
            "cvar_cost": cvar_cost,
            "var_threshold_zeta": zeta,
            "convergence_history": conv_history
        }