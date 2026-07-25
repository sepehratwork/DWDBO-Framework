"""
DWDBO Bi-Level Optimization Master Framework Solver.
Coordinates Wavelet Decomposition, TFT Forecasting, Upper-Level Adaptive AOA, 
and Lower-Level CVaR Risk Management into a unified architecture.
"""

from typing import Dict, Any
import numpy as np

from config import ModelConfig, BESSConfig, AOAConfig, RiskConfig, ConvergenceConfig
from src.data_processing.imputer import TimeSeriesKNNImputer
from src.data_processing.wavelet import WaveletSignalDecomposer
from src.models.trainer import TFTForecastingEngine
from src.power_system.ieee30_data import IEEE30BusSystem
from src.power_system.power_flow import OptimalPowerFlowSolver
from src.optimization.adaptive_aoa import AdaptiveAOASolver
from src.optimization.cvar_lower_level import CVaRRiskOptimizer


class DWDBOFrameworkSolver:
    """Integrated Master Controller executing the full end-to-end DWDBO framework."""

    def __init__(self):
        self.model_cfg = ModelConfig()
        self.bess_cfg = BESSConfig()
        self.aoa_cfg = AOAConfig()
        self.risk_cfg = RiskConfig()
        self.conv_cfg = ConvergenceConfig()

        self.imputer = TimeSeriesKNNImputer()
        self.decomposer = WaveletSignalDecomposer()
        self.tft_engine = TFTForecastingEngine(self.model_cfg)
        self.sys_data = IEEE30BusSystem()
        self.opf_solver = OptimalPowerFlowSolver(self.sys_data)
        self.aoa_solver = AdaptiveAOASolver(self.aoa_cfg, self.bess_cfg)
        self.cvar_optimizer = CVaRRiskOptimizer(self.risk_cfg)

    def run_pipeline(self, raw_data_df: Any) -> Dict[str, Any]:
        """
        Executes the full paper workflow:
        1. Data Imputation -> 2. Wavelet Decomposition -> 3. TFT Forecasting 
        -> 4. Upper-level BESS Siting/Sizing -> 5. Lower-level CVaR Control.
        """
        print("=== Step 1: Performing Missing Data Imputation (KNN) ===")
        df_clean = self.imputer.impute(raw_data_df)
        res_profile = df_clean['wind_power'].values

        print("\n=== Step 2: Signal Decomposition via Discrete Wavelet Transform (DWT) ===")
        p_long, p_short = self.decomposer.decompose(res_profile)

        print("\n=== Step 3: Dual-Path TFT Model Training and Forecasting ===")
        metrics, pred_long, pred_short = self.tft_engine.train_and_evaluate(p_long, p_short)
        print(f"TFT Prediction Accuracy -> MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R2: {metrics['R2']:.4f}")

        print("\n=== Step 4 & 5: Bi-Level Optimization (Upper Adaptive AOA + Lower CVaR) ===")
        
        # Target single-period dispatch sample
        sample_demand = self.sys_data.base_demand
        sample_p_long = float(np.mean(pred_long))
        sample_p_short = float(np.mean(pred_short))

        # Upper-Level Multi-objective fitness function
        def upper_level_fitness(x: np.ndarray) -> float:
            buses = x[:2].astype(int)
            capacities = x[2:]
            
            # Assume rated powers proportional to capacity
            bess_actions = {buses[0]: capacities[0] * 0.1, buses[1]: capacities[1] * 0.1}
            
            c_op, pg, v_dev, l_loss = self.opf_solver.solve_dispatch(sample_demand, sample_p_long, bess_actions)
            c_inv = np.sum(capacities) * 100.0  # Annualized investment cost proxy
            
            w = self.aoa_cfg.weights
            total_fitness = w[0] * c_op + w[1] * c_inv + w[2] * v_dev * 1000 + w[3] * l_loss * 1000
            return total_fitness

        # Execute Upper-Level Adaptive AOA
        best_x, best_fit, conv_curve = self.aoa_solver.optimize(upper_level_fitness)
        
        best_buses = best_x[:2].astype(int)
        best_capacities = best_x[2:]
        print(f"Optimal BESS Placement Buses: {best_buses}")
        print(f"Optimal BESS Capacities (MWh): {np.round(best_capacities, 2)}")

        # Execute Lower-Level CVaR Risk Dispatch
        optimal_bess_actions = {best_buses[0]: best_capacities[0] * 0.1, best_buses[1]: best_capacities[1] * 0.1}
        base_cost, _, _, _ = self.opf_solver.solve_dispatch(sample_demand, sample_p_long, optimal_bess_actions)
        scenarios = self.cvar_optimizer.generate_error_scenarios(sample_p_short)
        cvar_cost, zeta, _ = self.cvar_optimizer.optimize_cvar_dispatch(base_cost, scenarios)

        print(f"Lower-Level CVaR Cost (Alpha={self.risk_cfg.confidence_level_alpha}): ${cvar_cost:.2f}")
        print(f"Value-at-Risk Threshold (Zeta): ${zeta:.2f}")

        return {
            "metrics": metrics,
            "bess_buses": best_buses,
            "bess_capacities": best_capacities,
            "cvar_cost": cvar_cost,
            "convergence_history": conv_curve
        }