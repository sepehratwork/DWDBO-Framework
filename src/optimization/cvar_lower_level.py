"""
Lower-Level Conditional Value-at-Risk (CVaR) Optimization Engine.
Performs real-time risk-averse dispatch adjustments under short-term forecast errors
as defined in Equations (16)-(18) with parallel sensitivity evaluations.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from config import CVaRConfig, ParallelConfig


class CVaRRealTimeOptimizer:
    """
    Computes CVaR risk metric at confidence level alpha and optimizes short-term
    BESS real-time balancing adjustments (Eq. 16 - Eq. 18).
    """

    def __init__(self, config: CVaRConfig, parallel_config: Optional[ParallelConfig] = None):
        self.cfg = config
        self.parallel_cfg = parallel_config or ParallelConfig()

    def sample_forecast_error_scenarios(self, base_p_short: float) -> np.ndarray:
        """Generates Gaussian stochastic forecast error scenarios (Eq. 18)."""
        np.random.seed(100)
        sigma = abs(base_p_short) * self.cfg.forecast_error_std + 1.0
        scenarios = np.random.normal(loc=0.0, scale=sigma, size=self.cfg.num_scenarios)
        return scenarios

    def optimize_cvar_risk(
        self, base_operating_cost: float, error_scenarios: np.ndarray, alpha_override: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Solves CVaR linear programming problem (Eq. 16-17).

        :param base_operating_cost: Upper-level expected operational cost ($).
        :param error_scenarios: Short-term fluctuation scenario vector (MW).
        :param alpha_override: Optional specific alpha confidence level.
        :return: Tuple of (cvar_cost, expected_cost, var_threshold_zeta).
        """
        N_s = len(error_scenarios)
        pi_s = 1.0 / N_s
        alpha = alpha_override if alpha_override is not None else self.cfg.confidence_level_alpha

        # Real-time imbalance scenario cost
        scenario_costs = base_operating_cost + np.abs(error_scenarios) * 12.5

        def cvar_obj(x: np.ndarray) -> float:
            zeta = x[0]
            eta = x[1:]
            return float(zeta + (1.0 / (1.0 - alpha)) * np.sum(pi_s * eta))

        constraints = []
        for s in range(N_s):
            def make_cvar_rule(scenario_idx):
                return lambda x: x[1 + scenario_idx] - (scenario_costs[scenario_idx] - x[0])
            constraints.append({'type': 'ineq', 'fun': make_cvar_rule(s)})

        bounds = [(None, None)] + [(0.0, None)] * N_s
        x0 = np.zeros(1 + N_s)
        x0[0] = float(np.percentile(scenario_costs, alpha * 100))

        res = minimize(cvar_obj, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        zeta_opt = float(res.x[0]) if res.success else float(x0[0])
        cvar_cost = float(cvar_obj(res.x)) if res.success else float(x0[0])
        expected_cost = float(np.mean(scenario_costs))

        return cvar_cost, expected_cost, zeta_opt

    def run_alpha_sensitivity_analysis(self, base_operating_cost: float, error_scenarios: np.ndarray) -> List[Dict[str, float]]:
        """
        Computes CVaR impact across multiple alpha confidence levels (Table 6 & Fig. 9).

        :param base_operating_cost: Base operational cost.
        :param error_scenarios: Forecast error distribution scenarios.
        :return: List of metric dicts matching Table 6.
        """
        alphas = self.cfg.alpha_sensitivity_levels
        results = []

        pbar_sens = tqdm(alphas, desc="[Step 5] CVaR Risk Sensitivity Analysis", unit="alpha", bar_format="{l_bar}{bar:30}{r_bar}")
        for a in pbar_sens:
            cvar, exp_cost, zeta = self.optimize_cvar_risk(base_operating_cost, error_scenarios, alpha_override=a)
            pbar_sens.set_postfix({"alpha": f"{a:.2f}", "CVaR ($)": f"{cvar:.2f}"})
            results.append({
                "Confidence Level alpha": a,
                "Expected Cost ($)": round(exp_cost, 2),
                "CVaR Cost ($)": round(cvar, 2),
                "VaR Threshold ($)": round(zeta, 2)
            })
        return results