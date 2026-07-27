"""
Lower-Level Conditional Value-at-Risk (CVaR) Optimization Engine.
Performs real-time risk-averse dispatch adjustments under short-term forecast errors
as defined in Equations (16)-(18).
"""

from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from config import CVaRConfig


class CVaRRealTimeOptimizer:
    """
    Computes CVaR risk metric at confidence level alpha and optimizes short-term
    BESS real-time balancing adjustments.
    """

    def __init__(self, config: CVaRConfig):
        self.cfg = config

    def sample_forecast_error_scenarios(self, base_p_short: float) -> np.ndarray:
        """Generates Gaussian stochastic forecast error scenarios (Eq. 18)."""
        np.random.seed(100)
        sigma = abs(base_p_short) * self.cfg.forecast_error_std + 1.0
        scenarios = np.random.normal(loc=0.0, scale=sigma, size=self.cfg.num_scenarios)
        return scenarios

    def optimize_cvar_risk(
        self, base_operating_cost: float, error_scenarios: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Solves CVaR linear programming problem (Eq. 16-17).

        :param base_operating_cost: Upper-level expected cost ($).
        :param error_scenarios: Short-term fluctuation scenario vector (MW).
        :return: Tuple of (cvar_cost, expected_cost, var_threshold_zeta).
        """
        N_s = len(error_scenarios)
        pi_s = 1.0 / N_s
        alpha = self.cfg.confidence_level_alpha

        # Scenario total operational cost impact
        scenario_costs = base_operating_cost + error_scenarios * 12.5

        # Decision variables: x = [zeta, eta_1, ..., eta_NS]
        def cvar_obj(x):
            zeta = x[0]
            eta = x[1:]
            return zeta + (1.0 / (1.0 - alpha)) * np.sum(pi_s * eta)

        constraints = []
        for s in range(N_s):
            constraints.append({'type': 'ineq', 'fun': lambda x, s=s: x[1 + s] - (scenario_costs[s] - x[0])})

        bounds = [(None, None)] + [(0.0, None)] * N_s
        x0 = np.zeros(1 + N_s)
        x0[0] = np.percentile(scenario_costs, alpha * 100)

        res = minimize(cvar_obj, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        zeta_opt = float(res.x[0]) if res.success else float(x0[0])
        cvar_cost = float(cvar_obj(res.x)) if res.success else float(x0[0])
        expected_cost = float(np.mean(scenario_costs))

        return cvar_cost, expected_cost, zeta_opt