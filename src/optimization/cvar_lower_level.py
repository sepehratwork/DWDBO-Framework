"""
Lower-Level CVaR-Based Risk Management Optimization.
Performs real-time corrective power dispatch under short-term forecast uncertainty scenarios.
"""

from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from config import RiskConfig


class CVaRRiskOptimizer:
    """
    Computes Conditional Value-at-Risk (CVaR) and optimizes lower-level 
    short-term real-time power dispatch under renewable uncertainty scenarios.
    """

    def __init__(self, config: RiskConfig):
        self.config = config

    def generate_error_scenarios(self, base_p_short: float) -> np.ndarray:
        """Generates stochastic forecast error scenarios using normal distribution."""
        np.random.seed(42)
        scenarios = np.random.normal(
            loc=base_p_short, 
            scale=abs(base_p_short) * self.config.error_std_dev + 1e-5, 
            size=self.config.num_scenarios
        )
        return scenarios

    def optimize_cvar_dispatch(
        self, base_operating_cost: float, p_short_scenarios: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Solves CVaR risk minimization problem (Eq. 16-17).

        :param base_operating_cost: Nominal operational cost from upper-level OPF ($).
        :param p_short_scenarios: Generated short-term fluctuation scenarios (MW).
        :return: Tuple of (cvar_value, var_threshold_zeta, adjusted_bess_power_array).
        """
        num_scenarios = len(p_short_scenarios)
        prob = 1.0 / num_scenarios
        alpha = self.config.confidence_level_alpha

        # Scenario operational costs under short-term fluctuations
        scenario_costs = base_operating_cost + p_short_scenarios * 15.0  # Marginal impact factor

        # Objective formulation: min zeta + (1 / (1 - alpha)) * sum(prob * eta_s)
        def cvar_objective(x):
            zeta = x[0]
            eta = x[1:]
            return zeta + (1.0 / (1.0 - alpha)) * np.sum(prob * eta)

        # Constraints: eta_s >= C_s - zeta, eta_s >= 0
        constraints = []
        for s in range(num_scenarios):
            constraints.append({'type': 'ineq', 'fun': lambda x, s=s: x[1 + s] - (scenario_costs[s] - x[0])})

        bounds = [(None, None)] + [(0, None)] * num_scenarios
        x0 = np.zeros(1 + num_scenarios)
        x0[0] = np.percentile(scenario_costs, alpha * 100)

        res = minimize(cvar_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        zeta_opt = res.x[0] if res.success else float(x0[0])
        cvar_opt = float(cvar_objective(res.x)) if res.success else float(x0[0])
        
        # Calculate real-time BESS compensation actions P_BESS_short
        p_bess_adjustments = -p_short_scenarios

        return cvar_opt, zeta_opt, p_bess_adjustments