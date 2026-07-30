"""
Lower-Level Conditional Value-at-Risk (CVaR) Optimization Engine.
Performs real-time risk-averse dispatch adjustments under short-term forecast errors
as defined in Equations (16)-(18) with parallel sensitivity evaluations.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

from config import CVaRConfig, ParallelConfig


class CVaRRealTimeOptimizer:
    """
    Computes CVaR risk metric at confidence level alpha and optimizes short-term
    BESS real-time balancing adjustments.
    """

    def __init__(self, config: CVaRConfig, parallel_config: Optional[ParallelConfig] = None):
        self.cfg = config
        self.parallel_cfg = parallel_config or ParallelConfig()
        self.n_workers = self.parallel_cfg.get_effective_workers()

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

        :param base_operating_cost: Upper-level expected cost ($).
        :param error_scenarios: Short-term fluctuation scenario vector (MW).
        :param alpha_override: Optional specific alpha confidence level.
        :return: Tuple of (cvar_cost, expected_cost, var_threshold_zeta).
        """
        N_s = len(error_scenarios)
        pi_s = 1.0 / N_s
        alpha = alpha_override if alpha_override is not None else self.cfg.confidence_level_alpha

        scenario_costs = base_operating_cost + error_scenarios * 12.5

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

    def run_alpha_sensitivity_analysis(self, base_operating_cost: float, error_scenarios: np.ndarray) -> List[Dict[str, float]]:
        """
        Parallelized computation of CVaR impact across multiple alpha confidence levels (Table 6).

        :param base_operating_cost: Base operational cost.
        :param error_scenarios: Forecast error distribution scenarios.
        :return: List of metric dicts matching Table 6.
        """
        alphas = self.cfg.alpha_sensitivity_levels

        def compute_single_alpha(alpha_val):
            cvar, exp_cost, _ = self.optimize_cvar_risk(base_operating_cost, error_scenarios, alpha_override=alpha_val)
            return {
                "Confidence Level alpha": alpha_val,
                "Expected Cost ($)": round(exp_cost, 2),
                "CVaR Cost ($)": round(cvar, 2)
            }

        if self.n_workers > 1:
            results = Parallel(n_jobs=self.n_workers)(
                delayed(compute_single_alpha)(a) for a in alphas
            )
        else:
            results = [compute_single_alpha(a) for a in alphas]

        return results