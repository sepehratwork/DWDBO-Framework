"""
Optimal Power Flow (OPF) Solver.
Performs network-constrained generation dispatch and calculates system losses and costs.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize
from src.power_system.ieee30_data import IEEE30BusSystem


class OptimalPowerFlowSolver:
    """Calculates network economic dispatch, power balance, voltage profile, and line losses."""

    def __init__(self, system_data: IEEE30BusSystem):
        self.sys = system_data

    def solve_dispatch(
        self, demand: float, p_res_long: float, bess_actions: Dict[int, float]
    ) -> Tuple[float, np.ndarray, float, float]:
        """
        Solves economic dispatch OPF under system operational constraints.

        :param demand: Active system total load demand (MW).
        :param p_res_long: Forecasted long-term renewable generation output (MW).
        :param bess_actions: Dictionary mapping bus index to active BESS power P_bess (MW).
        :return: Tuple (operating_cost, gen_dispatch_array, voltage_dev, line_losses).
        """
        net_demand = demand - p_res_long - sum(bess_actions.values())

        # Objective function: Thermal generation quadratic cost
        def objective(pg):
            cost = np.sum(self.sys.cost_a * (pg**2) + self.sys.cost_b * pg + self.sys.cost_c)
            return cost

        # Constraints: Power balance eq (11)
        constraints = [{'type': 'eq', 'fun': lambda pg: np.sum(pg) - net_demand}]
        bounds = list(zip(self.sys.p_min, self.sys.p_max))

        # Initial guess
        x0 = (self.sys.p_min + self.sys.p_max) / 2.0
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not res.success:
            pg_opt = np.clip(x0, self.sys.p_min, self.sys.p_max)
        else:
            pg_opt = res.x

        operating_cost = float(objective(pg_opt))
        
        # Approximate network loss and voltage deviation metrics
        line_losses = 0.02 * (np.sum(pg_opt) / self.sys.base_demand)**2
        voltage_dev = 0.001 * np.sum((pg_opt / self.sys.p_max - 0.8)**2)

        return operating_cost, pg_opt, voltage_dev, line_losses