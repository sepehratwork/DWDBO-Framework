"""
Adaptive Arithmetic Optimization Algorithm (Adaptive AOA).
Upper-level solver for optimal BESS siting and sizing across network buses.
"""

from typing import Tuple, List, Callable
import numpy as np
from config import AOAConfig, BESSConfig


class AdaptiveAOASolver:
    """
    Adaptive Arithmetic Optimization Algorithm with time-varying MOA and MOP schedules.
    Optimizes BESS location vectors L and capacity vectors S.
    """

    def __init__(self, config: AOAConfig, bess_config: BESSConfig, num_bess_units: int = 2, total_buses: int = 30):
        self.config = config
        self.bess_cfg = bess_config
        self.num_bess = num_bess_units
        self.total_buses = total_buses
        self.dim = num_bess_units * 2  # Encoding: [L_1..L_N, S_1..S_N]

    def _initialize_population() -> np.ndarray:
        pop = np.zeros((self.config.population_size, self.dim))
        for i in range(self.config.population_size):
            # Locations (bus indices 1 to 30)
            pop[i, : self.num_bess] = np.random.choice(range(1, self.total_buses + 1), size=self.num_bess, replace=False)
            # Capacities (MWh)
            pop[i, self.num_bess :] = np.random.uniform(
                self.bess_cfg.capacity_bounds_mwh[0], self.bess_cfg.capacity_bounds_mwh[1], size=self.num_bess
            )
        return pop

    def optimize(self, fitness_function: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float, List[float]]:
        """
        Executes adaptive AOA optimization loop.

        :param fitness_function: Objective evaluation function.
        :return: Tuple of (best_solution, best_fitness, convergence_history).
        """
        pop = self._initialize_population()
        fitness = np.array([fitness_function(ind) for ind in pop])
        
        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence_history = [best_fit]

        for t in range(1, self.config.max_iterations + 1):
            # Dynamic coefficient updates (Eqs. 19 & 20)
            moa = self.config.moa_min + t * ((self.config.moa_max - self.config.moa_min) / self.config.max_iterations)
            mop = 1.0 - (t / self.config.max_iterations) ** (1.0 / self.config.alpha)

            for i in range(self.config.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                
                # Exploration Phase
                if r1 < mop:
                    if r2 > 0.5:
                        pop[i, :] = best_sol + r2 * (best_sol - pop[i, :]) * moa
                    else:
                        pop[i, :] = best_sol - r2 * (best_sol - pop[i, :]) * moa
                # Exploitation Phase
                else:
                    if r2 > 0.5:
                        pop[i, :] = best_sol * r2 * moa
                    else:
                        pop[i, :] = best_sol / (moa + 1e-8)

                # Boundary enforcing
                pop[i, : self.num_bess] = np.clip(np.round(pop[i, : self.num_bess]), 1, self.total_buses)
                pop[i, self.num_bess :] = np.clip(
                    pop[i, self.num_bess :], self.bess_cfg.capacity_bounds_mwh[0], self.bess_cfg.capacity_bounds_mwh[1]
                )

                # Fitness evaluation
                fit = fitness_function(pop[i, :])
                if fit < fitness[i]:
                    fitness[i] = fit
                    if fit < best_fit:
                        best_fit = fit
                        best_sol = pop[i, :].copy()

            convergence_history.append(best_fit)

        return best_sol, best_fit, convergence_history