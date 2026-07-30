"""
Adaptive Arithmetic Optimization Algorithm (Adaptive AOA).
Parallelized implementation of BESS placement and sizing problem with dynamic 
MOA and MOP schedules executing Algorithm 1 (Page 6).
"""

from typing import Tuple, List, Callable, Optional
import numpy as np
from joblib import Parallel, delayed

from config import AOAConfig, BESSConfig, ParallelConfig


class AdaptiveAOASolver:
    """
    Implements Adaptive AOA for optimal BESS siting and sizing in transmission networks
    with parallelized population evaluations.
    """

    def __init__(self, config: AOAConfig, bess_config: BESSConfig, total_buses: int = 30,
                 parallel_config: Optional[ParallelConfig] = None):
        """
        Initialize Adaptive AOA Solver.

        :param config: AOA hyperparameters configuration.
        :param bess_config: BESS physical capacity bounds configuration.
        :param total_buses: Number of power system network buses.
        :param parallel_config: Parallel execution parameters.
        """
        self.cfg = config
        self.bess_cfg = bess_config
        self.total_buses = total_buses
        self.num_bess = bess_config.num_units
        self.dim = self.num_bess * 2  # Vector X = [L_1, ..., L_N, S_1, ..., S_N] (Eq. 21)
        self.parallel_cfg = parallel_config or ParallelConfig()
        self.n_workers = self.parallel_cfg.get_effective_workers()

    def _initialize_population(self) -> np.ndarray:
        """Initializes population with spatial bus indices and continuous storage capacities."""
        pop = np.zeros((self.cfg.population_size, self.dim))
        for i in range(self.cfg.population_size):
            # Spatial dimension L_i (Bus indices)
            pop[i, : self.num_bess] = np.random.choice(
                range(1, self.total_buses + 1), size=self.num_bess, replace=False
            )
            # Capacity dimension S_i (MWh)
            pop[i, self.num_bess :] = np.random.uniform(
                self.bess_cfg.capacity_min_mwh, self.bess_cfg.capacity_max_mwh, size=self.num_bess
            )
        return pop

    def _evaluate_population_parallel(self, pop: np.ndarray, fitness_func: Callable[[np.ndarray], float]) -> np.ndarray:
        """
        Parallelized fitness function evaluation across candidate solutions.

        :param pop: Candidate population matrix.
        :param fitness_func: Multi-objective fitness function.
        :return: Array of fitness scores.
        """
        if self.n_workers > 1:
            fitness = Parallel(n_jobs=self.n_workers)(
                delayed(fitness_func)(ind) for ind in pop
            )
            return np.array(fitness)
        else:
            return np.array([fitness_func(ind) for ind in pop])

    def optimize(self, fitness_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float, List[float]]:
        """
        Executes Algorithm 1 Adaptive AOA optimization process.

        :param fitness_func: Multi-objective evaluation function F_BESS (Eq. 22).
        :return: Tuple of (best_candidate_X, best_fitness_value, convergence_curve).
        """
        pop = self._initialize_population()
        fitness = self._evaluate_population_parallel(pop, fitness_func)

        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        convergence_curve = [best_fit]

        for t in range(1, self.cfg.max_iterations + 1):
            # Dynamic MOA schedule (Eq. 19)
            moa = self.cfg.moa_min + t * ((self.cfg.moa_max - self.cfg.moa_min) / self.cfg.max_iterations)
            # Dynamic MOP schedule (Eq. 20)
            mop = 1.0 - (t / self.cfg.max_iterations) ** (1.0 / self.cfg.alpha)

            for i in range(self.cfg.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                x_new = pop[i, :].copy()

                # Exploration Phase (Algorithm 1)
                if r1 < mop:
                    x_new = pop[i, :] + r2 * (best_sol - pop[i, :]) * moa
                # Exploitation Phase (Algorithm 1)
                else:
                    x_new = best_sol + r2 * (best_sol - pop[i, :])

                # Enforce physical constraints and integer bus bounds
                x_new[: self.num_bess] = np.clip(np.round(x_new[: self.num_bess]), 1, self.total_buses)
                x_new[self.num_bess :] = np.clip(
                    x_new[self.num_bess :], self.bess_cfg.capacity_min_mwh, self.bess_cfg.capacity_max_mwh
                )

                fit_new = fitness_func(x_new)
                if fit_new < fitness[i]:
                    pop[i, :] = x_new
                    fitness[i] = fit_new
                    if fit_new < best_fit:
                        best_fit = fit_new
                        best_sol = x_new.copy()

            convergence_curve.append(best_fit)

        return best_sol, best_fit, convergence_curve