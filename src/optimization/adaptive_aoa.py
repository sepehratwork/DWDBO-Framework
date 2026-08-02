"""
Adaptive Arithmetic Optimization Algorithm (Adaptive AOA).
Parallelized implementation of BESS placement and sizing problem with dynamic 
MOA and MOP schedules executing Algorithm 1 (Page 6).
"""

from typing import Tuple, List, Callable, Optional
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from config import AOAConfig, BESSConfig, ParallelConfig


def _update_and_eval_individual(
    i: int,
    pop_i: np.ndarray,
    best_sol: np.ndarray,
    moa: float,
    mop: float,
    total_buses: int,
    num_bess: int,
    capacity_max_mwh: float,
    fitness_func: Callable[[np.ndarray], float]
) -> Tuple[int, np.ndarray, float]:
    """Helper top-level function for multiprocessing evaluation of a single individual."""
    r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
    x_new = pop_i.copy()

    # Exploration Phase (Algorithm 1)
    if r1 < mop:
        if r3 > 0.5:
            x_new = pop_i + r2 * (best_sol - pop_i) * moa
        else:
            x_new = pop_i - r2 * (best_sol - pop_i) * moa
    # Exploitation Phase (Algorithm 1)
    else:
        if r3 > 0.5:
            x_new = best_sol + r2 * (best_sol - pop_i)
        else:
            x_new = best_sol - r2 * (best_sol - pop_i)

    # Enforce physical constraints and integer bus bounds
    x_new[:num_bess] = np.clip(np.round(x_new[:num_bess]), 1, total_buses)
    x_new[num_bess:] = np.clip(x_new[num_bess:], 5.0, capacity_max_mwh)

    fit_new = fitness_func(x_new)
    return i, x_new, fit_new


class AdaptiveAOASolver:
    """
    Implements Adaptive AOA for optimal BESS siting and sizing in transmission networks
    with parallelized population updates and progress tracking.
    """

    def __init__(self, config: AOAConfig, bess_config: BESSConfig, total_buses: int = 30,
                 parallel_config: Optional[ParallelConfig] = None):
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
            pop[i, : self.num_bess] = np.random.choice(
                range(1, self.total_buses + 1), size=self.num_bess, replace=False
            )
            pop[i, self.num_bess :] = np.random.uniform(
                10.0, self.bess_cfg.capacity_max_mwh, size=self.num_bess
            )
        return pop

    def optimize(self, fitness_func: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float, List[float], List[float]]:
        """
        Executes Algorithm 1 Adaptive AOA optimization process using multiprocessing.

        :param fitness_func: Multi-objective evaluation function F_BESS (Eq. 22).
        :return: Tuple of (best_candidate_X, best_fitness_value, convergence_curve, population_fitness_distribution).
        """
        pop = self._initialize_population()
        
        # Parallel initial fitness evaluation
        if self.n_workers > 1:
            fitness = np.array(Parallel(n_jobs=self.n_workers)(
                delayed(fitness_func)(ind) for ind in pop
            ))
        else:
            fitness = np.array([fitness_func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])

        convergence_curve = [best_fit]
        all_evaluated_fitness = list(fitness)

        # Progress bar for AOA optimization
        pbar = tqdm(
            range(1, self.cfg.max_iterations + 1),
            desc="[Step 4/5] Adaptive AOA Optimization",
            unit="iter",
            leave=True
        )

        for t in pbar:
            # Dynamic MOA schedule (Eq. 19)
            moa = self.cfg.moa_min + t * ((self.cfg.moa_max - self.cfg.moa_min) / self.cfg.max_iterations)
            # Dynamic MOP schedule (Eq. 20)
            mop = 1.0 - (t / self.cfg.max_iterations) ** (1.0 / self.cfg.alpha)

            # Parallelized Population Update & Evaluation (Multiprocessing)
            if self.n_workers > 1:
                results = Parallel(n_jobs=self.n_workers)(
                    delayed(_update_and_eval_individual)(
                        i, pop[i], best_sol, moa, mop, self.total_buses, self.num_bess,
                        self.bess_cfg.capacity_max_mwh, fitness_func
                    )
                    for i in range(self.cfg.population_size)
                )
            else:
                results = [
                    _update_and_eval_individual(
                        i, pop[i], best_sol, moa, mop, self.total_buses, self.num_bess,
                        self.bess_cfg.capacity_max_mwh, fitness_func
                    )
                    for i in range(self.cfg.population_size)
                ]

            # Update population state
            for i, x_new, fit_new in results:
                all_evaluated_fitness.append(fit_new)
                if fit_new < fitness[i]:
                    pop[i, :] = x_new
                    fitness[i] = fit_new
                    if fit_new < best_fit:
                        best_fit = fit_new
                        best_sol = x_new.copy()

            convergence_curve.append(best_fit)

            # Update progress bar description with metrics
            pbar.set_postfix({
                "Best Cost ($)": f"{best_fit:.2f}",
                "MOA": f"{moa:.2f}",
                "MOP": f"{mop:.2f}"
            })

            # Early stopping check (Eq. 23)
            if t > 1:
                rel_improvement = abs(convergence_curve[-1] - convergence_curve[-2]) / (abs(convergence_curve[-2]) + 1e-8)
                if rel_improvement < 1e-6:
                    pbar.set_postfix({"Status": "Converged (tau < 1e-6)"})
                    break

        return best_sol, best_fit, convergence_curve, all_evaluated_fitness