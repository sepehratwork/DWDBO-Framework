"""
Adaptive Arithmetic Optimization Algorithm (Adaptive AOA).
Parallelized implementation of BESS placement and sizing problem with dynamic 
MOA and MOP schedules executing Algorithm 1 (Page 6).
"""

from typing import Tuple, List, Callable, Optional
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from config import AOAConfig, BESSConfig, ParallelConfig


def _eval_individual_worker(args: Tuple[int, np.ndarray, Callable[[np.ndarray], float]]) -> Tuple[int, np.ndarray, float]:
    """Top-level worker function for multiprocessing evaluation of candidate solutions."""
    idx, candidate, func = args
    fit = func(candidate)
    return idx, candidate, fit


class AdaptiveAOASolver:
    """
    Implements Adaptive AOA for optimal BESS siting and sizing in transmission networks
    with parallelized multiprocessing population evaluations.
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
        """Initializes population with spatial bus indices and continuous storage capacities (Eq. 21)."""
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
        fitness = np.zeros(self.cfg.population_size)

        # Parallel Initial Population Evaluation with Detailed Progress Bar
        print(f"[Adaptive AOA] Initializing population (Size={self.cfg.population_size}, Workers={self.n_workers})...")
        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(_eval_individual_worker, (i, pop[i], fitness_func))
                    for i in range(self.cfg.population_size)
                ]
                with tqdm(total=self.cfg.population_size, desc="[Step 2] Initial Population Eval", bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
                    for future in as_completed(futures):
                        idx, _, fit = future.result()
                        fitness[idx] = fit
                        pbar.update(1)
        else:
            for i in tqdm(range(self.cfg.population_size), desc="[Step 2] Initial Population Eval", bar_format="{l_bar}{bar:30}{r_bar}"):
                _, _, fit = _eval_individual_worker((i, pop[i], fitness_func))
                fitness[i] = fit

        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])

        convergence_curve = [best_fit]
        all_evaluated_fitness = list(fitness)

        # Main Optimization Loop with Persistent Process Pool
        if self.n_workers > 1:
            executor = ProcessPoolExecutor(max_workers=self.n_workers)
        else:
            executor = None

        try:
            pbar = tqdm(
                range(1, self.cfg.max_iterations + 1),
                desc="[Step 4] Adaptive AOA Siting & Sizing Optimization",
                unit="iter",
                bar_format="{l_bar}{bar:30}{r_bar}"
            )

            for t in pbar:
                # Adaptive MOA schedule (Eq. 19)
                moa = self.cfg.moa_min + t * ((self.cfg.moa_max - self.cfg.moa_min) / self.cfg.max_iterations)
                # Adaptive MOP schedule (Eq. 20)
                mop = 1.0 - (t / self.cfg.max_iterations) ** (1.0 / self.cfg.alpha)

                candidate_batch = []
                for i in range(self.cfg.population_size):
                    r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                    x_new = pop[i, :].copy()

                    if r1 < mop:
                        # Exploration Phase (Algorithm 1)
                        if r3 > 0.5:
                            x_new = pop[i, :] + r2 * (best_sol - pop[i, :]) * moa
                        else:
                            x_new = pop[i, :] - r2 * (best_sol - pop[i, :]) * moa
                    else:
                        # Exploitation Phase (Algorithm 1)
                        if r3 > 0.5:
                            x_new = best_sol + r2 * (best_sol - pop[i, :])
                        else:
                            x_new = best_sol - r2 * (best_sol - pop[i, :])

                    # Spatial and capacity bound constraints (Eq. 21)
                    buses = np.clip(np.round(x_new[: self.num_bess]), 1, self.total_buses)
                    if len(np.unique(buses)) < self.num_bess:
                        buses = np.random.choice(range(1, self.total_buses + 1), size=self.num_bess, replace=False)
                    x_new[: self.num_bess] = buses

                    x_new[self.num_bess :] = np.clip(
                        x_new[self.num_bess :], 5.0, self.bess_cfg.capacity_max_mwh
                    )
                    candidate_batch.append(x_new)

                # Parallel Candidate Batch Evaluation
                batch_evals = [0.0] * self.cfg.population_size
                if executor is not None:
                    futures = [
                        executor.submit(_eval_individual_worker, (i, cand, fitness_func))
                        for i, cand in enumerate(candidate_batch)
                    ]
                    for future in as_completed(futures):
                        idx, _, fit_val = future.result()
                        batch_evals[idx] = fit_val
                else:
                    for i, cand in enumerate(candidate_batch):
                        _, _, fit_val = _eval_individual_worker((i, cand, fitness_func))
                        batch_evals[i] = fit_val

                # Greedy Selection and Population Update
                for i in range(self.cfg.population_size):
                    x_new = candidate_batch[i]
                    fit_new = batch_evals[i]
                    all_evaluated_fitness.append(fit_new)

                    if fit_new < fitness[i]:
                        pop[i, :] = x_new
                        fitness[i] = fit_new
                        if fit_new < best_fit:
                            best_fit = fit_new
                            best_sol = x_new.copy()

                convergence_curve.append(best_fit)

                pbar.set_postfix({
                    "Best Fit": f"{best_fit:.2f}",
                    "MOA": f"{moa:.2f}",
                    "MOP": f"{mop:.2f}"
                })

                # Convergence Stopping Criterion (Eq. 23): |F^(k) - F^(k-1)| / (|F^(k-1)| + eps) < tau
                if t > 1:
                    rel_improvement = abs(convergence_curve[-1] - convergence_curve[-2]) / (abs(convergence_curve[-2]) + 1e-8)
                    if rel_improvement < 1e-6:
                        pbar.set_postfix_str(f"Converged at iter {t} (tau < 1e-6)")
                        break
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        return best_sol, best_fit, convergence_curve, all_evaluated_fitness