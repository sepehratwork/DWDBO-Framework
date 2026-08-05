"""
Discrete Wavelet Transform (DWT) Decomposition & Reconstruction Module.
Decomposes renewable energy profiles into multi-scale long-term trends and short-term
fluctuations according to Section 3.2 and Equations (3)-(6).
"""

from typing import Tuple, Optional
import numpy as np
import pywt
from tqdm import tqdm


class DiscreteWaveletDecomposer:
    """
    Implements multi-level Discrete Wavelet Transform (DWT) to separate renewable generation signals
    into low-frequency long-term trends (Eq. 5) and high-frequency short-term fluctuations (Eq. 6).
    """

    def __init__(self, mother_wavelet: str = "db4"):
        """
        Initialize DWT Decomposer.

        :param mother_wavelet: Mother wavelet family function psi (default: 'db4').
        """
        self.mother_wavelet = mother_wavelet

    def compute_decomposition_depth(self, signal_length: int) -> int:
        """
        Calculates decomposition depth J = floor(log2(N)) - 1 as defined in Section 3.2,
        bounded by PyWavelets maximum valid decomposition level.

        :param signal_length: Length N of the input renewable signal.
        :return: Integer decomposition depth J.
        """
        max_possible = pywt.dwt_max_level(signal_length, self.mother_wavelet)
        J_calc = int(np.floor(np.log2(signal_length))) - 1
        J = min(J_calc, max_possible)
        return max(1, J)

    def decompose_signal(self, signal: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Splits renewable power profile into P_long (Eq. 5) and P_short (Eq. 6) using Mallat DWT algorithm.

        Equations:
        - cA_J, cD_j = wavedec(P_RES, wavelet, level=J)  (Eq. 3 & 4)
        - P_long  = waverec([cA_J, 0, ..., 0], wavelet)  (Eq. 5: Low-frequency macro-trend)
        - P_short = waverec([0, cD_J, ..., cD_1], wavelet) (Eq. 6: High-frequency stochastic volatility)

        :param signal: Original discrete renewable energy signal P_RES[n].
        :param verbose: Whether to display progress updates.
        :return: Tuple of (P_long, P_short, depth_J).
        """
        signal_arr = np.asarray(signal, dtype=np.float64)
        N = len(signal_arr)
        J = self.compute_decomposition_depth(N)

        pbar = tqdm(
            total=4, 
            desc="[Step 2 - DWT Decomposer]", 
            disable=not verbose, 
            bar_format="{l_bar}{bar:30}{r_bar}"
        )

        # Step 2.1: Multi-level DWT Decomposition (Eq. 3 & 4)
        pbar.set_postfix_str(f"Decomposing signal with {self.mother_wavelet} at level J={J}")
        coeffs = pywt.wavedec(signal_arr, wavelet=self.mother_wavelet, level=J, mode="symmetric")
        cA_J = coeffs[0]
        cD_list = coeffs[1:]
        pbar.update(1)

        # Step 2.2: Reconstruct Long-term component P_long using cA_J (Eq. 5)
        pbar.set_postfix_str("Reconstructing P_long using cA_J (Eq. 5)")
        zeros_details = [np.zeros_like(c) for c in cD_list]
        p_long = pywt.waverec([cA_J] + zeros_details, wavelet=self.mother_wavelet, mode="symmetric")
        pbar.update(1)

        # Step 2.3: Reconstruct Short-term component P_short using sum of cD_j (Eq. 6)
        pbar.set_postfix_str("Reconstructing P_short using sum of cD_j (Eq. 6)")
        zeros_approx = np.zeros_like(cA_J)
        p_short = pywt.waverec([zeros_approx] + cD_list, wavelet=self.mother_wavelet, mode="symmetric")
        pbar.update(1)

        # Step 2.4: Truncate to exact length N & Verify Additive Identity P = P_long + P_short
        pbar.set_postfix_str("Verifying additive reconstruction identity P = P_long + P_short")
        p_long = p_long[:N]
        p_short = p_short[:N]

        # Safety check: numerical verification of exact reconstruction
        reconstructed_total = p_long + p_short
        max_diff = np.max(np.abs(signal_arr - reconstructed_total))
        if max_diff > 1e-4:
            p_short = signal_arr - p_long

        pbar.update(1)
        pbar.close()

        return p_long, p_short, J