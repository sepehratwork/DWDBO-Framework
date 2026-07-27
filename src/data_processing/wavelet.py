"""
Discrete Wavelet Transform (DWT) Decomposition & Reconstruction Module.
Decomposes renewable energy profiles into multi-scale long-term trends and short-term
fluctuations according to Equations (3)-(6).
"""

from typing import Tuple
import numpy as np
import pywt


class DiscreteWaveletDecomposer:
    """
    Implements multi-level Discrete Wavelet Transform (DWT) to separate renewable generation signals.
    """

    def __init__(self, mother_wavelet: str = "db4"):
        """
        Initialize DWT Decomposer.

        :param mother_wavelet: Mother wavelet family function psi (default: 'db4').
        """
        self.mother_wavelet = mother_wavelet

    def compute_decomposition_depth(self, signal_length: int) -> int:
        """
        Calculates decomposition depth J = floor(log2(N)) - 1 as defined in Section 2.2.

        :param signal_length: Length N of the input renewable signal.
        :return: Integer decomposition depth J.
        """
        J = int(np.floor(np.log2(signal_length))) - 1
        return max(1, J)

    def decompose_signal(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Splits renewable power profile into P_long (Eq. 5) and P_short (Eq. 6).

        :param signal: Original discrete renewable energy signal P_RES[n].
        :return: Tuple of (P_long, P_short, depth_J).
        """
        N = len(signal)
        J = self.compute_decomposition_depth(N)

        # Multi-level DWT decomposition
        coeffs = pywt.wavedec(signal, wavelet=self.mother_wavelet, level=J)

        # Reconstruct Long-term component P_long using approximation coefficients cA_J (Eq. 5)
        cA_J = coeffs[0]
        zeros_details = [np.zeros_like(c) for c in coeffs[1:]]
        p_long = pywt.waverec([cA_J] + zeros_details, wavelet=self.mother_wavelet)

        # Reconstruct Short-term component P_short summing detail coefficients cD_j (Eq. 6)
        zeros_approx = np.zeros_like(cA_J)
        p_short = pywt.waverec([zeros_approx] + coeffs[1:], wavelet=self.mother_wavelet)

        # Ensure output arrays exactly match input length N
        p_long = p_long[:N]
        p_short = p_short[:N]

        return p_long, p_short, J