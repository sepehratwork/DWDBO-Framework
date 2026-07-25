"""
Discrete Wavelet Transform (DWT) Decomposition & Reconstruction Module.
Splits renewable energy profiles into long-term trends and short-term fluctuations.
"""

from typing import Tuple
import numpy as np
import pywt


class WaveletSignalDecomposer:
    """
    Uses Discrete Wavelet Transform (DWT) to separate renewable generation signals
    into low-frequency (long-term trend) and high-frequency (short-term fluctuation) components.
    """

    def __init__(self, wavelet_name: str = "db4", level: int = 3):
        """
        Initialize DWT Decomposer.

        :param wavelet_name: Mother wavelet family (e.g., 'db4', 'haar').
        :param level: Wavelet decomposition level J.
        """
        self.wavelet_name = wavelet_name
        self.level = level

    def decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decomposes 1D renewable energy profile into long-term and short-term components.

        :param signal: Raw renewable power output array P_RES(t).
        :return: Tuple of (P_long, P_short) components with length matching input signal.
        """
        # Multi-level DWT decomposition
        coeffs = pywt.wavedec(signal, wavelet=self.wavelet_name, level=self.level)
        
        # Reconstruct long-term approximation (low-frequency: cA_J)
        cA_J = coeffs[0]
        zeros_details = [np.zeros_like(c) for c in coeffs[1:]]
        p_long = pywt.waverec([cA_J] + zeros_details, wavelet=self.wavelet_name)
        
        # Reconstruct short-term details (high-frequency: cD_1 to cD_J)
        zeros_approx = np.zeros_like(cA_J)
        p_short = pywt.waverec([zeros_approx] + coeffs[1:], wavelet=self.wavelet_name)

        # Match exact signal length in case of minor wavelet padding differences
        min_len = min(len(signal), len(p_long), len(p_short))
        return p_long[:min_len], p_short[:min_len]