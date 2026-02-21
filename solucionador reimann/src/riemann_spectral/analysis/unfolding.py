"""
Unfolding / normalización mediante N(T).
Fórmula Riemann–von Mangoldt: N(T) ~ (T/2π) log(T/2πe) + O(1).
Transforma ceros a densidad local ≈ 1.
"""

import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True)
def unfolding_riemann(gamma: np.ndarray) -> np.ndarray:
    """
    Unfolding diferencial: (gamma/2π) * (log(gamma/2π) - 1).
    Convierte ceros a escala con espaciado medio local ≈ 1.
    """
    return (gamma / (2 * np.pi)) * (np.log(gamma / (2 * np.pi)) - 1.0)


def N_T_approx(T: float) -> float:
    """
    Aproximación de la función de conteo N(T) (número de ceros con parte imaginaria en (0, T]).
    N(T) ≈ (T/2π) * log(T/2πe) + O(1).
    """
    if T <= 0:
        return 0.0
    return (T / (2 * np.pi)) * (np.log(T / (2 * np.pi * np.e)))


def unfolding_tercio_central(spectrum: np.ndarray) -> np.ndarray:
    """
    Unfolding numerico del tercio central: asigna posiciones 0, 1, ..., M-1.
    No usa la densidad real; solo para compatibilidad. Preferir unfolding_wigner_gue.
    """
    n = len(spectrum)
    start = n // 3
    end = 2 * (n // 3)
    if end <= start:
        return np.empty(0)
    return np.arange(end - start, dtype=np.float64)


def unfolding_wigner_gue(evals_raw: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Unfolding por la CDF del semicirculo de Wigner (GUE).
    evals_raw: autovalores en escala tipica [-2*sigma, 2*sigma].
    Devuelve u_i = N * F(evals_raw_i) con F = CDF del semicirculo, espaciado medio 1.
    Preserva fluctuaciones locales; usar tercio central despues para evitar bordes.
    """
    x = np.asarray(evals_raw, dtype=np.float64)
    n = len(x)
    # Semicirculo en [-2*sigma, 2*sigma]: rho(x) = (1/(2*pi*sigma^2))*sqrt(4*sigma^2 - x^2)
    # CDF: F(x) = 1/2 + (1/(2*pi))*( (x/(2*sigma))*sqrt(4*sigma^2 - x^2) + 4*sigma^2*arcsin(x/(2*sigma)) )/(2*sigma^2)
    # Con sigma=1: F(x) = 1/2 + (1/(4*pi))*( x*sqrt(4-x^2) + 4*arcsin(x/2) ), x in [-2,2]
    s = sigma
    x_clip = np.clip(x / (2 * s), -1.0, 1.0)
    sqrt_term = np.sqrt(np.maximum(4 * s * s - x * x, 0.0))
    # CDF semicirculo: F(x) = 1/2 + (1/(4*pi))*( x*sqrt(4*sigma^2-x^2) + 4*sigma^2*arcsin(x/(2*sigma)) )
    F = 0.5 + (1.0 / (4 * np.pi)) * (x * sqrt_term + 4 * s * s * np.arcsin(x_clip))
    return n * F
