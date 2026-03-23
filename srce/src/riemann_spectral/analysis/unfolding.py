# -*- coding: utf-8 -*-
"""
Unfolding / normalización mediante N(T).
Fórmula Riemann–von Mangoldt: N(T) ~ (T/2π) log(T/2πe) + O(1).
Transforma ceros a densidad local ≈ 1.

Nota sobre Numba:
    Este módulo usa @njit con fallback automático si Numba no está instalado,
    consistente con el resto del repo (rigidity.py, r_statistic.py, etc.).
    Sin Numba, unfolding_riemann corre en NumPy puro (misma salida, sin JIT).
"""

import numpy as np

# ── Numba con fallback ────────────────────────────────────────────────────────
try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False

    def njit(*args, **kwargs):
        """No-op decorator cuando Numba no está disponible."""
        def _dec(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _dec


# ── Unfolding de ceros de Riemann ─────────────────────────────────────────────

@njit(fastmath=True)
def unfolding_riemann(gamma: np.ndarray) -> np.ndarray:
    """
    Unfolding diferencial vía la función de conteo de Von Mangoldt.

    Aplica la transformación:
        x_n = N(γ_n)  ≈  (γ_n / 2π) · (log(γ_n / 2π) − 1)

    que convierte los ceros de Riemann a escala con espaciado medio local ≈ 1.
    Es la forma diferencial de N(T) ≈ (T/2π)·log(T/2πe).

    Args:
        gamma: Array de partes imaginarias de ceros de Riemann (γ_n > 0, ordenados).

    Returns:
        Array unfolded con densidad media ≈ 1.
    """
    return (gamma / (2 * np.pi)) * (np.log(gamma / (2 * np.pi)) - 1.0)


def N_T_approx(T: float) -> float:
    """
    Función de conteo N(T): número de ceros con parte imaginaria en (0, T].

    Aproximación de Riemann–von Mangoldt:
        N(T) ≈ (T/2π) · log(T/2πe) + O(log T)

    Args:
        T: Altura en la recta crítica (T > 0).

    Returns:
        Estimación de N(T). Devuelve 0.0 para T ≤ 0.
    """
    if T <= 0:
        return 0.0
    return (T / (2 * np.pi)) * (np.log(T / (2 * np.pi * np.e)))


# ── Unfolding numérico del tercio central ─────────────────────────────────────

def unfolding_tercio_central(spectrum: np.ndarray) -> np.ndarray:
    """
    Unfolding numérico del tercio central: asigna posiciones 0, 1, ..., M−1.

    No usa la densidad real; solo para compatibilidad con pipelines legacy.
    Preferir unfolding_wigner_gue para matrices aleatorias.

    Args:
        spectrum: Espectro ordenado (cualquier escala).

    Returns:
        Array de índices enteros para el tercio central.
    """
    n = len(spectrum)
    start = n // 3
    end = 2 * (n // 3)
    if end <= start:
        return np.empty(0)
    return np.arange(end - start, dtype=np.float64)


# ── Unfolding por CDF del semicírculo de Wigner (GUE) ────────────────────────

def unfolding_wigner_gue(evals_raw: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Unfolding por la CDF del semicírculo de Wigner (GUE).

    Para una matriz GUE de tamaño N, los autovalores siguen la ley del
    semicírculo en [-2σ, 2σ]. Esta función los mapea a escala con densidad ≈ 1
    aplicando la CDF del semicírculo escalada por N:

        u_i = N · F(x_i)

    donde F es la CDF del semicírculo de Wigner:
        F(x) = 1/2 + (1/4π) · [ x·√(4σ²−x²) + 4σ²·arcsin(x/2σ) ]

    Usar el tercio central después para evitar efectos de borde del semicírculo.

    Args:
        evals_raw: Autovalores en escala típica [−2σ, 2σ].
        sigma    : Parámetro de escala del semicírculo (default 1.0).

    Returns:
        Array unfolded con espaciado medio ≈ 1 en el centro del espectro.
    """
    x = np.asarray(evals_raw, dtype=np.float64)
    n = len(x)
    s = sigma
    x_clip = np.clip(x / (2 * s), -1.0, 1.0)
    sqrt_term = np.sqrt(np.maximum(4 * s * s - x * x, 0.0))
    F = 0.5 + (1.0 / (4 * np.pi)) * (x * sqrt_term + 4 * s * s * np.arcsin(x_clip))
    return n * F
