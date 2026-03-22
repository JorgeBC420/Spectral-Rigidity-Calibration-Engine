# -*- coding: utf-8 -*-
"""
src/riemann_spectral/analysis/spectral_form_factor.py
======================================================

**Convenciones (dos objetos distintos en la literatura):**

1. **Factor de forma coherente (oficial SRCE)** — ``spectral_form_factor``::

       K(τ) = (1/N) |∑ⱼ exp(i τ λⱼ)|²

   Usado en quantum chaos / SYK (dip–ramp–plateau). Fase **sin** factor 2π en τ.

2. **Factor de forma tipo Mehta** — ``spectral_form_factor_mehta``::

       K_raw(t) = (1/N²) |∑ₙ exp(2π i t γₙ)|²

   Comparación con ``spectral_form_factor_mehta_teorico`` (GUE/GOE/Poisson según Mehta).

── SPECTRAL FORM FACTOR K(t) Mehta ──────────────────────────────────────────

K(t) es la transformada de Fourier de la función de correlación de pares R₂(s):

    K(t) = ∫ R₂(s) · e^(2πi·s·t) ds

Predicciones teóricas (Mehta Cap. 9):
    Poisson : K(t) = 1  (plano, sin estructura)
    GUE     : K(t) = { |t|          si 0 ≤ |t| ≤ 1
                      { 1            si |t| > 1
    GOE     : K(t) = { 2|t| - |t|ln(1+2|t|)   (|t| < 1)
                      { 2 - |t|ln((2|t|+1)/(2|t|-1))  (|t| > 1)

El "dip" de K(t) para t pequeño revela las correlaciones de largo alcance.

── R-STATISTIC ───────────────────────────────────────────────────────────────

El r-statistic (ratio de espaciados consecutivos) es más robusto que P(s)
porque no requiere unfolding:

    rₙ = min(sₙ, sₙ₊₁) / max(sₙ, sₙ₊₁)    con sₙ = γₙ₊₁ - γₙ

Valores medios teóricos (Atas et al.; ver ``statistics.r_statistic``):
    Poisson : ⟨r⟩ = 2·ln(2) − 1 ≈ 0.386294…
    GOE     : ⟨r⟩ = 4 − 2√3 ≈ 0.535898…
    GUE     : ⟨r⟩ ≈ 0.60272166211556

No requiere unfolding → más robusto frente a variaciones de densidad.

Referencias:
    Mehta, M.L. (2004). Random Matrices, Caps. 6, 9.
    Atas et al. (2013). Phys. Rev. Lett. 110, 084101. (r-statistic)
    Haake, F. (2010). Quantum Signatures of Chaos.

Autor: Jorge BC & Claude
Versión: 1.0.0
"""

import numpy as np
from typing import Tuple, Optional

from ..statistics.r_statistic import (
    R_POISSON_EXACT as R_MEAN_POISSON,
    R_GOE_EXACT as R_MEAN_GOE,
    R_GUE_EXACT as R_MEAN_GUE,
)

# ============================================================================
# Numba (factor coherente)
# ============================================================================
try:
    from numba import njit as _njit_sff
except ImportError:
    def _njit_sff(*args, **kwargs):
        def _d(f):
            return f
        return _d if not args or not callable(args[0]) else args[0]


# ============================================================================
# R-STATISTIC
# ============================================================================

def r_statistic(
    levels: np.ndarray,
    return_distribution: bool = False,
) -> dict:
    """
    Calcula el r-statistic (ratio de espaciados consecutivos).

    No requiere unfolding — directamente aplicable a los ceros de Riemann
    sin preprocesamiento.

    Args:
        levels              : Niveles espectrales (ordenados o no).
        return_distribution : Si True, devuelve el array completo de rₙ.

    Returns:
        dict con:
            r_mean     : ⟨r⟩ media de los ratios
            r_std      : σ(r)
            r_median   : mediana
            clasificacion : 'GUE' | 'GOE' | 'Poisson' | 'Intermedio'
            distancia_gue     : |⟨r⟩ - r_GUE|
            distancia_goe     : |⟨r⟩ - r_GOE|
            distancia_poisson : |⟨r⟩ - r_Poisson|
            r_vals     : Array de rₙ (solo si return_distribution=True)

    Example:
        >>> gamma = np.array([14.1, 21.0, 25.0, 30.4, 32.9, ...])
        >>> resultado = r_statistic(gamma)
        >>> print(resultado['r_mean'], resultado['clasificacion'])
    """
    levels = np.sort(np.asarray(levels, dtype=np.float64))
    N = len(levels)

    if N < 4:
        raise ValueError(f"Se necesitan al menos 4 niveles, se tienen {N}.")

    # Espaciados consecutivos
    s = np.diff(levels)            # sₙ = γₙ₊₁ - γₙ,  longitud N-1
    s1 = s[:-1]                    # sₙ
    s2 = s[1:]                     # sₙ₊₁,  longitud N-2

    # r-statistic
    r_vals = np.minimum(s1, s2) / np.maximum(s1, s2)
    # Filtrar valores degenerados (espaciados cero)
    valid = (s1 > 0) & (s2 > 0)
    r_vals = r_vals[valid]

    if len(r_vals) < 3:
        raise ValueError("Demasiados espaciados degenerados (ceros consecutivos).")

    r_mean   = float(np.mean(r_vals))
    r_std    = float(np.std(r_vals))
    r_median = float(np.median(r_vals))

    # Clasificación por distancia al valor teórico
    dist_gue = abs(r_mean - R_MEAN_GUE)
    dist_goe = abs(r_mean - R_MEAN_GOE)
    dist_poi = abs(r_mean - R_MEAN_POISSON)

    dists = {"GUE": dist_gue, "GOE": dist_goe, "Poisson": dist_poi}
    clasificacion = min(dists, key=dists.get)

    # Si el más cercano está a más de 0.05, es "Intermedio"
    if min(dists.values()) > 0.05:
        clasificacion = "Intermedio"

    resultado = {
        "r_mean"            : r_mean,
        "r_std"             : r_std,
        "r_median"          : r_median,
        "n_ratios"          : len(r_vals),
        "clasificacion"     : clasificacion,
        "distancia_gue"     : dist_gue,
        "distancia_goe"     : dist_goe,
        "distancia_poisson" : dist_poi,
        "r_teorico_gue"     : R_MEAN_GUE,
        "r_teorico_goe"     : R_MEAN_GOE,
        "r_teorico_poisson" : R_MEAN_POISSON,
    }
    if return_distribution:
        resultado["r_vals"] = r_vals

    return resultado


def r_distribucion_teorica(
    r_grid: np.ndarray,
    ensemble: str = "GUE",
) -> np.ndarray:
    """
    Distribución teórica P(r) del r-statistic según Atas et al. (2013).

    Para r = min(sₙ, sₙ₊₁)/max(sₙ, sₙ₊₁) ∈ [0, 1]:

        P_β(r) = (1/Z_β) · (r + r²)^β / (1 + r + r²)^(1 + 3β/2)

    Con constantes de normalización calculadas numéricamente (∫₀¹ P dr = 1):
        Z_GOE (β=1) ≈ 0.1481
        Z_GUE (β=2) ≈ 0.0448

    Para Poisson:
        P(r) = 2 / (1 + r)²

    Valores medios teóricos (consistentes con ``R_MEAN_*``):
        ⟨r⟩_Poisson = 2 ln 2 − 1
        ⟨r⟩_GOE     = 4 − 2√3
        ⟨r⟩_GUE     ≈ 0.60272166211556

    Args:
        r_grid   : Array de valores r ∈ [0, 1].
        ensemble : 'GUE', 'GOE', o 'Poisson'.

    Returns:
        P(r) normalizada (∫₀¹ P dr = 1).
    """
    r = np.asarray(r_grid, dtype=np.float64)
    r = np.clip(r, 0.0, 1.0)

    if ensemble == "GUE":
        # β=2, Z_GUE ≈ 0.04478 (integral numérica de (r+r²)²/(1+r+r²)^4 en [0,1])
        Z = 0.044785
        return (r + r**2)**2 / ((1 + r + r**2)**4 * Z)

    elif ensemble == "GOE":
        # β=1, Z_GOE ≈ 0.14815 (integral numérica de (r+r²)/(1+r+r²)^(5/2) en [0,1])
        Z = 0.148148
        return (r + r**2) / ((1 + r + r**2)**(5.0/2.0) * Z)

    elif ensemble == "Poisson":
        return 2.0 / (1 + r) ** 2

    else:
        raise ValueError(f"ensemble debe ser 'GUE', 'GOE' o 'Poisson', no '{ensemble}'")


# ============================================================================
# SPECTRAL FORM FACTOR K(t)
# ============================================================================

def spectral_form_factor_mehta_teorico(
    t_grid: np.ndarray,
    ensemble: str = "GUE",
) -> np.ndarray:
    """
    Factor de forma espectral teórico K(t) para t ≥ 0 (convención Mehta, Cap. 9).

    Args:
        t_grid   : Array de tiempos t ≥ 0.
        ensemble : 'GUE', 'GOE', o 'Poisson'.

    Returns:
        K(t) para cada t en t_grid.
    """
    t = np.abs(np.asarray(t_grid, dtype=np.float64))

    if ensemble == "GUE":
        # K(t) = |t| para t ≤ 1, K(t) = 1 para t > 1
        return np.where(t <= 1.0, t, np.ones_like(t))

    elif ensemble == "GOE":
        # K(t) según Mehta (3ª ed.), Eq. 9.28
        K = np.zeros_like(t)
        mask1 = (t > 0) & (t < 1)
        mask2 = t >= 1
        t1 = t[mask1]
        t2 = t[mask2]
        K[mask1] = 2 * t1 - t1 * np.log(1 + 2 * t1)
        K[mask2] = 2 - t2 * np.log((2 * t2 + 1) / np.maximum(2 * t2 - 1, 1e-10))
        return K

    elif ensemble == "Poisson":
        # K(t) = 1 para todo t (sin correlaciones)
        return np.ones_like(t)

    else:
        raise ValueError(f"ensemble desconocido: '{ensemble}'")


def spectral_form_factor_mehta(
    levels: np.ndarray,
    t_max: float = 3.0,
    n_t: int = 200,
    smooth_sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula K(t) experimental (convención Mehta: fase 2π t γ, normalización 1/N²).

    Definición:
        K_raw(t) = (1/N²) · |∑ₙ exp(2πi·t·γₙ)|²

    Normalización:
        K(t) se re-escala para que K(t→∞) = 1, lo que permite la comparación
        directa con las curvas teóricas de GUE/GOE/Poisson.

    Notas:
        - El resultado de una sola realización es muy ruidoso.
        - Usar smooth_sigma >= 2 para visualización.
        - Para comparar con teoría se necesita promedio sobre muchas realizaciones.

    Args:
        levels       : Espectro unfolded (ordenado, densidad ≈ 1).
        t_max        : Tiempo máximo en unidades de Heisenberg.
        n_t          : Número de puntos en t.
        smooth_sigma : σ del kernel gaussiano de suavizado (None = sin suavizar).

    Returns:
        (t_grid, K_vals): t en [0, t_max], K normalizado a K(∞)=1.
    """
    levels = np.asarray(levels, dtype=np.float64)
    N = len(levels)

    if N < 10:
        raise ValueError(f"Se necesitan al menos 10 niveles, se tienen {N}.")

    t_grid = np.linspace(0.0, t_max, n_t)
    K_vals = np.zeros(n_t)

    for i, t in enumerate(t_grid):
        phase = 2 * np.pi * t * levels
        K_vals[i] = (np.sum(np.cos(phase)) ** 2 + np.sum(np.sin(phase)) ** 2) / (N * N)

    if smooth_sigma is not None and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter1d
            K_vals = gaussian_filter1d(K_vals, sigma=smooth_sigma)
        except ImportError:
            pass

    # Re-normalizar: K(t→∞) → 1
    # Para t grande el espectro actúa como Poisson → K → 1/N para una realización
    # Normalizar por la media de la cola (t > 0.7*t_max)
    cola_mask = t_grid > 0.7 * t_max
    if cola_mask.any():
        K_cola = np.mean(K_vals[cola_mask])
        if K_cola > 1e-15:
            K_vals = K_vals / K_cola

    return t_grid, K_vals


# Alias retrocompatible (antes ``spectral_form_factor_teorico`` = curva Mehta GUE/GOE/Poisson)
spectral_form_factor_teorico = spectral_form_factor_mehta_teorico


# ============================================================================
# FACTOR DE FORMA COHERENTE K(τ) = |∑ exp(iτλⱼ)|² / N  (definición oficial)
# ============================================================================

def form_factor_poisson(tau: np.ndarray, N: int) -> np.ndarray:
    """Referencia Poisson: K(τ) ≡ 1 (sin correlación entre niveles)."""
    return np.ones_like(tau)


def form_factor_gue_analytical(tau: np.ndarray, N: int) -> np.ndarray:
    """
    Aproximación escalar de Haake para GUE (dip / ramp / plateau); no es la
    sincronización exacta con la traza de R₂(s).
    """
    tau = np.asarray(tau)
    K = np.zeros_like(tau, dtype=np.float64)
    dip_mask = tau < 1
    K[dip_mask] = tau[dip_mask] ** 2
    ramp_mask = (tau >= 1) & (tau < N)
    K[ramp_mask] = tau[ramp_mask]
    K[tau >= N] = 1.0
    transition_mask = (tau >= 0.5) & (tau < 2)
    if np.any(transition_mask):
        t = tau[transition_mask]
        K[transition_mask] = t**2 + 0.3 * t
    return np.minimum(K, 1.0)


@_njit_sff
def _compute_form_factor_numba(spectrum, tau_grid):
    """
    K(τ) = |∑_j exp(iτE_j)|² / N.
    """
    N = len(spectrum)
    n_tau = len(tau_grid)
    K = np.zeros(n_tau, dtype=np.float64)
    for i in range(n_tau):
        tau = tau_grid[i]
        real_sum = 0.0
        imag_sum = 0.0
        for j in range(N):
            phase = tau * spectrum[j]
            real_sum += np.cos(phase)
            imag_sum += np.sin(phase)
        K[i] = (real_sum**2 + imag_sum**2) / N
    return K


def spectral_form_factor(
    spectrum: np.ndarray,
    tau_max: float = 20.0,
    n_points: int = 200,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Factor de forma espectral coherente (definición canónica usada en SRCE).

    K(τ) = (1/N) |∑ⱼ exp(i τ λⱼ)|²

    No confundir con ``spectral_form_factor_mehta`` (fase 2π t γ, normalización N²).
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    N = len(spectrum)
    tau_grid = np.linspace(0, tau_max, n_points)
    K = _compute_form_factor_numba(spectrum, tau_grid)
    if normalize and K[0] > 0:
        K = K / K[0]
    return tau_grid, K


def identify_regimes(
    tau: np.ndarray,
    K: np.ndarray,
    N: int,
) -> dict:
    dip_end = np.searchsorted(tau, 1.0)
    ramp_end = np.searchsorted(tau, N / 2)
    return {
        "dip_range": (0, dip_end),
        "ramp_range": (dip_end, ramp_end),
        "plateau_range": (ramp_end, len(tau)),
    }


def extract_ramp_slope(
    tau: np.ndarray,
    K: np.ndarray,
    N: int,
) -> float:
    regimes = identify_regimes(tau, K, N)
    ramp_start, ramp_end = regimes["ramp_range"]
    tau_ramp = tau[ramp_start:ramp_end]
    K_ramp = K[ramp_start:ramp_end]
    mask = (tau_ramp > 0) & (K_ramp > 0)
    tau_ramp = tau_ramp[mask]
    K_ramp = K_ramp[mask]
    if len(tau_ramp) < 2:
        return 0.0
    log_tau = np.log(tau_ramp)
    log_K = np.log(K_ramp)
    slope, _ = np.polyfit(log_tau, log_K, 1)
    return float(slope)


def compare_with_theory(
    spectrum: np.ndarray,
    ensemble: str = "GUE",
    tau_max: float = 20.0,
) -> dict:
    N = len(spectrum)
    tau, K_emp = spectral_form_factor(spectrum, tau_max=tau_max)
    if ensemble.lower() == "poisson":
        K_theory = form_factor_poisson(tau, N)
    elif ensemble.lower() == "gue":
        K_theory = form_factor_gue_analytical(tau, N)
    else:
        raise ValueError(f"Ensemble '{ensemble}' not supported")
    slope = extract_ramp_slope(tau, K_emp, N)
    return {
        "tau": tau,
        "K_empirical": K_emp,
        "K_theory": K_theory,
        "ramp_slope": slope,
        "ensemble": ensemble,
    }


# ============================================================================
# AUTO-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPECTRAL FORM FACTOR & R-STATISTIC — AUTO TEST")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Test 1: r-statistic Poisson
    print("\n[TEST 1] r-statistic Poisson")
    N = 5000
    levels_p = np.cumsum(rng.exponential(1.0, N))
    res_p = r_statistic(levels_p)
    print(f"  ⟨r⟩ = {res_p['r_mean']:.4f}  (esperado {R_MEAN_POISSON:.4f})")
    print(f"  Clasificación: {res_p['clasificacion']}")
    ok = abs(res_p["r_mean"] - R_MEAN_POISSON) < 0.02
    print(f"  {'✅' if ok else '❌'}")

    # Test 2: K(t) teórico GUE en t=0 y t=2
    print("\n[TEST 2] K(t) teórico GUE")
    t = np.array([0.0, 0.5, 1.0, 2.0])
    K = spectral_form_factor_mehta_teorico(t, "GUE")
    print(f"  K(0) = {K[0]:.4f}  (esperado 0.0)")
    print(f"  K(0.5) = {K[1]:.4f}  (esperado 0.5)")
    print(f"  K(1.0) = {K[2]:.4f}  (esperado 1.0)")
    print(f"  K(2.0) = {K[3]:.4f}  (esperado 1.0)")
    ok = abs(K[0]) < 0.01 and abs(K[1] - 0.5) < 0.01 and abs(K[3] - 1.0) < 0.01
    print(f"  {'✅' if ok else '❌'}")

    # Test 3: r_distribucion_teorica suma ≈ 1
    print("\n[TEST 3] P(r) GUE integra ≈ 1")
    r_grid = np.linspace(0, 1, 1000)
    dr = r_grid[1] - r_grid[0]
    for ens in ["GUE", "GOE", "Poisson"]:
        P = r_distribucion_teorica(r_grid, ens)
        integral = np.sum(P) * dr
        ok = abs(integral - 1.0) < 0.05
        print(f"  ∫P(r)dr [{ens}] = {integral:.4f}  {'✅' if ok else '❌'}")

    print("\n" + "=" * 60)
    print("✅ Tests completados")
