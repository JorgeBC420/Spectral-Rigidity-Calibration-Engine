# -*- coding: utf-8 -*-
"""
src/riemann_spectral/analysis/spectral_form_factor.py
======================================================

Factor de forma espectral K(t) y r-statistic.

── SPECTRAL FORM FACTOR K(t) ─────────────────────────────────────────────────

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

Valores medios teóricos:
    Poisson : ⟨r⟩ ≈ 0.3863  (= 2·ln(2) - 1)
    GUE     : ⟨r⟩ ≈ 0.5996
    GOE     : ⟨r⟩ ≈ 0.5307

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


# ============================================================================
# CONSTANTES TEÓRICAS
# ============================================================================

R_MEAN_POISSON = 2 * np.log(2) - 1    # ≈ 0.3863
R_MEAN_GOE     = 0.5307                # numérico (Atas et al. 2013)
R_MEAN_GUE     = 0.5996                # numérico (Atas et al. 2013)


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

    Valores medios teóricos:
        ⟨r⟩_Poisson ≈ 0.3863
        ⟨r⟩_GOE     ≈ 0.5307
        ⟨r⟩_GUE     ≈ 0.5996

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

def spectral_form_factor_teorico(
    t_grid: np.ndarray,
    ensemble: str = "GUE",
) -> np.ndarray:
    """
    Factor de forma espectral teórico K(t) para t ≥ 0.

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


def spectral_form_factor(
    levels: np.ndarray,
    t_max: float = 3.0,
    n_t: int = 200,
    smooth_sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula K(t) experimental a partir de un espectro unfolded.

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
    K = spectral_form_factor_teorico(t, "GUE")
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
