# -*- coding: utf-8 -*-
"""
src/riemann_spectral/statistics/r_statistic.py
===============================================

r-parameter (Oganesyan-Huse): ratio de spacings consecutivos.

El r-parameter es una estadística ROBUSTA para clasificar ensembles,
independiente del unfolding.

Definición:
    r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
    ⟨r⟩ = mean(r_i)

Valores EXACTOS (Atas et al., 2013):
    Poisson: ⟨r⟩ = 2ln(2) - 1 ≈ 0.38629436...
    GOE:     ⟨r⟩ = 4 - 2√3   ≈ 0.53589838...
    GUE:     ⟨r⟩ ≈ 0.60272166...  (constante R_GUE_EXACT en código)

Referencias:
    Oganesyan & Huse (2007). Phys. Rev. B 75, 155111
    Atas et al. (2013). Phys. Rev. Lett. 110, 084101

Autor: Jorge BC & Claude
Versión: 1.0.0
"""

import numpy as np
from typing import Dict, Tuple
from numba import njit

# ============================================================================
# CONSTANTES TEÓRICAS EXACTAS
# ============================================================================

# Valores analíticos exactos (Atas et al., 2013)
R_POISSON_EXACT = 2 * np.log(2) - 1             # ≈ 0.38629436111989
R_GOE_EXACT = 4 - 2 * np.sqrt(3)                # ≈ 0.53589838486224
# ⟨r⟩_GUE (β=2): valor cerrado estándar en la literatura (≈ 0.60272)
R_GUE_EXACT = 0.60272166211556

# Tolerancias típicas para clasificación
R_TOLERANCE = 0.05


# ============================================================================
# CÁLCULO DEL r-PARAMETER
# ============================================================================

@njit
def _compute_r_ratios_numba(spacings):
    """
    Calcula ratios r_i con Numba (10× más rápido).
    
    r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
    """
    N = len(spacings) - 1
    r_vals = np.empty(N, dtype=np.float64)
    
    for i in range(N):
        s_i = spacings[i]
        s_i1 = spacings[i + 1]
        
        if s_i < s_i1:
            r_vals[i] = s_i / s_i1
        else:
            r_vals[i] = s_i1 / s_i
    
    return r_vals


def compute_r_parameter(spectrum: np.ndarray) -> float:
    """
    Calcula el r-parameter ⟨r⟩.
    
    El r-parameter es INDEPENDIENTE del unfolding, solo requiere
    que el espectro esté ordenado.
    
    Args:
        spectrum: Eigenvalues ordenados
    
    Returns:
        ⟨r⟩: promedio del ratio de spacings consecutivos
    
    Example:
        >>> from riemann_spectral.data.generators import generar_gue_normalizado
        >>> eigenvalues = generar_gue_normalizado(rng, N=1000)
        >>> r_mean = compute_r_parameter(eigenvalues)
        >>> print(f"⟨r⟩ = {r_mean:.4f}")  # Esperado: ~0.603
    
    Notes:
        - NO requiere normalización de spacing
        - Robusto a errores de unfolding
        - Excelente para clasificación rápida
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    spectrum = np.sort(spectrum)
    
    # Calcular spacings
    spacings = np.diff(spectrum)
    
    # Calcular ratios con Numba
    r_vals = _compute_r_ratios_numba(spacings)
    
    return float(np.mean(r_vals))


def compute_r_distribution(
    spectrum: np.ndarray,
    n_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula la distribución completa P(r).
    
    Args:
        spectrum: Eigenvalues ordenados
        n_bins: Número de bins para histograma
    
    Returns:
        (r_centers, P_r): Centros de bins y densidad P(r)
    
    Example:
        >>> r_vals, P_r = compute_r_distribution(gue_spectrum)
        >>> plt.hist(r_vals, bins=50, density=True, alpha=0.6)
        >>> plt.xlabel('r')
        >>> plt.ylabel('P(r)')
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    spectrum = np.sort(spectrum)
    
    spacings = np.diff(spectrum)
    r_vals = _compute_r_ratios_numba(spacings)
    
    # Histograma
    hist, bin_edges = np.histogram(r_vals, bins=n_bins, range=(0, 1), density=True)
    r_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return r_centers, hist


# ============================================================================
# CLASIFICACIÓN DE ENSEMBLES
# ============================================================================

def classify_ensemble_by_r(
    spectrum: np.ndarray,
    tolerance: float = R_TOLERANCE,
) -> Dict:
    """
    Clasifica ensemble usando r-parameter.
    
    Args:
        spectrum: Eigenvalues ordenados
        tolerance: Tolerancia para clasificación
    
    Returns:
        dict con:
            - r_mean: ⟨r⟩ observado
            - ensemble: 'Poisson', 'GOE', 'GUE', o 'Unknown'
            - confidence: distancia al valor más cercano
            - distances: dict con distancias a cada ensemble
    
    Example:
        >>> result = classify_ensemble_by_r(gue_spectrum)
        >>> print(f"Clasificado como: {result['ensemble']}")
        >>> print(f"⟨r⟩ = {result['r_mean']:.4f}")
    """
    r_mean = compute_r_parameter(spectrum)
    
    # Distancias a cada ensemble
    distances = {
        'Poisson': abs(r_mean - R_POISSON_EXACT),
        'GOE': abs(r_mean - R_GOE_EXACT),
        'GUE': abs(r_mean - R_GUE_EXACT),
    }
    
    # Encontrar mínimo
    ensemble = min(distances, key=distances.get)
    min_distance = distances[ensemble]
    
    # Confidence
    if min_distance > tolerance:
        ensemble = 'Unknown'
        confidence = 0.0
    else:
        # Distancia normalizada
        confidence = 1.0 - (min_distance / tolerance)
    
    return {
        'r_mean': r_mean,
        'ensemble': ensemble,
        'confidence': confidence,
        'distances': distances,
        'theoretical_values': {
            'Poisson': R_POISSON_EXACT,
            'GOE': R_GOE_EXACT,
            'GUE': R_GUE_EXACT,
        },
    }


def compare_r_with_theory(
    spectrum: np.ndarray,
    ensemble: str,
) -> Dict:
    """
    Compara ⟨r⟩ observado con predicción teórica.
    
    Args:
        spectrum: Eigenvalues
        ensemble: 'Poisson', 'GOE', o 'GUE'
    
    Returns:
        dict con r_obs, r_theory, error_abs, error_rel
    """
    r_obs = compute_r_parameter(spectrum)
    
    # Valor teórico
    theoretical_values = {
        'Poisson': R_POISSON_EXACT,
        'GOE': R_GOE_EXACT,
        'GUE': R_GUE_EXACT,
    }
    
    if ensemble not in theoretical_values:
        raise ValueError(f"Ensemble '{ensemble}' not recognized")
    
    r_theory = theoretical_values[ensemble]
    
    error_abs = abs(r_obs - r_theory)
    error_rel = error_abs / r_theory
    
    return {
        'r_observed': r_obs,
        'r_theoretical': r_theory,
        'error_absolute': error_abs,
        'error_relative': error_rel,
        'ensemble': ensemble,
    }


# ============================================================================
# UTILIDADES
# ============================================================================

def r_parameter_convergence(
    spectrum: np.ndarray,
    n_points: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analiza convergencia de ⟨r⟩ con tamaño del espectro.
    
    Args:
        spectrum: Eigenvalues ordenados
        n_points: Número de puntos para el análisis
    
    Returns:
        (N_vals, r_vals): Tamaños y ⟨r⟩ correspondientes
    
    Example:
        >>> N_vals, r_vals = r_parameter_convergence(spectrum)
        >>> plt.plot(N_vals, r_vals)
        >>> plt.axhline(R_GUE_EXACT, color='r', linestyle='--')
        >>> plt.xlabel('N')
        >>> plt.ylabel('⟨r⟩')
    """
    spectrum = np.sort(spectrum)
    N_total = len(spectrum)
    
    # Tamaños a probar (logarítmicos)
    N_vals = np.logspace(
        np.log10(100),
        np.log10(N_total),
        n_points,
        dtype=int
    )
    
    r_vals = np.zeros(len(N_vals))
    
    for i, N in enumerate(N_vals):
        r_vals[i] = compute_r_parameter(spectrum[:N])
    
    return N_vals, r_vals


# ============================================================================
# AUTO-TEST
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("r-PARAMETER MODULE - DEMO & VALIDATION")
    print("="*70)
    
    import sys
    sys.path.insert(0, 'src')
    
    from riemann_spectral.data.generators import (
        generar_goe_normalizado,
        generar_gue_normalizado,
        generar_poisson,
    )
    
    # Test con los 3 ensembles
    print("\n[TEST 1] Validación contra valores exactos:")
    print("-" * 70)
    
    rng = np.random.default_rng(seed=42)
    N = 5000
    
    # Poisson
    poisson = generar_poisson(rng, N, densidad=1.0)
    r_poisson = compute_r_parameter(poisson)
    error_poisson = abs(r_poisson - R_POISSON_EXACT) / R_POISSON_EXACT
    
    print(f"Poisson:")
    print(f"  ⟨r⟩ observado: {r_poisson:.6f}")
    print(f"  ⟨r⟩ exacto:    {R_POISSON_EXACT:.6f}")
    print(f"  Error:         {100*error_poisson:.2f}%")
    print(f"  {'✓ PASADO' if error_poisson < 0.03 else '✗ FALLIDO'}")
    
    # GOE
    goe = generar_goe_normalizado(N, rng=np.random.default_rng(7))
    r_goe = compute_r_parameter(goe)
    error_goe = abs(r_goe - R_GOE_EXACT) / R_GOE_EXACT
    
    print(f"\nGOE:")
    print(f"  ⟨r⟩ observado: {r_goe:.6f}")
    print(f"  ⟨r⟩ exacto:    {R_GOE_EXACT:.6f}")
    print(f"  Error:         {100*error_goe:.2f}%")
    print(f"  {'✓ PASADO' if error_goe < 0.03 else '✗ FALLIDO'}")
    
    # GUE
    gue = generar_gue_normalizado(np.random.default_rng(99), N)
    r_gue = compute_r_parameter(gue)
    error_gue = abs(r_gue - R_GUE_EXACT) / R_GUE_EXACT
    
    print(f"\nGUE:")
    print(f"  ⟨r⟩ observado: {r_gue:.6f}")
    print(f"  ⟨r⟩ exacto:    {R_GUE_EXACT:.6f}")
    print(f"  Error:         {100*error_gue:.2f}%")
    print(f"  {'✓ PASADO' if error_gue < 0.03 else '✗ FALLIDO'}")
    
    # Test de clasificación
    print("\n" + "-" * 70)
    print("[TEST 2] Clasificación automática:")
    print("-" * 70)
    
    for name, spectrum in [('Poisson', poisson), ('GOE', goe), ('GUE', gue)]:
        result = classify_ensemble_by_r(spectrum)
        print(f"\n{name}:")
        print(f"  Clasificado como: {result['ensemble']}")
        print(f"  Confidence:       {100*result['confidence']:.1f}%")
    
    # Resultado final
    print("\n" + "="*70)
    all_passed = (error_poisson < 0.03 and error_goe < 0.03 and error_gue < 0.03)
    print(f"{'✅ TODOS LOS TESTS PASARON' if all_passed else '⚠️ ALGUNOS TESTS FALLARON'}")
    print("="*70)
