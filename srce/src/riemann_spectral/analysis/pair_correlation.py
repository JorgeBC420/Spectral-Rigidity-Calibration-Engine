# -*- coding: utf-8 -*-
"""
src/riemann_spectral/analysis/pair_correlation.py
==================================================

Función de correlación de pares R₂(s) (pair correlation function).

Mide la densidad de probabilidad de encontrar dos niveles separados
por una distancia s — para TODOS los pares, no solo vecinos.

Fórmula teórica GUE (Montgomery 1973, Dyson 1962):
    R₂(s) = 1 - (sin(πs) / (πs))²

Fórmula teórica GOE (Mehta, Random Matrices, Cap. 6):
    R₂(s) = 1 - sinc²(πs) + sinc'(πs) · ∫_s^∞ sinc(πt) dt

    donde sinc'(πs) es la derivada respecto a s de sin(πs)/(πs).
    Esta integral se evalúa numéricamente con scipy.integrate.quad.

Predicciones por ensemble:
    Poisson : R₂(s) = 1                    (sin correlaciones)
    GUE     : R₂(s) = 1 - (sin(πs)/πs)²   (repulsión cuadrática)
    GOE     : R₂(s) = fórmula completa Mehta (ver arriba)

Esta es la estadística que Hugh Montgomery conjeturó en 1973 y que
Freeman Dyson reconoció como la fórmula GUE exacta.

Nota sobre normalización en pair_correlation_fast:
    La función normaliza R₂ por la media de la cola (s > 0.7·s_max).
    Para que la cola sea representativa de la región sin correlaciones,
    usar s_max ≥ 5.0. Con s_max < 3 la "cola" puede estar dentro de la
    región con correlaciones fuertes de GUE (s ≈ 1–2), produciendo una
    normalización errónea.

Referencias:
    Montgomery, H.L. (1973). The pair correlation of zeros of the zeta function.
    Mehta, M.L. (2004). Random Matrices, Cap. 6.
    Odlyzko, A.M. (1987). Mathematics of Computation 48(177), 273–308.

Autor: Jorge BC & Claude
Versión: 1.1.0
"""

import numpy as np
from functools import lru_cache
from typing import Tuple, Optional

# scipy para la integral exacta de GOE (opcional con fallback)
try:
    from scipy.integrate import quad as _quad
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ============================================================================
# PREDICCIONES TEÓRICAS
# ============================================================================

def r2_teorica_gue(s: np.ndarray) -> np.ndarray:
    """
    Función de correlación de pares teórica del GUE.

    R₂(s) = 1 - (sin(πs) / (πs))²

    La función sinc tiene un cero en s=0 que se maneja por límite:
        lím_{s→0} sin(πs)/(πs) = 1  →  R₂(0) = 0

    Esto refleja la repulsión de niveles: la probabilidad de encontrar
    dos niveles en el mismo punto es cero.

    Args:
        s: Array de distancias (s ≥ 0).

    Returns:
        R₂(s) para cada valor de s.
    """
    s = np.asarray(s, dtype=np.float64)
    # sinc normalizado en numpy: np.sinc(x) = sin(πx)/(πx)
    sinc_vals = np.sinc(s)          # sin(πs)/(πs), sinc(0)=1 por definición
    return 1.0 - sinc_vals ** 2


@lru_cache(maxsize=512)
def _sinc_integral(s_val: float) -> float:
    """
    Evalúa ∫_s^∞ sin(πt)/(πt) dt numéricamente con caché LRU.

    Equivalente a (1/2 - Si(πs)/π) donde Si es la función seno integral.
    El resultado se cachea por valor exacto de s_val — las llamadas repetidas
    con la misma grilla (p.ej. np.linspace fijo entre plots o validaciones)
    reutilizan el resultado sin reejecutar quad.

    El caché es efectivo porque:
        - r2_teorica_goe típicamente se llama con arrays fijos (grillas de bins).
        - float(np.linspace(...)[i]) es reproducible entre llamadas.
        - maxsize=512 cubre grillas de hasta 512 puntos sin evictions.

    Tolerancias: epsabs=1e-8 y epsrel=1e-8 son suficientes para s ≥ 0.
    Para s pequeño (s < 0.5) la integral ≈ 0.5 − O(s²); quad converge
    limpiamente con limit=100. Usar limit=200 o tolerancias más estrictas
    no aporta para la precisión que necesita R₂(s).

    Args:
        s_val: Límite inferior de integración (s ≥ 0). Debe ser float puro,
               no np.float64, para que lru_cache funcione — el caller hace
               float(s[idx]) antes de llamar.

    Returns:
        Valor de ∫_s^∞ sinc(t) dt  (sinc normalizado: sin(πt)/(πt)).
    """
    if not _SCIPY:
        # Fallback sin scipy: aproximación asintótica de primer orden
        # ∫_s^∞ sinc(t) dt ≈ cos(πs)/(π²s)  para s >> 1  (integración por partes)
        # Para s ≤ 3 el error es significativo — instalar scipy para precisión.
        if s_val > 3.0:
            return np.cos(np.pi * s_val) / (np.pi ** 2 * s_val)
        return 0.0

    integral, _ = _quad(
        lambda t: np.sinc(t),   # np.sinc(t) = sin(πt)/(πt), sinc(0)=1
        s_val,
        np.inf,
        limit=100,
        epsabs=1e-8,
        epsrel=1e-8,
    )
    return integral


def r2_teorica_goe(s: np.ndarray) -> np.ndarray:
    """
    Función de correlación de pares teórica del GOE (fórmula exacta).

    Fórmula completa de Mehta (Random Matrices, Cap. 6, β=1):

        R₂(s) = 1 - sinc²(s) + sinc'(s) · ∫_s^∞ sinc(t) dt

    donde:
        sinc(s)  = sin(πs)/(πs)       [convención numpy: sinc normalizado]
        sinc'(s) = d/ds [sin(πs)/(πs)] = [cos(πs)/s - sin(πs)/(πs²)] · π  (s≠0)
        ∫_s^∞ sinc(t) dt              evaluado con scipy.integrate.quad

    La integral se calcula numéricamente punto a punto. Para s > 5, la
    contribución es despreciable (< 0.5%) y se trunca a 0.

    Para s ≈ 0: sinc'(0) = 0 (función par), por lo que el término integral
    no contribuye y R₂(0) = 0 (repulsión de niveles).

    Diferencia GOE vs GUE: GOE tiene correlaciones de rango más corto.
    La curva GOE está por ENCIMA de GUE para s ≳ 0.5 y converge a 1 más rápido.

    Nota: si scipy no está instalado, se usa una aproximación de cola para s > 3
    y 0 para s ≤ 3. Instalar scipy para resultados precisos.

    Args:
        s: Array de distancias (s ≥ 0).

    Returns:
        R₂(s) exacta para GOE.

    References:
        Mehta, M.L. (2004). Random Matrices, 3rd ed., Eq. (6.2.7).
    """
    s = np.asarray(s, dtype=np.float64)
    scalar_input = s.ndim == 0
    s = np.atleast_1d(s)

    # sinc(s) = sin(πs)/(πs)
    sinc_vals = np.sinc(s)

    # sinc'(s) = d/ds [sin(πs)/(πs)]
    # = [π·cos(πs)·(πs) - sin(πs)·π] / (πs)²   (regla del cociente)
    # = [cos(πs)/s - sin(πs)/(π·s²)]             (s ≠ 0)
    # En s=0: límite = 0 (sinc es par y suave en el origen)
    s_safe = np.where(np.abs(s) < 1e-10, 1e-10, s)
    sinc_deriv = np.where(
        np.abs(s) < 1e-10,
        0.0,
        np.cos(np.pi * s) / s_safe - np.sin(np.pi * s) / (np.pi * s_safe ** 2),
    )

    # ∫_s^∞ sinc(t) dt — evaluado punto a punto
    # Para s > 5 la integral es < 0.5% y se trunca a 0 por eficiencia
    integral_vals = np.zeros_like(s)
    for idx in range(len(s)):
        si = s[idx]
        if si < 5.0:
            integral_vals[idx] = _sinc_integral(float(si))
        # si >= 5: se deja en 0.0

    r2 = 1.0 - sinc_vals ** 2 + sinc_deriv * integral_vals

    # Clip físico: R₂ ∈ [0, 1.5] (pequeñas oscilaciones numéricas en la cola)
    result = np.clip(r2, 0.0, 1.5)

    return float(result[0]) if scalar_input else result


def r2_teorica_poisson(s: np.ndarray) -> np.ndarray:
    """
    Función de correlación de pares teórica de Poisson.

    R₂(s) = 1   (sin correlaciones de largo alcance)

    Args:
        s: Array de distancias.

    Returns:
        Array de unos del mismo shape que s.
    """
    return np.ones_like(np.asarray(s, dtype=np.float64))


# ============================================================================
# CÁLCULO EXPERIMENTAL
# ============================================================================

def pair_correlation(
    levels: np.ndarray,
    s_max: float = 5.0,
    bins: int = 100,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula R₂(s) experimental a partir de un espectro unfolded.

    Algoritmo:
        1. Calcula todas las distancias |γᵢ - γⱼ| para i < j.
        2. Construye histograma normalizado de esas distancias.
        3. Normaliza para que R₂(s) → 1 para s grande (Poisson).

    Complejidad: O(N²) en tiempo y memoria para N niveles.
    Para N > 3000, considerar pair_correlation_fast().

    Args:
        levels  : Espectro unfolded (ordenado, densidad ≈ 1).
        s_max   : Distancia máxima a considerar. Usar s_max ≥ 5.0 para que
                  la región de cola (s > 0.7·s_max) esté libre de correlaciones.
        bins    : Número de bins del histograma.
        normalize: Si True, normaliza para R₂(∞) → 1.

    Returns:
        (centers, r2_vals):
            centers  : Centros de los bins (array de s).
            r2_vals  : Valores de R₂(s) para cada bin.

    Example:
        >>> levels = np.cumsum(np.random.exponential(1.0, 2000))
        >>> s, r2 = pair_correlation(levels, s_max=5.0, bins=80)
    """
    levels = np.asarray(levels, dtype=np.float64)
    levels = np.sort(levels)
    N = len(levels)

    if N < 20:
        raise ValueError(f"Se necesitan al menos 20 niveles, se tienen {N}.")

    # Calcular diferencias — algoritmo O(N²)
    # Para N grande, pair_correlation_fast es más eficiente
    diffs = []
    for i in range(N):
        for j in range(i + 1, N):
            d = levels[j] - levels[i]
            if d >= s_max:
                break   # Los niveles están ordenados → podemos romper
            diffs.append(d)

    if len(diffs) < bins:
        raise ValueError(
            f"Muy pocas diferencias ({len(diffs)}) para {bins} bins. "
            f"Reducir s_max o aumentar N."
        )

    diffs = np.array(diffs)
    hist, edges = np.histogram(diffs, bins=bins, range=(0.0, s_max), density=False)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ds = edges[1] - edges[0]

    if normalize:
        # Normalizar: R₂(s) → 1 para s grande
        # La densidad esperada de pares es N*(N-1)/2 * (ds/s_max) por bin
        # Para espectro Poisson: <pares en bin> = N*(N-1)/2 * ds/s_max
        total_pares = N * (N - 1) / 2
        densidad_esperada = total_pares * ds / s_max
        r2_vals = hist.astype(float) / (densidad_esperada if densidad_esperada > 0 else 1.0)
    else:
        r2_vals = hist.astype(float)

    return centers, r2_vals


def pair_correlation_fast(
    levels: np.ndarray,
    s_max: float = 5.0,
    bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Versión vectorizada de pair_correlation. Más eficiente para N > 500.

    Normalización: R₂(s) → 1 para s grande (referencia Poisson).
    Se normaliza dividiendo por la densidad de fondo estimada en la cola
    del histograma (s > 0.7 * s_max).

    Advertencia sobre s_max:
        La normalización asume que la región s > 0.7·s_max está libre de
        correlaciones. Para GUE/GOE, las correlaciones decaen a 1 para s ≳ 3.
        Usar s_max ≥ 5.0 garantiza que la cola (s > 3.5) sea representativa.
        Con s_max < 3, la normalización puede ser errónea.

    Args:
        levels: Espectro unfolded ordenado (densidad ≈ 1 después del unfolding).
        s_max : Distancia máxima. Recomendado: s_max ≥ 5.0.
        bins  : Número de bins.

    Returns:
        (centers, r2_vals): Misma semántica que pair_correlation().
    """
    levels = np.sort(np.asarray(levels, dtype=np.float64))
    N = len(levels)
    L = levels[-1] - levels[0]

    edges = np.linspace(0.0, s_max, bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ds = edges[1] - edges[0]
    counts = np.zeros(bins, dtype=np.float64)

    for i in range(N - 1):
        diffs_i = levels[i + 1:] - levels[i]
        diffs_i = diffs_i[diffs_i < s_max]
        if len(diffs_i) == 0:
            break
        bin_idx = np.searchsorted(edges[1:], diffs_i, side='right')
        bin_idx = np.clip(bin_idx, 0, bins - 1)
        np.add.at(counts, bin_idx, 1)

    # Normalización por densidad de fondo:
    # Para espectro de densidad ρ = N/L, la tasa esperada de pares es N·ρ·ds
    rho = N / L if L > 0 else 1.0
    raw = counts / (N * rho * ds) if (N * rho * ds) > 0 else counts

    # Re-normalizar por la media de la cola para que R₂(s grande) = 1
    cola_mask = centers > 0.7 * s_max
    cola_mean = np.mean(raw[cola_mask]) if cola_mask.any() else 1.0
    r2_vals = raw / cola_mean if cola_mean > 1e-10 else raw

    return centers, r2_vals


# ============================================================================
# DIAGNÓSTICO / BONDAD DE AJUSTE
# ============================================================================

def chi2_r2_vs_gue(
    centers: np.ndarray,
    r2_obs: np.ndarray,
    s_min: float = 0.5,
    s_max_fit: float = 4.0,
) -> dict:
    """
    Calcula χ² entre R₂ observada y la predicción GUE.

    Solo usa el rango [s_min, s_max_fit] donde la señal es fiable.

    Args:
        centers    : Array de s (centros de bins).
        r2_obs     : R₂ experimental.
        s_min      : Límite inferior del ajuste (evitar s≈0).
        s_max_fit  : Límite superior.

    Returns:
        dict con:
            chi2_reducido : χ²/grados_libertad
            r2_teorica    : Predicción GUE en el rango usado
            mask          : Máscara booleana de puntos usados
    """
    mask = (centers >= s_min) & (centers <= s_max_fit) & np.isfinite(r2_obs)
    if mask.sum() < 5:
        return {"chi2_reducido": np.nan, "r2_teorica": np.array([]), "mask": mask}

    s_fit = centers[mask]
    obs_fit = r2_obs[mask]
    teo_fit = r2_teorica_gue(s_fit)

    # χ² con varianza estimada como Poisson: var ≈ teo
    var_est = np.where(teo_fit > 0.01, teo_fit, 0.01)
    chi2 = float(np.sum((obs_fit - teo_fit) ** 2 / var_est))
    dof = max(mask.sum() - 1, 1)

    return {
        "chi2_reducido": chi2 / dof,
        "chi2_total": chi2,
        "dof": dof,
        "r2_teorica": teo_fit,
        "s_fit": s_fit,
        "mask": mask,
    }


# ============================================================================
# VARIANTE g(r) MONTGOMERY–ODLYZKO (ZIP / Numba)
# ============================================================================
# Aliases de nomenclatura y histograma alternativo sin romper la API anterior.

pair_correlation_gue = r2_teorica_gue
pair_correlation_poisson = r2_teorica_poisson
pair_correlation_goe = r2_teorica_goe

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        def _wrap(f):
            return f

        return _wrap if not args else args[0]


@njit
def _compute_distances_numba(spectrum, r_max):
    """Calcula distancias |E_i-E_j| con early exit (Numba)."""
    N = len(spectrum)
    distances = []
    for i in range(N):
        for j in range(i + 1, N):
            d = spectrum[j] - spectrum[i]
            if d > r_max:
                break
            distances.append(d)
    return np.array(distances)


def pair_correlation_histogram_numba(
    spectrum: np.ndarray,
    r_max: float = 10.0,
    n_bins: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histograma de g(r) con normalización de cola (variante del paquete zip).

    No reemplaza a ``pair_correlation`` / ``pair_correlation_fast`` (API SRCE).
    """
    spectrum = np.sort(np.asarray(spectrum, dtype=np.float64))
    distances = _compute_distances_numba(spectrum, r_max)
    hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max))
    r = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    total_pairs = len(spectrum) * (len(spectrum) - 1) / 2
    g = hist / (total_pairs * bin_width)
    tail_mean = np.mean(g[n_bins // 2:])
    if tail_mean > 0:
        g = g / tail_mean
    return r, g


# ============================================================================
# AUTO-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PAIR CORRELATION — AUTO TEST")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Test 1: Poisson debe dar R₂ ≈ 1
    print("\n[TEST 1] Poisson → R₂ ≈ 1")
    N = 3000
    levels_p = np.cumsum(rng.exponential(1.0, N))
    s, r2 = pair_correlation_fast(levels_p, s_max=5.0, bins=80)
    r2_cola = np.mean(r2[(s > 2.0) & (s < 4.5)])
    print(f"  R₂ cola (s > 2): {r2_cola:.4f}  (esperado ≈ 1.0)")
    print(f"  {'✅' if abs(r2_cola - 1.0) < 0.2 else '❌'}")

    # Test 2: GUE debe dar R₂(s→0) ≈ 0
    print("\n[TEST 2] Teórica GUE en s=0")
    s_test = np.array([0.0, 0.5, 1.0, 2.0])
    r2_t = r2_teorica_gue(s_test)
    print(f"  R₂_GUE(0) = {r2_t[0]:.4f}  (esperado 0.0)")
    print(f"  R₂_GUE(1) = {r2_t[2]:.4f}  (esperado 0.0, nodo)")
    print(f"  R₂_GUE(2) = {r2_t[3]:.4f}  (esperado ≈ 1.0)")
    print(f"  {'✅' if abs(r2_t[0]) < 0.01 and abs(r2_t[3] - 1.0) < 0.1 else '❌'}")

    # Test 3: GOE exacta — verificar propiedades cualitativas
    print("\n[TEST 3] GOE exacta — propiedades")
    s_test2 = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    r2_goe = r2_teorica_goe(s_test2)
    r2_gue_ref = r2_teorica_gue(s_test2)
    print(f"  R₂_GOE(0) = {r2_goe[0]:.4f}  (esperado ≈ 0.0)")
    print(f"  R₂_GOE vs GUE en s=0.5: GOE={r2_goe[1]:.4f} > GUE={r2_gue_ref[1]:.4f}  (GOE > GUE esperado)")
    print(f"  R₂_GOE(3) = {r2_goe[4]:.4f}  (esperado ≈ 1.0)")
    ok_goe = (
        r2_goe[0] < 0.05
        and r2_goe[1] > r2_gue_ref[1]
        and abs(r2_goe[4] - 1.0) < 0.1
    )
    print(f"  {'✅' if ok_goe else '❌'}")
    if not _SCIPY:
        print("  ⚠️  scipy no disponible — GOE usa aproximación de cola para s>3")

    print("\n" + "=" * 60)
    print("✅ Tests completados")
