# -*- coding: utf-8 -*-
"""
src/riemann_spectral/analysis/empirical_unfolding.py
=====================================================

Unfolding empírico: KDE, spline y polinomial.

El **unfolding** transforma un espectro crudo a densidad local ≈ 1 para
que las estadísticas espectrales (Δ₃, Σ², P(s), r-statistic) sean
comparables entre sistemas. Hay dos familias:

Unfolding analítico
    Usa la fórmula teórica exacta de la densidad (p.ej. semicírculo de
    Wigner para GUE, Riemann–von Mangoldt para ceros de Riemann).
    Ventaja: sin sesgo de estimación. Desventaja: requiere conocer ρ(x).

Unfolding empírico
    Estima ρ(x) directamente del espectro. Tres métodos aquí:

    KDE (Kernel Density Estimation)
        Suavizado gaussiano de la densidad. Bueno para espectros largos.
        Robusto pero sensible al ancho de banda h.

    Spline cúbico
        Ajusta una spline a la CDF empírica suavizada. Muy preciso
        cuando la densidad varía suavemente. Puede oscilar en bordes.

    Polinomial
        Ajusta un polinomio de grado k a la CDF. Más estable pero
        menos flexible. Útil como baseline o para N pequeño.

Comparación
    ``compare_unfolding_methods`` ejecuta los tres métodos + el analítico
    y devuelve métricas (⟨s⟩, σ(s), r-statistic) para cada uno, lo que
    permite ver cuál introduce menos distorsión sobre los ceros de Riemann.

Referencias
    Haake, F. (2010). Quantum Signatures of Chaos, Cap. 3.
    Mehta, M.L. (2004). Random Matrices, Cap. 1.
    Pappalardi et al. (2022). arXiv:2209.01551 — comparación de métodos.

Autor: Jorge BC & Claude
Versión: 1.0.0
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


# ============================================================================
# MÉTODOS DE UNFOLDING EMPÍRICO
# ============================================================================

def unfold_kde(
    spectrum: np.ndarray,
    bandwidth: Optional[float] = None,
    n_grid: int = 2000,
) -> np.ndarray:
    """
    Unfolding por estimación de densidad con kernel gaussiano (KDE).

    Algoritmo:
        1. Estimar ρ(x) suavizando el histograma con un kernel gaussiano.
        2. Integrar ρ(x) para obtener la CDF empírica N(x).
        3. Evaluar N(xᵢ) para cada autovalor → espectro unfolded.

    El ancho de banda h controla el nivel de suavizado:
        h pequeño → sigue la densidad local (puede oscilar)
        h grande  → densidad global (puede perder variaciones reales)

    Si bandwidth=None se usa la regla de Silverman:
        h = 0.9 · σ · N^(-1/5)

    Args:
        spectrum : Espectro ordenado (cualquier escala).
        bandwidth: Ancho de banda del kernel gaussiano. None = Silverman.
        n_grid   : Resolución de la grilla de estimación.

    Returns:
        Espectro unfolded con densidad media ≈ 1.

    Example:
        >>> gamma = CACHE.obtener(2000)
        >>> unfolded = unfold_kde(gamma)
        >>> print(np.mean(np.diff(unfolded)))  # ≈ 1.0
    """
    spectrum = np.sort(np.asarray(spectrum, dtype=np.float64))
    N = len(spectrum)

    if N < 10:
        raise ValueError(f"Se necesitan al menos 10 puntos, se tienen {N}.")

    # Ancho de banda: regla de Silverman si no se especifica
    if bandwidth is None:
        std = np.std(spectrum)
        if std < 1e-10:
            std = 1.0
        bandwidth = 0.9 * std * N ** (-0.2)

    # Grilla uniforme que cubre el espectro con margen
    margin = 3 * bandwidth
    x_min = spectrum[0]  - margin
    x_max = spectrum[-1] + margin
    x_grid = np.linspace(x_min, x_max, n_grid)
    dx = x_grid[1] - x_grid[0]

    # Densidad KDE: histograma suavizado
    counts, edges = np.histogram(spectrum, bins=n_grid,
                                  range=(x_min, x_max), density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sigma_bins = bandwidth / dx          # σ en unidades de bins
    rho = gaussian_filter1d(counts.astype(float), sigma=sigma_bins)
    rho = np.maximum(rho, 0.0)

    # CDF por integración trapezoidal acumulada
    cdf_grid = np.zeros(n_grid)
    for i in range(1, n_grid):
        cdf_grid[i] = cdf_grid[i - 1] + 0.5 * (rho[i] + rho[i - 1]) * dx

    # Normalizar CDF a [0, N]
    cdf_max = cdf_grid[-1]
    if cdf_max < 1e-10:
        raise ValueError("KDE produjo densidad nula. Verificar el espectro.")
    cdf_grid = cdf_grid * N / cdf_max

    # Interpolar CDF en los puntos del espectro
    unfolded = np.interp(spectrum, centers, cdf_grid)
    return unfolded


def unfold_spline(
    spectrum: np.ndarray,
    n_knots: int = 50,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """
    Unfolding por spline cúbica ajustada a la CDF empírica suavizada.

    Algoritmo:
        1. Calcular la CDF escalonada: F(xᵢ) = (i + 0.5) / N.
        2. Suavizar la CDF con un gaussiano para eliminar oscilaciones.
        3. Ajustar una CubicSpline a n_knots puntos uniformes.
        4. Evaluar la spline en los autovalores originales.

    Ventaja sobre KDE: la spline interpola exactamente los puntos de
    referencia y garantiza monotonía local. Mejor para densidades suaves.

    Args:
        spectrum    : Espectro ordenado.
        n_knots     : Número de nudos de la spline (más = más flexible).
        smooth_sigma: Suavizado gaussiano de la CDF antes del ajuste.

    Returns:
        Espectro unfolded con densidad media ≈ 1.
    """
    spectrum = np.sort(np.asarray(spectrum, dtype=np.float64))
    N = len(spectrum)

    if N < 20:
        raise ValueError(f"Spline necesita al menos 20 puntos, se tienen {N}.")

    # CDF escalonada (índice centrado)
    i_vals = np.arange(N, dtype=float)
    cdf_raw = (i_vals + 0.5) / N * N    # escala [0.5, N-0.5]

    # Suavizar la CDF (en espacio de índices)
    cdf_smooth = gaussian_filter1d(cdf_raw, sigma=smooth_sigma)

    # Seleccionar n_knots uniformemente distribuidos en el interior
    idx_knots = np.linspace(0, N - 1, n_knots, dtype=int)
    x_knots = spectrum[idx_knots]
    y_knots = cdf_smooth[idx_knots]

    # Eliminar duplicados en x_knots (pueden aparecer en espectros degenerados)
    _, unique_idx = np.unique(x_knots, return_index=True)
    x_knots = x_knots[unique_idx]
    y_knots = y_knots[unique_idx]

    if len(x_knots) < 4:
        raise ValueError("Muy pocos nudos únicos para ajustar spline. "
                         "El espectro puede tener muchos valores repetidos.")

    # Forzar monotonía estricta en y_knots
    for i in range(1, len(y_knots)):
        if y_knots[i] <= y_knots[i - 1]:
            y_knots[i] = y_knots[i - 1] + 1e-6

    cs = CubicSpline(x_knots, y_knots, extrapolate=True)
    unfolded = cs(spectrum)

    # Garantizar monotonía en la salida
    for i in range(1, N):
        if unfolded[i] <= unfolded[i - 1]:
            unfolded[i] = unfolded[i - 1] + 1e-8

    return unfolded


def unfold_polynomial(
    spectrum: np.ndarray,
    degree: int = 7,
) -> np.ndarray:
    """
    Unfolding polinomial: ajusta un polinomio de grado k a la CDF empírica.

    El polinomio actúa como un filtro pasa-bajos sobre la CDF: elimina
    las fluctuaciones de corto alcance (las estadísticas que queremos
    medir) y conserva la tendencia global (densidad promedio).

    Es el método más simple y clásico (usado en física nuclear desde los
    años 60). Menos flexible que KDE/spline pero muy estable.

    Args:
        spectrum: Espectro ordenado.
        degree  : Grado del polinomio (tipicamente 5–10).

    Returns:
        Espectro unfolded con densidad media ≈ 1.
    """
    spectrum = np.sort(np.asarray(spectrum, dtype=np.float64))
    N = len(spectrum)

    if N < degree + 2:
        raise ValueError(f"Se necesitan al menos {degree + 2} puntos para grado {degree}.")

    # CDF escalonada normalizada a [0, N]
    i_vals = np.arange(N, dtype=float)
    cdf_vals = (i_vals + 0.5) / N * N

    # Normalizar x a [-1, 1] para estabilidad numérica del polinomio
    x_min, x_max = spectrum[0], spectrum[-1]
    x_range = x_max - x_min
    if x_range < 1e-10:
        raise ValueError("Espectro degenerado (todos los valores iguales).")

    x_norm = 2.0 * (spectrum - x_min) / x_range - 1.0  # en [-1, 1]

    # Ajuste polinomial
    coeffs = np.polyfit(x_norm, cdf_vals, degree)
    unfolded = np.polyval(coeffs, x_norm)

    # Garantizar monotonía
    for i in range(1, N):
        if unfolded[i] <= unfolded[i - 1]:
            unfolded[i] = unfolded[i - 1] + 1e-8

    return unfolded


# ============================================================================
# MÉTRICAS DE CALIDAD DEL UNFOLDING
# ============================================================================

def unfolding_metrics(unfolded: np.ndarray, label: str = "") -> dict:
    """
    Calcula métricas de calidad de un espectro unfolded.

    Un buen unfolding debe producir:
        ⟨s⟩ ≈ 1.0         (densidad unitaria)
        σ(s) ∈ [0.3, 1.1]  (dispersión razonable según ensemble)
        r_mean cercano a algún ensemble conocido

    Args:
        unfolded: Espectro unfolded (densidad ≈ 1).
        label   : Nombre del método (para logging).

    Returns:
        dict con mean_s, std_s, r_mean, n_points, is_valid.
    """
    unfolded = np.asarray(unfolded, dtype=np.float64)
    spacings = np.diff(unfolded)
    spacings = spacings[spacings > 0]  # Filtrar degenerados

    if len(spacings) < 5:
        return {
            "label": label, "mean_s": np.nan, "std_s": np.nan,
            "r_mean": np.nan, "n_points": len(unfolded), "is_valid": False,
        }

    mean_s = float(np.mean(spacings))
    std_s  = float(np.std(spacings))

    # r-statistic (sin unfolding adicional, usa los spacings directamente)
    r_vals = (np.minimum(spacings[:-1], spacings[1:]) /
              np.maximum(spacings[:-1], spacings[1:]))
    r_vals = r_vals[np.isfinite(r_vals)]
    r_mean = float(np.mean(r_vals)) if len(r_vals) > 0 else np.nan

    # Validación básica
    is_valid = (0.7 < mean_s < 1.3) and np.isfinite(std_s)

    return {
        "label"   : label,
        "mean_s"  : mean_s,
        "std_s"   : std_s,
        "r_mean"  : r_mean,
        "n_points": len(unfolded),
        "is_valid": is_valid,
    }


# ============================================================================
# COMPARACIÓN DE MÉTODOS
# ============================================================================

def compare_unfolding_methods(
    spectrum: np.ndarray,
    analytic_fn=None,
    kde_bandwidth: Optional[float] = None,
    spline_knots: int = 50,
    poly_degree: int = 7,
    recorte: float = 0.1,
) -> Dict[str, dict]:
    """
    Compara los cuatro métodos de unfolding sobre el mismo espectro.

    Ejecuta KDE, spline, polinomial y (opcionalmente) el método analítico,
    calcula métricas para cada uno y devuelve un dict con los resultados.

    Args:
        spectrum     : Espectro crudo ordenado.
        analytic_fn  : Función de unfolding analítico (p.ej. unfolding_riemann).
                       Si None, se omite el método analítico.
        kde_bandwidth: Ancho de banda KDE (None = Silverman).
        spline_knots : Nudos de la spline.
        poly_degree  : Grado del polinomio.
        recorte      : Fracción a eliminar en cada extremo antes de calcular
                       métricas (reduce efectos de borde).

    Returns:
        dict con clave = nombre del método, valor = dict de métricas +
        el espectro unfolded completo en "unfolded".

    Example:
        >>> gamma = CACHE.obtener(2000)
        >>> from riemann_spectral.analysis.unfolding import unfolding_riemann
        >>> results = compare_unfolding_methods(gamma, analytic_fn=unfolding_riemann)
        >>> for name, r in results.items():
        ...     print(f"{name}: <s>={r['mean_s']:.4f}  r={r['r_mean']:.4f}")
    """
    spectrum = np.sort(np.asarray(spectrum, dtype=np.float64))
    results = {}

    methods = {
        "KDE"       : lambda s: unfold_kde(s, bandwidth=kde_bandwidth),
        "Spline"    : lambda s: unfold_spline(s, n_knots=spline_knots),
        "Polinomial": lambda s: unfold_polynomial(s, degree=poly_degree),
    }
    if analytic_fn is not None:
        methods["Analítico"] = analytic_fn

    for name, fn in methods.items():
        try:
            unfolded_full = fn(spectrum)
            unfolded_full = np.asarray(unfolded_full, dtype=np.float64)

            # Recortar extremos para métricas
            n = len(unfolded_full)
            i0 = int(n * recorte)
            i1 = n - i0
            unfolded_central = unfolded_full[i0:i1]

            metrics = unfolding_metrics(unfolded_central, label=name)
            metrics["unfolded"]         = unfolded_full
            metrics["unfolded_central"] = unfolded_central
            results[name] = metrics

        except Exception as e:
            results[name] = {
                "label": name, "mean_s": np.nan, "std_s": np.nan,
                "r_mean": np.nan, "n_points": len(spectrum),
                "is_valid": False, "error": str(e),
                "unfolded": None, "unfolded_central": None,
            }

    return results


# ============================================================================
# DIAGNÓSTICO DE ESPACIADOS
# ============================================================================

def spacing_histogram(
    unfolded: np.ndarray,
    bins: int = 50,
    recorte: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histograma de espaciados normalizados P(s) a partir de espectro unfolded.

    Args:
        unfolded: Espectro unfolded.
        bins    : Número de bins del histograma.
        recorte : Fracción a eliminar en extremos.

    Returns:
        (centers, density): centros de bins y densidad normalizada.
    """
    unfolded = np.sort(np.asarray(unfolded, dtype=np.float64))
    n = len(unfolded)
    i0 = int(n * recorte)
    i1 = n - i0
    central = unfolded[i0:i1]

    spacings = np.diff(central)
    spacings = spacings[spacings > 0]
    if len(spacings) == 0:
        return np.array([]), np.array([])

    # Normalizar ⟨s⟩ = 1
    s_mean = np.mean(spacings)
    if s_mean > 1e-10:
        spacings = spacings / s_mean

    hist, edges = np.histogram(spacings, bins=bins, range=(0, 4), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


# ============================================================================
# AUTO-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EMPIRICAL UNFOLDING — AUTO TEST")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N = 3000

    # Espectro de prueba: Poisson (cumsum exponencial)
    spectrum_poisson = np.cumsum(rng.exponential(1.0, N))

    print(f"\nEspectro Poisson, N={N}")
    print(f"  Rango: [{spectrum_poisson[0]:.1f}, {spectrum_poisson[-1]:.1f}]")

    from riemann_spectral.analysis.unfolding import unfolding_riemann

    results = compare_unfolding_methods(
        spectrum_poisson,
        analytic_fn=None,   # Poisson no tiene analítico natural
        recorte=0.1,
    )

    print()
    print(f"{'Método':<14} {'<s>':<8} {'σ(s)':<8} {'r_mean':<8} {'Válido'}")
    print("-" * 50)
    for name, r in results.items():
        v = "✅" if r["is_valid"] else "❌"
        ms  = f"{r['mean_s']:.4f}" if np.isfinite(r['mean_s'])  else "NaN"
        ss  = f"{r['std_s']:.4f}"  if np.isfinite(r['std_s'])   else "NaN"
        rm  = f"{r['r_mean']:.4f}" if np.isfinite(r['r_mean'])  else "NaN"
        print(f"  {name:<12} {ms:<8} {ss:<8} {rm:<8} {v}")

    print()
    print("Esperado: <s> ≈ 1.0 para todos los métodos con Poisson")
    print("=" * 60)
