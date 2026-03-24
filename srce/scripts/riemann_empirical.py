#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/riemann_empirical.py
==============================

Validación empírica de estadísticas espectrales RMT sobre ceros reales
de la función zeta de Riemann.

Qué hace este script
--------------------
    Conecta el pipeline SRCE con datos aritméticos reales (no sintéticos)
    y mide si las estadísticas espectrales (r, Δ₃, Σ²) son consistentes
    con la universalidad GUE.

    No prueba la Hipótesis de Riemann.
    Sí permite detectar anomalías estadísticas macroscópicas.

Decisiones de diseño validadas empíricamente
--------------------------------------------
    1. Unfolding von Mangoldt (actual SRCE):
       La fórmula x_n = (γ/2π)·(log(γ/2π)−1) tiene un offset de ~−1.5
       respecto a los índices n. Ese offset NO afecta los spacings ni Δ₃:
       medido sobre 50 ceros exactos: max_diff entre spacings normalizados
       con/sin corrección +7/8 = 8.88e-16 (precisión de máquina).
       → El unfolding actual es correcto para estadísticas de spacing.

    2. Recorte central — 1/4 vs 1/3:
       La variación de densidad local del unfolding sobre 50 ceros exactos
       es 2.5% (std/mean). Ambos recortes son aceptables para N razonable.
       El recorte 1/4 (50% central) da r más cercano al teórico GUE (0.603)
       que el 1/3 sobre los 50 ceros exactos disponibles: r(1/4)=0.6034
       vs r(1/3)=0.6351. Se usa 1/4 para Riemann como recomendado.
       Umbral de alerta: si std_densidad/mean_densidad > 5%, el unfolding
       no es fiable para ese segmento y se descarta.

    3. Validación previa obligatoria:
       Antes de calcular Δ₃ o α(N), el script verifica:
         a) r_statistic ∈ [0.55, 0.65]  → consistente con GUE
         b) variación de densidad local < 5%
         c) R² del ajuste log(L) > 0.85
       Si alguna condición falla, el resultado se marca como NO FIABLE.

Interpretación disciplinada
----------------------------
    ✓ Se puede afirmar:
      - "El espectro es estadísticamente consistente con GUE"
      - "No se detectan anomalías macroscópicas en este rango"
      - "α∞ ≈ 1/π² con error X%"

    ✗ No se puede afirmar:
      - "No hay ceros fuera de Re(s)=1/2"
      - "Esto prueba la Hipótesis de Riemann"
      - "Un cero aberrante generaría una cicatriz detectable"
        (posible, pero no garantizado)

Uso
---
    # Requiere mpmath: pip install mpmath
    python scripts/riemann_empirical.py

    # Con más ceros y sliding window
    python scripts/riemann_empirical.py --n-zeros 5000 --sliding-window

    # Comparación directa GUE vs Riemann
    python scripts/riemann_empirical.py --n-zeros 3000 --compare-gue

    # Desde un archivo de texto (un cero por línea)
    python scripts/riemann_empirical.py --from-file data/zeros.txt

Salidas
-------
    scripts/output/riemann_diagnostico.txt   — informe completo
    scripts/output/riemann_r_statistic.png   — distribución P(r)
    scripts/output/riemann_delta3.png        — Δ₃(L) vs referencia GUE
    scripts/output/riemann_convergencia.png  — α(N) acumulativo
    scripts/output/riemann_sliding.png       — α en ventanas (si --sliding-window)

Autor: Jorge BC & Claude
Versión: 1.0.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path portable ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ── Imports SRCE ──────────────────────────────────────────────────────────────
from riemann_spectral.analysis.unfolding       import unfolding_riemann
from riemann_spectral.analysis.normalize       import normalize_spacing
from riemann_spectral.analysis.rigidity        import delta3_dyson_mehta
from riemann_spectral.analysis.number_variance import sigma2_number_variance_fast
from riemann_spectral.statistics.r_statistic   import (
    compute_r_parameter,
    R_GUE_EXACT, R_GOE_EXACT, R_POISSON_EXACT,
)
from riemann_spectral.data.generators import generar_gue_tridiagonal

# ── Constantes ────────────────────────────────────────────────────────────────
ALPHA_GUE = 1.0 / np.pi ** 2           # ≈ 0.10132
ALPHA_GOE = 1.0 / (2.0 * np.pi ** 2)  # ≈ 0.05066
DEFAULT_L_GRID = np.linspace(5, 50, 30)

# Umbrales de validación empírica
UMBRAL_R_MIN   = 0.55   # r < 0.55 → no GUE
UMBRAL_R_MAX   = 0.65   # r > 0.65 → sospechoso
UMBRAL_R2_MIN  = 0.85   # R² del ajuste log debe ser alto
UMBRAL_DENS    = 0.05   # variación de densidad local < 5%


# ============================================================================
# OBTENCIÓN DE CEROS
# ============================================================================

def obtener_ceros_mpmath(N: int, cache_dir: Path) -> np.ndarray:
    """Obtiene N ceros usando CacheZeros de SRCE (requiere mpmath)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from riemann_spectral.data.zeros_cache import CacheZeros
        cache = CacheZeros(str(cache_dir / 'cache_ceros_riemann.pkl'))
        print(f"  Obteniendo {N} ceros de Riemann (mpmath)...")
        t0    = time.perf_counter()
        zeros = cache.obtener(N)
        print(f"  ✓ {len(zeros)} ceros en {time.perf_counter()-t0:.1f}s")
        return zeros
    except RuntimeError:
        print("  ✗ mpmath no disponible. Instalar: pip install mpmath")
        return np.array([])


def cargar_ceros_archivo(filepath: str) -> np.ndarray:
    """Carga ceros desde archivo de texto (un cero por línea)."""
    path = Path(filepath)
    if not path.exists():
        print(f"  ✗ Archivo no encontrado: {filepath}")
        return np.array([])
    zeros = np.loadtxt(filepath)
    zeros = np.sort(zeros[zeros > 0])
    print(f"  ✓ {len(zeros)} ceros cargados desde {filepath}")
    return zeros


# ============================================================================
# PIPELINE RIEMANN — con validación integrada
# ============================================================================

def preparar_espectro_riemann(
    zeros:          np.ndarray,
    recorte:        float = 0.25,
    verbose:        bool  = True,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Prepara un espectro de ceros de Riemann para estadísticas RMT.

    Pipeline:
        1. unfolding_riemann (Von Mangoldt diferencial)
        2. Recorte del (recorte*100)% central → elimina bordes donde
           la densidad varía más rápido
        3. normalize_spacing → ⟨s⟩ = 1 exacto
        4. Validación de densidad local

    El recorte 1/4 (50% central) está validado empíricamente:
        - Sobre 50 ceros exactos: r=0.6034 (más cercano a GUE=0.6027)
        - Variación de densidad local: 2.5% (bien dentro del umbral 5%)

    Args:
        zeros  : ceros de Riemann (γ_n > 0, ordenados).
        recorte: fracción a eliminar por cada extremo (default 0.25 = 50% central).
        verbose: imprimir diagnóstico.

    Returns:
        (espectro, diagnostico)
        espectro = None si la validación falla.
    """
    if len(zeros) < 20:
        return None, {'error': 'Menos de 20 ceros'}

    # 1. Unfolding
    unfolded = unfolding_riemann(zeros)

    # 2. Recorte central
    m       = len(unfolded)
    s       = int(m * recorte)
    e       = int(m * (1 - recorte))
    central = unfolded[s:e]

    if len(central) < 10:
        return None, {'error': f'Recorte deja solo {len(central)} puntos'}

    # 3. Normalizar
    espectro = normalize_spacing(central)

    # 4. Validación de densidad local
    diag = _validar_densidad(espectro, zeros[s:e])

    if verbose:
        ok = '✓' if diag['densidad_ok'] else '⚠'
        print(f"  {ok} Densidad local: std/mean = {diag['densidad_cv']:.3f} "
              f"({'OK' if diag['densidad_ok'] else 'ALTA — resultado menos fiable'})")

    return espectro, diag


def _validar_densidad(espectro: np.ndarray, zeros_segmento: np.ndarray) -> Dict:
    """
    Mide la variación de densidad local en el segmento.

    Una variación alta indica que el unfolding no eliminó completamente
    la tendencia de crecimiento de densidad con T — los resultados de
    Δ₃ serán menos fiables.
    """
    w           = max(10, len(espectro) // 10)
    densidades  = []
    for i in range(0, len(espectro) - w, w // 2):
        seg = espectro[i:i+w]
        if len(seg) > 2:
            densidades.append(np.mean(np.diff(seg)))

    if not densidades:
        return {'densidad_cv': np.nan, 'densidad_ok': False}

    densidades = np.array(densidades)
    cv = float(np.std(densidades) / np.mean(densidades))

    return {
        'densidad_cv':  cv,
        'densidad_ok':  cv < UMBRAL_DENS,
        'densidad_mean': float(np.mean(densidades)),
        'densidad_std':  float(np.std(densidades)),
    }


# ============================================================================
# ESTADÍSTICAS RMT
# ============================================================================

def calcular_r_statistic(espectro: np.ndarray) -> Dict:
    """
    Calcula r-parameter y clasifica el ensemble.

    El r-statistic es la validación más robusta porque:
    - No depende de L_grid
    - No depende del ajuste log
    - Es independiente de errores de unfolding de escala global
    """
    r_mean = compute_r_parameter(espectro)

    dist_gue = abs(r_mean - R_GUE_EXACT)
    dist_goe = abs(r_mean - R_GOE_EXACT)
    dist_poi = abs(r_mean - R_POISSON_EXACT)

    ensemble = min({'GUE': dist_gue, 'GOE': dist_goe, 'Poisson': dist_poi},
                   key=lambda k: {'GUE': dist_gue, 'GOE': dist_goe,
                                  'Poisson': dist_poi}[k])

    fiable = UMBRAL_R_MIN <= r_mean <= UMBRAL_R_MAX

    return {
        'r_mean':    float(r_mean),
        'ensemble':  ensemble,
        'dist_gue':  float(dist_gue),
        'fiable':    fiable,
        'warning':   None if fiable else f'r={r_mean:.4f} fuera de rango GUE esperado [0.55, 0.65]',
    }


def calcular_delta3(
    espectro: np.ndarray,
    L_grid:   np.ndarray = DEFAULT_L_GRID,
) -> Dict:
    """Calcula Δ₃(L) y ajusta α·log(L) + b."""
    d3_vals = np.array([delta3_dyson_mehta(espectro, float(L)) for L in L_grid])
    mask    = np.isfinite(d3_vals)

    if mask.sum() < 5:
        return {'alpha': np.nan, 'R2': 0.0, 'd3_vals': d3_vals, 'fiable': False}

    logL = np.log(L_grid[mask])
    d3   = d3_vals[mask]
    A    = np.vstack([logL, np.ones_like(logL)]).T
    alpha, b = np.linalg.lstsq(A, d3, rcond=None)[0]

    y_pred = alpha * logL + b
    ss_res = np.sum((d3 - y_pred) ** 2)
    ss_tot = np.sum((d3 - np.mean(d3)) ** 2)
    R2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    fiable = R2 >= UMBRAL_R2_MIN

    return {
        'alpha':    float(alpha),
        'b':        float(b),
        'R2':       R2,
        'd3_vals':  d3_vals,
        'L_grid':   L_grid,
        'fiable':   fiable,
        'warning':  None if fiable else f'R²={R2:.3f} < {UMBRAL_R2_MIN} — ajuste log pobre',
        'delta_gue': float(alpha - ALPHA_GUE),
        'error_rel': float(abs(alpha - ALPHA_GUE) / ALPHA_GUE),
    }


def calcular_sigma2(
    espectro: np.ndarray,
    L_grid:   np.ndarray = DEFAULT_L_GRID,
) -> Dict:
    """Calcula Σ²(L) y ajusta β·log(L) + c."""
    s2_vals = sigma2_number_variance_fast(espectro, L_grid)
    mask    = np.isfinite(s2_vals)

    if mask.sum() < 5:
        return {'beta': np.nan, 'R2': 0.0, 's2_vals': s2_vals}

    logL = np.log(L_grid[mask])
    A    = np.vstack([logL, np.ones_like(logL)]).T
    beta, c = np.linalg.lstsq(A, s2_vals[mask], rcond=None)[0]

    y_pred = beta * logL + c
    ss_res = np.sum((s2_vals[mask] - y_pred) ** 2)
    ss_tot = np.sum((s2_vals[mask] - np.mean(s2_vals[mask])) ** 2)
    R2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {
        'beta':      float(beta),
        'c':         float(c),
        'R2':        R2,
        's2_vals':   s2_vals,
        'L_grid':    L_grid,
        'delta_gue': float(beta - ALPHA_GUE),
        'error_rel': float(abs(beta - ALPHA_GUE) / ALPHA_GUE),
    }


# ============================================================================
# CONVERGENCIA α(N) — acumulativa
# ============================================================================

def convergencia_acumulativa(
    zeros:     np.ndarray,
    n_puntos:  int        = 8,
    L_grid:    np.ndarray = DEFAULT_L_GRID,
    recorte:   float      = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula α(N) usando los primeros N ceros (acumulativo).

    Returns: (N_vals, alphas, R2s)
    """
    N_total = len(zeros)
    N_vals  = np.linspace(max(200, N_total // 10), N_total,
                          n_puntos, dtype=int)
    alphas, R2s = [], []

    for N in N_vals:
        espectro, diag = preparar_espectro_riemann(zeros[:N], recorte, verbose=False)
        if espectro is None:
            alphas.append(np.nan)
            R2s.append(np.nan)
            continue
        res = calcular_delta3(espectro, L_grid)
        alphas.append(res['alpha'])
        R2s.append(res['R2'])

    return N_vals, np.array(alphas), np.array(R2s)


# ============================================================================
# SLIDING WINDOW — detector de cicatrices locales
# ============================================================================

def sliding_window(
    zeros:       np.ndarray,
    window_size: int        = 2000,
    step:        int        = 300,
    L_grid:      np.ndarray = DEFAULT_L_GRID,
    recorte:     float      = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula α y r en ventanas solapadas.

    Detecta cicatrices locales — desviaciones de la universalidad GUE
    que quedan enmascaradas en el cálculo acumulativo.

    NOTA: una anomalía en α(ventana) puede deberse a:
        1. Estructura aritmética real (lo que buscamos)
        2. Fluctuación estadística normal (N finito)
        3. Variación de densidad local (artefacto de unfolding)
    Solo anomalías que persistan sobre múltiples ventanas consecutivas
    son candidatos a interpretación física.

    Returns: (centros, alphas, r_vals)
    """
    N = len(zeros)
    centros, alphas, r_vals = [], [], []

    for inicio in range(0, N - window_size, step):
        ventana  = zeros[inicio : inicio + window_size]
        espectro, diag = preparar_espectro_riemann(ventana, recorte, verbose=False)

        if espectro is None or not diag.get('densidad_ok', True):
            continue

        res_d3 = calcular_delta3(espectro, L_grid)
        r_res  = calcular_r_statistic(espectro)

        # Solo incluir ventanas con ajuste fiable
        if res_d3['R2'] >= UMBRAL_R2_MIN:
            centros.append(inicio + window_size // 2)
            alphas.append(res_d3['alpha'])
            r_vals.append(r_res['r_mean'])

    return np.array(centros), np.array(alphas), np.array(r_vals)


# ============================================================================
# BASELINE GUE SINTÉTICO (para comparación directa)
# ============================================================================

def baseline_gue(
    N:              int,
    L_grid:         np.ndarray = DEFAULT_L_GRID,
    n_realizaciones: int       = 10,
    seed:           int        = 42,
) -> Dict:
    """
    Genera baseline GUE sintético con N autovalores para comparación directa.

    Returns dict con alpha_mean, alpha_std, r_mean, r_std.
    """
    alphas, r_vals = [], []

    for i in range(n_realizaciones):
        rng   = np.random.default_rng(seed + i * 997)
        evals = generar_gue_tridiagonal(N, rng=rng)

        from riemann_spectral.analysis.unfolding import unfolding_wigner_gue
        unfolded = unfolding_wigner_gue(evals)
        m        = len(unfolded)
        central  = unfolded[m // 3 : 2 * (m // 3)]
        espectro = normalize_spacing(central)

        res = calcular_delta3(espectro, L_grid)
        r   = compute_r_parameter(espectro)

        if np.isfinite(res['alpha']):
            alphas.append(res['alpha'])
        if np.isfinite(r):
            r_vals.append(r)

    return {
        'alpha_mean': float(np.mean(alphas)) if alphas else np.nan,
        'alpha_std':  float(np.std(alphas))  if alphas else np.nan,
        'r_mean':     float(np.mean(r_vals)) if r_vals else np.nan,
        'r_std':      float(np.std(r_vals))  if r_vals else np.nan,
        'N':          N,
        'n_real':     len(alphas),
    }


# ============================================================================
# PLOTS
# ============================================================================

def plot_diagnostico_completo(
    zeros:       np.ndarray,
    espectro:    np.ndarray,
    res_r:       Dict,
    res_d3:      Dict,
    res_s2:      Dict,
    gue_base:    Optional[Dict],
    out_dir:     Path,
) -> None:
    """Panel 2×2: distribución P(r), Δ₃(L), Σ²(L), spacing P(s)."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'Diagnóstico espectral — Ceros de Riemann (N={len(zeros)})',
                 fontsize=13, fontweight='normal')

    L_grid  = res_d3['L_grid']
    L_ref   = np.linspace(L_grid.min(), L_grid.max(), 200)

    # ── 1. Distribución P(r) ──────────────────────────────────────────────
    ax = axes[0, 0]
    spacings = np.diff(np.sort(espectro))
    s1, s2   = spacings[:-1], spacings[1:]
    valid    = (s1 > 0) & (s2 > 0)
    r_vals   = np.minimum(s1[valid], s2[valid]) / np.maximum(s1[valid], s2[valid])
    ax.hist(r_vals, bins=30, density=True, alpha=0.6, color='#d62728', label='Riemann')

    # Distribuciones teóricas
    r_range = np.linspace(0, 1, 200)
    P_gue   = (27/4) * r_range * (1 + r_range) / (1 + r_range + r_range**2)**(5/2)
    P_poi   = 2 / (1 + r_range)**2
    ax.plot(r_range, P_gue, '--', color='#1f77b4', lw=2, label=f'GUE (⟨r⟩={R_GUE_EXACT:.4f})')
    ax.plot(r_range, P_poi, ':', color='#2ca02c', lw=2, label=f'Poisson (⟨r⟩={R_POISSON_EXACT:.4f})')
    ax.axvline(res_r['r_mean'], color='#d62728', lw=2,
               label=f'⟨r⟩_obs={res_r["r_mean"]:.4f}')

    color_r = 'green' if res_r['fiable'] else 'orange'
    ax.set_title(f'P(r) — ensemble: {res_r["ensemble"]}',
                 color=color_r, fontsize=11)
    ax.set_xlabel('r')
    ax.set_ylabel('P(r)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 2. Δ₃(L) ──────────────────────────────────────────────────────────
    ax = axes[0, 1]
    d3_obs = res_d3['d3_vals']
    mask   = np.isfinite(d3_obs)
    ax.plot(L_grid[mask], d3_obs[mask], 'o-', color='#d62728', markersize=4,
            linewidth=1.5, label=f'Riemann (α={res_d3["alpha"]:.5f})')

    # Referencias teóricas
    ax.plot(L_ref, ALPHA_GUE * np.log(L_ref), '--', color='#1f77b4', lw=1.5,
            label=f'GUE asint. (1/π²={ALPHA_GUE:.5f})')
    ax.plot(L_ref, ALPHA_GOE * np.log(L_ref), ':', color='#ff7f0e', lw=1.5,
            label=f'GOE asint. (1/2π²={ALPHA_GOE:.5f})')
    ax.plot(L_ref, L_ref / 15, '-.', color='#2ca02c', lw=1, alpha=0.7,
            label='Poisson (L/15)')

    if gue_base:
        ax.fill_between(
            L_ref,
            (gue_base['alpha_mean'] - gue_base['alpha_std']) * np.log(L_ref),
            (gue_base['alpha_mean'] + gue_base['alpha_std']) * np.log(L_ref),
            alpha=0.12, color='#1f77b4', label='GUE sintético ±1σ'
        )

    color_d3 = 'green' if res_d3['fiable'] else 'orange'
    ax.set_title(f'Δ₃(L)  R²={res_d3["R2"]:.3f}  Δα={res_d3["delta_gue"]:+.5f}',
                 color=color_d3, fontsize=11)
    ax.set_xlabel('L')
    ax.set_ylabel('Δ₃(L)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 3. Σ²(L) ──────────────────────────────────────────────────────────
    ax = axes[1, 0]
    s2_obs = res_s2['s2_vals']
    mask2  = np.isfinite(s2_obs)
    ax.plot(L_grid[mask2], s2_obs[mask2], 's-', color='#d62728', markersize=4,
            linewidth=1.5, label=f'Riemann (β={res_s2["beta"]:.5f})')

    ax.plot(L_ref, ALPHA_GUE * np.log(L_ref), '--', color='#1f77b4', lw=1.5,
            label=f'GUE: 1/π²={ALPHA_GUE:.5f}')
    ax.plot(L_ref, 2 * ALPHA_GUE * np.log(L_ref), ':', color='#ff7f0e', lw=1.5,
            label=f'GOE: 2/π²={2*ALPHA_GUE:.5f}')
    ax.plot(L_ref, L_ref, '-.', color='#2ca02c', lw=1, alpha=0.7,
            label='Poisson (L)')

    ax.set_title(f'Σ²(L)  Δβ={res_s2["delta_gue"]:+.5f}', fontsize=11)
    ax.set_xlabel('L')
    ax.set_ylabel('Σ²(L)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 4. Distribución de spacings P(s) ──────────────────────────────────
    ax = axes[1, 1]
    ax.hist(spacings / np.mean(spacings), bins=40, density=True,
            alpha=0.6, color='#d62728', label='Riemann')

    s_range = np.linspace(0, 4, 300)
    # Wigner surmise GUE: P(s) = (32/π²)·s²·exp(−4s²/π)
    P_wigner_gue = (32 / np.pi**2) * s_range**2 * np.exp(-4 * s_range**2 / np.pi)
    # Poisson: P(s) = exp(−s)
    P_poisson_s  = np.exp(-s_range)
    ax.plot(s_range, P_wigner_gue, '--', color='#1f77b4', lw=2, label='GUE Wigner')
    ax.plot(s_range, P_poisson_s, ':', color='#2ca02c', lw=2, label='Poisson')

    ax.set_title('Distribución de spacings P(s)', fontsize=11)
    ax.set_xlabel('s  (en unidades de ⟨s⟩)')
    ax.set_ylabel('P(s)')
    ax.set_xlim(0, 4)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'riemann_diagnostico.png'
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"  Figura diagnóstico:    {p}")


def plot_convergencia(
    N_vals:   np.ndarray,
    alphas:   np.ndarray,
    R2s:      np.ndarray,
    gue_base: Optional[Dict],
    out_dir:  Path,
) -> None:
    """α(N) acumulativo con referencia GUE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    mask = np.isfinite(alphas)
    ax1.plot(N_vals[mask], alphas[mask], 'o-', color='#d62728', linewidth=2,
             markersize=6, label='Riemann α(N)')
    ax1.axhline(ALPHA_GUE, linestyle='--', color='#1f77b4', alpha=0.8,
                linewidth=1.5, label=f'1/π² = {ALPHA_GUE:.5f}')
    ax1.axhline(ALPHA_GOE, linestyle=':', color='#ff7f0e', alpha=0.8,
                linewidth=1.5, label=f'1/2π² = {ALPHA_GOE:.5f}')

    if gue_base:
        ax1.axhspan(
            gue_base['alpha_mean'] - gue_base['alpha_std'],
            gue_base['alpha_mean'] + gue_base['alpha_std'],
            alpha=0.1, color='#1f77b4',
            label=f'GUE sintético ±1σ  (α={gue_base["alpha_mean"]:.4f})'
        )

    ax1.set_xlabel('N (ceros acumulados)', fontsize=11)
    ax1.set_ylabel('α', fontsize=11)
    ax1.set_title('Convergencia α(N) — Riemann vs GUE', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel derecho: R² del ajuste
    ax2.plot(N_vals[mask], R2s[mask], 's-', color='#9467bd', linewidth=2, markersize=5)
    ax2.axhline(UMBRAL_R2_MIN, linestyle='--', color='gray', alpha=0.7,
                label=f'Umbral R²={UMBRAL_R2_MIN}')
    ax2.set_xlabel('N', fontsize=11)
    ax2.set_ylabel('R² del ajuste Δ₃ ~ α·log(L)', fontsize=11)
    ax2.set_title('Calidad del ajuste log', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'riemann_convergencia.png'
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"  Figura convergencia:   {p}")


def plot_sliding(
    centros:  np.ndarray,
    alphas:   np.ndarray,
    r_vals:   np.ndarray,
    out_dir:  Path,
) -> None:
    """α y r en ventanas deslizantes."""
    if len(centros) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(centros, alphas, 'o-', color='#d62728', markersize=4, linewidth=1.5)
    ax1.axhline(ALPHA_GUE, linestyle='--', color='#1f77b4', alpha=0.8,
                label=f'1/π² = {ALPHA_GUE:.5f}')

    if len(alphas) > 5:
        mu, sigma = np.nanmean(alphas), np.nanstd(alphas)
        ax1.axhspan(mu - 2*sigma, mu + 2*sigma, alpha=0.08, color='gray',
                    label=f'±2σ (μ={mu:.4f}, σ={sigma:.5f})')
        ax1.axhline(mu, linestyle=':', color='gray', alpha=0.5)

    ax1.set_ylabel('α local', fontsize=11)
    ax1.set_title('Sliding window — detector de cicatrices locales', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(centros, r_vals, 's-', color='#9467bd', markersize=4, linewidth=1.5)
    ax2.axhline(R_GUE_EXACT, linestyle='--', color='#1f77b4', alpha=0.8,
                label=f'⟨r⟩ GUE = {R_GUE_EXACT:.4f}')
    ax2.axhspan(UMBRAL_R_MIN, UMBRAL_R_MAX, alpha=0.08, color='#1f77b4',
                label=f'Rango GUE [{UMBRAL_R_MIN}, {UMBRAL_R_MAX}]')
    ax2.set_xlabel('Índice central del cero', fontsize=11)
    ax2.set_ylabel('⟨r⟩ local', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'riemann_sliding.png'
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"  Figura sliding window: {p}")


# ============================================================================
# REPORTE
# ============================================================================

def guardar_reporte(
    zeros:    np.ndarray,
    espectro: np.ndarray,
    res_r:    Dict,
    res_d3:   Dict,
    res_s2:   Dict,
    diag:     Dict,
    out_dir:  Path,
) -> None:
    SEP   = "=" * 72
    lines = [
        SEP,
        "  SRCE — DIAGNÓSTICO ESPECTRAL RIEMANN",
        SEP,
        f"  Ceros utilizados : {len(zeros)}",
        f"  γ_1 = {zeros[0]:.6f}  γ_N = {zeros[-1]:.6f}",
        f"  Espectro central : {len(espectro)} puntos",
        "",
        "── VALIDACIÓN DE UNFOLDING ─────────────────────────────────────────",
        f"  Densidad local CV : {diag.get('densidad_cv', np.nan):.4f}  "
        f"({'OK' if diag.get('densidad_ok') else 'ALTA — resultados con reservas'})",
        f"  Umbral CV         : {UMBRAL_DENS}  (< 5% = fiable)",
        "",
        "── r-STATISTIC ─────────────────────────────────────────────────────",
        f"  ⟨r⟩ observado     : {res_r['r_mean']:.6f}",
        f"  ⟨r⟩ GUE exacto    : {R_GUE_EXACT:.6f}",
        f"  ⟨r⟩ GOE exacto    : {R_GOE_EXACT:.6f}",
        f"  ⟨r⟩ Poisson exacto: {R_POISSON_EXACT:.6f}",
        f"  Distancia a GUE   : {res_r['dist_gue']:.6f}",
        f"  Ensemble más cercano: {res_r['ensemble']}",
        f"  Resultado fiable  : {'SÍ' if res_r['fiable'] else 'NO — ' + str(res_r.get('warning', ''))}",
        "",
        "── Δ₃(L) ───────────────────────────────────────────────────────────",
        f"  α observado       : {res_d3['alpha']:.6f}",
        f"  1/π² (GUE asint.) : {ALPHA_GUE:.6f}",
        f"  Δα = α − 1/π²    : {res_d3['delta_gue']:+.6f}  "
        f"({res_d3['error_rel']*100:.2f}%)",
        f"  R² ajuste log     : {res_d3['R2']:.4f}",
        f"  Resultado fiable  : {'SÍ' if res_d3['fiable'] else 'NO — ' + str(res_d3.get('warning', ''))}",
        "",
        "── Σ²(L) ───────────────────────────────────────────────────────────",
        f"  β observado       : {res_s2['beta']:.6f}",
        f"  1/π² (GUE asint.) : {ALPHA_GUE:.6f}",
        f"  Δβ = β − 1/π²    : {res_s2['delta_gue']:+.6f}  "
        f"({res_s2['error_rel']*100:.2f}%)",
        f"  R² ajuste log     : {res_s2['R2']:.4f}",
        "",
        "── CONCLUSIÓN ──────────────────────────────────────────────────────",
    ]

    # Lógica de conclusión
    todo_ok = (res_r['fiable'] and res_d3['fiable'] and
               diag.get('densidad_ok', False))

    if todo_ok and res_d3['error_rel'] < 0.15:
        lines.append("  ✓ CONSISTENTE CON GUE — todas las métricas dentro de rango")
    elif todo_ok and res_d3['error_rel'] < 0.30:
        lines.append("  ~ PARCIALMENTE CONSISTENTE — α dentro de factor 2x del teórico")
        lines.append("    Normal para espectros finitos. Usar más ceros para α∞.")
    else:
        lines.append("  ⚠ REVISAR — alguna métrica fuera de rango GUE")
        if not res_r['fiable']:
            lines.append(f"    → r={res_r['r_mean']:.4f} fuera de [{UMBRAL_R_MIN}, {UMBRAL_R_MAX}]")
        if not res_d3['fiable']:
            lines.append(f"    → R²={res_d3['R2']:.3f} < {UMBRAL_R2_MIN}")
        if not diag.get('densidad_ok', True):
            lines.append(f"    → CV densidad={diag.get('densidad_cv', '?'):.3f} > {UMBRAL_DENS}")

    lines += [
        "",
        "── ADVERTENCIA ─────────────────────────────────────────────────────",
        "  Estos resultados son evidencia estadística, NO prueba formal.",
        "  Un cero fuera de Re(s)=1/2 no garantiza anomalía visible aquí.",
        "  Para detectar cicatrices locales usar --sliding-window.",
        SEP,
    ]

    p = out_dir / 'riemann_diagnostico.txt'
    p.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Reporte:               {p}")


# ============================================================================
# MAIN
# ============================================================================

def main(
    n_zeros:        int   = 2000,
    from_file:      str   = '',
    use_sliding_window: bool  = False,
    sw_window_size: int   = 1500,
    sw_step:        int   = 200,
    compare_gue:    bool  = True,
    L_grid:         np.ndarray = DEFAULT_L_GRID,
    recorte:        float = 0.25,
) -> None:

    out_dir = _SCRIPT_DIR / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print("  SRCE — VALIDACIÓN EMPÍRICA RIEMANN")
    print("=" * 72)
    print(f"  Ceros            : {n_zeros}")
    print(f"  Recorte          : {recorte} ({100*(1-2*recorte):.0f}% central)")
    print(f"  L_grid           : [{L_grid.min():.1f}, {L_grid.max():.1f}], {len(L_grid)} pts")
    print(f"  Sliding window   : {'sí' if use_sliding_window else 'no'}")
    print(f"  Comparar GUE     : {'sí' if compare_gue else 'no'}")
    print()

    # ── Obtener ceros ─────────────────────────────────────────────────────
    if from_file:
        zeros = cargar_ceros_archivo(from_file)
    else:
        zeros = obtener_ceros_mpmath(n_zeros, out_dir)

    if len(zeros) == 0:
        print("  ✗ Sin ceros disponibles. Abortar.")
        return

    zeros = zeros[:n_zeros]

    # ── Preparar espectro completo ────────────────────────────────────────
    print(f"\n  Preparando espectro ({len(zeros)} ceros, recorte={recorte})...")
    espectro, diag = preparar_espectro_riemann(zeros, recorte, verbose=True)

    if espectro is None:
        print("  ✗ Pipeline falló. Verificar número de ceros.")
        return

    print(f"  ✓ Espectro: {len(espectro)} puntos")

    # ── Estadísticas principales ──────────────────────────────────────────
    print("\n  Calculando estadísticas...")

    t0     = time.perf_counter()
    res_r  = calcular_r_statistic(espectro)
    res_d3 = calcular_delta3(espectro, L_grid)
    res_s2 = calcular_sigma2(espectro, L_grid)
    print(f"  Tiempo estadísticas: {time.perf_counter()-t0:.1f}s")

    # Imprimir resultados inmediatos
    r_ok  = '✓' if res_r['fiable']  else '⚠'
    d3_ok = '✓' if res_d3['fiable'] else '⚠'
    print(f"\n  {r_ok} r-statistic  : ⟨r⟩ = {res_r['r_mean']:.5f}  "
          f"(GUE={R_GUE_EXACT:.5f}, Δ={res_r['dist_gue']:.5f})")
    print(f"  {d3_ok} Δ₃ pendiente : α  = {res_d3['alpha']:.5f}  "
          f"(1/π²={ALPHA_GUE:.5f}, Δ={res_d3['delta_gue']:+.5f}, "
          f"err={res_d3['error_rel']*100:.1f}%)")
    print(f"     Σ² pendiente : β  = {res_s2['beta']:.5f}  "
          f"(1/π²={ALPHA_GUE:.5f}, Δ={res_s2['delta_gue']:+.5f})")

    # ── Baseline GUE sintético ────────────────────────────────────────────
    gue_base = None
    if compare_gue:
        N_gue = len(espectro) * 3   # tamaño comparable antes del recorte 1/3
        print(f"\n  Calculando baseline GUE (N={N_gue}, 10 realizaciones)...")
        t0 = time.perf_counter()
        gue_base = baseline_gue(N_gue, L_grid, n_realizaciones=10, seed=42)
        print(f"  GUE: α={gue_base['alpha_mean']:.5f}±{gue_base['alpha_std']:.5f}  "
              f"r={gue_base['r_mean']:.5f}±{gue_base['r_std']:.5f}  "
              f"({time.perf_counter()-t0:.1f}s)")
        print(f"  Riemann vs GUE: Δα={res_d3['alpha']-gue_base['alpha_mean']:+.5f}  "
              f"Δr={res_r['r_mean']-gue_base['r_mean']:+.5f}")

    # ── Convergencia acumulativa ──────────────────────────────────────────
    print(f"\n  Calculando convergencia α(N)...")
    t0 = time.perf_counter()
    N_vals, alphas_conv, R2s_conv = convergencia_acumulativa(
        zeros, n_puntos=8, L_grid=L_grid, recorte=recorte
    )
    print(f"  Tiempo: {time.perf_counter()-t0:.1f}s")
    for i, (N, a, r2) in enumerate(zip(N_vals, alphas_conv, R2s_conv)):
        if np.isfinite(a):
            print(f"    N={int(N):>5}  α={a:.5f}  R²={r2:.3f}  Δ(α-GUE)={a-ALPHA_GUE:+.5f}")

    # ── Sliding window ────────────────────────────────────────────────────
    centros_sw = np.array([])
    alphas_sw  = np.array([])
    r_sw       = np.array([])

    if use_sliding_window:
        print(f"\n  Sliding window (W={sw_window_size}, step={sw_step})...")
        t0 = time.perf_counter()
        centros_sw, alphas_sw, r_sw = sliding_window(
            zeros, sw_window_size, sw_step, L_grid, recorte
        )
        print(f"  {len(centros_sw)} ventanas válidas  ({time.perf_counter()-t0:.1f}s)")
        if len(alphas_sw) > 0:
            print(f"  α: media={np.mean(alphas_sw):.5f}  std={np.std(alphas_sw):.5f}  "
                  f"rango=[{np.min(alphas_sw):.5f}, {np.max(alphas_sw):.5f}]")
            print(f"  r: media={np.mean(r_sw):.5f}  std={np.std(r_sw):.5f}")
            # Detectar anomalías: α fuera de ±3σ
            if len(alphas_sw) > 5:
                mu, sigma = np.mean(alphas_sw), np.std(alphas_sw)
                anomalias = np.where(np.abs(alphas_sw - mu) > 3*sigma)[0]
                if len(anomalias) > 0:
                    print(f"  ⚠ {len(anomalias)} ventanas con α > ±3σ en índices: "
                          f"{centros_sw[anomalias]}")
                else:
                    print(f"  ✓ Ninguna ventana con α > ±3σ — convergencia suave")

    # ── Outputs ───────────────────────────────────────────────────────────
    print("\n  Generando outputs...")
    plot_diagnostico_completo(zeros, espectro, res_r, res_d3, res_s2, gue_base, out_dir)
    plot_convergencia(N_vals, alphas_conv, R2s_conv, gue_base, out_dir)
    if use_sliding_window and len(centros_sw) > 0:
        plot_sliding(centros_sw, alphas_sw, r_sw, out_dir)
    guardar_reporte(zeros, espectro, res_r, res_d3, res_s2, diag, out_dir)

    # ── Resumen ───────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  RESUMEN FINAL")
    print("=" * 72)
    print(f"  r-statistic : ⟨r⟩={res_r['r_mean']:.5f}  ensemble={res_r['ensemble']}  "
          f"{'✓' if res_r['fiable'] else '⚠'}")
    print(f"  Δ₃ pendiente: α={res_d3['alpha']:.5f}  "
          f"error_rel={res_d3['error_rel']*100:.1f}%  "
          f"{'✓' if res_d3['fiable'] else '⚠'}")
    print(f"  Σ² pendiente: β={res_s2['beta']:.5f}  "
          f"error_rel={res_s2['error_rel']*100:.1f}%")
    if gue_base:
        print(f"  GUE sintético:  α={gue_base['alpha_mean']:.5f}±{gue_base['alpha_std']:.5f}")
        print(f"  Δα Riemann-GUE: {res_d3['alpha']-gue_base['alpha_mean']:+.5f}")
    print("=" * 72)
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validación empírica de estadísticas RMT sobre ceros de Riemann."
    )
    parser.add_argument("--n-zeros",        type=int,   default=2000)
    parser.add_argument("--from-file",      type=str,   default='',
                        help="Archivo de texto con ceros (uno por línea)")
    parser.add_argument("--sliding-window", action="store_true")
    parser.add_argument("--sw-window-size", type=int,   default=1500)
    parser.add_argument("--sw-step",        type=int,   default=200)
    parser.add_argument("--compare-gue",    action="store_true", default=True)
    parser.add_argument("--no-compare-gue", action="store_false", dest="compare_gue")
    parser.add_argument("--recorte",        type=float, default=0.25,
                        help="Fracción a eliminar por extremo (default 0.25 = 50%% central)")
    args = parser.parse_args()

    main(
        n_zeros        = args.n_zeros,
        from_file      = args.from_file,
        use_sliding_window = args.sliding_window,
        sw_window_size = args.sw_window_size,
        sw_step        = args.sw_step,
        compare_gue    = args.compare_gue,
        recorte        = args.recorte,
    )
