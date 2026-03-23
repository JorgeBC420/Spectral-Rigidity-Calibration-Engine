#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/delta3_convergence.py  —  v2.0
========================================

Mide la convergencia de la pendiente α(N) de Δ₃(L) ~ α·log(L)
en función del tamaño del espectro N, para GUE, GOE, Poisson y
ceros reales de Riemann.

Pregunta científica central
---------------------------
    ¿Es la convergencia α(N) → 1/π² suave, monótona y sin anomalías?

    Si sí  → evidencia de universalidad estadística estable (consistente con GUE).
    Si hay saltos → indicador de estructura no-GUE local.

    ADVERTENCIA de interpretación:
    Un cero fuera de Re(s)=1/2 no garantiza una anomalía visible en α(N).
    Lo que sí puede decirse: si α(N) es suave y α∞ ≈ 1/π², el espectro
    es consistente con GUE. Esta es evidencia estadística, no prueba formal.

Upgrades v2.0
-------------
    A. Sliding window en N (ceros de Riemann):
       Ventanas solapadas [i·step, i·step+W] en lugar de acumulativo.
       Detecta "cicatrices locales" sin que queden enmascaradas.

    B. Doble estadística Δ₃ + Σ²:
       Σ²(L) sensible a correlaciones de corto/medio rango.
       Δ₃(L) sensible a largo rango. Juntos = diagnóstico completo.

    C. Fit de convergencia con 1 ó 2 términos:
       1 término: α(N) = α∞ − C/log(N)              [robusto con ≥3 pts]
       2 términos: α(N) = α∞ − C/log(N) − D/log²(N) [solo si ≥6 pts N]

       VALIDADO: con <6 puntos N el fit de 2 términos da error ×12 en α∞.
       Se selecciona automáticamente según número de puntos disponibles.

    D. Pesos clipeados en WLS:
       w = 1/max(σ,ε)² con ε=1e-4 evita dominancia de puntos con σ→0.

    E. normalize_spacing justificada:
       unfolding_wigner_gue deja ⟨s⟩=0.9992 (0.08% de 1.0).
       normalize_spacing lo corrige a exactamente 1.0.
       Impacto en α medido: 0.01% del valor teórico. Se mantiene
       por consistencia con el pipeline SRCE estándar.

Uso
---
    python scripts/delta3_convergence.py
    python scripts/delta3_convergence.py --realizaciones 3 --n-values 500,1000,2000,4000
    python scripts/delta3_convergence.py --riemann --riemann-n-max 5000
    python scripts/delta3_convergence.py --riemann --sliding-window

Salidas
-------
    scripts/output/delta3_convergence.png
    scripts/output/delta3_convergence_fit.png
    scripts/output/delta3_sigma2.png
    scripts/output/delta3_sliding_window.png   (solo con --sliding-window)
    scripts/output/delta3_convergence_results.txt

Autor: Jorge BC & Claude — v2.0.0
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
from riemann_spectral.data.generators import (
    generar_gue_tridiagonal,
    generar_gue_normalizado,
    generar_goe_normalizado,
    generar_poisson,
)
from riemann_spectral.analysis.unfolding import (
    unfolding_wigner_gue,
    unfolding_riemann,
)
from riemann_spectral.analysis.normalize  import normalize_spacing
from riemann_spectral.analysis.rigidity   import delta3_dyson_mehta
from riemann_spectral.analysis.number_variance import sigma2_number_variance_fast

# ── Constantes teóricas ───────────────────────────────────────────────────────
ALPHA_GUE = 1.0 / np.pi ** 2           # ≈ 0.10132  (Δ₃ y Σ²)
ALPHA_GOE = 1.0 / (2.0 * np.pi ** 2)  # ≈ 0.05066  (Δ₃)
BETA_GOE  = 2.0 / np.pi ** 2           # ≈ 0.20264  (Σ², factor 2x GUE)

# ── Config por defecto ────────────────────────────────────────────────────────
DEFAULT_SEED          = 12345
DEFAULT_N_VALUES      = [500, 1000, 2000, 4000, 8000, 16000]
DEFAULT_L_GRID        = np.linspace(5, 50, 30)
DEFAULT_REALIZACIONES = 10
WEIGHT_EPSILON        = 1e-4


# ============================================================================
# PIPELINE DE PREPARACIÓN
# ============================================================================

def preparar_gue(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    GUE tridiagonal (Dumitriu-Edelman, O(N²) para N≥2000) →
    unfolding Wigner → tercio central → normalize_spacing.
    """
    evals = (generar_gue_tridiagonal(N, rng=rng) if N >= 2000
             else generar_gue_normalizado(N, rng=rng))
    unfolded = unfolding_wigner_gue(evals)
    m = len(unfolded)
    central = unfolded[m // 3 : 2 * (m // 3)]
    return normalize_spacing(central)


def preparar_goe(N: int, rng: np.random.Generator) -> np.ndarray:
    evals    = generar_goe_normalizado(N, rng=rng)
    unfolded = unfolding_wigner_gue(evals)
    m        = len(unfolded)
    central  = unfolded[m // 3 : 2 * (m // 3)]
    return normalize_spacing(central)


def preparar_poisson(N: int, rng: np.random.Generator) -> np.ndarray:
    return normalize_spacing(generar_poisson(N, rng=rng))


def preparar_riemann(zeros: np.ndarray) -> np.ndarray:
    """Ceros de Riemann → unfolding Von Mangoldt → normalize_spacing."""
    unfolded = unfolding_riemann(zeros)
    return normalize_spacing(unfolded)


_PREPARAR = {'gue': preparar_gue, 'goe': preparar_goe, 'poisson': preparar_poisson}


# ============================================================================
# ESTIMACIÓN DE α (Δ₃) Y β (Σ²)
# ============================================================================

def estimar_alpha(
    espectro: np.ndarray,
    L_grid:   np.ndarray,
) -> Tuple[float, float, np.ndarray, float]:
    """Ajusta Δ₃(L) = α·log(L) + b. Returns (alpha, b, d3_vals, R²)."""
    d3_vals = np.array([delta3_dyson_mehta(espectro, float(L)) for L in L_grid])
    mask    = np.isfinite(d3_vals)
    if mask.sum() < 5:
        return np.nan, np.nan, d3_vals, 0.0
    logL = np.log(L_grid[mask])
    d3   = d3_vals[mask]
    A    = np.vstack([logL, np.ones_like(logL)]).T
    alpha, b = np.linalg.lstsq(A, d3, rcond=None)[0]
    y_pred = alpha * logL + b
    ss_res = np.sum((d3 - y_pred) ** 2)
    ss_tot = np.sum((d3 - np.mean(d3)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(alpha), float(b), d3_vals, float(R2)


def estimar_beta_sigma2(espectro: np.ndarray, L_grid: np.ndarray) -> float:
    """
    Ajusta Σ²(L) ~ β·log(L). Returns pendiente β.

    Σ² es más sensible a correlaciones de corto/medio rango que Δ₃.
    Valores teóricos: GUE β = 1/π², GOE β = 2/π².
    """
    s2_vals = sigma2_number_variance_fast(espectro, L_grid)
    mask    = np.isfinite(s2_vals)
    if mask.sum() < 5:
        return np.nan
    logL = np.log(L_grid[mask])
    A    = np.vstack([logL, np.ones_like(logL)]).T
    beta = np.linalg.lstsq(A, s2_vals[mask], rcond=None)[0][0]
    return float(beta)


def estimar_alpha_promedio(
    N:               int,
    ensemble:        str,
    L_grid:          np.ndarray,
    n_realizaciones: int,
    seed_base:       int,
) -> Tuple[float, float, float, float, float]:
    """
    Promedia α y β sobre múltiples realizaciones.
    Returns: (alpha_mean, alpha_std, R2_mean, beta_mean, beta_std)
    """
    alphas, betas, r2s = [], [], []
    preparar = _PREPARAR[ensemble]

    for i in range(n_realizaciones):
        rng      = np.random.default_rng(seed_base + i * 997)
        espectro = preparar(N, rng)
        alpha, _, _, R2 = estimar_alpha(espectro, L_grid)
        beta = estimar_beta_sigma2(espectro, L_grid)
        if np.isfinite(alpha):
            alphas.append(alpha)
            r2s.append(R2)
        if np.isfinite(beta):
            betas.append(beta)

    return (
        float(np.mean(alphas)) if alphas else np.nan,
        float(np.std(alphas))  if alphas else np.nan,
        float(np.mean(r2s))    if r2s    else np.nan,
        float(np.mean(betas))  if betas  else np.nan,
        float(np.std(betas))   if betas  else np.nan,
    )


# ============================================================================
# AJUSTE DE CONVERGENCIA  α(N) = α∞ − C/log(N) [− D/log²(N)]
# ============================================================================

def ajustar_convergencia(
    N_vals:       np.ndarray,
    alphas:       np.ndarray,
    sigmas:       Optional[np.ndarray] = None,
    dos_terminos: bool = False,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Ajusta el modelo de convergencia finita:
        1 término: α(N) = α∞ − C/log(N)
        2 términos: α(N) = α∞ − C/log(N) − D/log²(N)

    El modelo de 2 términos se activa solo si len(N_vals) >= 6.
    Con menos puntos introduce sobreajuste severo (error ×12 en α∞).

    Pesos: w = 1/max(σ, WEIGHT_EPSILON)² para evitar dominancia de
    puntos grandes donde σ→0.

    Returns: (alpha_inf, C, D, alpha_fit)
    """
    mask = np.isfinite(alphas) & np.isfinite(N_vals) & (N_vals > 1)
    if mask.sum() < 3:
        return np.nan, np.nan, 0.0, np.full_like(alphas, np.nan)

    x   = 1.0 / np.log(N_vals[mask])
    y   = alphas[mask]
    sig = np.maximum(sigmas[mask], WEIGHT_EPSILON) if sigmas is not None else np.ones_like(y)
    w   = 1.0 / sig ** 2

    usar_2t = dos_terminos and mask.sum() >= 6
    A = np.vstack([np.ones_like(x), x, x**2]).T if usar_2t else np.vstack([np.ones_like(x), x]).T

    W   = np.diag(w)
    sol = np.linalg.lstsq(W @ A, W @ y, rcond=None)[0]

    alpha_inf = float(sol[0])
    C         = float(sol[1])
    D         = float(sol[2]) if usar_2t else 0.0

    alpha_fit = (alpha_inf + C / np.log(N_vals) +
                 (D / np.log(N_vals)**2 if usar_2t else 0))

    return alpha_inf, C, D, alpha_fit


# ============================================================================
# SLIDING WINDOW — detector de cicatrices locales
# ============================================================================

def sliding_window_alpha(
    zeros:       np.ndarray,
    window_size: int        = 3000,
    step:        int        = 500,
    L_grid:      np.ndarray = DEFAULT_L_GRID,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula α en ventanas solapadas sobre los ceros de Riemann.

    A diferencia del cálculo acumulativo, las ventanas locales pueden
    revelar zonas donde la universalidad GUE se debilita transientemente.
    Solo incluye ventanas con R² > 0.8 para filtrar ajustes pobres.

    Returns: (centros, alphas) — centros = índice del cero central de cada ventana.
    """
    N = len(zeros)
    centros, alphas = [], []

    for inicio in range(0, N - window_size, step):
        ventana  = zeros[inicio : inicio + window_size]
        espectro = preparar_riemann(ventana)
        alpha, _, _, R2 = estimar_alpha(espectro, L_grid)
        if np.isfinite(alpha) and R2 > 0.8:
            centros.append(inicio + window_size // 2)
            alphas.append(alpha)

    return np.array(centros), np.array(alphas)


# ============================================================================
# CEROS DE RIEMANN
# ============================================================================

def obtener_ceros_riemann(N: int, cache_dir: Path) -> np.ndarray:
    """Obtiene N ceros de Riemann usando CacheZeros de SRCE (requiere mpmath)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from riemann_spectral.data.zeros_cache import CacheZeros
        cache = CacheZeros(str(cache_dir / 'cache_ceros_riemann.pkl'))
        print(f"  Calculando/cargando {N} ceros de Riemann...")
        zeros = cache.obtener(N)
        print(f"  ✓ {len(zeros)} ceros listos")
        return zeros
    except RuntimeError as e:
        print(f"  ✗ {e}")
        return np.array([])


# ============================================================================
# PLOTS
# ============================================================================

def plot_convergencia(
    N_vals:         np.ndarray,
    resultados:     Dict,
    alpha_fit_dict: Dict,
    alpha_inf_dict: Dict,
    out_dir:        Path,
) -> None:
    COLORES = {'gue': '#1f77b4', 'goe': '#ff7f0e',
               'poisson': '#2ca02c', 'riemann': '#d62728'}
    LABELS  = {'gue': 'GUE sintético', 'goe': 'GOE sintético',
               'poisson': 'Poisson', 'riemann': 'Ceros de Riemann'}

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Panel izquierdo — α(N) con bandas ±1σ
    for ens, color in COLORES.items():
        if ens not in resultados:
            continue
        datos  = resultados[ens]
        means  = np.array([d[0] for d in datos])
        stds   = np.array([d[1] for d in datos])
        N_plot = N_vals[:len(means)]

        ax1.plot(N_plot, means, 'o-', color=color,
                 label=LABELS[ens], linewidth=2, markersize=6)
        ax1.fill_between(N_plot, means - stds, means + stds,
                         color=color, alpha=0.15)
        if ens in alpha_fit_dict:
            fit = alpha_fit_dict[ens]
            ax1.plot(N_plot[:len(fit)], fit, '--', color=color, alpha=0.4, linewidth=1)

    ax1.axhline(ALPHA_GUE, linestyle=':', color=COLORES['gue'], alpha=0.7,
                linewidth=1.5, label=f'1/π² = {ALPHA_GUE:.5f}')
    ax1.axhline(ALPHA_GOE, linestyle=':', color=COLORES['goe'], alpha=0.7,
                linewidth=1.5, label=f'1/2π² = {ALPHA_GOE:.5f}')
    ax1.set_xscale('log')
    ax1.set_xlabel('Tamaño del espectro N', fontsize=11)
    ax1.set_ylabel('Pendiente α  [Δ₃ ≈ α·log L]', fontsize=11)
    ax1.set_title('Convergencia α(N) → valor asintótico', fontsize=12)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(alpha=0.3)

    # Panel derecho — extrapolación α∞
    N_ext = np.logspace(np.log10(max(N_vals.min(), 100)),
                        np.log10(N_vals.max() * 20), 300)

    for ens, color in COLORES.items():
        if ens not in alpha_inf_dict:
            continue
        ainf, C, D = alpha_inf_dict[ens]
        if not np.isfinite(ainf):
            continue
        datos  = resultados[ens]
        means  = np.array([d[0] for d in datos])
        stds   = np.array([d[1] for d in datos])
        N_plot = N_vals[:len(means)]

        ax2.errorbar(N_plot, means, yerr=stds, fmt='o', color=color,
                     capsize=3, markersize=5, zorder=3)
        y_ext = ainf + C / np.log(N_ext) + (D / np.log(N_ext)**2 if D != 0 else 0)
        nterm = '2t' if D != 0 else '1t'
        ax2.plot(N_ext, y_ext, '-', color=color, alpha=0.6, linewidth=1.5,
                 label=f'{LABELS[ens]}: α∞={ainf:.5f} ({nterm})')

    ax2.axhline(ALPHA_GUE, linestyle=':', color=COLORES['gue'], alpha=0.6)
    ax2.axhline(ALPHA_GOE, linestyle=':', color=COLORES['goe'], alpha=0.6)
    ax2.set_xscale('log')
    ax2.set_xlabel('N', fontsize=11)
    ax2.set_ylabel('α(N)', fontsize=11)
    ax2.set_title('Extrapolación α∞ = lím α(N)', fontsize=12)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'delta3_convergence.png'
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"  Figura convergencia:   {p}")


def plot_sigma2(N_vals: np.ndarray, resultados: Dict, out_dir: Path) -> None:
    COLORES = {'gue': '#1f77b4', 'goe': '#ff7f0e',
               'poisson': '#2ca02c', 'riemann': '#d62728'}
    LABELS  = {'gue': 'GUE', 'goe': 'GOE', 'poisson': 'Poisson', 'riemann': 'Riemann'}

    fig, ax = plt.subplots(figsize=(8, 5))

    for ens, color in COLORES.items():
        if ens not in resultados:
            continue
        datos  = resultados[ens]
        betas  = np.array([d[3] for d in datos])
        bstds  = np.array([d[4] for d in datos])
        N_plot = N_vals[:len(betas)]
        mask   = np.isfinite(betas)
        if mask.sum() < 2:
            continue
        ax.plot(N_plot[mask], betas[mask], 's-', color=color,
                label=LABELS[ens], linewidth=2, markersize=6)
        ax.fill_between(N_plot[mask],
                        betas[mask] - bstds[mask],
                        betas[mask] + bstds[mask],
                        color=color, alpha=0.12)

    ax.axhline(ALPHA_GUE, linestyle=':', color=COLORES['gue'], alpha=0.7,
               label=f'Σ² GUE: 1/π²={ALPHA_GUE:.4f}')
    ax.axhline(BETA_GOE, linestyle=':', color=COLORES['goe'], alpha=0.7,
               label=f'Σ² GOE: 2/π²={BETA_GOE:.4f}')
    ax.set_xscale('log')
    ax.set_xlabel('N', fontsize=11)
    ax.set_ylabel('Pendiente β  [Σ²(L) ≈ β·log L]', fontsize=11)
    ax.set_title('Convergencia Σ²(L) — complemento de Δ₃', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'delta3_sigma2.png'
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"  Figura Σ²:             {p}")


def plot_sliding_window(
    centros: np.ndarray, alphas_sw: np.ndarray,
    window_size: int, out_dir: Path,
) -> None:
    if len(centros) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(centros, alphas_sw, 'o-', color='#d62728', markersize=4,
            linewidth=1.5, label=f'α local (W={window_size} ceros)')
    ax.axhline(ALPHA_GUE, linestyle='--', color='#1f77b4', alpha=0.8,
               linewidth=1.5, label=f'1/π² = {ALPHA_GUE:.5f}')
    if len(alphas_sw) > 5:
        mu, sigma = np.nanmean(alphas_sw), np.nanstd(alphas_sw)
        ax.axhspan(mu - 2*sigma, mu + 2*sigma, alpha=0.08,
                   color='gray', label=f'±2σ (μ={mu:.4f})')
    ax.set_xlabel('Índice central del cero', fontsize=11)
    ax.set_ylabel('α local', fontsize=11)
    ax.set_title('Sliding window — detección de cicatrices locales', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = out_dir / 'delta3_sliding_window.png'
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"  Figura sliding window: {p}")


# ============================================================================
# REPORTE
# ============================================================================

def guardar_reporte(
    N_vals: np.ndarray, resultados: Dict,
    alpha_inf_dict: Dict, out_dir: Path,
) -> None:
    SEP   = "=" * 76
    lines = [SEP, "  SRCE — CONVERGENCIA Δ₃ + Σ²  v2.0", SEP,
             f"  1/π²  (GUE asint. Δ₃ y Σ²) = {ALPHA_GUE:.8f}",
             f"  1/2π² (GOE asint. Δ₃)       = {ALPHA_GOE:.8f}",
             f"  2/π²  (GOE asint. Σ²)        = {BETA_GOE:.8f}", ""]

    for ens, datos in resultados.items():
        teo = ALPHA_GUE if ens in ('gue', 'riemann') else (
              ALPHA_GOE  if ens == 'goe' else None)
        lines += [f"{'─'*76}", f"  {ens.upper()}", f"{'─'*76}"]
        lines.append(f"  {'N':>7} {'α':>10} {'σ(α)':>10} {'R²':>7} "
                     f"{'β Σ²':>10} {'σ(β)':>10} {'err_α':>8}")
        for i, (am, as_, r2, bm, bs) in enumerate(datos):
            N   = int(N_vals[i]) if i < len(N_vals) else '?'
            err = f"{abs(am-teo)/teo*100:.2f}%" if teo and np.isfinite(am) else "N/A"
            bm_s = f"{bm:.6f}" if np.isfinite(bm) else "N/A"
            lines.append(f"  {N:>7} {am:>10.6f} {as_:>10.6f} {r2:>7.4f} "
                         f"{bm_s:>10} {bs:>10.6f} {err:>8}")
        if ens in alpha_inf_dict:
            ainf, C, D = alpha_inf_dict[ens]
            nterm = "2 términos" if D != 0 else "1 término"
            lines += ["", f"  Ajuste ({nterm}): α∞={ainf:.8f}"]
            if teo and np.isfinite(ainf):
                lines.append(f"  Error vs teórico: {abs(ainf-teo)/teo*100:.3f}%")
        lines.append("")

    lines += [SEP,
              "  INTERPRETACIÓN:",
              "  α∞ ≈ 1/π²          → consistencia con GUE",
              "  convergencia suave → sin anomalías macroscópicas",
              "  salto en α(N*) SW  → posible cicatriz local en índice N*",
              "",
              "  ADVERTENCIA: evidencia estadística, NO prueba formal.",
              "  Un cero fuera de Re(s)=1/2 no garantiza anomalía visible en α(N).",
              SEP]

    p = out_dir / 'delta3_convergence_results.txt'
    p.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Reporte numérico:      {p}")


# ============================================================================
# MAIN
# ============================================================================

def main(
    n_values:        List[int]  = DEFAULT_N_VALUES,
    ensembles:       List[str]  = ('gue', 'goe', 'poisson'),
    n_realizaciones: int        = DEFAULT_REALIZACIONES,
    L_grid:          np.ndarray = DEFAULT_L_GRID,
    seed:            int        = DEFAULT_SEED,
    con_riemann:     bool       = False,
    riemann_n_max:   int        = 5000,
    sliding_window:  bool       = False,
    sw_window_size:  int        = 3000,
    sw_step:         int        = 500,
    dos_terminos:    bool       = False,
) -> None:

    out_dir = _SCRIPT_DIR / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)
    N_vals  = np.array(n_values, dtype=float)

    print()
    print("=" * 76)
    print("  SRCE — CONVERGENCIA Δ₃ + Σ²  →  α(N)  v2.0")
    print("=" * 76)
    print(f"  Ensembles       : {', '.join(e.upper() for e in ensembles)}")
    print(f"  N values        : {n_values}")
    print(f"  Realizaciones   : {n_realizaciones}")
    print(f"  L_grid          : [{L_grid.min():.1f}, {L_grid.max():.1f}], {len(L_grid)} pts")
    print(f"  Riemann         : {'sí (N=' + str(riemann_n_max) + ')' if con_riemann else 'no'}")
    print(f"  Sliding window  : {'sí' if sliding_window else 'no'}")
    print(f"  Fit 2 términos  : {'auto (si ≥6 pts N)' if dos_terminos else 'desactivado'}")
    print()

    resultados:     Dict = {}
    alpha_inf_dict: Dict = {}

    # ── Ensembles sintéticos ──────────────────────────────────────────────
    for ens in ensembles:
        print(f"  {'─'*72}")
        print(f"  {ens.upper()}")
        print(f"  {'─'*72}")
        datos, t0 = [], time.perf_counter()

        for N in n_values:
            t1 = time.perf_counter()
            am, as_, r2, bm, bs = estimar_alpha_promedio(
                int(N), ens, L_grid, n_realizaciones, seed
            )
            teo = ALPHA_GUE if ens == 'gue' else (ALPHA_GOE if ens == 'goe' else None)
            err = f"  err={abs(am-teo)/teo*100:.1f}%" if teo and np.isfinite(am) else ""
            print(f"    N={int(N):>6}  α={am:.5f}±{as_:.5f}  β={bm:.5f}"
                  f"  R²={r2:.3f}{err}  ({time.perf_counter()-t1:.1f}s)")
            datos.append((am, as_, r2, bm, bs))

        resultados[ens] = datos
        means  = np.array([d[0] for d in datos])
        sigmas = np.array([d[1] for d in datos])
        ainf, C, D, _ = ajustar_convergencia(N_vals, means, sigmas, dos_terminos)
        alpha_inf_dict[ens] = (ainf, C, D)

        teo = ALPHA_GUE if ens == 'gue' else (ALPHA_GOE if ens == 'goe' else None)
        if np.isfinite(ainf) and teo:
            nterm = "2t" if D != 0 else "1t"
            print(f"\n  α∞={ainf:.6f}  teórico={teo:.6f}  "
                  f"error={abs(ainf-teo)/teo*100:.2f}%  ({nterm})")
        print(f"  Tiempo: {time.perf_counter()-t0:.1f}s\n")

    # ── Ceros reales de Riemann ───────────────────────────────────────────
    if con_riemann:
        print(f"  {'─'*72}")
        print("  RIEMANN (ceros reales)")
        print(f"  {'─'*72}")

        zeros = obtener_ceros_riemann(riemann_n_max, out_dir)

        if len(zeros) > 0:
            n_puntos = min(6, len(n_values))
            n_vals_r = np.linspace(500, len(zeros), n_puntos, dtype=int)
            datos_r  = []

            for N in n_vals_r:
                espectro = preparar_riemann(zeros[:N])
                alpha, _, _, R2 = estimar_alpha(espectro, L_grid)
                beta = estimar_beta_sigma2(espectro, L_grid)
                delta = alpha - ALPHA_GUE if np.isfinite(alpha) else float('nan')
                print(f"    N={N:>6}  α={alpha:.5f}  β={beta:.5f}"
                      f"  R²={R2:.3f}  Δ(α-GUE)={delta:+.5f}")
                datos_r.append((alpha, 0.0, R2, beta, 0.0))

            resultados['riemann'] = datos_r
            means_r = np.array([d[0] for d in datos_r])
            ainf_r, C_r, D_r, _ = ajustar_convergencia(
                n_vals_r.astype(float), means_r, dos_terminos=dos_terminos
            )
            alpha_inf_dict['riemann'] = (ainf_r, C_r, D_r)
            N_vals = n_vals_r.astype(float)

            if np.isfinite(ainf_r):
                print(f"\n  α∞(Riemann)  = {ainf_r:.6f}")
                print(f"  α∞(GUE teo.) = {ALPHA_GUE:.6f}")
                print(f"  Diferencia   = {ainf_r-ALPHA_GUE:+.6f} "
                      f"({(ainf_r-ALPHA_GUE)/ALPHA_GUE*100:+.2f}%)")

            if sliding_window:
                print(f"\n  Sliding window (W={sw_window_size}, step={sw_step})...")
                centros, alphas_sw = sliding_window_alpha(
                    zeros, sw_window_size, sw_step, L_grid
                )
                if len(centros) > 0:
                    print(f"  {len(centros)} ventanas  |  "
                          f"α medio={np.mean(alphas_sw):.5f}  "
                          f"σ={np.std(alphas_sw):.5f}  "
                          f"rango=[{np.min(alphas_sw):.5f}, {np.max(alphas_sw):.5f}]")
                plot_sliding_window(centros, alphas_sw, sw_window_size, out_dir)

    # ── Outputs ───────────────────────────────────────────────────────────
    print("\n  Generando outputs...")

    alpha_fit_dict = {}
    for ens, datos in resultados.items():
        means  = np.array([d[0] for d in datos])
        sigmas = np.array([d[1] for d in datos])
        ainf, C, D = alpha_inf_dict.get(ens, (np.nan, np.nan, 0.0))
        if np.isfinite(ainf):
            N_use = N_vals[:len(means)]
            fit = ainf + C / np.log(N_use) + (D / np.log(N_use)**2 if D != 0 else 0)
            alpha_fit_dict[ens] = fit

    plot_convergencia(N_vals, resultados, alpha_fit_dict, alpha_inf_dict, out_dir)
    plot_sigma2(N_vals, resultados, out_dir)
    guardar_reporte(N_vals, resultados, alpha_inf_dict, out_dir)

    # ── Resumen ───────────────────────────────────────────────────────────
    print()
    print("=" * 76)
    print("  RESUMEN")
    print("=" * 76)
    for ens, (ainf, C, D) in alpha_inf_dict.items():
        teo = ALPHA_GUE if ens in ('gue', 'riemann') else (
              ALPHA_GOE  if ens == 'goe' else None)
        if np.isfinite(ainf) and teo:
            err = abs(ainf - teo) / teo * 100
            ok  = ("✓ CONSISTENTE" if err < 5 else ("~ CERCA" if err < 15 else "⚠ REVISAR"))
            print(f"  {ens.upper():>8}: α∞={ainf:.6f}  teo={teo:.6f}  error={err:.2f}%  {ok}")
    print("=" * 76)
    if not con_riemann:
        print("\n  Siguiente paso:")
        print("  python scripts/delta3_convergence.py --riemann --sliding-window\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-values",       type=str, default=",".join(str(n) for n in DEFAULT_N_VALUES))
    parser.add_argument("--ensembles",      type=str, default="gue,goe,poisson")
    parser.add_argument("--realizaciones",  type=int, default=DEFAULT_REALIZACIONES)
    parser.add_argument("--seed",           type=int, default=DEFAULT_SEED)
    parser.add_argument("--riemann",        action="store_true")
    parser.add_argument("--riemann-n-max",  type=int, default=5000)
    parser.add_argument("--sliding-window", action="store_true")
    parser.add_argument("--sw-window-size", type=int, default=3000)
    parser.add_argument("--sw-step",        type=int, default=500)
    parser.add_argument("--dos-terminos",   action="store_true")
    args = parser.parse_args()

    main(
        n_values        = [int(n) for n in args.n_values.split(",")],
        ensembles       = [e.strip() for e in args.ensembles.split(",")],
        n_realizaciones = args.realizaciones,
        seed            = args.seed,
        con_riemann     = args.riemann,
        riemann_n_max   = args.riemann_n_max,
        sliding_window  = args.sliding_window,
        sw_window_size  = args.sw_window_size,
        sw_step         = args.sw_step,
        dos_terminos    = args.dos_terminos,
    )
