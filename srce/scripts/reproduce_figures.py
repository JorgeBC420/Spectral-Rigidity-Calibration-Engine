#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_figures.py
====================

Script para reproducir TODAS las figuras científicas del SRCE.

Garantiza reproducibilidad completa usando seeds fijas.

Uso (desde la carpeta ``srce/``)::
    python scripts/reproduce_figures.py --output figures/

Genera:
    - figure1_spacing_distributions.png
    - figure2_delta3_comparison.png
    - figure3_sigma2_ordering.png
    - figure4_r_parameter_validation.png
    - figure5_pair_correlation.png
    - figure6_spectral_form_factor.png

Autor: Jorge BC
Versión: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import sys

# Setup paths (repo root = padre de scripts/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from riemann_spectral.data.generators import (
    generar_poisson,
    generar_goe_normalizado,
    generar_gue_normalizado,
)
from riemann_spectral.analysis.normalize import normalize_spacing
from riemann_spectral.analysis.rigidity import delta3_dyson_mehta
from riemann_spectral.analysis.number_variance import (
    sigma2_number_variance_fast,
    sigma2_theoretical,
)
from riemann_spectral.statistics.r_statistic import (
    compute_r_parameter,
    R_POISSON_EXACT,
    R_GOE_EXACT,
    R_GUE_EXACT,
)
from riemann_spectral.analysis.pair_correlation import (
    pair_correlation,
    pair_correlation_gue,
    pair_correlation_poisson,
)
from riemann_spectral.analysis.spectral_form_factor import (
    spectral_form_factor,
    extract_ramp_slope,
)

# Configuración
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Estilo
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


# ============================================================================
# FIGURA 1: DISTRIBUCIONES DE SPACING
# ============================================================================

def figure1_spacing_distributions(output_dir: Path):
    """
    P(s) para Poisson, GOE, GUE vs teoría.
    """
    logger.info("Generando Figura 1: Distribuciones de Spacing...")
    
    # Seeds fijas
    N = 2000
    
    # Generar ensembles
    poisson = generar_poisson(N, rng=np.random.default_rng(42))
    goe = generar_goe_normalizado(N, rng=np.random.default_rng(7))
    gue = generar_gue_normalizado(N, rng=np.random.default_rng(99))
    
    # Normalizar
    poisson_norm = normalize_spacing(poisson)
    goe_norm = normalize_spacing(goe)
    gue_norm = normalize_spacing(gue)
    
    # Spacings
    s_poisson = np.diff(poisson_norm)
    s_goe = np.diff(goe_norm)
    s_gue = np.diff(gue_norm)
    
    # Teoría
    s_theory = np.linspace(0, 4, 200)
    P_poisson_theory = np.exp(-s_theory)
    P_goe_theory = (np.pi / 2) * s_theory * np.exp(-np.pi * s_theory**2 / 4)
    P_gue_theory = (32 / np.pi**2) * s_theory**2 * np.exp(-4 * s_theory**2 / np.pi)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Poisson
    axes[0].hist(s_poisson, bins=50, density=True, alpha=0.6, label='Datos', color='C0')
    axes[0].plot(s_theory, P_poisson_theory, 'r-', linewidth=2, label='Teoría')
    axes[0].set_xlabel('Spacing s')
    axes[0].set_ylabel('P(s)')
    axes[0].set_title('Poisson Ensemble')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # GOE
    axes[1].hist(s_goe, bins=50, density=True, alpha=0.6, label='Datos', color='C2')
    axes[1].plot(s_theory, P_goe_theory, 'r-', linewidth=2, label='Wigner β=1')
    axes[1].set_xlabel('Spacing s')
    axes[1].set_title('GOE Ensemble')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # GUE
    axes[2].hist(s_gue, bins=50, density=True, alpha=0.6, label='Datos', color='C1')
    axes[2].plot(s_theory, P_gue_theory, 'r-', linewidth=2, label='Wigner β=2')
    axes[2].set_xlabel('Spacing s')
    axes[2].set_title('GUE Ensemble')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_spacing_distributions.png', dpi=300, bbox_inches='tight')
    logger.info("  ✓ Guardada: figure1_spacing_distributions.png")
    plt.close()


# ============================================================================
# FIGURA 2: Δ₃ COMPARISON
# ============================================================================

def figure2_delta3_comparison(output_dir: Path):
    """
    Δ₃(L) para los tres ensembles con predicciones teóricas.
    """
    logger.info("Generando Figura 2: Comparación Δ₃...")
    
    N = 2000
    
    # Generar
    poisson = generar_poisson(N, rng=np.random.default_rng(42))
    goe = generar_goe_normalizado(N, rng=np.random.default_rng(7))
    gue = generar_gue_normalizado(N, rng=np.random.default_rng(99))
    
    # Normalizar
    poisson_norm = normalize_spacing(poisson)
    goe_norm = normalize_spacing(goe)
    gue_norm = normalize_spacing(gue)
    
    # Grid L
    L_grid = np.linspace(5, 40, 20)
    
    # Calcular Δ₃
    d3_poisson = np.array([delta3_dyson_mehta(poisson_norm, L) for L in L_grid])
    d3_goe = np.array([delta3_dyson_mehta(goe_norm, L) for L in L_grid])
    d3_gue = np.array([delta3_dyson_mehta(gue_norm, L) for L in L_grid])
    
    # Teoría
    d3_poisson_theory = L_grid / 15
    d3_goe_theory = (1 / (2 * np.pi**2)) * np.log(L_grid)
    d3_gue_theory = (1 / np.pi**2) * np.log(L_grid)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(L_grid, d3_poisson, 'o-', label='Poisson (datos)', markersize=5, color='C0')
    plt.plot(L_grid, d3_poisson_theory, '--', label='Poisson (L/15)', linewidth=2, color='C0')
    
    plt.plot(L_grid, d3_goe, 's-', label='GOE (datos)', markersize=5, color='C2')
    plt.plot(L_grid, d3_goe_theory, '--', label='GOE teórico', linewidth=2, color='C2')
    
    plt.plot(L_grid, d3_gue, '^-', label='GUE (datos)', markersize=5, color='C1')
    plt.plot(L_grid, d3_gue_theory, '--', label='GUE teórico', linewidth=2, color='C1')
    
    plt.xlabel('Longitud de ventana L', fontsize=12)
    plt.ylabel('Δ₃(L)', fontsize=12)
    plt.title('Rigidez Espectral de Dyson-Mehta', fontsize=14, fontweight='bold')
    plt.legend(ncol=2)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_delta3_comparison.png', dpi=300, bbox_inches='tight')
    logger.info("  ✓ Guardada: figure2_delta3_comparison.png")
    plt.close()


# ============================================================================
# FIGURA 3: Σ² ORDERING
# ============================================================================

def figure3_sigma2_ordering(output_dir: Path):
    """
    Number variance mostrando orden universal Poisson > GOE > GUE.
    """
    logger.info("Generando Figura 3: Orden Universal Σ²...")
    
    N = 5000
    L_grid = np.linspace(2, 40, 30)
    
    # Generar
    poisson = generar_poisson(N, rng=np.random.default_rng(42))
    goe = generar_goe_normalizado(N, rng=np.random.default_rng(7))
    gue = generar_gue_normalizado(N, rng=np.random.default_rng(99))
    
    # Normalizar
    poisson_norm = normalize_spacing(poisson)
    goe_norm = normalize_spacing(goe)
    gue_norm = normalize_spacing(gue)
    
    # Calcular Σ²
    sigma2_poisson = sigma2_number_variance_fast(poisson_norm, L_grid)
    sigma2_goe = sigma2_number_variance_fast(goe_norm, L_grid)
    sigma2_gue = sigma2_number_variance_fast(gue_norm, L_grid)
    
    # Teoría
    sigma2_poisson_theory = sigma2_theoretical(L_grid, "Poisson")
    sigma2_goe_theory = sigma2_theoretical(L_grid, "GOE", use_full_formula=True)
    sigma2_gue_theory = sigma2_theoretical(L_grid, "GUE", use_full_formula=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(L_grid, sigma2_poisson, 'o-', label='Poisson (datos)', markersize=4, alpha=0.7, color='C0')
    plt.plot(L_grid, sigma2_poisson_theory, '--', label='Poisson (L)', linewidth=2, color='C0')
    
    plt.plot(L_grid, sigma2_goe, 's-', label='GOE (datos)', markersize=4, alpha=0.7, color='C2')
    plt.plot(L_grid, sigma2_goe_theory, '--', label='GOE teórico', linewidth=2, color='C2')
    
    plt.plot(L_grid, sigma2_gue, '^-', label='GUE (datos)', markersize=4, alpha=0.7, color='C1')
    plt.plot(L_grid, sigma2_gue_theory, '--', label='GUE teórico', linewidth=2, color='C1')
    
    plt.xlabel('Longitud L', fontsize=12)
    plt.ylabel('Σ²(L)', fontsize=12)
    plt.title('Number Variance - Orden Universal', fontsize=14, fontweight='bold')
    plt.legend(ncol=2)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_sigma2_ordering.png', dpi=300, bbox_inches='tight')
    logger.info("  ✓ Guardada: figure3_sigma2_ordering.png")
    plt.close()


# ============================================================================
# FIGURA 4: r-PARAMETER VALIDATION
# ============================================================================

def figure4_r_parameter_validation(output_dir: Path):
    """
    Validación de r-parameter contra valores exactos.
    """
    logger.info("Generando Figura 4: Validación r-parameter...")
    
    N = 5000
    
    # Generar ensembles
    poisson = generar_poisson(N, rng=np.random.default_rng(42))
    goe = generar_goe_normalizado(N, rng=np.random.default_rng(7))
    gue = generar_gue_normalizado(N, rng=np.random.default_rng(99))
    
    # Calcular ⟨r⟩
    r_poisson = compute_r_parameter(poisson)
    r_goe = compute_r_parameter(goe)
    r_gue = compute_r_parameter(gue)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ensembles = ['Poisson', 'GOE', 'GUE']
    r_observed = [r_poisson, r_goe, r_gue]
    r_theoretical = [R_POISSON_EXACT, R_GOE_EXACT, R_GUE_EXACT]
    
    x = np.arange(len(ensembles))
    width = 0.35
    
    ax.bar(x - width/2, r_observed, width, label='Observado', alpha=0.8)
    ax.bar(x + width/2, r_theoretical, width, label='Teórico', alpha=0.8)
    
    ax.set_ylabel('⟨r⟩', fontsize=12)
    ax.set_title('Validación r-parameter (N=5000)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ensembles)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Mostrar errores
    for i, (obs, theo) in enumerate(zip(r_observed, r_theoretical)):
        error = abs(obs - theo) / theo * 100
        ax.text(i, max(obs, theo) + 0.02, f'{error:.1f}%', 
                ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_r_parameter_validation.png', dpi=300, bbox_inches='tight')
    logger.info("  ✓ Guardada: figure4_r_parameter_validation.png")
    plt.close()


# ============================================================================
# FIGURA 5: PAIR CORRELATION
# ============================================================================

def figure5_pair_correlation(output_dir: Path):
    """
    g(r) Montgomery-Odlyzko para GUE vs Poisson.
    """
    logger.info("Generando Figura 5: Pair Correlation...")
    
    N = 2000
    
    # Generar
    poisson = generar_poisson(N, rng=np.random.default_rng(42))
    gue = generar_gue_normalizado(N, rng=np.random.default_rng(99))
    
    # Normalizar
    poisson_norm = normalize_spacing(poisson)
    gue_norm = normalize_spacing(gue)
    
    # Calcular g(r)
    r_gue, g_gue = pair_correlation(gue_norm, s_max=5.0, bins=100)
    r_poisson, g_poisson = pair_correlation(poisson_norm, s_max=5.0, bins=100)
    
    # Teoría
    g_gue_theory = pair_correlation_gue(r_gue)
    g_poisson_theory = pair_correlation_poisson(r_poisson)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(r_gue, g_gue, 'o-', label='GUE (datos)', markersize=3, alpha=0.7, color='C1')
    plt.plot(r_gue, g_gue_theory, '--', label='GUE teórico (Montgomery-Odlyzko)', 
             linewidth=2, color='C1')
    
    plt.plot(r_poisson, g_poisson, 's-', label='Poisson (datos)', markersize=3, alpha=0.7, color='C0')
    plt.plot(r_poisson, g_poisson_theory, '--', label='Poisson teórico', 
             linewidth=2, color='C0')
    
    plt.xlabel('r', fontsize=12)
    plt.ylabel('g(r)', fontsize=12)
    plt.title('Pair Correlation Function', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_pair_correlation.png', dpi=300, bbox_inches='tight')
    logger.info("  ✓ Guardada: figure5_pair_correlation.png")
    plt.close()


# ============================================================================
# FIGURA 6: SPECTRAL FORM FACTOR
# ============================================================================

def figure6_spectral_form_factor(output_dir: Path):
    """
    K(τ) mostrando dip-ramp-plateau para GUE vs Poisson.
    """
    logger.info("Generando Figura 6: Spectral Form Factor...")
    
    N = 2000
    
    # Generar
    poisson = generar_poisson(N, rng=np.random.default_rng(43))
    gue = generar_gue_normalizado(N, rng=np.random.default_rng(99))
    
    # Normalizar
    poisson_norm = normalize_spacing(poisson)
    gue_norm = normalize_spacing(gue)
    
    # Calcular K(τ)
    tau, K_gue = spectral_form_factor(gue_norm, tau_max=50, n_points=100)
    _, K_poisson = spectral_form_factor(poisson_norm, tau_max=50, n_points=100)
    
    # Pendientes
    slope_gue = extract_ramp_slope(tau, K_gue, N)
    slope_poisson = extract_ramp_slope(tau, K_poisson, N)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    plt.loglog(tau, K_gue, 'o-', label=f'GUE (slope={slope_gue:.2f})', 
               markersize=4, alpha=0.7, color='C1')
    plt.loglog(tau, K_poisson, 's-', label=f'Poisson (slope={slope_poisson:.2f})', 
               markersize=4, alpha=0.7, color='C0')
    
    # Líneas de referencia
    tau_ref = np.logspace(0, 2, 50)
    plt.loglog(tau_ref, tau_ref**2, ':', color='gray', alpha=0.5, label='K ~ τ² (dip)')
    plt.loglog(tau_ref, tau_ref, ':', color='gray', alpha=0.5, label='K ~ τ (ramp)')
    
    plt.xlabel('τ', fontsize=12)
    plt.ylabel('K(τ)', fontsize=12)
    plt.title('Spectral Form Factor - Dip-Ramp-Plateau', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure6_spectral_form_factor.png', dpi=300, bbox_inches='tight')
    logger.info("  ✓ Guardada: figure6_spectral_form_factor.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Reproducir figuras científicas del SRCE')
    parser.add_argument('--output', type=str, default='figures/',
                       help='Directorio de salida para figuras')
    parser.add_argument('--figures', nargs='+', type=int,
                       help='Figuras a generar (default: todas)')
    
    args = parser.parse_args()
    
    # Crear directorio
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("="*70)
    logger.info(" REPRODUCIENDO FIGURAS CIENTÍFICAS - SRCE")
    logger.info("="*70)
    logger.info(f"\nDirectorio de salida: {output_dir.absolute()}\n")
    
    # Figuras a generar
    figures = {
        1: figure1_spacing_distributions,
        2: figure2_delta3_comparison,
        3: figure3_sigma2_ordering,
        4: figure4_r_parameter_validation,
        5: figure5_pair_correlation,
        6: figure6_spectral_form_factor,
    }
    
    if args.figures:
        selected = {k: v for k, v in figures.items() if k in args.figures}
    else:
        selected = figures
    
    # Generar figuras
    for num, func in selected.items():
        try:
            func(output_dir)
        except Exception as e:
            logger.error(f"  ✗ Error en figura {num}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "="*70)
    logger.info(f"✅ Completado. {len(selected)} figuras generadas en {output_dir}/")
    logger.info("="*70)


if __name__ == "__main__":
    main()
