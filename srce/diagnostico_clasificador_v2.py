#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIAGNÓSTICO COMPLETO DEL CLASIFICADOR DE ENSEMBLES
===================================================

Protocolo de 3 niveles:
1. Validación de densidad espectral (⟨s⟩ = 1)
2. Estadística robusta (r-parameter)
3. Refinamiento del clasificador Δ₃

Autor: Jorge BC & Claude
Fecha: 2026-03-07
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from typing import Tuple

# Importar módulos del proyecto
import sys
sys.path.insert(0, 'src')

from riemann_spectral.analysis.unfolding import unfolding_wigner_gue
from riemann_spectral.engine.ensemble_classifier import (
    EnsembleClassifier,
    PENDIENTE_GUE,
    PENDIENTE_GOE,
)

# Constantes teóricas
R_PARAM_POISSON = 0.386
R_PARAM_GOE = 0.5359
R_PARAM_GUE = 0.6027

print("="*80)
print(" DIAGNÓSTICO COMPLETO DEL CLASIFICADOR DE ENSEMBLES SRCE")
print("="*80)
print()

# ============================================================================
# GENERACIÓN DE ENSEMBLES (REPRODUCIR TESTS EXACTOS)
# ============================================================================

print("[SETUP] Generando ensembles...")
print("-" * 80)

# GUE N=1200 (seed=99, igual que conftest.py)
print("→ GUE N=1200 (matriz Hermítica compleja)")
rng_gue = np.random.default_rng(seed=99)
N_gue = 1200
A_gue = rng_gue.standard_normal((N_gue, N_gue)) + 1j * rng_gue.standard_normal((N_gue, N_gue))
H_gue = (A_gue + A_gue.conj().T) / (2 * np.sqrt(N_gue))
ev_gue_raw = np.sort(la.eigvalsh(H_gue))
print(f"  Eigenvalores: N={len(ev_gue_raw)}, rango=[{ev_gue_raw.min():.3f}, {ev_gue_raw.max():.3f}]")

# GOE N=1200 (seed=7, igual que conftest.py)
print("\n→ GOE N=1200 (matriz simétrica real)")
rng_goe = np.random.default_rng(seed=7)
N_goe = 1200
A_goe = rng_goe.standard_normal((N_goe, N_goe))
H_goe = (A_goe + A_goe.T) / (2 * np.sqrt(N_goe))
ev_goe_raw = np.sort(la.eigvalsh(H_goe))
print(f"  Eigenvalores: N={len(ev_goe_raw)}, rango=[{ev_goe_raw.min():.3f}, {ev_goe_raw.max():.3f}]")

# Unfolding inicial (igual que conftest.py)
print("\n→ Aplicando unfolding de Wigner (semicírculo)")
u_gue = unfolding_wigner_gue(ev_gue_raw)
u_goe = unfolding_wigner_gue(ev_goe_raw)

# Tercio central (igual que conftest.py)
print("→ Extrayendo tercio central [N/3 : 2N/3]")
n_gue = len(u_gue)
central_gue = u_gue[n_gue // 3: 2 * (n_gue // 3)]
gue_unfolded_original = central_gue - central_gue[0]

n_goe = len(u_goe)
central_goe = u_goe[n_goe // 3: 2 * (n_goe // 3)]
goe_unfolded_original = central_goe - central_goe[0]

print(f"  GUE tercio central: N={len(gue_unfolded_original)}")
print(f"  GOE tercio central: N={len(goe_unfolded_original)}")

print("\n" + "="*80)
print(" NIVEL 1 — VALIDACIÓN DE DENSIDAD ESPECTRAL")
print("="*80)
print()

def validar_densidad(spectrum: np.ndarray, label: str) -> Tuple[float, float, bool]:
    """
    Valida que ⟨s⟩ ≈ 1 después del unfolding.
    
    Returns:
        (mean_spacing, std_spacing, is_valid)
    """
    spacings = np.diff(spectrum)
    mean_s = np.mean(spacings)
    std_s = np.std(spacings)
    is_valid = 0.95 <= mean_s <= 1.05
    
    print(f"[{label}]")
    print(f"  ⟨s⟩ = {mean_s:.6f}  (esperado: 1.000)")
    print(f"  σ(s) = {std_s:.6f}")
    print(f"  Rango aceptable: [0.95, 1.05]")
    print(f"  Estado: {'✓ VÁLIDO' if is_valid else '✗ FUERA DE RANGO'}")
    
    if not is_valid:
        print(f"  ⚠️  Desviación de {abs(mean_s - 1.0)*100:.1f}% detectada")
    
    return mean_s, std_s, is_valid

# Validar GUE original
mean_gue_orig, std_gue_orig, valid_gue_orig = validar_densidad(
    gue_unfolded_original, "GUE (unfolding original)"
)

print()

# Validar GOE original
mean_goe_orig, std_goe_orig, valid_goe_orig = validar_densidad(
    goe_unfolded_original, "GOE (unfolding original)"
)

# ============================================================================
# RENORMALIZACIÓN SI ES NECESARIO
# ============================================================================

print("\n" + "-"*80)
print("RENORMALIZACIÓN EXPLÍCITA")
print("-"*80)

gue_unfolded = gue_unfolded_original.copy()
goe_unfolded = goe_unfolded_original.copy()

if not valid_gue_orig:
    print(f"\n→ Aplicando renormalización a GUE (⟨s⟩ = {mean_gue_orig:.6f})")
    gue_unfolded = (gue_unfolded_original - gue_unfolded_original[0]) / mean_gue_orig
    mean_gue_new = np.mean(np.diff(gue_unfolded))
    print(f"  Nuevo ⟨s⟩ = {mean_gue_new:.6f}")
else:
    print("\n→ GUE ya tiene densidad correcta, no requiere renormalización")

if not valid_goe_orig:
    print(f"\n→ Aplicando renormalización a GOE (⟨s⟩ = {mean_goe_orig:.6f})")
    goe_unfolded = (goe_unfolded_original - goe_unfolded_original[0]) / mean_goe_orig
    mean_goe_new = np.mean(np.diff(goe_unfolded))
    print(f"  Nuevo ⟨s⟩ = {mean_goe_new:.6f}")
else:
    print("\n→ GOE ya tiene densidad correcta, no requiere renormalización")

print("\n" + "="*80)
print(" NIVEL 2 — ESTADÍSTICA ROBUSTA (r-parameter)")
print("="*80)
print()

def calcular_r_parameter(spectrum: np.ndarray, label: str) -> float:
    """
    Calcula el ratio de spacings consecutivos (r-parameter).
    
    r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
    
    Valores universales:
        Poisson ≈ 0.386
        GOE     ≈ 0.5359
        GUE     ≈ 0.6027
    """
    spacings = np.diff(spectrum)
    
    # Evitar división por cero
    s_i = spacings[:-1]
    s_i1 = spacings[1:]
    
    # r_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1})
    r_vals = np.minimum(s_i, s_i1) / np.maximum(s_i, s_i1)
    r_mean = np.mean(r_vals)
    r_std = np.std(r_vals)
    
    print(f"[{label}]")
    print(f"  ⟨r⟩ = {r_mean:.4f}  ±  {r_std:.4f}")
    print(f"  Comparación con teoría:")
    print(f"    Poisson: {R_PARAM_POISSON:.4f}  (distancia: {abs(r_mean - R_PARAM_POISSON):.4f})")
    print(f"    GOE    : {R_PARAM_GOE:.4f}  (distancia: {abs(r_mean - R_PARAM_GOE):.4f})")
    print(f"    GUE    : {R_PARAM_GUE:.4f}  (distancia: {abs(r_mean - R_PARAM_GUE):.4f})")
    
    # Clasificación por r-parameter
    dist_poisson = abs(r_mean - R_PARAM_POISSON)
    dist_goe = abs(r_mean - R_PARAM_GOE)
    dist_gue = abs(r_mean - R_PARAM_GUE)
    
    min_dist = min(dist_poisson, dist_goe, dist_gue)
    if min_dist == dist_poisson:
        clasificacion = "Poisson"
    elif min_dist == dist_goe:
        clasificacion = "GOE"
    else:
        clasificacion = "GUE"
    
    print(f"  → Clasificación por r-parameter: {clasificacion}")
    
    return r_mean

# Calcular r-parameter para ambos
r_gue = calcular_r_parameter(gue_unfolded, "GUE (renormalizado)")
print()
r_goe = calcular_r_parameter(goe_unfolded, "GOE (renormalizado)")

print("\n" + "="*80)
print(" NIVEL 3 — REFINAMIENTO DEL CLASIFICADOR Δ₃")
print("="*80)
print()

# Clasificador con rango más asintótico
print("→ Creando clasificador con rango asintótico")
print("  L_min = 10  (evita correcciones de orden bajo)")
print("  L_max = 50  (mejor estadística)")
print()

clf_refinado = EnsembleClassifier(L_min=10.0, L_max=50.0, n_puntos=20)

# Clasificar GUE
print("[Clasificación GUE con rango refinado]")
print("-" * 80)
res_gue = clf_refinado.clasificar(gue_unfolded, label="GUE")
print(f"  Ensemble detectado   : {res_gue.ensemble}")
print(f"  Pendiente observada  : {res_gue.pendiente_obs:.6f}")
print(f"  Pendiente teórica GUE: {PENDIENTE_GUE:.6f}  (1/π²)")
print(f"  Error relativo       : {100 * res_gue.error_relativo:.1f}%")
print(f"  R² ajuste log        : {res_gue.R2_log:.4f}")
print(f"  R² ajuste lineal     : {res_gue.R2_lineal:.4f}")

print()

# Clasificar GOE
print("[Clasificación GOE con rango refinado]")
print("-" * 80)
res_goe = clf_refinado.clasificar(goe_unfolded, label="GOE")
print(f"  Ensemble detectado   : {res_goe.ensemble}")
print(f"  Pendiente observada  : {res_goe.pendiente_obs:.6f}")
print(f"  Pendiente teórica GOE: {PENDIENTE_GOE:.6f}  (1/2π²)")
print(f"  Error relativo       : {100 * res_goe.error_relativo:.1f}%")
print(f"  R² ajuste log        : {res_goe.R2_log:.4f}")
print(f"  R² ajuste lineal     : {res_goe.R2_lineal:.4f}")

print()

# Ratio de pendientes
ratio_pendientes = res_goe.pendiente_obs / res_gue.pendiente_obs
print(f"[Ratio GOE/GUE]")
print(f"  Observado: {ratio_pendientes:.4f}")
print(f"  Teórico  : 0.5000")
print(f"  Error    : {abs(ratio_pendientes - 0.5):.4f}")
if abs(ratio_pendientes - 0.5) < 0.1:
    print(f"  Estado   : ✓ Consistente con teoría")
else:
    print(f"  Estado   : ✗ Desviación significativa")

print("\n" + "="*80)
print(" VISUALIZACIÓN — DISTRIBUCIÓN DE SPACINGS")
print("="*80)
print()

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Diagnóstico de Ensembles: Distribución de Spacings', fontsize=16, fontweight='bold')

# GUE - Histograma
ax = axes[0, 0]
s_gue = np.diff(gue_unfolded)
ax.hist(s_gue, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(np.mean(s_gue), color='red', linestyle='--', linewidth=2, label=f'⟨s⟩ = {np.mean(s_gue):.3f}')
ax.set_title('GUE: Distribución de Spacings', fontweight='bold')
ax.set_xlabel('Spacing s')
ax.set_ylabel('Densidad')
ax.legend()
ax.grid(alpha=0.3)

# Predicción teórica GUE (Wigner surmise)
s_theory = np.linspace(0, s_gue.max(), 200)
# Wigner surmise GUE (β=2): P(s) = (32/π²) s² exp(-4s²/π)
P_gue_theory = (32 / np.pi**2) * s_theory**2 * np.exp(-4 * s_theory**2 / np.pi)
ax.plot(s_theory, P_gue_theory, 'r-', linewidth=2, label='Wigner surmise (β=2)', alpha=0.7)
ax.legend()

# GOE - Histograma
ax = axes[0, 1]
s_goe = np.diff(goe_unfolded)
ax.hist(s_goe, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
ax.axvline(np.mean(s_goe), color='red', linestyle='--', linewidth=2, label=f'⟨s⟩ = {np.mean(s_goe):.3f}')
ax.set_title('GOE: Distribución de Spacings', fontweight='bold')
ax.set_xlabel('Spacing s')
ax.set_ylabel('Densidad')
ax.legend()
ax.grid(alpha=0.3)

# Predicción teórica GOE (Wigner surmise)
# Wigner surmise GOE (β=1): P(s) = (π/2) s exp(-πs²/4)
P_goe_theory = (np.pi / 2) * s_theory * np.exp(-np.pi * s_theory**2 / 4)
ax.plot(s_theory, P_goe_theory, 'r-', linewidth=2, label='Wigner surmise (β=1)', alpha=0.7)
ax.legend()

# GUE - Curva Δ₃(L)
ax = axes[1, 0]
L_vals = res_gue.L_grid
d3_vals = res_gue.d3_valores
mask = np.isfinite(d3_vals)
ax.plot(L_vals[mask], d3_vals[mask], 'bo-', label='Δ₃ observado', markersize=5)
# Predicción teórica
L_theory = np.linspace(L_vals[mask].min(), L_vals[mask].max(), 100)
d3_gue_theory = (1 / np.pi**2) * np.log(L_theory)
ax.plot(L_theory, d3_gue_theory, 'r--', linewidth=2, label='Teoría: (1/π²)log(L)')
ax.set_title(f'GUE: Δ₃(L) vs L (pendiente={res_gue.pendiente_obs:.4f})', fontweight='bold')
ax.set_xlabel('L')
ax.set_ylabel('Δ₃(L)')
ax.legend()
ax.grid(alpha=0.3)

# GOE - Curva Δ₃(L)
ax = axes[1, 1]
L_vals = res_goe.L_grid
d3_vals = res_goe.d3_valores
mask = np.isfinite(d3_vals)
ax.plot(L_vals[mask], d3_vals[mask], 'go-', label='Δ₃ observado', markersize=5)
# Predicción teórica
d3_goe_theory = (1 / (2 * np.pi**2)) * np.log(L_theory)
ax.plot(L_theory, d3_goe_theory, 'r--', linewidth=2, label='Teoría: (1/2π²)log(L)')
ax.set_title(f'GOE: Δ₃(L) vs L (pendiente={res_goe.pendiente_obs:.4f})', fontweight='bold')
ax.set_xlabel('L')
ax.set_ylabel('Δ₃(L)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostico_ensembles.png', dpi=150, bbox_inches='tight')
print("→ Figura guardada: diagnostico_ensembles.png")
print()

# ============================================================================
# RESUMEN Y CONCLUSIONES
# ============================================================================

print("="*80)
print(" RESUMEN Y CONCLUSIONES")
print("="*80)
print()

print("[Nivel 1: Densidad Espectral]")
if valid_gue_orig and valid_goe_orig:
    print("  ✓ Ambos ensembles tienen densidad correcta ⟨s⟩ ≈ 1")
else:
    print("  ✗ Problemas de densidad detectados:")
    if not valid_gue_orig:
        print(f"    - GUE: ⟨s⟩ = {mean_gue_orig:.6f} (esperado: 1.000)")
    if not valid_goe_orig:
        print(f"    - GOE: ⟨s⟩ = {mean_goe_orig:.6f} (esperado: 1.000)")
    print("  → Renormalización aplicada exitosamente")

print()
print("[Nivel 2: r-parameter]")
print(f"  GUE: ⟨r⟩ = {r_gue:.4f}  (esperado: {R_PARAM_GUE:.4f})")
print(f"  GOE: ⟨r⟩ = {r_goe:.4f}  (esperado: {R_PARAM_GOE:.4f})")
dist_r_gue = abs(r_gue - R_PARAM_GUE)
dist_r_goe = abs(r_goe - R_PARAM_GOE)
if dist_r_gue < 0.05 and dist_r_goe < 0.05:
    print("  ✓ Ambos ensembles correctamente clasificados por r-parameter")
else:
    print("  ⚠️  Desviaciones detectadas en r-parameter")

print()
print("[Nivel 3: Clasificador Δ₃ Refinado]")
print(f"  GUE: pendiente = {res_gue.pendiente_obs:.6f} (teórico: {PENDIENTE_GUE:.6f})")
print(f"       error = {100 * res_gue.error_relativo:.1f}%")
print(f"  GOE: pendiente = {res_goe.pendiente_obs:.6f} (teórico: {PENDIENTE_GOE:.6f})")
print(f"       error = {100 * res_goe.error_relativo:.1f}%")
print(f"  Ratio GOE/GUE = {ratio_pendientes:.4f} (teórico: 0.5000)")

print()
print("[Diagnóstico Final]")
if res_gue.error_relativo < 0.30 and res_goe.error_relativo < 0.30:
    print("  ✓ Clasificador funciona correctamente con rango refinado")
    print("  ✓ Problema original: rango L=[5,25] demasiado bajo")
elif abs(ratio_pendientes - 0.5) > 0.5:
    print("  ✗ Ratio invertido persiste")
    print("  → Posible bug en generación o etiquetado de ensembles")
else:
    print("  ⚠️  Convergencia parcial - aumentar N recomendado")

print()
print("="*80)
print(" FIN DEL DIAGNÓSTICO")
print("="*80)
print()
print("Revisa la figura 'diagnostico_ensembles.png' para análisis visual.")
