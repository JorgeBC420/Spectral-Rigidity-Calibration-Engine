#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DIAGNÓSTICO V3 - CON NORMALIZACIÓN FORZADA INTEGRADA
=====================================================

Pipeline CORREGIDO:
    raw eigenvalues
         ↓
    unfolding
         ↓
    extraer región bulk
         ↓
    normalize_spacing  ← NUEVO
         ↓
    calcular r, Δ₃, Σ²

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

# ============================================================================
# FUNCIÓN DE NORMALIZACIÓN (INTEGRADA AL PIPELINE)
# ============================================================================

def normalize_spacing(spectrum: np.ndarray) -> np.ndarray:
    """
    Normaliza un espectro para que tenga spacing medio = 1.
    
    OBLIGATORIO después del unfolding para garantizar que todas las
    estadísticas (r, Δ₃, Σ²) usen la escala correcta.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    
    if len(spectrum) < 2:
        return spectrum
    
    s_mean = np.mean(np.diff(spectrum))
    
    if abs(s_mean) < 1e-12:
        raise ValueError("Spacing medio ≈ 0. Espectro degenerado.")
    
    return (spectrum - spectrum[0]) / s_mean


def check_spacing_sanity(spectrum: np.ndarray, label: str) -> None:
    """Imprime sanity check del spacing."""
    spacings = np.diff(spectrum)
    mean_s = np.mean(spacings)
    std_s = np.std(spacings)
    
    status = "✓" if 0.95 <= mean_s <= 1.05 else "✗"
    
    print(f"[{label}]")
    print(f"  ⟨s⟩ = {mean_s:.6f}  {status}")
    print(f"  σ(s) = {std_s:.6f}")
    
    # Hints de ensemble basados en σ(s)
    if 0.38 < std_s < 0.46:
        print(f"  → σ(s) sugiere GUE")
    elif 0.50 < std_s < 0.56:
        print(f"  → σ(s) sugiere GOE")
    elif 0.95 < std_s < 1.05:
        print(f"  → σ(s) sugiere Poisson")


print("="*80)
print(" DIAGNÓSTICO V3 - PIPELINE CORREGIDO CON NORMALIZACIÓN")
print("="*80)
print()

# ============================================================================
# GENERACIÓN DE ENSEMBLES
# ============================================================================

print("[SETUP] Generando ensembles...")
print("-" * 80)

# GUE N=1200 (seed=99)
print("→ GUE N=1200 (matriz Hermítica compleja)")
rng_gue = np.random.default_rng(seed=99)
N_gue = 1200
A_gue = rng_gue.standard_normal((N_gue, N_gue)) + 1j * rng_gue.standard_normal((N_gue, N_gue))
# CORRECCIÓN: Normalización estándar de Wigner
H_gue = (A_gue + A_gue.conj().T) / np.sqrt(2 * N_gue)
ev_gue_raw = np.sort(la.eigvalsh(H_gue))

# GOE N=1200 (seed=7)
print("→ GOE N=1200 (matriz simétrica real)")
rng_goe = np.random.default_rng(seed=7)
N_goe = 1200
A_goe = rng_goe.standard_normal((N_goe, N_goe))
# CORRECCIÓN: Normalización estándar de Wigner
H_goe = (A_goe + A_goe.T) / np.sqrt(2 * N_goe)
ev_goe_raw = np.sort(la.eigvalsh(H_goe))

print()

# ============================================================================
# PIPELINE CORREGIDO
# ============================================================================

print("="*80)
print(" PIPELINE CORREGIDO - GUE")
print("="*80)

print("\n[1] Unfolding de Wigner...")
u_gue = unfolding_wigner_gue(ev_gue_raw)
check_spacing_sanity(u_gue, "GUE - después de unfolding")

print("\n[2] Extrayendo tercio central...")
n_gue = len(u_gue)
central_gue = u_gue[n_gue // 3: 2 * (n_gue // 3)]
print(f"   Tamaño: {len(central_gue)} puntos")
check_spacing_sanity(central_gue, "GUE - tercio central")

print("\n[3] Normalización FORZADA (normalize_spacing)...")
gue_unfolded = normalize_spacing(central_gue)
check_spacing_sanity(gue_unfolded, "GUE - NORMALIZADO")

print("\n" + "="*80)
print(" PIPELINE CORREGIDO - GOE")
print("="*80)

print("\n[1] Unfolding de Wigner...")
u_goe = unfolding_wigner_gue(ev_goe_raw)
check_spacing_sanity(u_goe, "GOE - después de unfolding")

print("\n[2] Extrayendo tercio central...")
n_goe = len(u_goe)
central_goe = u_goe[n_goe // 3: 2 * (n_goe // 3)]
print(f"   Tamaño: {len(central_goe)} puntos")
check_spacing_sanity(central_goe, "GOE - tercio central")

print("\n[3] Normalización FORZADA (normalize_spacing)...")
goe_unfolded = normalize_spacing(central_goe)
check_spacing_sanity(goe_unfolded, "GOE - NORMALIZADO")

# ============================================================================
# ESTADÍSTICAS ESPECTRALES
# ============================================================================

print("\n" + "="*80)
print(" ESTADÍSTICA r-PARAMETER")
print("="*80)

def calcular_r_parameter(spectrum: np.ndarray, label: str) -> float:
    """Calcula r-parameter."""
    spacings = np.diff(spectrum)
    s_i = spacings[:-1]
    s_i1 = spacings[1:]
    r_vals = np.minimum(s_i, s_i1) / np.maximum(s_i, s_i1)
    r_mean = np.mean(r_vals)
    
    print(f"\n[{label}]")
    print(f"  ⟨r⟩ = {r_mean:.4f}")
    print(f"  Comparación:")
    print(f"    Poisson: {R_PARAM_POISSON:.4f}  (dist: {abs(r_mean - R_PARAM_POISSON):.4f})")
    print(f"    GOE    : {R_PARAM_GOE:.4f}  (dist: {abs(r_mean - R_PARAM_GOE):.4f})")
    print(f"    GUE    : {R_PARAM_GUE:.4f}  (dist: {abs(r_mean - R_PARAM_GUE):.4f})")
    
    return r_mean

r_gue = calcular_r_parameter(gue_unfolded, "GUE")
r_goe = calcular_r_parameter(goe_unfolded, "GOE")

# ============================================================================
# CLASIFICADOR Δ₃
# ============================================================================

print("\n" + "="*80)
print(" CLASIFICADOR Δ₃ (Dyson-Mehta)")
print("="*80)

clf = EnsembleClassifier(L_min=10.0, L_max=50.0, n_puntos=20)

print("\n[GUE]")
print("-" * 80)
res_gue = clf.clasificar(gue_unfolded, label="GUE")
print(f"  Clasificado como     : {res_gue.ensemble}")
print(f"  Pendiente observada  : {res_gue.pendiente_obs:.6f}")
print(f"  Pendiente teórica    : {PENDIENTE_GUE:.6f}")
print(f"  Error relativo       : {100 * res_gue.error_relativo:.1f}%")
print(f"  R² log               : {res_gue.R2_log:.4f}")

print("\n[GOE]")
print("-" * 80)
res_goe = clf.clasificar(goe_unfolded, label="GOE")
print(f"  Clasificado como     : {res_goe.ensemble}")
print(f"  Pendiente observada  : {res_goe.pendiente_obs:.6f}")
print(f"  Pendiente teórica    : {PENDIENTE_GOE:.6f}")
print(f"  Error relativo       : {100 * res_goe.error_relativo:.1f}%")
print(f"  R² log               : {res_goe.R2_log:.4f}")

ratio = res_goe.pendiente_obs / res_gue.pendiente_obs
print("\n[Ratio GOE/GUE]")
print(f"  Observado: {ratio:.4f}")
print(f"  Teórico  : 0.5000")
print(f"  Estado   : {'✓ OK' if abs(ratio - 0.5) < 0.15 else '✗ PROBLEMA'}")

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

print("\n" + "="*80)
print(" GENERANDO VISUALIZACIÓN")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Diagnóstico V3 - Pipeline Corregido', fontsize=16, fontweight='bold')

# GUE - Spacings
ax = axes[0, 0]
s_gue = np.diff(gue_unfolded)
ax.hist(s_gue, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
ax.axvline(np.mean(s_gue), color='red', linestyle='--', linewidth=2, 
           label=f'⟨s⟩ = {np.mean(s_gue):.3f}')
ax.set_title(f'GUE: Spacings (σ={np.std(s_gue):.3f})', fontweight='bold')
ax.set_xlabel('Spacing s')
ax.set_ylabel('Densidad')
ax.legend()
ax.grid(alpha=0.3)

# GOE - Spacings
ax = axes[0, 1]
s_goe = np.diff(goe_unfolded)
ax.hist(s_goe, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
ax.axvline(np.mean(s_goe), color='red', linestyle='--', linewidth=2,
           label=f'⟨s⟩ = {np.mean(s_goe):.3f}')
ax.set_title(f'GOE: Spacings (σ={np.std(s_goe):.3f})', fontweight='bold')
ax.set_xlabel('Spacing s')
ax.set_ylabel('Densidad')
ax.legend()
ax.grid(alpha=0.3)

# GUE - Δ₃(L)
ax = axes[1, 0]
L_vals = res_gue.L_grid
d3_vals = res_gue.d3_valores
mask = np.isfinite(d3_vals)
ax.plot(L_vals[mask], d3_vals[mask], 'bo-', label='Δ₃ observado', markersize=5)
L_theory = np.linspace(L_vals[mask].min(), L_vals[mask].max(), 100)
d3_theory = (1 / np.pi**2) * np.log(L_theory)
ax.plot(L_theory, d3_theory, 'r--', linewidth=2, label='Teoría GUE')
ax.set_title(f'GUE: Δ₃(L) (pendiente={res_gue.pendiente_obs:.4f})', fontweight='bold')
ax.set_xlabel('L')
ax.set_ylabel('Δ₃(L)')
ax.legend()
ax.grid(alpha=0.3)

# GOE - Δ₃(L)
ax = axes[1, 1]
L_vals = res_goe.L_grid
d3_vals = res_goe.d3_valores
mask = np.isfinite(d3_vals)
ax.plot(L_vals[mask], d3_vals[mask], 'go-', label='Δ₃ observado', markersize=5)
# Recalcular L_theory para GOE
L_theory_goe = np.linspace(L_vals[mask].min(), L_vals[mask].max(), 100)
d3_theory = (1 / (2 * np.pi**2)) * np.log(L_theory_goe)
ax.plot(L_theory_goe, d3_theory, 'r--', linewidth=2, label='Teoría GOE')
ax.set_title(f'GOE: Δ₃(L) (pendiente={res_goe.pendiente_obs:.4f})', fontweight='bold')
ax.set_xlabel('L')
ax.set_ylabel('Δ₃(L)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostico_v3_corregido.png', dpi=150, bbox_inches='tight')
print("→ Figura guardada: diagnostico_v3_corregido.png")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print(" RESUMEN FINAL")
print("="*80)

print("\n[Pipeline]")
print("  ✓ Normalización forzada aplicada")
print("  ✓ Todas las densidades ⟨s⟩ ≈ 1.000")

print("\n[r-parameter]")
print(f"  GUE: {r_gue:.4f} (esperado: {R_PARAM_GUE:.4f})")
print(f"  GOE: {r_goe:.4f} (esperado: {R_PARAM_GOE:.4f})")

print("\n[Δ₃ pendientes]")
print(f"  GUE: {res_gue.pendiente_obs:.6f} (esperado: {PENDIENTE_GUE:.6f})")
print(f"       error = {100 * res_gue.error_relativo:.1f}%")
print(f"  GOE: {res_goe.pendiente_obs:.6f} (esperado: {PENDIENTE_GOE:.6f})")
print(f"       error = {100 * res_goe.error_relativo:.1f}%")
print(f"  Ratio: {ratio:.4f} (esperado: 0.5000)")

print("\n[Diagnóstico]")
if res_gue.error_relativo < 0.20 and res_goe.error_relativo < 0.20:
    print("  ✅ ÉXITO: Normalización corrigió el problema")
    print("  ✅ Ambos ensembles clasificados correctamente")
else:
    print("  ⚠️  Persisten desviaciones")
    print(f"     Error GUE: {100 * res_gue.error_relativo:.1f}%")
    print(f"     Error GOE: {100 * res_goe.error_relativo:.1f}%")

print("\n" + "="*80)
