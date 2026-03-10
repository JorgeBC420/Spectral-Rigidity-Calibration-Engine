#!/usr/bin/env python3
"""
DIAGNÓSTICO PROFUNDO: VERIFICACIÓN DE DENSIDAD EN CADA ETAPA
=============================================================

Verifica que ⟨s⟩ = 1 se mantenga en TODO el pipeline:
1. Después del unfolding
2. Después de extraer tercio central
3. Después de renormalización
4. Dentro del bulk usado por Δ₃

"""

import numpy as np
import numpy.linalg as la

import sys
sys.path.insert(0, 'src')

from riemann_spectral.analysis.unfolding import unfolding_wigner_gue
from riemann_spectral.engine.ensemble_classifier import EnsembleClassifier

print("="*80)
print("DIAGNÓSTICO PROFUNDO: DENSIDAD EN CADA ETAPA")
print("="*80)

# Generar GOE (el problemático)
print("\n[1] Generando GOE N=1200...")
rng_goe = np.random.default_rng(seed=7)
N_goe = 1200
A_goe = rng_goe.standard_normal((N_goe, N_goe))
H_goe = (A_goe + A_goe.T) / (2 * np.sqrt(N_goe))
ev_goe_raw = np.sort(la.eigvalsh(H_goe))

print(f"   Eigenvalues raw: N={len(ev_goe_raw)}")
print(f"   Rango: [{ev_goe_raw.min():.3f}, {ev_goe_raw.max():.3f}]")

# Etapa 1: Unfolding
print("\n[2] Aplicando unfolding de Wigner...")
u_goe = unfolding_wigner_gue(ev_goe_raw)
s_after_unfold = np.diff(u_goe)
mean_s1 = np.mean(s_after_unfold)
print(f"   ⟨s⟩ después de unfolding: {mean_s1:.6f}")
print(f"   {'✓ OK' if 0.95 < mean_s1 < 1.05 else '✗ PROBLEMA'}")

# Etapa 2: Tercio central
print("\n[3] Extrayendo tercio central...")
n_goe = len(u_goe)
central_goe = u_goe[n_goe // 3: 2 * (n_goe // 3)]
s_after_central = np.diff(central_goe)
mean_s2 = np.mean(s_after_central)
print(f"   Tercio central: N={len(central_goe)}")
print(f"   ⟨s⟩ del tercio central: {mean_s2:.6f}")
print(f"   {'✓ OK' if 0.95 < mean_s2 < 1.05 else '✗ PROBLEMA'}")

# Etapa 3: Normalización a cero
print("\n[4] Normalizando a x[0]=0...")
goe_unfolded_v1 = central_goe - central_goe[0]
s_after_shift = np.diff(goe_unfolded_v1)
mean_s3 = np.mean(s_after_shift)
print(f"   ⟨s⟩ después de shift: {mean_s3:.6f}")
print(f"   {'✓ OK' if 0.95 < mean_s3 < 1.05 else '✗ PROBLEMA'}")

# Etapa 4: Renormalización forzada (si es necesario)
print("\n[5] Renormalización forzada...")
if abs(mean_s3 - 1.0) > 0.05:
    print(f"   Aplicando corrección: dividir por {mean_s3:.6f}")
    goe_unfolded = (goe_unfolded_v1 - goe_unfolded_v1[0]) / mean_s3
    s_after_renorm = np.diff(goe_unfolded)
    mean_s4 = np.mean(s_after_renorm)
    print(f"   ⟨s⟩ después de renormalizar: {mean_s4:.6f}")
    print(f"   {'✓ OK' if 0.95 < mean_s4 < 1.05 else '✗ PROBLEMA'}")
else:
    print("   No se requiere renormalización")
    goe_unfolded = goe_unfolded_v1
    mean_s4 = mean_s3

# Etapa 5: Verificar bulk usado por Δ₃
print("\n[6] Verificando bulk usado por Δ₃ (percentil 10-90)...")
n = len(goe_unfolded)
s_bulk_start = n // 10
e_bulk_end = 9 * n // 10

bulk_spectrum = goe_unfolded[s_bulk_start:e_bulk_end]
s_bulk = np.diff(bulk_spectrum)
mean_s_bulk = np.mean(s_bulk)

print(f"   Bulk: índices [{s_bulk_start}, {e_bulk_end}] ({len(bulk_spectrum)} puntos)")
print(f"   ⟨s⟩ en el bulk: {mean_s_bulk:.6f}")
print(f"   {'✓ OK' if 0.95 < mean_s_bulk < 1.05 else '✗ PROBLEMA CRÍTICO'}")

# Etapa 6: Probar Δ₃ directamente
print("\n[7] Calculando Δ₃(L) para L=10,20,30,40,50...")
from src.riemann_spectral.analysis.rigidity import delta3_dyson_mehta

L_vals = [10, 20, 30, 40, 50]
d3_vals = []

for L in L_vals:
    d3 = delta3_dyson_mehta(goe_unfolded, L)
    d3_vals.append(d3)
    
    # Predicción teórica GOE
    d3_teo_goe = (1 / (2 * np.pi**2)) * np.log(L)
    error = abs(d3 - d3_teo_goe) / d3_teo_goe if d3_teo_goe > 0 else np.inf
    
    print(f"   L={L:2d}  Δ₃={d3:.4f}  Teórico_GOE={d3_teo_goe:.4f}  Error={100*error:.1f}%")

# Etapa 7: Ajuste de pendiente manual
print("\n[8] Ajuste manual de pendiente log...")
L_array = np.array(L_vals)
d3_array = np.array(d3_vals)

log_L = np.log(L_array)
p = np.polyfit(log_L, d3_array, 1)
pendiente_manual = p[0]
intercepto_manual = p[1]

print(f"   Pendiente observada: {pendiente_manual:.6f}")
print(f"   Pendiente teórica GOE: {1/(2*np.pi**2):.6f}")
print(f"   Error: {100*abs(pendiente_manual - 1/(2*np.pi**2))/(1/(2*np.pi**2)):.1f}%")

# Etapa 8: Verificar con el clasificador
print("\n[9] Verificando con EnsembleClassifier...")
clf = EnsembleClassifier(L_min=10.0, L_max=50.0, n_puntos=20)
res = clf.clasificar(goe_unfolded, label="GOE")

print(f"   Clasificado como: {res.ensemble}")
print(f"   Pendiente: {res.pendiente_obs:.6f}")
print(f"   R² log: {res.R2_log:.4f}")
print(f"   R² lineal: {res.R2_lineal:.4f}")

# Resumen
print("\n" + "="*80)
print("RESUMEN DE DENSIDADES")
print("="*80)
print(f"  [1] Después unfolding:       ⟨s⟩ = {mean_s1:.6f}")
print(f"  [2] Tercio central:          ⟨s⟩ = {mean_s2:.6f}  ← PROBLEMA AQUÍ")
print(f"  [3] Después shift:           ⟨s⟩ = {mean_s3:.6f}")
print(f"  [4] Después renormalización: ⟨s⟩ = {mean_s4:.6f}")
print(f"  [5] Bulk (p10-p90):          ⟨s⟩ = {mean_s_bulk:.6f}")
print()

if abs(mean_s2 - 1.0) > 0.05:
    print("❌ PROBLEMA DETECTADO:")
    print(f"   El tercio central tiene ⟨s⟩ = {mean_s2:.6f}")
    print("   Esto significa que el unfolding de Wigner está fallando.")
    print()
    print("   CAUSA PROBABLE:")
    print("   - unfolding_wigner_gue() asume distribución semicírculo")
    print("   - GOE y GUE tienen la MISMA densidad de estados (semicírculo)")
    print("   - Pero el tercio central puede tener densidad local diferente")
    print()
    print("   SOLUCIÓN:")
    print("   Aplicar normalización forzada SIEMPRE:")
    print("   unfolded = (central - central[0]) / np.mean(np.diff(central))")
else:
    print("✅ Todas las densidades son correctas")

print("="*80)
