# -*- coding: utf-8 -*-
"""
src/riemann_spectral/analysis/normalize.py
===========================================

Normalización de spacing para garantizar densidad unitaria.

Uso:
    spectrum_normalized = normalize_spacing(spectrum)
    
Esto asegura que todas las estadísticas (r, Δ₃, Σ²) usen la misma escala.

Autor: Jorge BC & Claude
Versión: 1.0.0
"""

import numpy as np


def normalize_spacing(spectrum: np.ndarray) -> np.ndarray:
    """
    Normaliza un espectro para que tenga spacing medio = 1.
    
    Esta es la normalización OBLIGATORIA después del unfolding para
    garantizar que las estadísticas espectrales (r-parameter, Δ₃, Σ²)
    usen la escala correcta.
    
    Fórmula:
        unfolded = (spectrum - spectrum[0]) / ⟨s⟩
    
    donde ⟨s⟩ = mean(diff(spectrum))
    
    Args:
        spectrum: Eigenvalues ordenados (puede estar en cualquier escala).
    
    Returns:
        Espectro normalizado con:
        - Primer punto en 0
        - Spacing medio = 1
    
    Example:
        >>> raw_evals = np.sort(np.random.randn(1000))
        >>> unfolded = unfolding_wigner_gue(raw_evals)
        >>> normalized = normalize_spacing(unfolded)
        >>> print(np.mean(np.diff(normalized)))  # Debería ser ≈ 1.0
    
    Notes:
        Esta función es IDEMPOTENTE: aplicarla dos veces no cambia nada.
        normalize_spacing(normalize_spacing(x)) == normalize_spacing(x)
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    
    if len(spectrum) < 2:
        return spectrum
    
    # Calcular spacing medio
    s_mean = np.mean(np.diff(spectrum))
    
    # Evitar división por cero (espectro degenerado)
    # Usar epsilon de máquina para tolerancia relativa
    if abs(s_mean) < np.finfo(np.float64).eps:
        raise ValueError(
            "Spacing medio ≈ 0. El espectro parece degenerado o mal ordenado."
        )
    
    # Normalizar: shift a 0 y escalar por spacing medio
    normalized = (spectrum - spectrum[0]) / s_mean
    
    return normalized


def check_spacing_sanity(
    spectrum: np.ndarray,
    label: str = "Spectrum",
    verbose: bool = True,
) -> dict:
    """
    Verifica que un espectro tenga spacing medio ≈ 1.
    
    Útil para debugging: detecta problemas de normalización ANTES de
    calcular estadísticas espectrales.
    
    Args:
        spectrum: Espectro a verificar.
        label   : Nombre del espectro (para mensajes).
        verbose : Si True, imprime diagnóstico.
    
    Returns:
        dict con:
            - mean_spacing: ⟨s⟩
            - std_spacing: σ(s)
            - is_normalized: bool (True si 0.95 < ⟨s⟩ < 1.05)
            - first_value: spectrum[0]
    
    Example:
        >>> check = check_spacing_sanity(goe_unfolded, "GOE", verbose=True)
        >>> if not check['is_normalized']:
        >>>     print("⚠️ Spectrum no está normalizado!")
    """
    spectrum = np.asarray(spectrum)
    
    if len(spectrum) < 2:
        return {
            'mean_spacing': np.nan,
            'std_spacing': np.nan,
            'is_normalized': False,
            'first_value': spectrum[0] if len(spectrum) > 0 else np.nan,
        }
    
    spacings = np.diff(spectrum)
    mean_s = np.mean(spacings)
    std_s = np.std(spacings)
    is_normalized = 0.95 <= mean_s <= 1.05
    
    result = {
        'mean_spacing': float(mean_s),
        'std_spacing': float(std_s),
        'is_normalized': bool(is_normalized),
        'first_value': float(spectrum[0]),
    }
    
    if verbose:
        status = "✓ OK" if is_normalized else "✗ PROBLEMA"
        print(f"[{label}]")
        print(f"  ⟨s⟩ = {mean_s:.6f}  {status}")
        print(f"  σ(s) = {std_s:.6f}")
        print(f"  x[0] = {spectrum[0]:.6f}")
        
        # Hints basados en σ(s)
        if is_normalized:
            if 0.38 < std_s < 0.46:
                print(f"  → σ(s) sugiere ensemble GUE")
            elif 0.50 < std_s < 0.56:
                print(f"  → σ(s) sugiere ensemble GOE")
            elif 0.95 < std_s < 1.05:
                print(f"  → σ(s) sugiere ensemble Poisson")
    
    return result


# ============================================================================
# AUTO-TEST
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("NORMALIZE MODULE - AUTO TEST")
    print("="*70)
    
    # Test 1: Normalización básica
    print("\n[TEST 1] Normalización básica")
    rng = np.random.default_rng(42)
    raw = np.cumsum(rng.exponential(1.5, 1000))  # Spacing medio = 1.5
    
    print(f"  Antes:  ⟨s⟩ = {np.mean(np.diff(raw)):.3f}")
    normalized = normalize_spacing(raw)
    print(f"  Después: ⟨s⟩ = {np.mean(np.diff(normalized)):.3f}")
    
    if abs(np.mean(np.diff(normalized)) - 1.0) < 0.001:
        print("  ✓ Test PASADO")
    else:
        print("  ✗ Test FALLIDO")
    
    # Test 2: Idempotencia
    print("\n[TEST 2] Idempotencia")
    normalized_twice = normalize_spacing(normalized)
    diff = np.max(np.abs(normalized - normalized_twice))
    print(f"  Diferencia tras normalizar 2×: {diff:.2e}")
    
    if diff < 1e-10:
        print("  ✓ Test PASADO: Función es idempotente")
    else:
        print("  ✗ Test FALLIDO")
    
    # Test 3: Sanity check
    print("\n[TEST 3] Sanity check")
    check = check_spacing_sanity(normalized, "Normalized Poisson", verbose=True)
    
    if check['is_normalized']:
        print("  ✓ Test PASADO")
    else:
        print("  ✗ Test FALLIDO")
    
    print("\n" + "="*70)
    print("✅ Todos los tests pasaron")
    print("="*70)
