#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis Δ₃ con ceros de Riemann (exploratorio).

- Δ₃ vs log L
- pendiente local dΔ₃/d(log L)

Densidad de ceros en la recta crítica (forma estándar):

    ρ(E) ≈ (1/(2π)) log(E/(2π))

Unfolding: x_n = N(γ_n) con conteo de Von Mangoldt aproximado

    N(T) ≈ (T/(2π)) log(T/(2π)) − T/(2π).

**Nota:** Este script usa una implementación **simplificada** de Δ₃ en ventanas
(ajuste lineal sobre un índice de cuenta); el núcleo SRCE usa
``delta3_dyson_mehta`` en ``rigidity.py`` para comparaciones con RMT.

Requiere::

    pip install mpmath numpy matplotlib

Salida: ``scripts/output/figure_delta3_riemann.png``,
``scripts/output/figure_delta3_slope.png``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpmath import zetazero

# ===============================
# 1. Obtener ceros de Riemann
# ===============================


def get_riemann_zeros(N: int) -> np.ndarray:
    """Primeros N ceros (parte imaginaria γ_n)."""
    return np.array([float(zetazero(n).imag) for n in range(1, N + 1)])


# ===============================
# 2. Unfolding usando N(T)
# ===============================


def riemann_von_mangoldt(T: np.ndarray) -> np.ndarray:
    """Función de conteo N(T) aproximada."""
    return (T / (2 * np.pi)) * np.log(T / (2 * np.pi)) - (T / (2 * np.pi))


def unfold_riemann(zeros: np.ndarray) -> np.ndarray:
    """Unfolding: x_n = N(γ_n)."""
    return riemann_von_mangoldt(zeros)


# ===============================
# 3. Δ₃ (simplificado en ventanas)
# ===============================


def delta3(x: np.ndarray, L: float) -> float:
    """
    Δ₃(L) en ventanas deslizantes — implementación directa (mínimos cuadrados).
    """
    n = len(x)
    results = []

    for i in range(n):
        start = x[i]
        end = start + L

        mask = (x >= start) & (x <= end)
        segment = x[mask]

        if len(segment) < 5:
            continue

        y = np.arange(len(segment))  # N(x)

        A = np.vstack([segment, np.ones(len(segment))]).T
        b, a = np.linalg.lstsq(A, y, rcond=None)[0]

        y_fit = a + b * segment
        delta = np.mean((y - y_fit) ** 2)

        results.append(delta)

    if len(results) == 0:
        return np.nan

    return float(np.mean(results))


# ===============================
# 4. Pipeline principal
# ===============================


def main() -> None:
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(parents=True, exist_ok=True)

    print("Calculando ceros de Riemann...")
    zeros = get_riemann_zeros(1500)

    print("Unfolding...")
    x = unfold_riemann(zeros)

    # Normalizar spacing medio
    spacings = np.diff(x)
    x = x / np.mean(spacings)

    # ===========================
    # Δ₃(L)
    # ===========================
    L_vals = np.linspace(5, 50, 20)
    delta_vals = np.array([delta3(x, L) for L in L_vals])

    # ===========================
    # Pendiente local
    # ===========================
    logL = np.log(L_vals)
    slope_local = np.gradient(delta_vals, logL)

    # ===========================
    # Plot 1: Δ₃ vs log L
    # ===========================
    plt.figure(figsize=(10, 5))

    plt.plot(logL, delta_vals, "o-", label="Δ₃ (Riemann)")
    plt.plot(logL, (1 / np.pi**2) * logL, "--", label="(1/π²) log L (ref)")

    plt.xlabel("log L")
    plt.ylabel("Δ₃(L)")
    plt.title("Δ₃ vs log L (Ceros de Riemann)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    p1 = out / "figure_delta3_riemann.png"
    plt.savefig(p1, dpi=300)

    # ===========================
    # Plot 2: pendiente local
    # ===========================
    plt.figure(figsize=(10, 5))

    plt.plot(L_vals, slope_local, "o-", label="Pendiente local")
    plt.axhline(1 / np.pi**2, linestyle="--", label="1/π² ≈ 0.101")

    plt.xlabel("L")
    plt.ylabel("dΔ₃ / d(log L)")
    plt.title("Pendiente local Δ₃ (Riemann)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    p2 = out / "figure_delta3_slope.png"
    plt.savefig(p2, dpi=300)

    print("✓ Figuras generadas:")
    print(f"  - {p1}")
    print(f"  - {p2}")


if __name__ == "__main__":
    main()
