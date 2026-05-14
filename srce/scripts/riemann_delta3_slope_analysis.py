#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
riemann_delta3_slope_analysis.py

Análisis de:
- Δ₃(L) vs log L
- Pendiente local dΔ₃ / d(log L)

Usa ceros de Riemann reales (mpmath) y el núcleo SRCE ``delta3_dyson_mehta``.

Nota: Un unfolding estricto de ceros de ζ usa x_n = N(γ_n) con N(T) de Von Mangoldt;
aquí se aplica ``normalize_spacing`` sobre las γ_n como aproximación operativa
(ver también ``docs/THEORY.md`` sobre ventanas finitas).

Salida: ``scripts/output/riemann_delta3_log.png``,
``scripts/output/riemann_delta3_slope.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpmath import zetazero

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from riemann_spectral.analysis.normalize import normalize_spacing
from riemann_spectral.analysis.rigidity import delta3_dyson_mehta


def get_riemann_zeros(n_zeros: int = 2000) -> np.ndarray:
    zeros = [float(zetazero(n).imag) for n in range(1, n_zeros + 1)]
    return np.array(zeros)


def local_log_slope(L: np.ndarray, delta3: np.ndarray) -> np.ndarray:
    """dΔ₃ / d(log L) = L * dΔ₃/dL."""
    d_delta = np.gradient(delta3, L)
    return L * d_delta


def main() -> None:
    out = _SCRIPT_DIR / "output"
    out.mkdir(parents=True, exist_ok=True)

    print("Análisis Δ₃ con ceros de Riemann (mpmath + delta3_dyson_mehta)")

    zeros = get_riemann_zeros(2000)
    unfolded = normalize_spacing(zeros)

    L_grid = np.linspace(5, 50, 25)
    delta3_vals = np.array([delta3_dyson_mehta(unfolded, float(L)) for L in L_grid])

    slope_vals = local_log_slope(L_grid, delta3_vals)
    gue_slope_theory = 1.0 / (np.pi ** 2)

    log_l = np.log(L_grid)

    plt.figure(figsize=(10, 6))
    plt.plot(log_l, delta3_vals, "o-", label="Riemann zeros")

    coeffs = np.polyfit(log_l, delta3_vals, 1)
    fit = np.polyval(coeffs, log_l)
    plt.plot(log_l, fit, "--", label=f"Fit slope ≈ {coeffs[0]:.4f}")

    plt.xlabel("log L")
    plt.ylabel("Δ₃(L)")
    plt.title("Δ₃ vs log L (Riemann Zeros)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p_log = out / "riemann_delta3_log.png"
    plt.savefig(p_log, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(L_grid, slope_vals, "o-", label="Pendiente local")
    plt.axhline(
        gue_slope_theory,
        color="r",
        linestyle="--",
        label=f"GUE asint. = 1/π² ≈ {gue_slope_theory:.4f}",
    )
    plt.xlabel("L")
    plt.ylabel("dΔ₃ / d(log L)")
    plt.title("Pendiente local de Δ₃ (Riemann Zeros)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    p_slope = out / "riemann_delta3_slope.png"
    plt.savefig(p_slope, dpi=300)
    plt.close()

    print("Figuras generadas:")
    print(f"  - {p_log}")
    print(f"  - {p_slope}")
    print("\nResultado clave:")
    print(f"  Pendiente global (OLS log L) ≈ {coeffs[0]:.6f}")
    print(f"  1/π² (solo referencia asintótica) ≈ {gue_slope_theory:.6f}")


if __name__ == "__main__":
    main()
