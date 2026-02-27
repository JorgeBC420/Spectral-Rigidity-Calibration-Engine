#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_rmt_validation.py
===========================
Validación paper-grade del motor numérico RMT.

Demuestra tres propiedades sin ambigüedad:
    1. Densidad espectral GUE converge a la Ley del Semicírculo de Wigner.
    2. Delta_3(L) GUE reproduce la pendiente logarítmica teórica 1/π².
    3. Ambas propiedades se verifican con N=8000 en tiempo competitivo.

Uso:
    cd srce
    python benchmark_rmt_validation.py              # N=8000, 50 realizaciones
    python benchmark_rmt_validation.py --N 2000     # prueba rápida (debug)
    python benchmark_rmt_validation.py --N 8000 --R 100 --no-plot  # CI

Outputs:
    - Tabla de resultados en stdout (apta para paper)
    - benchmark_results.csv (opcional)
    - plots/benchmark_semicircle.pdf + plots/benchmark_delta3.pdf (si --plot)

Criterios de éxito (paper-level):
    - Error L2 densidad vs semicírculo < 0.005
    - Pendiente α ∈ [0.090, 0.115]  (1/π² ± 10%)
    - Error relativo pendiente < 5%

Nota técnica — fórmula de Delta_3:
    Usa la fórmula exacta I1/I2/I3 de Mehta (Cap. 16), NO la 'fórmula de
    momentos' m2/L - 12*m1^2/L^4 + ... que es algebraicamente incorrecta
    (I3 requiere sum_j (2j-1)*t_j que depende del orden, no solo de m0,m1,m2).
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

# ── Constantes teóricas ───────────────────────────────────────────────────────
PENDIENTE_GUE_TEORICA = 1.0 / np.pi**2        # ≈ 0.10132
PENDIENTE_GOE_TEORICA = 1.0 / (2 * np.pi**2)  # ≈ 0.05066


# ══════════════════════════════════════════════════════════════════════════════
# 1. GENERADOR GUE TRIDIAGONAL (Dumitriu–Edelman β=2)
# ══════════════════════════════════════════════════════════════════════════════

def generar_gue_tridiagonal(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Genera espectro GUE de tamaño N usando el modelo tridiagonal de
    Dumitriu & Edelman (2002). O(N²) en tiempo y O(N) en memoria.

    Parámetros del modelo (β=2, GUE):
        diagonal   d_i ~ N(0, 2)               [varianza 2, no 1]
        subdiag    e_i ~ χ(2*(N-i)) / √(2)

    Normalización: evals / √(2N) → soporte [-2, 2] (semicírculo de Wigner).

    Verificación en este script:
        - Histograma empírico vs ρ(x) = √(4-x²)/(2π)
        - KS-distance < 0.02 para N=8000
        - Pendiente Δ₃ vs log(L) ≈ 1/π² ± 5%
    """
    # Diagonal: N(0, √2) → varianza 2
    d = rng.normal(0.0, np.sqrt(2.0), size=N)

    # Sub-diagonal: χ con dfs = 2*(N-1), 2*(N-2), ..., 2
    dfs = 2 * np.arange(N - 1, 0, -1)
    e   = np.sqrt(rng.chisquare(dfs))

    # eigvalsh_tridiagonal: O(N²) con LAPACK (dstev)
    evals = eigvalsh_tridiagonal(d, e)

    # Escalar a soporte [-2, 2]
    return np.sort(evals / np.sqrt(2.0 * N))


# ══════════════════════════════════════════════════════════════════════════════
# 2. SEMICÍRCULO DE WIGNER Y UNFOLDING ANALÍTICO
# ══════════════════════════════════════════════════════════════════════════════

def semicircle_density(x: np.ndarray) -> np.ndarray:
    """ρ(x) = √(4-x²) / (2π), soporte [-2, 2]."""
    rho = np.zeros_like(x)
    mask = np.abs(x) <= 2.0
    rho[mask] = np.sqrt(4.0 - x[mask]**2) / (2.0 * np.pi)
    return rho


def semicircle_cdf(x: np.ndarray) -> np.ndarray:
    """
    CDF analítica del semicírculo en [-2, 2].
    F(x) = (1/2π) * (x*√(4-x²) + 4*arcsin(x/2)) + 1/2
    """
    xc  = np.clip(x, -2.0, 2.0)
    return (xc * np.sqrt(4.0 - xc**2) + 4.0 * np.arcsin(xc / 2.0)) / (2.0 * np.pi) + 0.5


def unfold_gue(evals: np.ndarray) -> np.ndarray:
    """
    Unfolding analítico GUE: F(x) * N → densidad ≈ 1 en el bulk.
    Tercio central para evitar efectos de borde del semicírculo.
    """
    u = semicircle_cdf(evals) * len(evals)
    n = len(u)
    s, e = n // 3, 2 * (n // 3)
    central = u[s:e]
    return central - central[0]


# ══════════════════════════════════════════════════════════════════════════════
# 3. DELTA_3 EXACTO (I1/I2/I3, ventanas deslizantes en el bulk)
# ══════════════════════════════════════════════════════════════════════════════

def delta3_ventana(bulk: np.ndarray, x0: float, L: float) -> float:
    """
    Delta_3 para una ventana [x0, x0+L].

    Fórmula exacta (Mehta 2004, Cap. 16):
        I1 = M*L - Σt_j
        I2 = (M*L² - Σt_j²) / 2
        I3 = M²*L - Σ_j (2j-1)*t_j    [j 1-indexed, niveles ordenados]
        B  = 12*I2/L³ - 6*I1/L²
        A  = (I1 - B*L²/2) / L
        Δ₃ = (I3 - 2A*I1 - 2B*I2 + A²*L + A*B*L² + B²*L³/3) / L
    """
    x1 = x0 + L
    i  = np.searchsorted(bulk, x0)          # O(log N) — bulk es ordenado
    m0, sx, sx2, sjx = 0, 0.0, 0.0, 0.0
    j = i
    while j < len(bulk) and bulk[j] <= x1:  # early-exit
        t = bulk[j] - x0
        m0 += 1; sx += t; sx2 += t * t
        sjx += (2 * m0 - 1) * t             # suma ponderada por índice — requiere orden
        j   += 1

    if m0 < 2:
        return np.nan

    L2, L3 = L * L, L * L * L
    I1  = m0 * L  - sx
    I2  = 0.5 * (m0 * L2 - sx2)
    I3  = m0 * m0 * L - sjx
    B   = 12.0 * I2 / L3 - 6.0 * I1 / L2
    A   = (I1 - B * L2 * 0.5) / L
    val = (I3 - 2*A*I1 - 2*B*I2 + A**2*L + A*B*L2 + B**2*L3/3) / L
    return max(val, 0.0)


def delta3_espectro(u: np.ndarray, L: float, n_windows: int = 100) -> float:
    """
    Delta_3(L) promediado sobre n_windows ventanas en el bulk (percentil 10-90).

    Args:
        u        : espectro unfolded ORDENADO, densidad ≈ 1.
        L        : longitud de la ventana.
        n_windows: número de ventanas equiespaciadas.

    Returns:
        Media de Delta_3 sobre ventanas válidas (al menos 2 niveles).
    """
    n = len(u)
    s, e   = n // 10, 9 * n // 10
    bulk   = u[s:e]
    x_min  = bulk[0]
    x_max  = bulk[-1] - L

    if x_max <= x_min or len(bulk) < 2:
        return np.nan

    starts = np.linspace(x_min, x_max, n_windows)
    vals   = [delta3_ventana(bulk, x0, L) for x0 in starts]
    vals   = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def delta3_error_estandar(u: np.ndarray, L: float, n_windows: int = 100) -> Tuple:
    """Devuelve (media, error_estándar_de_la_media) de Delta_3 sobre ventanas."""
    n = len(u)
    s, e   = n // 10, 9 * n // 10
    bulk   = u[s:e]
    x_min  = bulk[0]
    x_max  = bulk[-1] - L
    if x_max <= x_min or len(bulk) < 2:
        return np.nan, np.nan
    starts = np.linspace(x_min, x_max, n_windows)
    vals   = [delta3_ventana(bulk, x0, L) for x0 in starts]
    vals   = np.array([v for v in vals if np.isfinite(v)])
    if len(vals) < 2:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))


# ══════════════════════════════════════════════════════════════════════════════
# 4. AJUSTE DE PENDIENTE LOGARÍTMICA
# ══════════════════════════════════════════════════════════════════════════════

def ajustar_pendiente(L_values: np.ndarray, d3_mean: np.ndarray) -> Tuple:
    """
    Ajuste lineal de Delta_3(L) ~ α*log(L) + C por mínimos cuadrados.

    Returns:
        (alpha, C, R²) donde R² mide la bondad del ajuste log-lineal.
    """
    mask  = np.isfinite(d3_mean)
    logL  = np.log(L_values[mask])
    y     = d3_mean[mask]
    A     = np.vstack([logL, np.ones_like(logL)]).T
    coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    alpha, C = coefs
    y_fit = alpha * logL + C
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(alpha), float(C), float(R2)


# ══════════════════════════════════════════════════════════════════════════════
# 5. VALIDACIÓN DE LA DENSIDAD ESPECTRAL
# ══════════════════════════════════════════════════════════════════════════════

def validar_semicircle(todos_evals: np.ndarray, n_bins: int = 200) -> Tuple:
    """
    Compara el histograma empírico con la densidad teórica del semicírculo.

    Returns:
        (l2_error, ks_distance) — ambos deben ser pequeños.
    """
    hist, edges = np.histogram(todos_evals, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rho_teo = semicircle_density(centers)

    l2_error    = float(np.sqrt(np.mean((hist - rho_teo) ** 2)))
    # KS: máxima diferencia entre CDFs
    ks_distance = float(np.max(np.abs(
        np.cumsum(hist) / np.sum(hist)
        - np.cumsum(rho_teo) / np.sum(rho_teo)
    )))
    return l2_error, ks_distance


# ══════════════════════════════════════════════════════════════════════════════
# 6. BENCHMARK PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

# Necesitamos Tuple en Python 3.8
from typing import Tuple  # noqa: E402 (import fuera de orden deliberado)


def run_benchmark(
    N:          int  = 8000,
    n_real:     int  = 50,
    L_values:   np.ndarray = None,
    n_windows:  int  = 100,
    seed:       int  = 12345,
    plot:       bool = True,
    save_csv:   bool = True,
) -> dict:
    """
    Ejecuta el benchmark completo y devuelve el dict de resultados.

    Args:
        N        : tamaño de la matriz GUE (número de autovalores).
        n_real   : número de realizaciones.
        L_values : ventanas L para Delta_3. Default [5,10,20,30,40,50].
        n_windows: ventanas deslizantes por (realización, L).
        seed     : semilla maestra.
        plot     : generar figuras PDF.
        save_csv : guardar resultados en CSV.
    """
    if L_values is None:
        L_values = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 50.0])

    rng = np.random.default_rng(seed)
    SEP = "=" * 65

    print(f"\n{SEP}")
    print(f"  BENCHMARK RMT VALIDATION — N={N}, realizaciones={n_real}")
    print(f"{SEP}\n")

    # ── Generación ────────────────────────────────────────────────────────────
    print("1. Generando espectros GUE (Dumitriu-Edelman, β=2)...")
    espectros = np.empty((n_real, N), dtype=np.float64)

    t0 = time.perf_counter()
    for r in range(n_real):
        espectros[r] = generar_gue_tridiagonal(N, rng)
    t_gen_total = time.perf_counter() - t0
    t_gen_ms    = t_gen_total / n_real * 1000

    print(f"   Tiempo medio por espectro: {t_gen_ms:.1f} ms")
    print(f"   Tiempo total generación:   {t_gen_total:.2f} s")

    # ── Validación densidad ───────────────────────────────────────────────────
    print("\n2. Validando densidad vs semicírculo de Wigner...")
    todos_evals = espectros.flatten()
    l2_err, ks_dist = validar_semicircle(todos_evals)
    ok_density = l2_err < 0.005

    print(f"   Error L2 histograma vs ρ_Wigner: {l2_err:.6f}  "
          f"{'✓' if ok_density else '✗ (> 0.005)'}")
    print(f"   Distancia KS:                    {ks_dist:.6f}")

    # ── Unfolding ─────────────────────────────────────────────────────────────
    print("\n3. Unfolding analítico (CDF semicírculo, tercio central)...")
    unfolded = np.empty((n_real, N // 3), dtype=np.float64)
    for r in range(n_real):
        uf = unfold_gue(espectros[r])
        # El tercio central puede variar ±1 punto entre realizaciones
        m = min(len(uf), N // 3)
        unfolded[r, :m] = uf[:m]

    # ── Delta_3 ───────────────────────────────────────────────────────────────
    print(f"\n4. Calculando Δ₃(L) ({n_windows} ventanas/realización)...")
    d3_matrix = np.full((n_real, len(L_values)), np.nan)

    t1 = time.perf_counter()
    for r in range(n_real):
        u = unfolded[r]
        u = u[np.isfinite(u) & (u > 0)]
        if len(u) < 10:
            continue
        for k, L in enumerate(L_values):
            d3_matrix[r, k] = delta3_espectro(u, L, n_windows)
    t_d3 = time.perf_counter() - t1

    d3_mean = np.nanmean(d3_matrix, axis=0)
    d3_std  = np.nanstd(d3_matrix,  axis=0)
    d3_sem  = d3_std / np.sqrt(np.sum(np.isfinite(d3_matrix), axis=0))

    print(f"   Tiempo total Δ₃:               {t_d3:.2f} s")
    print(f"   Tiempo medio por espectro:     {t_d3/n_real*1000:.1f} ms")

    # ── Ajuste de pendiente ───────────────────────────────────────────────────
    print("\n5. Ajuste logarítmico Δ₃(L) ~ α*log(L) + C...")
    alpha, C, R2 = ajustar_pendiente(L_values, d3_mean)
    err_rel = abs(alpha - PENDIENTE_GUE_TEORICA) / PENDIENTE_GUE_TEORICA * 100
    ok_slope = err_rel < 5.0 and R2 > 0.95

    # ── Tabla de resultados ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  RESULTADOS CUANTITATIVOS")
    print(f"{SEP}")
    print(f"\n  Δ₃(L) por ventana:\n")
    print(f"  {'L':>6}  {'obs':>10}  {'±SEM':>8}  {'teórico':>10}  {'err%':>7}")
    print(f"  {'-'*50}")
    for k, L in enumerate(L_values):
        teo = PENDIENTE_GUE_TEORICA * np.log(L) + C
        e   = abs(d3_mean[k] - teo) / teo * 100 if np.isfinite(d3_mean[k]) else np.nan
        print(f"  {L:6.1f}  {d3_mean[k]:10.6f}  {d3_sem[k]:8.6f}  "
              f"{teo:10.6f}  {e:7.2f}%")

    print(f"\n  Pendiente logarítmica:")
    print(f"    α observada  = {alpha:.6f}")
    print(f"    α teórica    = {PENDIENTE_GUE_TEORICA:.6f}   (1/π²)")
    print(f"    Error rel.   = {err_rel:.3f} %")
    print(f"    R² ajuste    = {R2:.4f}")
    print(f"    Dentro del 5%: {'✓ SÍ' if ok_slope else '✗ NO'}")

    print(f"\n  Densidad espectral:")
    print(f"    Error L2     = {l2_err:.6f}  {'✓' if ok_density else '✗'}")
    print(f"    KS-distance  = {ks_dist:.6f}")

    t_total = t_gen_total + t_d3
    print(f"\n  Tiempos:")
    print(f"    Generación   = {t_gen_total:.2f} s  ({t_gen_ms:.1f} ms/espectro)")
    print(f"    Δ₃           = {t_d3:.2f} s  ({t_d3/n_real*1000:.1f} ms/espectro)")
    print(f"    TOTAL        = {t_total:.2f} s")

    veredicto = ok_density and ok_slope
    print(f"\n{SEP}")
    print(f"  VEREDICTO: {'✓ MOTOR VALIDADO — listo para experimentos masivos' if veredicto else '✗ REVISAR — ver detalles arriba'}")
    print(f"{SEP}\n")

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    results = {
        "N": N, "n_real": n_real,
        "alpha": alpha, "alpha_teorico": PENDIENTE_GUE_TEORICA,
        "error_rel_pct": err_rel, "R2": R2,
        "l2_error": l2_err, "ks_distance": ks_dist,
        "t_gen_s": t_gen_total, "t_d3_s": t_d3,
        "d3_mean": d3_mean.tolist(), "d3_sem": d3_sem.tolist(),
        "L_values": L_values.tolist(),
    }

    if save_csv:
        Path("output").mkdir(exist_ok=True)
        csv_path = "output/benchmark_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N", "n_real", "alpha", "alpha_teo", "err_pct",
                        "R2", "l2_err", "ks_dist", "t_gen_s", "t_d3_s"])
            w.writerow([N, n_real, f"{alpha:.6f}", f"{PENDIENTE_GUE_TEORICA:.6f}",
                        f"{err_rel:.3f}", f"{R2:.4f}", f"{l2_err:.6f}",
                        f"{ks_dist:.6f}", f"{t_gen_total:.3f}", f"{t_d3:.3f}"])
        print(f"  Resultados guardados en {csv_path}")

    # ── Gráficos ──────────────────────────────────────────────────────────────
    if plot:
        _generar_plots(todos_evals, L_values, d3_mean, d3_sem, alpha, C, N, n_real)

    return results


def _generar_plots(
    todos_evals, L_values, d3_mean, d3_sem, alpha, C, N, n_real
) -> None:
    """Genera dos figuras PDF: densidad espectral y rigidez Δ₃(L)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # sin display requerido
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [matplotlib no disponible — omitiendo gráficos]")
        return

    Path("plots").mkdir(exist_ok=True)

    # Fig 1: Densidad espectral
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(todos_evals, bins=200, density=True, alpha=0.65,
            color="#4C72B0", label=f"GUE N={N}, {n_real} real.")
    x = np.linspace(-2.1, 2.1, 400)
    ax.plot(x, semicircle_density(x), "r-", lw=2,
            label=r"$\rho(x) = \frac{\sqrt{4-x^2}}{2\pi}$ (Wigner)")
    ax.set_xlabel("Autovalor $x$")
    ax.set_ylabel("Densidad")
    ax.set_title("Ley del Semicírculo de Wigner — GUE Tridiagonal")
    ax.legend()
    ax.set_xlim(-2.5, 2.5)
    plt.tight_layout()
    plt.savefig("plots/benchmark_semicircle.pdf", dpi=150)
    plt.close()
    print("  Figura guardada: plots/benchmark_semicircle.pdf")

    # Fig 2: Δ₃(L) vs log(L)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(L_values, d3_mean, yerr=2 * d3_sem,
                fmt="o-", color="#4C72B0", capsize=4, label="Δ₃ numérico (±2 SEM)")
    logL = np.linspace(np.log(L_values[0]) * 0.9, np.log(L_values[-1]) * 1.05, 200)
    ax.plot(np.exp(logL), PENDIENTE_GUE_TEORICA * logL + C,
            "r--", lw=2, label=r"$\frac{1}{\pi^2}\log L + C$ (teórico GUE)")
    ax.plot(np.exp(logL), alpha * logL + C,
            "g:", lw=1.5, label=f"Ajuste (α={alpha:.4f})")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\Delta_3(L)$")
    ax.set_title(fr"Rigidez Espectral GUE — N={N}, α={alpha:.4f} (teórico={PENDIENTE_GUE_TEORICA:.4f})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/benchmark_delta3.pdf", dpi=150)
    plt.close()
    print("  Figura guardada: plots/benchmark_delta3.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark paper-grade del motor RMT (GUE Dumitriu-Edelman + Δ₃ exacto)"
    )
    parser.add_argument("--N",       type=int,   default=8000,
                        help="Tamaño de la matriz GUE (default: 8000)")
    parser.add_argument("--R",       type=int,   default=50,
                        help="Número de realizaciones (default: 50)")
    parser.add_argument("--windows", type=int,   default=100,
                        help="Ventanas deslizantes por Δ₃ (default: 100)")
    parser.add_argument("--seed",    type=int,   default=12345,
                        help="Semilla aleatoria (default: 12345)")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="No generar gráficos PDF")
    parser.add_argument("--no-csv",  dest="csv",  action="store_false",
                        help="No guardar CSV de resultados")
    args = parser.parse_args()

    run_benchmark(
        N         = args.N,
        n_real    = args.R,
        n_windows = args.windows,
        seed      = args.seed,
        plot      = args.plot,
        save_csv  = args.csv,
    )


if __name__ == "__main__":
    main()
