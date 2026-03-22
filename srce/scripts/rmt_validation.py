#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación RMT frente al código SRCE (solo importa desde ``src/``).

Ejecutar desde la carpeta ``srce/``::

    python -u scripts/rmt_validation.py

O con ruta absoluta. Ver ``docs/VALIDATION_RMT.md`` para interpretación.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from scipy import stats
from scipy.linalg import eigvalsh

from riemann_spectral.analysis.rigidity import delta3_dyson_mehta
from riemann_spectral.analysis.unfolding import unfolding_wigner_gue
from riemann_spectral.analysis.normalize import normalize_spacing
from riemann_spectral.analysis.number_variance import sigma2_number_variance_fast
from riemann_spectral.analysis.spectral_form_factor import spectral_form_factor, extract_ramp_slope
from riemann_spectral.statistics.r_statistic import (
    compute_r_parameter,
    R_POISSON_EXACT,
    R_GOE_EXACT,
    R_GUE_EXACT,
)

RNG_SEED = 20250323
N_REAL = 10
L_MIN, L_MAX = 5.0, 50.0
N_L = 25
L_GRID = np.linspace(L_MIN, L_MAX, N_L)


def bulk_matrix_ensemble(rng: np.random.Generator, n: int, beta: str) -> np.ndarray:
    a = rng.standard_normal((n, n))
    if beta == "gue":
        a = a + 1j * rng.standard_normal((n, n))
        h = (a + a.conj().T) / (2 * np.sqrt(n))
    else:
        h = (a + a.T) / (2 * np.sqrt(n))
    ev = np.sort(eigvalsh(h))
    u = unfolding_wigner_gue(ev)
    m = len(u)
    central = u[m // 3 : 2 * (m // 3)]
    central = normalize_spacing(central)
    return central - central[0]


def bulk_poisson(rng: np.random.Generator) -> np.ndarray:
    esp = rng.exponential(1.0, size=12000)
    pos = np.cumsum(esp)
    pos = pos[pos <= 8000.0]
    m = len(pos)
    central = pos[m // 3 : 2 * (m // 3)] - pos[m // 3]
    central = normalize_spacing(central)
    return central - central[0]


def pdf_poisson(s: np.ndarray) -> np.ndarray:
    return np.exp(-np.maximum(s, 0.0))


def pdf_goe_wigner(s: np.ndarray) -> np.ndarray:
    return (np.pi / 2) * s * np.exp(-np.pi * s ** 2 / 4.0)


def pdf_gue_wigner(s: np.ndarray) -> np.ndarray:
    return (32.0 / np.pi ** 2) * s ** 2 * np.exp(-4.0 * s ** 2 / np.pi)


def build_cdf_from_pdf(pdf_fn, smax=8.0, n=8000):
    sg = np.linspace(0, smax, n)
    p = pdf_fn(sg)
    p[0] = 0.0
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(sg))])
    cdf /= cdf[-1]

    def cdf_interp(x):
        x = np.asarray(x, dtype=float)
        return np.interp(x, sg, cdf, left=0.0, right=1.0)

    return cdf_interp


def spacing_normalized(spec: np.ndarray) -> np.ndarray:
    s = np.diff(spec)
    m = np.mean(s)
    return s / m if m > 0 else s


def ks_and_l2(data: np.ndarray, pdf_fn, cdf_fn, bins=np.linspace(0, 4, 41)):
    ks, p = stats.kstest(data, cdf_fn)
    hc, _ = np.histogram(data, bins=bins, density=True)
    c = 0.5 * (bins[1:] + bins[:-1])
    theo = pdf_fn(c)
    theo[c < 0] = 0.0
    l2 = float(np.sqrt(np.mean((hc - theo) ** 2)))
    return ks, p, l2


def mean_delta3_curve(spec_fn, rng_base: int, n_matrix: int) -> np.ndarray:
    acc = np.zeros(len(L_GRID))
    for k in range(N_REAL):
        rng = np.random.default_rng(rng_base + k * 9973)
        spec = spec_fn(rng, n_matrix)
        for j, L in enumerate(L_GRID):
            acc[j] += delta3_dyson_mehta(spec, float(L))
    return acc / N_REAL


def main():
    np.set_printoptions(precision=5, suppress=True)
    print("=" * 72)
    print("SRCE — Validación RMT")
    print(f"RNG_SEED={RNG_SEED}, N_real={N_REAL}, L∈[{L_MIN},{L_MAX}] ({N_L} pts)")
    print("=" * 72)

    cdf_poi = build_cdf_from_pdf(pdf_poisson)
    cdf_goe = build_cdf_from_pdf(pdf_goe_wigner)
    cdf_gue = build_cdf_from_pdf(pdf_gue_wigner)

    n_main = 2000

    sp = spacing_normalized(bulk_poisson(np.random.default_rng(RNG_SEED)))
    sg = spacing_normalized(bulk_matrix_ensemble(np.random.default_rng(RNG_SEED + 1), n_main, "gue"))
    so = spacing_normalized(bulk_matrix_ensemble(np.random.default_rng(RNG_SEED + 2), n_main, "goe"))

    ks_p, p_p, l2_p = ks_and_l2(sp, pdf_poisson, cdf_poi)
    ks_gue, p_gue, l2_gue = ks_and_l2(sg, pdf_gue_wigner, cdf_gue)
    ks_goe, p_goe, l2_goe = ks_and_l2(so, pdf_goe_wigner, cdf_goe)

    def mean_r(get_spec, n_mat):
        out = []
        for k in range(N_REAL):
            rng = np.random.default_rng(RNG_SEED + 100 + k * 7919)
            out.append(compute_r_parameter(get_spec(rng, n_mat)))
        return float(np.mean(out)), float(np.std(out, ddof=0))

    mr_p, sd_p = mean_r(lambda r, n: bulk_poisson(r), n_main)
    mr_g, sd_g = mean_r(lambda r, n: bulk_matrix_ensemble(r, n, "gue"), n_main)
    mr_o, sd_o = mean_r(lambda r, n: bulk_matrix_ensemble(r, n, "goe"), n_main)

    err_r_p = abs(mr_p - R_POISSON_EXACT)
    err_r_g = abs(mr_g - R_GUE_EXACT)
    err_r_o = abs(mr_o - R_GOE_EXACT)

    L2g = np.linspace(L_MIN, L_MAX, 20)

    def sigma2_mean(get_spec, n_mat):
        acc = np.zeros(len(L2g))
        for k in range(N_REAL):
            rng = np.random.default_rng(RNG_SEED + 300 + k * 7919)
            acc += sigma2_number_variance_fast(get_spec(rng, n_mat), L2g)
        return acc / N_REAL

    s2p = sigma2_mean(lambda r, n: bulk_poisson(r), n_main)
    s2g = sigma2_mean(lambda r, n: bulk_matrix_ensemble(r, n, "gue"), n_main)
    slope_s2_p, _ = np.polyfit(L2g, s2p, 1)
    slope_s2_g_log, _ = np.polyfit(np.log(L2g), s2g, 1)

    d3p = mean_delta3_curve(lambda r, n: bulk_poisson(r), RNG_SEED + 500, n_main)
    d3g = mean_delta3_curve(lambda r, n: bulk_matrix_ensemble(r, n, "gue"), RNG_SEED + 600, n_main)
    d3o = mean_delta3_curve(lambda r, n: bulk_matrix_ensemble(r, n, "goe"), RNG_SEED + 700, n_main)

    logL = np.log(L_GRID)
    a_gue, _ = np.polyfit(logL, d3g, 1)
    a_goe, _ = np.polyfit(logL, d3o, 1)
    grad_g = np.gradient(d3g, logL)
    grad_o = np.gradient(d3o, logL)
    mean_grad_g = float(np.mean(grad_g))
    mean_grad_o = float(np.mean(grad_o))

    teor_p = L_GRID / 15.0
    err_d3_p = float(np.mean(np.abs(d3p - teor_p) / np.maximum(teor_p, 1e-15)))
    global_order = float(np.mean(d3o)) < float(np.mean(d3g))

    def sff_flags(spec):
        tau, K = spectral_form_factor(spec, tau_max=35.0, n_points=350, normalize=True)
        early = float(np.mean(K[1 : max(2, len(tau) // 50)]))
        mxq = float(np.max(K[: len(tau) // 4]))
        dip = early < 0.5 * mxq if mxq > 1e-12 else False
        ramp = extract_ramp_slope(tau, K, len(spec))
        tail = float(np.mean(K[-25:]))
        plat = 0.3 < tail < 2.5
        return dip, ramp, tail, plat

    fp = sff_flags(bulk_poisson(np.random.default_rng(RNG_SEED + 900)))
    fg = sff_flags(bulk_matrix_ensemble(np.random.default_rng(RNG_SEED + 901), n_main, "gue"))
    fo = sff_flags(bulk_matrix_ensemble(np.random.default_rng(RNG_SEED + 902), n_main, "goe"))

    print("\n### TABLA RESUMEN (N=2000, 10 realiz. donde aplica)\n")
    print("| Métrica | Poisson | GOE | GUE | Estado |")
    print("|---------|---------|-----|-----|--------|")

    ok_p = err_d3_p < 0.10
    ok_gue_slope = 0.045 <= float(a_gue) <= 0.055
    ok_r = err_r_p < 0.02 and err_r_o < 0.02 and err_r_g < 0.02

    print(f"| Δ₃ vs L/15 (mean rel. err.) | {err_d3_p:.4f} | — | — | {'OK' if ok_p else 'revisar'} |")
    print(f"| dΔ₃/d(log L) OLS *a* | — | {a_goe:.4f} | {a_gue:.4f} | GUE∈[0.045,0.055]: {'OK' if ok_gue_slope else 'marginal'} |")
    print(f"| mean ∂Δ₃/∂(log L) (grad.) | — | {mean_grad_o:.4f} | {mean_grad_g:.4f} | — |")
    print(f"| ⟨Δ₃⟩ GOE < ⟨Δ₃⟩ GUE | — | {'sí' if global_order else 'no'} | — | {'OK' if global_order else 'revisar'} |")
    print(f"| Σ²: tendencia | dΣ²/dL≈{slope_s2_p:.3f} | — | dΣ²/d log L≈{slope_s2_g_log:.4f} | Poisson~lineal; GUE~log |")
    print(f"| P(s) KS | {ks_p:.4f} | {ks_goe:.4f} | {ks_gue:.4f} | p: {p_p:.2e} / {p_goe:.2e} / {p_gue:.2e} |")
    print(f"| P(s) L² | {l2_p:.4f} | {l2_goe:.4f} | {l2_gue:.4f} | menor mejor |")
    print(f"| ⟨r⟩ (mean 10 real.) | {mr_p:.4f} | {mr_o:.4f} | {mr_g:.4f} | err vs exacto |")
    print(f"| ⟨r⟩ |error| | {err_r_p:.4f} | {err_r_o:.4f} | {err_r_g:.4f} | {'OK' if ok_r else 'revisar'} |")
    print(f"| SFF: dip / ramp / plat. | dip={fp[0]} | dip={fo[0]} | dip={fg[0]} | cualitativo |")

    checks = [
        ok_p,
        ok_gue_slope,
        global_order,
        slope_s2_g_log > 0,
        0.7 < slope_s2_p < 1.25,
        ok_r,
    ]
    n_ok = sum(checks)
    if n_ok >= 5:
        diag = "CONSISTENTE con RMT (régimen finito)"
    elif n_ok >= 3:
        diag = "PARCIALMENTE CONSISTENTE"
    else:
        diag = "INCONSISTENTE (revisar muestra / ventanas)"

    print("\n### DIAGNÓSTICO\n")
    print(diag)
    print(f"(checks heurísticos: {n_ok}/{len(checks)})")
    print("\nNotas:")
    print("- Δ₃: no se exige coincidencia con 1/π²; pendiente efectiva ~0.05 en L finito es esperable.")
    print("- SFF: dip/ramp/plateau dependen de N y normalización K/K(0).")
    print("- Métricas robustas: ⟨r⟩, P(s) KS. Inestables: pendiente Δ₃ GOE vs GUE en ventana corta.")

    print("\n### EXTRA: estabilidad pendiente OLS Δ₃ (GUE) vs N\n")
    print("| N | a (d₃ ~ a log L + b) |")
    print("|---|--:|")
    for n_sz in (1000, 2000, 4000):
        d3 = mean_delta3_curve(lambda r, nn: bulk_matrix_ensemble(r, nn, "gue"), RNG_SEED + 800 + n_sz, n_sz)
        a, _ = np.polyfit(np.log(L_GRID), d3, 1)
        print(f"| {n_sz} | {a:.6f} |")

    print("\nListo. Ver docs/VALIDATION_RMT.md.")


if __name__ == "__main__":
    main()
