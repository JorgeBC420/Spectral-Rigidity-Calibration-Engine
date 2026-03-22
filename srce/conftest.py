# -*- coding: utf-8 -*-
"""
tests/conftest.py
=================
Configuración global de pytest. Fixtures compartidas entre todos los módulos.
RNG: np.random.default_rng en todas las fixtures — sin estado global.

Nota de auditoría (2026-03-10):
  - Normalización GOE/GUE: (A + A.T) / (2*sqrt(N))  ← CORRECTO para esta CDF
    unfolding_wigner_gue usa semicírculo en [-2, 2], por lo que el radio
    natural de H = (A+A†)/(2√N) ≈ 2 es exactamente el esperado → <s> ≈ 1.
    La alternativa /sqrt(2N) produce radio ≈ 2.8 → <s> ≈ 1.4 (incorrecto).
  - normalize_spacing agregado como salvaguarda post-unfolding.
"""

import os
import sys

import numpy as np
import pytest
import scipy.linalg as la

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from riemann_spectral.analysis.unfolding  import unfolding_wigner_gue
from riemann_spectral.analysis.normalize  import normalize_spacing


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test lento (diagonalización grande)")
    config.addinivalue_line("markers", "rmt: test de teoría de matrices aleatorias")
    config.addinivalue_line("markers", "integration: test de integración end-to-end")


# ── Fixtures de sesión ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng_session():
    """RNG de sesión con seed fija. Usar solo para datos auxiliares rápidos."""
    return np.random.default_rng(seed=2025)


@pytest.fixture(scope="session")
def poisson_unfolded():
    """
    Proceso Poisson densidad 1 largo — generado UNA VEZ para toda la sesión.
    cumsum(Exp(1)) → tercio central → re-centrado. Ya en escala unfolded.
    """
    rng      = np.random.default_rng(seed=42)
    longitud = 5000.0
    espacios = rng.exponential(1.0, size=int(longitud * 1.3))
    pos      = np.cumsum(espacios)
    pos      = pos[pos <= longitud]
    n        = len(pos)
    u        = pos[n // 3: 2 * (n // 3)] - pos[n // 3]
    return u


@pytest.fixture(scope="session")
def gue_unfolded():
    """
    GUE N grande → unfolding Wigner → tercio central → normalize_spacing.

    Normalización: H = (A + A†) / (2*sqrt(N))
    Produce radio del semicírculo ≈ 2.0, compatible con unfolding_wigner_gue
    que usa CDF del semicírculo en [-2, 2] con sigma=1.
    Resultado: <s> ≈ 1.0 después del unfolding (verificado empíricamente).
    """
    rng = np.random.default_rng(seed=99)
    N   = 2000
    A   = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H   = (A + A.conj().T) / (2 * np.sqrt(N))
    ev  = np.sort(la.eigvalsh(H))
    u   = unfolding_wigner_gue(ev)
    n   = len(u)
    central = u[n // 3: 2 * (n // 3)]
    central = normalize_spacing(central)              # salvaguarda: forzar <s>=1
    return central - central[0]


@pytest.fixture(scope="session")
def goe_unfolded():
    """
    GOE N grande → unfolding Wigner → tercio central → normalize_spacing.

    Normalización: H = (A + A.T) / (2*sqrt(N))
    Misma lógica que GUE: radio ≈ 2.0, compatible con unfolding_wigner_gue.
    """
    rng = np.random.default_rng(seed=7)
    N   = 2000
    A   = rng.standard_normal((N, N))
    H   = (A + A.T) / (2 * np.sqrt(N))
    ev  = np.sort(la.eigvalsh(H))
    u   = unfolding_wigner_gue(ev)
    n   = len(u)
    central = u[n // 3: 2 * (n // 3)]
    central = normalize_spacing(central)              # salvaguarda: forzar <s>=1
    return central - central[0]


@pytest.fixture(scope="session")
def poisson_ensemble_promedio():
    """
    Promedio de Delta_3 sobre 60 realizaciones Poisson.
    Devuelve (d3_mean, L_grid) para test parametrizado de L/15.
    """
    from riemann_spectral.analysis.rigidity import delta3_dyson_mehta

    rng    = np.random.default_rng(seed=11)
    L_grid = np.array([5.0, 10.0, 20.0, 30.0])
    acum   = {L: [] for L in L_grid}

    for _ in range(60):
        espacios = rng.exponential(1.0, size=8000)
        pos      = np.cumsum(espacios)
        pos      = pos[pos <= 6000.0]
        n        = len(pos)
        u        = pos[n // 3: 2 * (n // 3)] - pos[n // 3]
        if len(u) < 50:
            continue
        for L in L_grid:
            val = delta3_dyson_mehta(u, L)
            if np.isfinite(val):
                acum[L].append(val)

    d3_mean = np.array([
        float(np.mean(acum[L])) if acum[L] else np.nan
        for L in L_grid
    ])
    return d3_mean, L_grid
