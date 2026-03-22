# -*- coding: utf-8 -*-
"""
Validación contra predicciones teóricas (P(s), ⟨r⟩, etc.).

Ejecutar desde ``srce``::
    PYTHONPATH=src pytest test_theoretical_validation.py -v

Nota: ``R_GUE_EXACT`` es la constante tabulada (~0.60272); la
expresión `(27/4) - 6√3` del zip original es incorrecta (negativa).
"""

import os
import sys

import numpy as np
import pytest
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from riemann_spectral.data.generators import (
    generar_poisson,
    generar_gue_normalizado,
    generar_goe_normalizado,
)
from riemann_spectral.analysis.normalize import normalize_spacing


# Constantes exactas (Atas et al., 2013 — GUE como valor numérico estándar)
R_POISSON_EXACT = 2 * np.log(2) - 1
R_GOE_EXACT = 4 - 2 * np.sqrt(3)
R_GUE_EXACT = 0.60272166211556


class TheoreticalValidationTest:
    """Utilidades comunes para tests teóricos."""

    @staticmethod
    def compute_spacing_distribution(spectrum: np.ndarray, n_bins: int = 50):
        spacings = np.diff(spectrum)
        spacings = spacings / np.mean(spacings)
        hist, bins = np.histogram(spacings, bins=n_bins, density=True)
        centers = (bins[:-1] + bins[1:]) / 2
        return centers, hist

    @staticmethod
    def statistical_error_l1(empirical: np.ndarray, theoretical: np.ndarray) -> float:
        return float(np.mean(np.abs(empirical - theoretical)))

    @staticmethod
    def statistical_error_ks(data: np.ndarray, cdf_func):
        result = stats.kstest(data, cdf_func)
        return result.statistic, result.pvalue


class TestPoissonSpacing(TheoreticalValidationTest):
    def test_poisson_spacing_distribution(self):
        rng = np.random.default_rng(seed=42)
        N = 5000
        levels = generar_poisson(N, rng=rng)
        centers, hist = self.compute_spacing_distribution(levels, n_bins=50)
        theory = np.exp(-centers)
        error = self.statistical_error_l1(hist, theory)
        spacings = np.diff(levels) / np.mean(np.diff(levels))
        _, p_value = self.statistical_error_ks(
            spacings,
            lambda x: 1 - np.exp(-x),
        )
        assert error < 0.15
        assert p_value > 0.01

    def test_poisson_spacing_variance(self):
        rng = np.random.default_rng(seed=43)
        N = 10000
        levels = generar_poisson(N, rng=rng)
        spacings = np.diff(levels) / np.mean(np.diff(levels))
        var_observed = np.var(spacings)
        assert abs(var_observed - 1.0) < 0.10


class TestGUESpacing(TheoreticalValidationTest):
    @staticmethod
    def wigner_surmise_gue(s: np.ndarray) -> np.ndarray:
        return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)

    def test_gue_spacing_wigner_surmise(self):
        rng = np.random.default_rng(seed=99)
        N = 1200
        eigenvalues = generar_gue_normalizado(N, rng=rng)
        spectrum = normalize_spacing(eigenvalues)
        centers, hist = self.compute_spacing_distribution(spectrum, n_bins=60)
        theory = self.wigner_surmise_gue(centers)
        error = self.statistical_error_l1(hist, theory)
        assert error < 0.20

    def test_gue_spacing_repulsion_exponent(self):
        rng = np.random.default_rng(seed=100)
        N = 2000
        eigenvalues = generar_gue_normalizado(N, rng=rng)
        spectrum = normalize_spacing(eigenvalues)
        spacings = np.diff(spectrum)
        small_spacings = spacings[spacings < 0.1]
        fraction = len(small_spacings) / len(spacings)
        expected_fraction = 0.0013
        assert abs(fraction - expected_fraction) < 0.005


class TestGOESpacing(TheoreticalValidationTest):
    @staticmethod
    def wigner_surmise_goe(s: np.ndarray) -> np.ndarray:
        return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)

    def test_goe_spacing_wigner_surmise(self):
        rng = np.random.default_rng(seed=7)
        N = 1200
        eigenvalues = generar_goe_normalizado(N, rng=rng)
        spectrum = normalize_spacing(eigenvalues)
        centers, hist = self.compute_spacing_distribution(spectrum, n_bins=60)
        theory = self.wigner_surmise_goe(centers)
        error = self.statistical_error_l1(hist, theory)
        assert error < 0.20

    def test_goe_spacing_repulsion_exponent(self):
        rng = np.random.default_rng(seed=8)
        N = 2000
        eigenvalues = generar_goe_normalizado(N, rng=rng)
        spectrum = normalize_spacing(eigenvalues)
        spacings = np.diff(spectrum)
        small_spacings = spacings[spacings < 0.1]
        fraction = len(small_spacings) / len(spacings)
        expected_fraction = 0.0039
        assert abs(fraction - expected_fraction) < 0.008


class TestRParameterExact(TheoreticalValidationTest):
    @staticmethod
    def compute_r_parameter(spectrum: np.ndarray) -> float:
        spacings = np.diff(spectrum)
        s_i = spacings[:-1]
        s_i1 = spacings[1:]
        r_vals = np.minimum(s_i, s_i1) / np.maximum(s_i, s_i1)
        return float(np.mean(r_vals))

    def test_poisson_r_exact(self):
        rng = np.random.default_rng(seed=44)
        N = 10000
        levels = generar_poisson(N, rng=rng)
        r_obs = self.compute_r_parameter(levels)
        error = abs(r_obs - R_POISSON_EXACT) / R_POISSON_EXACT
        assert error < 0.03

    def test_gue_r_exact(self):
        rng = np.random.default_rng(seed=101)
        N = 2000
        eigenvalues = generar_gue_normalizado(N, rng=rng)
        spectrum = normalize_spacing(eigenvalues)
        r_obs = self.compute_r_parameter(spectrum)
        error = abs(r_obs - R_GUE_EXACT) / R_GUE_EXACT
        assert error < 0.05

    def test_goe_r_exact(self):
        rng = np.random.default_rng(seed=9)
        N = 2000
        eigenvalues = generar_goe_normalizado(N, rng=rng)
        spectrum = normalize_spacing(eigenvalues)
        r_obs = self.compute_r_parameter(spectrum)
        error = abs(r_obs - R_GOE_EXACT) / R_GOE_EXACT
        assert error < 0.05


@pytest.mark.parametrize("N", [200, 500, 1000, 2000, 5000])
class TestConvergenceWithN(TheoreticalValidationTest):
    def test_poisson_convergence(self, N):
        rng = np.random.default_rng(seed=42)
        levels = generar_poisson(N, rng=rng)
        centers, hist = self.compute_spacing_distribution(levels, n_bins=40)
        theory = np.exp(-centers)
        error = self.statistical_error_l1(hist, theory)
        expected_error = 3.0 / np.sqrt(N)
        assert error < 2 * expected_error
