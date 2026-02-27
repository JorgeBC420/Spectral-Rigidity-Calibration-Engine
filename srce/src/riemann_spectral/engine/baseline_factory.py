# -*- coding: utf-8 -*-
"""
src/riemann_spectral/engine/baseline_factory.py
================================================
BaselineFactory: genera baselines empíricos (GUE, Poisson) para comparación.
Produce distribuciones de métricas bajo cada baseline para Z-scores sin sesgo.

Historial:
    v1.0 — versión original (unfolding incorrecto para GUE)
    v1.1 — corrección: GUE usa unfolding_wigner_gue + tercio central
    v1.2 — migración completa a np.random.default_rng (sin estado global)
            Usa SeedSequence para derivar sub-semillas independientes.
            Eliminada dependencia de np.random.seed() global.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

from ..analysis.unfolding import unfolding_riemann, unfolding_wigner_gue
from ..analysis.rigidity import (
    espaciado_minimo,
    varianza_numero,
    delta3_dyson_mehta,
)


# ── Generadores internos con default_rng (sin estado global) ─────────────────

def _generar_gue(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    GUE via Generator. H = (A + A†) / (2√N), A_ij complejo i.i.d. N(0,1).
    Autovalores en semicírculo de Wigner, re-escalados a media=N/2, std=1.
    """
    A  = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H  = (A + A.conj().T) / (2 * np.sqrt(N))
    ev = la.eigvalsh(H)
    media = np.mean(ev)
    desv  = np.std(ev)
    if desv < 1e-10:
        desv = 1.0
    return np.sort((ev - media) / desv + N / 2)


def _generar_poisson(N: int, rng: np.random.Generator) -> np.ndarray:
    """Poisson uniforme en [0, N]. Espaciado medio ≈ 1 → densidad ≈ 1."""
    return np.sort(rng.uniform(0.0, float(N), N))


# ── Helpers de unfolding por tipo de ensemble ────────────────────────────────

def _unfold_gue(gamma: np.ndarray) -> np.ndarray:
    """
    Unfolding correcto para GUE: CDF semicírculo de Wigner + tercio central.
    Pipeline: unfolding_wigner_gue → tercio central → re-centrar a 0.
    """
    u = unfolding_wigner_gue(gamma)
    n = len(u)
    start, end = n // 3, 2 * (n // 3)
    if end <= start:
        return np.empty(0)
    central = u[start:end]
    return central - central[0]


def _unfold_poisson(gamma: np.ndarray) -> np.ndarray:
    """Poisson en [0,N]: densidad ≈ 1 por construcción, sin transformación."""
    if not np.all(np.isfinite(gamma)):
        return np.empty(0)
    return gamma


def _unfold_riemann(gamma: np.ndarray) -> np.ndarray:
    """Unfolding von Mangoldt. Solo para ceros de Riemann."""
    u = unfolding_riemann(gamma)
    return u if np.all(np.isfinite(u)) else np.empty(0)


# ── Funciones de métrica por ensemble ────────────────────────────────────────

def _metric_d_min_gue(gamma: np.ndarray) -> float:
    u = _unfold_gue(gamma)
    if len(u) < 2:
        return np.nan
    d, _ = espaciado_minimo(u)
    return float(d)


def _metric_d_min_poisson(gamma: np.ndarray) -> float:
    u = _unfold_poisson(gamma)
    if len(u) < 2:
        return np.nan
    d, _ = espaciado_minimo(u)
    return float(d)


def _metric_varianza_gue(gamma: np.ndarray, L: float = 2.0) -> float:
    u = _unfold_gue(gamma)
    return varianza_numero(u, L) if len(u) >= 2 else np.nan


def _metric_varianza_poisson(gamma: np.ndarray, L: float = 2.0) -> float:
    u = _unfold_poisson(gamma)
    return varianza_numero(u, L) if len(u) >= 2 else np.nan


def _metric_delta3_gue(gamma: np.ndarray, L: float = 2.0) -> float:
    u = _unfold_gue(gamma)
    return delta3_dyson_mehta(u, L) if len(u) >= 2 else np.nan


def _metric_delta3_poisson(gamma: np.ndarray, L: float = 2.0) -> float:
    u = _unfold_poisson(gamma)
    return delta3_dyson_mehta(u, L) if len(u) >= 2 else np.nan


# ── BaselineFactory ───────────────────────────────────────────────────────────

class BaselineFactory:
    """
    Genera baselines empíricos para métricas de rigidez.
    Compara datos reales (Riemann) contra GUE y Poisson mediante
    distribuciones empíricas de cada métrica sobre muchas realizaciones.

    RNG (v1.2):
        Usa np.random.SeedSequence + default_rng exclusivamente.
        Cada realización recibe su propio Generator derivado e independiente:
            - Sin estado global: np.random.seed() nunca se llama
            - Reproducible: misma seed → mismos resultados siempre
            - Thread-safe: cada Generator es independiente

    Unfolding por ensemble:
        GUE     → unfolding_wigner_gue() + tercio central
        Poisson → sin transformar (densidad ≈ 1 por construcción)
        Riemann → unfolding_riemann() [aplicado en ZScoreEngine, no aquí]

    Args:
        num_realizaciones_gue    : realizaciones GUE por llamada a baseline_*.
        num_realizaciones_poisson: realizaciones Poisson por llamada.
        L_varianza               : ventana L para varianza del número.
        L_delta3                 : ventana L para Delta_3.
        seed                     : semilla maestra (int o None = aleatorio).
    """

    def __init__(
        self,
        num_realizaciones_gue:     int           = 100,
        num_realizaciones_poisson: int           = 100,
        L_varianza:                float         = 2.0,
        L_delta3:                  float         = 2.0,
        seed:                      Optional[int] = None,
    ):
        self.num_gue     = num_realizaciones_gue
        self.num_poisson = num_realizaciones_poisson
        self.L_varianza  = L_varianza
        self.L_delta3    = L_delta3
        # SeedSequence: derivar sub-semillas reproducibles e independientes
        self._seed_seq   = np.random.SeedSequence(seed)

    def _make_rngs(self, n: int) -> List[np.random.Generator]:
        """
        Produce n Generators independientes derivados de la semilla maestra.
        Deterministas si seed != None; cada llamada produce Generators frescos.
        """
        return [np.random.default_rng(s) for s in self._seed_seq.spawn(n)]

    def _sample_gue(self, N: int, realizaciones: int) -> List[np.ndarray]:
        """
        Genera `realizaciones` espectros GUE de tamaño N_gue = 3*N.
        El tercio central de cada uno tendrá ~N puntos útiles.
        """
        N_gue = max(N * 3, 30)
        return [_generar_gue(N_gue, rng) for rng in self._make_rngs(realizaciones)]

    def _sample_poisson(self, N: int, realizaciones: int) -> List[np.ndarray]:
        """Genera `realizaciones` espectros Poisson de N puntos en [0, N]."""
        return [_generar_poisson(N, rng) for rng in self._make_rngs(realizaciones)]

    # ── Baselines públicos ────────────────────────────────────────────────────

    def baseline_d_min(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Distribución empírica de d_min: (array_GUE, array_Poisson)."""
        gue_s = self._sample_gue(N, self.num_gue)
        poi_s = self._sample_poisson(N, self.num_poisson)
        return (
            np.array([_metric_d_min_gue(g)    for g in gue_s]),
            np.array([_metric_d_min_poisson(p) for p in poi_s]),
        )

    def baseline_varianza_numero(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Distribución empírica de varianza del número (ventana L_varianza)."""
        gue_s = self._sample_gue(N, self.num_gue)
        poi_s = self._sample_poisson(N, self.num_poisson)
        return (
            np.array([_metric_varianza_gue(g, self.L_varianza)    for g in gue_s]),
            np.array([_metric_varianza_poisson(p, self.L_varianza) for p in poi_s]),
        )

    def baseline_delta3(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Distribución empírica de Delta_3 (ventana L_delta3)."""
        gue_s = self._sample_gue(N, self.num_gue)
        poi_s = self._sample_poisson(N, self.num_poisson)
        return (
            np.array([_metric_delta3_gue(g, self.L_delta3)    for g in gue_s]),
            np.array([_metric_delta3_poisson(p, self.L_delta3) for p in poi_s]),
        )

    def get_baseline_stats(
        self,
        N:       int,
        metrica: str = "d_min",
    ) -> Dict[str, Dict[str, float]]:
        """
        Estadísticos (media, std) del baseline para GUE y Poisson.
        metrica: 'd_min' | 'varianza_numero' | 'delta3'
        """
        dispatch = {
            "d_min":           self.baseline_d_min,
            "varianza_numero": self.baseline_varianza_numero,
            "delta3":          self.baseline_delta3,
        }
        if metrica not in dispatch:
            raise ValueError(
                f"métrica desconocida: '{metrica}'. Opciones: {list(dispatch.keys())}"
            )
        gue_vals, poi_vals = dispatch[metrica](N)
        gue_vals = gue_vals[np.isfinite(gue_vals)]
        poi_vals = poi_vals[np.isfinite(poi_vals)]
        return {
            "gue": {
                "mean": float(np.mean(gue_vals)),
                "std":  float(np.std(gue_vals)) if len(gue_vals) > 1 else 0.0,
            },
            "poisson": {
                "mean": float(np.mean(poi_vals)),
                "std":  float(np.std(poi_vals)) if len(poi_vals) > 1 else 0.0,
            },
        }
