"""
BaselineFactory: genera baselines empíricos (GUE, Poisson) para comparación.
Produce distribuciones de métricas bajo cada baseline para Z-scores sin sesgo.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..data.generators import generar_gue_normalizado, generar_poisson
from ..analysis.unfolding import unfolding_riemann
from ..analysis.rigidity import (
    espaciado_minimo,
    varianza_numero,
    delta3_dyson_mehta,
)


def _metric_d_min(gamma: np.ndarray) -> float:
    u = unfolding_riemann(gamma)
    if not np.all(np.isfinite(u)):
        u = gamma
    d, _ = espaciado_minimo(u)
    return d


def _metric_varianza(gamma: np.ndarray, L: float = 2.0) -> float:
    u = unfolding_riemann(gamma)
    if not np.all(np.isfinite(u)):
        return np.nan
    return varianza_numero(u, L)


def _metric_delta3(gamma: np.ndarray, L: float = 2.0) -> float:
    u = unfolding_riemann(gamma)
    if not np.all(np.isfinite(u)):
        return np.nan
    return delta3_dyson_mehta(u, L)


class BaselineFactory:
    """
    Genera baselines empíricos para métricas de rigidez.
    Permite comparar datos reales (p. ej. Riemann) contra GUE y Poisson
    mediante distribuciones de la métrica sobre muchas realizaciones.
    """

    def __init__(
        self,
        num_realizaciones_gue: int = 100,
        num_realizaciones_poisson: int = 100,
        L_varianza: float = 2.0,
        L_delta3: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.num_gue = num_realizaciones_gue
        self.num_poisson = num_realizaciones_poisson
        self.L_varianza = L_varianza
        self.L_delta3 = L_delta3
        self._seed = seed

    def _sample_gue(self, N: int, realizaciones: int) -> List[np.ndarray]:
        out = []
        for i in range(realizaciones):
            s = self._seed + i if self._seed is not None else None
            out.append(generar_gue_normalizado(N, seed=s))
        return out

    def _sample_poisson(self, N: int, realizaciones: int) -> List[np.ndarray]:
        out = []
        for i in range(realizaciones):
            s = self._seed + i if self._seed is not None else None
            out.append(generar_poisson(N, seed=s))
        return out

    def baseline_d_min(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (media_gue, std_gue), (media_poisson, std_poisson) para d_min.
        En la práctica se devuelve (array de valores GUE, array de valores Poisson).
        """
        gue_samples = self._sample_gue(N, self.num_gue)
        poi_samples = self._sample_poisson(N, self.num_poisson)
        gue_vals = np.array([_metric_d_min(g) for g in gue_samples])
        poi_vals = np.array([_metric_d_min(p) for p in poi_samples])
        return gue_vals, poi_vals

    def baseline_varianza_numero(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Baselines para varianza del número (ventana L_varianza)."""
        gue_samples = self._sample_gue(N, self.num_gue)
        poi_samples = self._sample_poisson(N, self.num_poisson)
        gue_vals = np.array([_metric_varianza(g, self.L_varianza) for g in gue_samples])
        poi_vals = np.array([_metric_varianza(p, self.L_varianza) for p in poi_samples])
        return gue_vals, poi_vals

    def baseline_delta3(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Baselines para Delta_3(L_delta3)."""
        gue_samples = self._sample_gue(N, self.num_gue)
        poi_samples = self._sample_poisson(N, self.num_poisson)
        gue_vals = np.array([_metric_delta3(g, self.L_delta3) for g in gue_samples])
        poi_vals = np.array([_metric_delta3(p, self.L_delta3) for p in poi_samples])
        return gue_vals, poi_vals

    def get_baseline_stats(
        self,
        N: int,
        metrica: str = "d_min",
    ) -> Dict[str, Dict[str, float]]:
        """
        Estadísticos de baseline (media, std) para GUE y Poisson.
        métrica: 'd_min', 'varianza_numero', 'delta3'
        """
        if metrica == "d_min":
            gue_vals, poi_vals = self.baseline_d_min(N)
        elif metrica == "varianza_numero":
            gue_vals, poi_vals = self.baseline_varianza_numero(N)
        elif metrica == "delta3":
            gue_vals, poi_vals = self.baseline_delta3(N)
        else:
            raise ValueError(f"métrica desconocida: {metrica}")

        gue_vals = gue_vals[np.isfinite(gue_vals)]
        poi_vals = poi_vals[np.isfinite(poi_vals)]
        return {
            "gue": {"mean": float(np.mean(gue_vals)), "std": float(np.std(gue_vals)) if len(gue_vals) > 1 else 0.0},
            "poisson": {"mean": float(np.mean(poi_vals)), "std": float(np.std(poi_vals)) if len(poi_vals) > 1 else 0.0},
        }
