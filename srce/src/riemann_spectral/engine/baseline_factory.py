"""
BaselineFactory: genera baselines empíricos (GUE, Poisson) para comparación.
Produce distribuciones de métricas bajo cada baseline para Z-scores sin sesgo.

CORRECCIÓN (v1.1):
    Las métricas GUE ahora usan unfolding_wigner_gue + tercio central, que es el
    procedimiento correcto para matrices GUE. unfolding_riemann() es específico
    de los ceros de Riemann (fórmula N(T) de von Mangoldt) y NO debe aplicarse
    a autovalores GUE. Poisson usa unfolding empírico por CDF uniforme.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..data.generators import generar_gue_normalizado, generar_poisson
from ..analysis.unfolding import unfolding_riemann, unfolding_wigner_gue
from ..analysis.rigidity import (
    espaciado_minimo,
    varianza_numero,
    delta3_dyson_mehta,
)


# ── Helpers de unfolding por tipo de ensemble ────────────────────────────────

def _unfold_gue(gamma: np.ndarray) -> np.ndarray:
    """
    Unfolding correcto para espectros GUE.

    Pipeline:
        1. unfolding_wigner_gue: aplica la CDF del semicírculo de Wigner,
           produciendo densidad ≈ 1 en el bulk.
        2. Tercio central: descarta el primer y último tercio para evitar
           los efectos de borde del semicírculo (donde la densidad cae a 0).

    Returns:
        Array unfolded del tercio central, re-centrado a 0.
        Puede ser vacío si N es muy pequeño.
    """
    u = unfolding_wigner_gue(gamma)
    n = len(u)
    start = n // 3
    end = 2 * (n // 3)
    if end <= start:
        return np.empty(0)
    central = u[start:end]
    return central - central[0]  # re-centrar para que ventanas empiecen en 0


def _unfold_poisson(gamma: np.ndarray) -> np.ndarray:
    """
    Unfolding empírico para Poisson: rescala por la CDF empírica de modo que
    el espaciado medio sea 1. Equivalente a rangos / (N-1) * (N-1).

    Para Poisson uniforme en [0, N], los valores ya tienen densidad ≈ 1,
    así que devolvemos gamma directamente (ya está en escala natural).
    """
    if not np.all(np.isfinite(gamma)):
        return np.empty(0)
    # generar_poisson produce puntos uniformes en [0, N], espaciado medio ≈ 1
    return gamma


def _unfold_riemann(gamma: np.ndarray) -> np.ndarray:
    """Unfolding Riemann–von Mangoldt. Solo usar para ceros de Riemann."""
    u = unfolding_riemann(gamma)
    if not np.all(np.isfinite(u)):
        return np.empty(0)
    return u


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
    if len(u) < 2:
        return np.nan
    return varianza_numero(u, L)


def _metric_varianza_poisson(gamma: np.ndarray, L: float = 2.0) -> float:
    u = _unfold_poisson(gamma)
    if len(u) < 2:
        return np.nan
    return varianza_numero(u, L)


def _metric_delta3_gue(gamma: np.ndarray, L: float = 2.0) -> float:
    u = _unfold_gue(gamma)
    if len(u) < 2:
        return np.nan
    return delta3_dyson_mehta(u, L)


def _metric_delta3_poisson(gamma: np.ndarray, L: float = 2.0) -> float:
    u = _unfold_poisson(gamma)
    if len(u) < 2:
        return np.nan
    return delta3_dyson_mehta(u, L)


# ── BaselineFactory ───────────────────────────────────────────────────────────

class BaselineFactory:
    """
    Genera baselines empíricos para métricas de rigidez.
    Permite comparar datos reales (p. ej. Riemann) contra GUE y Poisson
    mediante distribuciones de la métrica sobre muchas realizaciones.

    Nota sobre unfolding:
        - GUE   → unfolding_wigner_gue() + tercio central (correcto para RMT)
        - Poisson → sin transformar (ya en escala con densidad ≈ 1)
        - Riemann → unfolding_riemann() (N(T) von Mangoldt, solo para ceros)
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
        """Genera realizaciones GUE. Pide N más grande para que el tercio central tenga tamaño ~N."""
        # Generamos 3*N autovalores para que el tercio central tenga ~N puntos
        N_gue = max(N * 3, 30)
        out = []
        for i in range(realizaciones):
            s = self._seed + i if self._seed is not None else None
            out.append(generar_gue_normalizado(N_gue, seed=s))
        return out

    def _sample_poisson(self, N: int, realizaciones: int) -> List[np.ndarray]:
        out = []
        for i in range(realizaciones):
            s = self._seed + i if self._seed is not None else None
            out.append(generar_poisson(N, seed=s))
        return out

    def baseline_d_min(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Distribución empírica de d_min bajo GUE y Poisson.
        Retorna (array_GUE, array_Poisson) de valores de d_min por realización.
        """
        gue_samples = self._sample_gue(N, self.num_gue)
        poi_samples = self._sample_poisson(N, self.num_poisson)
        gue_vals = np.array([_metric_d_min_gue(g) for g in gue_samples])
        poi_vals = np.array([_metric_d_min_poisson(p) for p in poi_samples])
        return gue_vals, poi_vals

    def baseline_varianza_numero(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Baselines para varianza del número (ventana L_varianza)."""
        gue_samples = self._sample_gue(N, self.num_gue)
        poi_samples = self._sample_poisson(N, self.num_poisson)
        gue_vals = np.array([_metric_varianza_gue(g, self.L_varianza) for g in gue_samples])
        poi_vals = np.array([_metric_varianza_poisson(p, self.L_varianza) for p in poi_samples])
        return gue_vals, poi_vals

    def baseline_delta3(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """Baselines para Delta_3(L_delta3)."""
        gue_samples = self._sample_gue(N, self.num_gue)
        poi_samples = self._sample_poisson(N, self.num_poisson)
        gue_vals = np.array([_metric_delta3_gue(g, self.L_delta3) for g in gue_samples])
        poi_vals = np.array([_metric_delta3_poisson(p, self.L_delta3) for p in poi_samples])
        return gue_vals, poi_vals

    def get_baseline_stats(
        self,
        N: int,
        metrica: str = "d_min",
    ) -> Dict[str, Dict[str, float]]:
        """
        Estadísticos de baseline (media, std) para GUE y Poisson.
        metrica: 'd_min', 'varianza_numero', 'delta3'
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
            "gue": {
                "mean": float(np.mean(gue_vals)),
                "std": float(np.std(gue_vals)) if len(gue_vals) > 1 else 0.0,
            },
            "poisson": {
                "mean": float(np.mean(poi_vals)),
                "std": float(np.std(poi_vals)) if len(poi_vals) > 1 else 0.0,
            },
        }
