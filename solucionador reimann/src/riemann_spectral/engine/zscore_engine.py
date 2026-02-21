"""
Z-Score Engine: detección de anomalías comparando datos reales contra baselines empíricos.
Umbral de alta confianza (p. ej. 5 sigma) para registrar hallazgos.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from ..analysis.unfolding import unfolding_riemann
from ..analysis.rigidity import espaciado_minimo, varianza_numero, delta3_dyson_mehta
from .baseline_factory import BaselineFactory


class ZScoreEngine:
    """
    Compara métricas de un espectro real (p. ej. Riemann) contra baselines
    GUE/Poisson generados por BaselineFactory. Calcula Z-scores y detecta anomalías.
    """

    def __init__(
        self,
        baseline_factory: Optional[BaselineFactory] = None,
        sigma_umbral: float = 5.0,
        L_varianza: float = 2.0,
        L_delta3: float = 2.0,
    ):
        self.factory = baseline_factory or BaselineFactory(
            num_realizaciones_gue=100,
            num_realizaciones_poisson=100,
            L_varianza=L_varianza,
            L_delta3=L_delta3,
        )
        self.sigma_umbral = sigma_umbral
        self.L_varianza = L_varianza
        self.L_delta3 = L_delta3

    def _valor_d_min(self, gamma: np.ndarray) -> float:
        u = unfolding_riemann(gamma)
        if not np.all(np.isfinite(u)):
            u = gamma
        d, _ = espaciado_minimo(u)
        return d

    def _valor_varianza(self, gamma: np.ndarray) -> float:
        u = unfolding_riemann(gamma)
        if not np.all(np.isfinite(u)):
            return np.nan
        return varianza_numero(u, self.L_varianza)

    def _valor_delta3(self, gamma: np.ndarray) -> float:
        u = unfolding_riemann(gamma)
        if not np.all(np.isfinite(u)):
            return np.nan
        return delta3_dyson_mehta(u, self.L_delta3)

    @staticmethod
    def _z_score(valor: float, media: float, std: float) -> float:
        if std <= 0:
            return 0.0
        return (valor - media) / std

    def evaluar(
        self,
        gamma: np.ndarray,
        N: int,
        metricas: Optional[list] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evalúa espectro gamma (N ceros) y devuelve para cada métrica:
        valor, z_score vs GUE, z_score vs Poisson, anomalía (bool).
        """
        if metricas is None:
            metricas = ["d_min", "varianza_numero", "delta3"]
        resultados = {}
        for met in metricas:
            if met == "d_min":
                valor = self._valor_d_min(gamma)
                gue_vals, poi_vals = self.factory.baseline_d_min(N)
            elif met == "varianza_numero":
                valor = self._valor_varianza(gamma)
                gue_vals, poi_vals = self.factory.baseline_varianza_numero(N)
            elif met == "delta3":
                valor = self._valor_delta3(gamma)
                gue_vals, poi_vals = self.factory.baseline_delta3(N)
            else:
                continue
            if not np.isfinite(valor):
                resultados[met] = {"valor": np.nan, "z_gue": np.nan, "z_poisson": np.nan, "anomalia": False}
                continue
            gue_vals = gue_vals[np.isfinite(gue_vals)]
            poi_vals = poi_vals[np.isfinite(poi_vals)]
            mean_gue = np.mean(gue_vals)
            std_gue = np.std(gue_vals) if len(gue_vals) > 1 else 1e-10
            mean_poi = np.mean(poi_vals)
            std_poi = np.std(poi_vals) if len(poi_vals) > 1 else 1e-10
            z_gue = self._z_score(valor, mean_gue, std_gue)
            z_poisson = self._z_score(valor, mean_poi, std_poi)
            anomalia = abs(z_gue) >= self.sigma_umbral or abs(z_poisson) >= self.sigma_umbral
            resultados[met] = {
                "valor": float(valor),
                "z_gue": float(z_gue),
                "z_poisson": float(z_poisson),
                "anomalia": anomalia,
                "mean_gue": float(mean_gue),
                "std_gue": float(std_gue),
                "mean_poisson": float(mean_poi),
                "std_poisson": float(std_poi),
            }
        return resultados

    def hay_anomalia(self, resultados: Dict[str, Dict[str, float]]) -> bool:
        """True si alguna métrica marcó anomalía."""
        return any(r.get("anomalia", False) for r in resultados.values())
