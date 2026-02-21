"""
Z-Score Engine: detección de anomalías comparando datos reales contra baselines empíricos.
Umbral de alta confianza (p. ej. 5 sigma) para registrar hallazgos.

CORRECCIÓN (v1.1):
    Las métricas del espectro REAL (Riemann) siguen usando unfolding_riemann(), que es
    correcto para los ceros de Riemann. El unfolding de los baselines GUE/Poisson es
    responsabilidad de BaselineFactory (ya corregido a Wigner/empírico). ZScoreEngine
    solo necesita ser consistente en cómo trata el espectro de entrada.
"""

from typing import Dict, List, Optional

import numpy as np

from ..analysis.unfolding import unfolding_riemann
from ..analysis.rigidity import espaciado_minimo, varianza_numero, delta3_dyson_mehta
from .baseline_factory import BaselineFactory


class ZScoreEngine:
    """
    Compara métricas de un espectro real (p. ej. Riemann) contra baselines
    GUE/Poisson generados por BaselineFactory. Calcula Z-scores y detecta anomalías.

    Protocolo de unfolding:
        - Espectro de entrada (gamma): se asume que son ceros de Riemann (partes
          imaginarias), por lo que se aplica unfolding_riemann() (N(T) von Mangoldt).
          Si se pasa otro tipo de espectro, el llamador es responsable de hacer
          el unfolding previo y pasar un espectro ya en escala de densidad ≈ 1.
        - Baselines GUE/Poisson: el unfolding correcto (Wigner/empírico) es aplicado
          internamente por BaselineFactory. No se toca aquí.

    Consistencia garantizada:
        El unfolding del espectro real y el de los baselines producen ahora
        distribuciones en la misma escala (densidad local ≈ 1), haciendo los
        Z-scores estadísticamente válidos.
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

    # ── Métodos de métrica para el espectro REAL ─────────────────────────────

    def _unfold_entrada(self, gamma: np.ndarray) -> Optional[np.ndarray]:
        """
        Aplica unfolding_riemann al espectro de entrada y valida el resultado.
        Retorna None si el unfolding produce valores no finitos.
        """
        u = unfolding_riemann(gamma)
        if not np.all(np.isfinite(u)):
            return None
        return u

    def _valor_d_min(self, gamma: np.ndarray) -> float:
        u = self._unfold_entrada(gamma)
        if u is None:
            return np.nan
        d, _ = espaciado_minimo(u)
        return float(d)

    def _valor_varianza(self, gamma: np.ndarray) -> float:
        u = self._unfold_entrada(gamma)
        if u is None:
            return np.nan
        return varianza_numero(u, self.L_varianza)

    def _valor_delta3(self, gamma: np.ndarray) -> float:
        u = self._unfold_entrada(gamma)
        if u is None:
            return np.nan
        return delta3_dyson_mehta(u, self.L_delta3)

    # ── Z-score ──────────────────────────────────────────────────────────────

    @staticmethod
    def _z_score(valor: float, media: float, std: float) -> float:
        if std <= 0:
            return 0.0
        return (valor - media) / std

    # ── Evaluación principal ─────────────────────────────────────────────────

    def evaluar(
        self,
        gamma: np.ndarray,
        N: int,
        metricas: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Evalúa el espectro gamma (ceros de Riemann) y devuelve para cada métrica:
            valor         : métrica calculada sobre el espectro real unfolded
            z_gue         : Z-score vs baseline GUE (unfolding Wigner + tercio central)
            z_poisson     : Z-score vs baseline Poisson (escala natural)
            anomalia      : True si |z_gue| >= sigma_umbral o |z_poisson| >= sigma_umbral
            mean_gue      : media del baseline GUE
            std_gue       : desv. estándar del baseline GUE
            mean_poisson  : media del baseline Poisson
            std_poisson   : desv. estándar del baseline Poisson

        Args:
            gamma   : array de partes imaginarias de los ceros de Riemann (NO unfolded).
            N       : número de ceros (len(gamma)); usado para generar baselines del mismo tamaño.
            metricas: lista de métricas a evaluar. Por defecto ['d_min', 'varianza_numero', 'delta3'].
        """
        if metricas is None:
            metricas = ["d_min", "varianza_numero", "delta3"]

        resultados: Dict[str, Dict] = {}

        for met in metricas:
            # Calcular valor sobre espectro real
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
                resultados[met] = {
                    "valor": np.nan,
                    "z_gue": np.nan,
                    "z_poisson": np.nan,
                    "anomalia": False,
                }
                continue

            # Limpiar NaN de baselines (realizaciones fallidas)
            gue_vals = gue_vals[np.isfinite(gue_vals)]
            poi_vals = poi_vals[np.isfinite(poi_vals)]

            mean_gue = float(np.mean(gue_vals)) if len(gue_vals) > 0 else np.nan
            std_gue  = float(np.std(gue_vals))  if len(gue_vals) > 1 else 1e-10
            mean_poi = float(np.mean(poi_vals)) if len(poi_vals) > 0 else np.nan
            std_poi  = float(np.std(poi_vals))  if len(poi_vals) > 1 else 1e-10

            z_gue     = self._z_score(valor, mean_gue, std_gue)
            z_poisson = self._z_score(valor, mean_poi, std_poi)
            anomalia  = abs(z_gue) >= self.sigma_umbral or abs(z_poisson) >= self.sigma_umbral

            resultados[met] = {
                "valor":        float(valor),
                "z_gue":        float(z_gue),
                "z_poisson":    float(z_poisson),
                "anomalia":     anomalia,
                "mean_gue":     mean_gue,
                "std_gue":      std_gue,
                "mean_poisson": mean_poi,
                "std_poisson":  std_poi,
            }

        return resultados

    def hay_anomalia(self, resultados: Dict[str, Dict]) -> bool:
        """True si alguna métrica marcó anomalía."""
        return any(r.get("anomalia", False) for r in resultados.values())
