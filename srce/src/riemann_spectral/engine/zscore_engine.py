# -*- coding: utf-8 -*-
"""
Z-Score Engine: detección de anomalías comparando datos reales contra baselines.

v1.2 — Endurecimiento estadístico:
    evaluar() ahora devuelve, además de z-scores:
        p_gue, p_poisson    : p-valor empírico de dos colas (no paramétrico).
        ic_gue, ic_poisson  : intervalo de confianza 95% del baseline empírico.
        confianza_gue,
        confianza_poi       : 'alto' (p<0.01) | 'medio' (p<0.05) | 'bajo'.

    Compatibilidad hacia atrás: todos los campos anteriores siguen presentes.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..analysis.unfolding import unfolding_riemann
from ..analysis.rigidity  import espaciado_minimo, varianza_numero, delta3_dyson_mehta
from .baseline_factory    import BaselineFactory


# ── Helpers estadísticos ──────────────────────────────────────────────────────

def _p_valor_dos_colas(valor: float, baseline: np.ndarray) -> float:
    """
    P-valor empírico de dos colas (no paramétrico).

    Fracción de realizaciones del baseline cuya distancia a la media del
    baseline es >= la distancia del valor observado a esa misma media.

        p = #{|x_i - mu| >= |valor - mu|} / N

    No asume normalidad. Para baseline < 50 realizaciones el p-valor tiene
    alta varianza — usar con precaución y aumentar num_realizaciones si se
    requieren p-valores confiables.
    """
    if len(baseline) == 0:
        return 1.0
    mu       = np.mean(baseline)
    dist_obs = abs(valor - mu)
    return float(np.mean(np.abs(baseline - mu) >= dist_obs))


def _intervalo_confianza(baseline: np.ndarray) -> Tuple[float, float]:
    """
    Intervalo de confianza 95% empírico (percentiles 2.5 y 97.5).
    Devuelve (nan, nan) si baseline tiene menos de 4 realizaciones.
    """
    if len(baseline) < 4:
        return (np.nan, np.nan)
    return (
        float(np.percentile(baseline, 2.5)),
        float(np.percentile(baseline, 97.5)),
    )


def _clasificar_confianza(p_valor: float) -> str:
    """
    Etiqueta cualitativa según p-valor de dos colas:
        'alto'  : p < 0.01  — evidencia fuerte de diferencia con el baseline
        'medio' : p < 0.05  — evidencia moderada
        'bajo'  : p >= 0.05 — evidencia débil
    """
    if p_valor < 0.01:
        return "alto"
    if p_valor < 0.05:
        return "medio"
    return "bajo"


# ── ZScoreEngine ──────────────────────────────────────────────────────────────

class ZScoreEngine:
    """
    Compara métricas de un espectro real (ceros de Riemann) contra baselines
    GUE/Poisson generados por BaselineFactory.

    Calcula Z-scores, p-valores empíricos, intervalos de confianza y etiquetas.

    Protocolo de unfolding:
        gamma de entrada = partes imaginarias de ceros de Riemann (T > 0).
        unfolding_riemann() se aplica internamente.
        Unfolding de baselines GUE/Poisson = responsabilidad de BaselineFactory.
    """

    def __init__(
        self,
        baseline_factory: Optional[BaselineFactory] = None,
        sigma_umbral:     float = 5.0,
        L_varianza:       float = 2.0,
        L_delta3:         float = 2.0,
    ):
        self.factory      = baseline_factory or BaselineFactory(
            num_realizaciones_gue     = 100,
            num_realizaciones_poisson = 100,
            L_varianza                = L_varianza,
            L_delta3                  = L_delta3,
        )
        self.sigma_umbral = sigma_umbral
        self.L_varianza   = L_varianza
        self.L_delta3     = L_delta3

    def _unfold(self, gamma: np.ndarray) -> Optional[np.ndarray]:
        u = unfolding_riemann(gamma)
        return u if np.all(np.isfinite(u)) else None

    def _valor_d_min(self, gamma: np.ndarray) -> float:
        u = self._unfold(gamma)
        if u is None:
            return np.nan
        d, _ = espaciado_minimo(u)
        return float(d)

    def _valor_varianza(self, gamma: np.ndarray) -> float:
        u = self._unfold(gamma)
        return np.nan if u is None else varianza_numero(u, self.L_varianza)

    def _valor_delta3(self, gamma: np.ndarray) -> float:
        u = self._unfold(gamma)
        return np.nan if u is None else delta3_dyson_mehta(u, self.L_delta3)

    @staticmethod
    def _z_score(valor: float, media: float, std: float) -> float:
        return 0.0 if std <= 0 else (valor - media) / std

    def evaluar(
        self,
        gamma:    np.ndarray,
        N:        int,
        metricas: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Evalúa el espectro gamma y devuelve estadísticas completas por métrica.

        Campos del dict por métrica:

            [ORIGINALES — retrocompatibles]
            valor, z_gue, z_poisson, anomalia,
            mean_gue, std_gue, mean_poisson, std_poisson

            [NUEVOS — v1.2]
            p_gue         : p-valor dos colas vs baseline GUE  ∈ [0, 1]
            p_poisson     : p-valor dos colas vs baseline Poisson
            ic_gue        : (percentil_2.5, percentil_97.5) baseline GUE
            ic_poisson    : ídem baseline Poisson
            confianza_gue : 'alto' | 'medio' | 'bajo'
            confianza_poi : 'alto' | 'medio' | 'bajo'

        Args:
            gamma   : partes imaginarias de ceros de Riemann (NO unfolded).
            N       : número de ceros (determina tamaño del baseline).
            metricas: subset de ['d_min', 'varianza_numero', 'delta3'].
        """
        if metricas is None:
            metricas = ["d_min", "varianza_numero", "delta3"]

        _nan_resultado = {
            "valor": np.nan, "z_gue": np.nan, "z_poisson": np.nan,
            "anomalia": False, "mean_gue": np.nan, "std_gue": np.nan,
            "mean_poisson": np.nan, "std_poisson": np.nan,
            "p_gue": np.nan, "p_poisson": np.nan,
            "ic_gue": (np.nan, np.nan), "ic_poisson": (np.nan, np.nan),
            "confianza_gue": "bajo", "confianza_poi": "bajo",
        }

        resultados: Dict[str, Dict] = {}

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
                resultados[met] = dict(_nan_resultado)
                continue

            gue_vals = gue_vals[np.isfinite(gue_vals)]
            poi_vals = poi_vals[np.isfinite(poi_vals)]

            mean_gue = float(np.mean(gue_vals)) if len(gue_vals) > 0 else np.nan
            std_gue  = float(np.std(gue_vals))  if len(gue_vals) > 1 else 1e-10
            mean_poi = float(np.mean(poi_vals)) if len(poi_vals) > 0 else np.nan
            std_poi  = float(np.std(poi_vals))  if len(poi_vals) > 1 else 1e-10

            z_gue     = self._z_score(valor, mean_gue, std_gue)
            z_poisson = self._z_score(valor, mean_poi, std_poi)
            anomalia  = abs(z_gue) >= self.sigma_umbral or abs(z_poisson) >= self.sigma_umbral

            p_gue = _p_valor_dos_colas(valor, gue_vals)
            p_poi = _p_valor_dos_colas(valor, poi_vals)
            ic_gue = _intervalo_confianza(gue_vals)
            ic_poi = _intervalo_confianza(poi_vals)

            resultados[met] = {
                # Campos originales
                "valor":         float(valor),
                "z_gue":         float(z_gue),
                "z_poisson":     float(z_poisson),
                "anomalia":      anomalia,
                "mean_gue":      mean_gue,
                "std_gue":       std_gue,
                "mean_poisson":  mean_poi,
                "std_poisson":   std_poi,
                # Nuevos v1.2
                "p_gue":         p_gue,
                "p_poisson":     p_poi,
                "ic_gue":        ic_gue,
                "ic_poisson":    ic_poi,
                "confianza_gue": _clasificar_confianza(p_gue),
                "confianza_poi": _clasificar_confianza(p_poi),
            }

        return resultados

    def hay_anomalia(self, resultados: Dict[str, Dict]) -> bool:
        """True si alguna métrica marcó anomalía (|z| >= sigma_umbral)."""
        return any(r.get("anomalia", False) for r in resultados.values())
