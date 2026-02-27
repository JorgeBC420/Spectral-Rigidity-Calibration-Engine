# -*- coding: utf-8 -*-
"""
src/riemann_spectral/engine/ensemble_classifier.py
===================================================
Clasificador de ensemble basado en la pendiente de Delta_3(L) vs log(L).

Dado un espectro unfolded, estima la pendiente de Dyson-Mehta y la compara
contra los valores teóricos conocidos:

    Ensemble      Delta_3(L)              Pendiente en log(L)
    ──────────    ──────────────────────  ───────────────────
    Poisson       L / 15                  ~ L (lineal, no log)
    GOE           (1/2π²) · log(L)        ≈ 0.05066
    GUE           (1/π²)  · log(L)        ≈ 0.10132
    Uniforme      ~0  (rígido)             ≈ 0

Referencias:
    Mehta, M.L. "Random Matrices" (3ª ed.), Cap. 16–17.
    Bohigas, Giannoni, Schmit (1984) — conjetura BGS: sistemas caóticos → GUE/GOE.

Uso:
    from riemann_spectral.engine.ensemble_classifier import EnsembleClassifier
    clf = EnsembleClassifier()
    resultado = clf.clasificar(gamma_unfolded)
    print(resultado)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..analysis.rigidity import delta3_dyson_mehta


# ── Constantes teóricas ──────────────────────────────────────────────────────

PENDIENTE_GUE     = 1.0 / (np.pi ** 2)          # ≈ 0.10132
PENDIENTE_GOE     = 1.0 / (2.0 * np.pi ** 2)    # ≈ 0.05066
PENDIENTE_POISSON = None                          # lineal en L, no log
PENDIENTE_UNIFORME = 0.0                          # rígido

# Etiquetas canónicas
ENSEMBLE_GUE      = "GUE"
ENSEMBLE_GOE      = "GOE"
ENSEMBLE_POISSON  = "Poisson"
ENSEMBLE_UNIFORME = "Uniforme"
ENSEMBLE_MIXTO    = "Mixto/Indefinido"


# ── Resultado de clasificación ────────────────────────────────────────────────

@dataclass
class ResultadoClasificacion:
    """
    Resultado completo de la clasificación de ensemble.

    Atributos:
        ensemble        : ensemble más probable ('GUE', 'GOE', 'Poisson', etc.)
        pendiente_obs   : pendiente ajustada de Delta_3 vs log(L)
        pendiente_teorica: pendiente teórica del ensemble asignado
        error_relativo  : |obs - teo| / teo  (para GUE/GOE); None si Poisson
        R2_log          : bondad del ajuste log (1=perfecto)
        R2_lineal       : bondad del ajuste lineal (para detectar Poisson)
        d3_valores      : array de Delta_3(L) observados
        L_grid          : grid de L usado
        scores          : dict con distancias a cada ensemble
        advertencias    : lista de warnings del clasificador
    """
    ensemble:          str
    pendiente_obs:     float
    pendiente_teorica: Optional[float]
    error_relativo:    Optional[float]
    R2_log:            float
    R2_lineal:         float
    d3_valores:        np.ndarray
    L_grid:            np.ndarray
    scores:            Dict[str, float] = field(default_factory=dict)
    advertencias:      List[str]        = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Ensemble clasificado : {self.ensemble}",
            f"Pendiente observada  : {self.pendiente_obs:.6f}",
        ]
        if self.pendiente_teorica is not None:
            lines.append(f"Pendiente teórica    : {self.pendiente_teorica:.6f}")
        if self.error_relativo is not None:
            lines.append(f"Error relativo       : {100 * self.error_relativo:.1f}%")
        lines += [
            f"R² ajuste log        : {self.R2_log:.4f}",
            f"R² ajuste lineal     : {self.R2_lineal:.4f}",
            "Scores vs ensembles  :",
        ]
        for ens, sc in sorted(self.scores.items(), key=lambda x: x[1]):
            lines.append(f"    {ens:<12s}: {sc:.6f}")
        if self.advertencias:
            lines.append("Advertencias:")
            for w in self.advertencias:
                lines.append(f"    ⚠  {w}")
        return "\n".join(lines)


# ── EnsembleClassifier ────────────────────────────────────────────────────────

class EnsembleClassifier:
    """
    Clasifica un espectro unfolded como GUE, GOE, Poisson o Uniforme
    basándose en la pendiente de Delta_3(L) vs log(L).

    Pipeline:
        1. Calcula Delta_3(L) para una grilla de L.
        2. Ajusta por mínimos cuadrados:
               (a) Delta_3 ~ a·log(L) + b   → pendiente_log = a
               (b) Delta_3 ~ c·L + d         → pendiente_lineal = c
        3. Compara pendiente_log contra GUE (1/π²) y GOE (1/2π²).
        4. Si R²_lineal >> R²_log → Poisson.
        5. Si pendiente ≈ 0 → Uniforme.
        6. Asigna ensemble por distancia mínima ponderada.

    Args:
        L_min       : valor mínimo de L para el ajuste (default 5.0).
        L_max       : valor máximo de L (default 30.0).
        n_puntos    : número de puntos en la grilla de L (default 20).
        tol_uniforme: umbral de pendiente para clasificar como Uniforme.
        tol_poisson : umbral de R²_lineal relativo para clasificar como Poisson.
    """

    # Pendientes teóricas para comparación
    _TEORICOS: Dict[str, Optional[float]] = {
        ENSEMBLE_GUE:      PENDIENTE_GUE,
        ENSEMBLE_GOE:      PENDIENTE_GOE,
        ENSEMBLE_UNIFORME: PENDIENTE_UNIFORME,
    }

    def __init__(
        self,
        L_min:        float = 5.0,
        L_max:        float = 30.0,
        n_puntos:     int   = 20,
        tol_uniforme: float = 0.005,
        tol_poisson:  float = 0.85,
    ):
        if L_min <= 0 or L_max <= L_min:
            raise ValueError(f"Se requiere 0 < L_min < L_max, recibido: [{L_min}, {L_max}]")
        if n_puntos < 4:
            raise ValueError(f"n_puntos debe ser >= 4, recibido: {n_puntos}")

        self.L_min        = L_min
        self.L_max        = L_max
        self.n_puntos     = n_puntos
        self.tol_uniforme = tol_uniforme
        self.tol_poisson  = tol_poisson
        self._L_grid      = np.linspace(L_min, L_max, n_puntos)

    # ── API pública ───────────────────────────────────────────────────────────

    def clasificar(
        self,
        gamma_unfolded: np.ndarray,
        label: str = "Espectro",
    ) -> ResultadoClasificacion:
        """
        Clasifica el espectro unfolded.

        Args:
            gamma_unfolded: espectro ya en escala de densidad ≈ 1.
                            Para Riemann: aplicar unfolding_riemann primero.
                            Para GUE/GOE: aplicar unfolding_wigner_gue + tercio central.
            label         : nombre del espectro para mensajes de advertencia.

        Returns:
            ResultadoClasificacion con ensemble, pendiente, R², scores.
        """
        advertencias: List[str] = []

        if not np.all(np.isfinite(gamma_unfolded)):
            advertencias.append("gamma_unfolded contiene NaN/Inf — resultados no confiables")
            gamma_unfolded = gamma_unfolded[np.isfinite(gamma_unfolded)]

        if len(gamma_unfolded) < self.L_max + 10:
            advertencias.append(
                f"Espectro corto (N={len(gamma_unfolded)}) para L_max={self.L_max}. "
                "Considera reducir L_max o usar más ceros."
            )

        # 1. Calcular Delta_3 en la grilla
        d3_vals = self._calcular_d3(gamma_unfolded, advertencias)

        # 2. Ajustes
        pendiente_log, intercepto_log, R2_log       = self._ajuste_log(d3_vals)
        pendiente_lin, intercepto_lin, R2_lineal     = self._ajuste_lineal(d3_vals)

        # 3. Clasificar
        ensemble, pendiente_teo = self._decidir_ensemble(
            pendiente_log, R2_log, R2_lineal, advertencias
        )

        # 4. Scores (distancia normalizada a cada ensemble)
        scores = self._calcular_scores(pendiente_log)

        # 5. Error relativo vs teórico
        error_rel: Optional[float] = None
        if pendiente_teo is not None and pendiente_teo > 0:
            error_rel = abs(pendiente_log - pendiente_teo) / pendiente_teo

        return ResultadoClasificacion(
            ensemble          = ensemble,
            pendiente_obs     = float(pendiente_log),
            pendiente_teorica = pendiente_teo,
            error_relativo    = error_rel,
            R2_log            = float(R2_log),
            R2_lineal         = float(R2_lineal),
            d3_valores        = d3_vals,
            L_grid            = self._L_grid.copy(),
            scores            = scores,
            advertencias      = advertencias,
        )

    def validar_pendiente_goe(
        self,
        gamma_unfolded: np.ndarray,
        tolerancia: float = 0.30,
    ) -> Tuple[bool, float, float]:
        """
        Validación rápida: ¿la pendiente de Delta_3 es consistente con GOE?

        Compara la pendiente observada contra PENDIENTE_GOE = 1/(2π²) ≈ 0.05066.

        Args:
            gamma_unfolded: espectro unfolded con densidad ≈ 1.
            tolerancia    : error relativo máximo admitido (default 30%).

        Returns:
            (es_goe, pendiente_obs, error_relativo)
        """
        resultado = self.clasificar(gamma_unfolded)
        es_goe    = resultado.error_relativo is not None and resultado.error_relativo <= tolerancia
        return es_goe, resultado.pendiente_obs, resultado.error_relativo or np.nan

    def validar_pendiente_gue(
        self,
        gamma_unfolded: np.ndarray,
        tolerancia: float = 0.30,
    ) -> Tuple[bool, float, float]:
        """
        Validación rápida: ¿la pendiente de Delta_3 es consistente con GUE?

        Compara contra PENDIENTE_GUE = 1/π² ≈ 0.10132.
        """
        resultado = self.clasificar(gamma_unfolded)
        err       = resultado.error_relativo
        es_gue    = (
            resultado.ensemble == ENSEMBLE_GUE
            and err is not None
            and err <= tolerancia
        )
        return es_gue, resultado.pendiente_obs, err or np.nan

    # ── Métodos internos ──────────────────────────────────────────────────────

    def _calcular_d3(
        self,
        gamma: np.ndarray,
        advertencias: List[str],
    ) -> np.ndarray:
        """Calcula Delta_3(L) para cada L en _L_grid. Rellena NaN si falla."""
        d3 = np.full(len(self._L_grid), np.nan)
        for i, L in enumerate(self._L_grid):
            val = delta3_dyson_mehta(gamma, L)
            if np.isfinite(val):
                d3[i] = val
        n_nan = np.sum(np.isnan(d3))
        if n_nan > len(d3) // 3:
            advertencias.append(
                f"{n_nan}/{len(d3)} valores de Delta_3 son NaN. "
                "El espectro puede ser demasiado corto para la grilla de L."
            )
        return d3

    @staticmethod
    def _r2(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
        """R² entre observado y predicho, ignorando NaN."""
        mask  = np.isfinite(y_obs) & np.isfinite(y_pred)
        if mask.sum() < 2:
            return 0.0
        ss_res = np.sum((y_obs[mask] - y_pred[mask]) ** 2)
        ss_tot = np.sum((y_obs[mask] - np.mean(y_obs[mask])) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    def _ajuste_log(
        self,
        d3_vals: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Ajusta Delta_3 ~ a·log(L) + b por mínimos cuadrados."""
        mask   = np.isfinite(d3_vals) & (self._L_grid > 0)
        if mask.sum() < 3:
            return 0.0, 0.0, 0.0
        log_L  = np.log(self._L_grid[mask])
        d3_ok  = d3_vals[mask]
        p      = np.polyfit(log_L, d3_ok, 1)
        y_pred = np.polyval(p, np.log(self._L_grid))
        R2     = self._r2(d3_vals, y_pred)
        return float(p[0]), float(p[1]), R2

    def _ajuste_lineal(
        self,
        d3_vals: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Ajusta Delta_3 ~ c·L + d por mínimos cuadrados."""
        mask   = np.isfinite(d3_vals)
        if mask.sum() < 3:
            return 0.0, 0.0, 0.0
        L_ok   = self._L_grid[mask]
        d3_ok  = d3_vals[mask]
        p      = np.polyfit(L_ok, d3_ok, 1)
        y_pred = np.polyval(p, self._L_grid)
        R2     = self._r2(d3_vals, y_pred)
        return float(p[0]), float(p[1]), R2

    def _decidir_ensemble(
        self,
        pendiente_log: float,
        R2_log:        float,
        R2_lineal:     float,
        advertencias:  List[str],
    ) -> Tuple[str, Optional[float]]:
        """
        Lógica de decisión por prioridad:
            1. pendiente ≈ 0            → Uniforme
            2. R²_lineal >> R²_log      → Poisson
            3. pendiente cercana a GOE  → GOE
            4. pendiente cercana a GUE  → GUE
            5. fallback                 → Mixto
        """
        # 1. Uniforme
        if abs(pendiente_log) < self.tol_uniforme:
            return ENSEMBLE_UNIFORME, PENDIENTE_UNIFORME

        # 2. Poisson: ajuste lineal claramente mejor que log
        if R2_lineal > self.tol_poisson and R2_lineal > R2_log + 0.1:
            advertencias.append(
                f"R²_lineal={R2_lineal:.3f} >> R²_log={R2_log:.3f}: comportamiento Poisson."
            )
            return ENSEMBLE_POISSON, None

        # 3 & 4. GUE vs GOE: distancia normalizada a cada pendiente teórica
        dist_gue = abs(pendiente_log - PENDIENTE_GUE) / PENDIENTE_GUE
        dist_goe = abs(pendiente_log - PENDIENTE_GOE) / PENDIENTE_GOE

        if dist_goe < dist_gue:
            if dist_goe > 0.50:
                advertencias.append(
                    f"Pendiente={pendiente_log:.5f} asignada a GOE pero con error "
                    f"relativo {100*dist_goe:.1f}% > 50%."
                )
            return ENSEMBLE_GOE, PENDIENTE_GOE
        else:
            if dist_gue > 0.50:
                advertencias.append(
                    f"Pendiente={pendiente_log:.5f} asignada a GUE pero con error "
                    f"relativo {100*dist_gue:.1f}% > 50%."
                )
            return ENSEMBLE_GUE, PENDIENTE_GUE

    def _calcular_scores(self, pendiente_log: float) -> Dict[str, float]:
        """
        Score por ensemble = distancia absoluta de la pendiente observada
        a la teórica. Menor score = ensemble más probable.
        Poisson no tiene pendiente log, se usa proxy de 10× GUE.
        """
        return {
            ENSEMBLE_GUE:      abs(pendiente_log - PENDIENTE_GUE),
            ENSEMBLE_GOE:      abs(pendiente_log - PENDIENTE_GOE),
            ENSEMBLE_UNIFORME: abs(pendiente_log - PENDIENTE_UNIFORME),
            ENSEMBLE_POISSON:  abs(pendiente_log - 10 * PENDIENTE_GUE),  # proxy lineal
        }
