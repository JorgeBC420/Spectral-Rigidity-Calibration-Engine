# -*- coding: utf-8 -*-
"""
Pipeline RMT de alto nivel — envuelve módulos analysis/ y engine/ existentes.

Pensado para datasets externos y para el dashboard ampliado (sin duplicar matemática).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..analysis.empirical_unfolding import (
    unfold_kde,
    unfold_polynomial,
    unfold_spline,
)
from ..analysis.normalize import normalize_spacing
from ..analysis.number_variance import sigma2_number_variance_fast
from ..analysis.pair_correlation import pair_correlation_fast, chi2_r2_vs_gue
from ..analysis.rigidity import delta3_dyson_mehta
from ..engine.ensemble_classifier import EnsembleClassifier
from ..statistics.r_statistic import compute_r_parameter, classify_ensemble_by_r


DISCLAIMER_ES = (
    "SRCE es un motor estadístico para auditoría de regularidades espectrales. "
    "No demuestra la Hipótesis de Riemann ni conclusiones físicas absolutas. "
    "Los resultados dependen del unfolding, del tamaño de muestra y del régimen de ventanas finitas."
)


@dataclass
class RMTAuditResult:
    """Resumen de una corrida de auditoría sobre un espectro 1D."""

    n_levels: int
    r_mean: float
    r_ensemble: str
    unfolding_method: str
    classifier_ensemble: str
    classifier_confidence_pct: float
    alpha_delta3: float
    chi2_r2_gue: Optional[float]
    spacing_mean: float
    advertencias: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


def confidence_from_scores(scores: Dict[str, float]) -> float:
    """Convierte distancias (menor = mejor) en confianza % del ensemble ganador."""
    if not scores:
        return 0.0
    vals = np.array([max(v, 1e-12) for v in scores.values()], dtype=float)
    inv = 1.0 / vals
    return float(100.0 * inv.max() / inv.sum())


def unfold_spectrum(
    levels: np.ndarray,
    method: str = "spline",
    **kwargs: Any,
) -> np.ndarray:
    """Unfolding empírico: polynomial | spline | kde."""
    m = method.lower()
    if m == "polynomial":
        return unfold_polynomial(levels, **kwargs)
    if m == "kde":
        return unfold_kde(levels, **kwargs)
    return unfold_spline(levels, **kwargs)


def run_rmt_audit(
    levels: np.ndarray,
    unfolding: str = "spline",
    seed: Optional[int] = None,
) -> RMTAuditResult:
    """
    Pipeline compacto: unfolding → r → Δ₃ pendiente → clasificador → P(s) χ².

    Args:
        levels: niveles ordenados (ceros, autovalores, etc.).
        unfolding: polynomial | spline | kde.
        seed: reservado para extensiones Monte Carlo (futuro).
    """
    advertencias: List[str] = []
    if seed is not None:
        np.random.seed(seed)

    levels = np.asarray(levels, dtype=float)
    levels = levels[np.isfinite(levels)]
    if len(levels) < 20:
        advertencias.append(f"Muestra pequeña (N={len(levels)}); métricas inestables.")

    try:
        unfolded = unfold_spectrum(levels, method=unfolding)
    except Exception as exc:
        advertencias.append(f"Unfolding falló ({unfolding}): {exc}")
        unfolded = normalize_spacing(levels)

    r_info = classify_ensemble_by_r(unfolded)
    r_mean = float(r_info["r_mean"])
    r_ens = str(r_info["ensemble"])

    clf = EnsembleClassifier()
    res = clf.clasificar(unfolded, label="externo")
    conf = confidence_from_scores(res.scores)

    # Pendiente α de Δ₃ vs log L (misma convención que scripts)
    L_grid = res.L_grid
    d3 = res.d3_valores
    mask = np.isfinite(d3) & (L_grid > 0)
    alpha = 0.0
    if mask.sum() >= 3:
        alpha = float(np.polyfit(np.log(L_grid[mask]), d3[mask], 1)[0])

    chi2: Optional[float] = None
    try:
        s_grid, r2_obs, _ = pair_correlation_fast(unfolded, s_max=3.0, bins=40)
        chi2, _ = chi2_r2_vs_gue(s_grid, r2_obs)
    except Exception:
        advertencias.append("No se pudo calcular χ² de R₂(s) vs GUE.")

    sp = np.diff(unfolded)
    sp = sp[sp > 0]

    return RMTAuditResult(
        n_levels=len(levels),
        r_mean=r_mean,
        r_ensemble=r_ens,
        unfolding_method=unfolding,
        classifier_ensemble=res.ensemble,
        classifier_confidence_pct=conf,
        alpha_delta3=alpha,
        chi2_r2_gue=chi2,
        spacing_mean=float(np.mean(sp)) if len(sp) else float("nan"),
        advertencias=advertencias + list(res.advertencias),
        extra={
            "disclaimer": DISCLAIMER_ES,
            "scores": res.scores,
            "R2_log": res.R2_log,
            "R2_lineal": res.R2_lineal,
        },
    )
