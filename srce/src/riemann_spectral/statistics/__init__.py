# -*- coding: utf-8 -*-
"""Estadísticas espectrales modulares (r-parameter, etc.)."""

from .r_statistic import (
    R_GOE_EXACT,
    R_GUE_EXACT,
    R_POISSON_EXACT,
    R_TOLERANCE,
    classify_ensemble_by_r,
    compare_r_with_theory,
    compute_r_distribution,
    compute_r_parameter,
    r_parameter_convergence,
)

__all__ = [
    "R_GOE_EXACT",
    "R_GUE_EXACT",
    "R_POISSON_EXACT",
    "R_TOLERANCE",
    "classify_ensemble_by_r",
    "compare_r_with_theory",
    "compute_r_distribution",
    "compute_r_parameter",
    "r_parameter_convergence",
]
