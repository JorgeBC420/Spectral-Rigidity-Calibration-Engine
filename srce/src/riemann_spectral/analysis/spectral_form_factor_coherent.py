# -*- coding: utf-8 -*-
"""
Compatibilidad: K(τ) coherente vive en ``spectral_form_factor.py``.

La implementación canónica es ``spectral_form_factor.spectral_form_factor``:
``K(τ) = |∑ⱼ exp(iτλⱼ)|² / N``. El factor tipo Mehta ``K(t)`` con fase ``2π t γ``
está en ``spectral_form_factor_mehta``. Este módulo solo reexporta símbolos
para imports antiguos sin duplicar lógica.
"""

from .spectral_form_factor import (
    compare_with_theory,
    extract_ramp_slope,
    form_factor_gue_analytical,
    form_factor_poisson,
    identify_regimes,
    spectral_form_factor,
)

__all__ = [
    "compare_with_theory",
    "extract_ramp_slope",
    "form_factor_gue_analytical",
    "form_factor_poisson",
    "identify_regimes",
    "spectral_form_factor",
]
