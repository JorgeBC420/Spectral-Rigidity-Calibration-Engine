# Analysis: unfolding, rigidez, espectral, pair_correlation, spectral_form_factor

from .unfolding import unfolding_riemann, N_T_approx
from .spectral import (
    calcular_jacobiano_kernel,
    energia_log_gas,
    analizar_espectro_completo,
    analizar_modo_blando,
    clasificar_modo_blando,
)
from .rigidity import (
    calcular_espaciados,
    espaciado_minimo,
    varianza_numero,
    delta3_dyson_mehta,
    ecuacion_espaciado_minimo_correcta,
    descomponer_termino_regular,
)
from .pair_correlation import (
    pair_correlation_fast,
    r2_teorica_gue,
    r2_teorica_goe,
    r2_teorica_poisson,
    chi2_r2_vs_gue,
)
from .spectral_form_factor import (
    r_statistic,
    r_distribucion_teorica,
    spectral_form_factor,
    spectral_form_factor_mehta,
    spectral_form_factor_mehta_teorico,
    spectral_form_factor_teorico,
    R_MEAN_GUE,
    R_MEAN_GOE,
    R_MEAN_POISSON,
)

__all__ = [
    "unfolding_riemann", "N_T_approx",
    "calcular_jacobiano_kernel", "energia_log_gas",
    "analizar_espectro_completo", "analizar_modo_blando", "clasificar_modo_blando",
    "calcular_espaciados", "espaciado_minimo", "varianza_numero",
    "delta3_dyson_mehta", "ecuacion_espaciado_minimo_correcta",
    "descomponer_termino_regular",
    "pair_correlation_fast", "r2_teorica_gue", "r2_teorica_goe",
    "r2_teorica_poisson", "chi2_r2_vs_gue",
    "r_statistic", "r_distribucion_teorica",
    "spectral_form_factor",
    "spectral_form_factor_mehta",
    "spectral_form_factor_mehta_teorico",
    "spectral_form_factor_teorico",
    "R_MEAN_GUE", "R_MEAN_GOE", "R_MEAN_POISSON",
]
