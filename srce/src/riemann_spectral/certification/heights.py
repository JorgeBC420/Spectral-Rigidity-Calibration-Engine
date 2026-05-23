# -*- coding: utf-8 -*-
"""
Alturas Im(s) sin perder precisión en T ancla + offset dt.

Regla: en log_T > FLOAT_SAFE_LOG_T no usar float(T_ancla) para certificación;
usar mpmath.mpf y cadenas de alta precisión hacia backends que lo admitan.
"""

from __future__ import annotations

from typing import Tuple, Union

import mpmath as mp

# Por encima de ~10^12 float64 pierde dígitos en la parte entera de T.
FLOAT_SAFE_LOG_T = 12.0


def t_im_from_offset(
    T_anchor: Union[mp.mpf, str],
    dt: float,
    dps: int = 30,
) -> mp.mpf:
    """Im(s) = T_anchor + dt con aritmética mpf."""
    with mp.workdps(dps):
        T_mp = mp.mpf(str(T_anchor)) if not isinstance(T_anchor, mp.mpf) else T_anchor
        return T_mp + mp.mpf(str(dt))


def window_im_mpf(
    T_anchor: mp.mpf,
    dt_inicio: float,
    dt_fin: float,
    dps: int,
) -> Tuple[mp.mpf, mp.mpf]:
    """Extremos Im(s) de la ventana en mpf."""
    with mp.workdps(dps):
        t0 = T_anchor + mp.mpf(str(dt_inicio))
        t1 = T_anchor + mp.mpf(str(dt_fin))
    return t0, t1


def im_str_for_backend(t_mpf: mp.mpf, dps: int) -> str:
    """Cadena decimal para Arb/mpmath sin pasar por float(T) gigante."""
    with mp.workdps(dps):
        return mp.nstr(t_mpf, dps)


def im_float_if_safe(t_mpf: mp.mpf, log_T: float, dps: int) -> Tuple[float, bool]:
    """
    Convierte a float solo si log_T está en rango seguro.

    Returns:
        (valor, es_seguro) — si es_seguro=False, el float es solo diagnóstico grueso.
    """
    if log_T <= FLOAT_SAFE_LOG_T:
        return float(t_mpf), True
    with mp.workdps(min(dps, 80)):
        coarse = float(mp.nstr(t_mpf, min(30, int(log_T) + 10)))
    return coarse, False
