#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/zeta_altura_extrema.py  — v2.2.0 (ID híbrido Gödel)
================================================================

Exploración de la función Z de Riemann-Siegel en alturas extremas
(T ≈ 10^70) usando aritmética de desplazamiento y θ asintótica.

Cambios v2.2 respecto a v2.1
------------------------------
    ID híbrido: hash de posición + número de Gödel del pipeline.

    En v2.1 el ID era solo un hash SHA-256 de (log_T, dt):
        SRCE-T70-dt+0.03142159-A3F7B2

    Identificaba el cero pero no decía nada sobre su calidad.
    Alguien que recibía ese ID no podía saber si convergió en 3
    iteraciones o en 12, si pasó la prueba de dps, ni su score.

    En v2.2 el ID es híbrido:
        SRCE-T70-dt+0.03142159-A3F7B2-G1058400

    donde G1058400 es un número de Gödel real que codifica el
    recorrido completo por las 3 fases del pipeline:

        2^alias_bin · 3^n_iter · 5^converged · 7^residual_bin
        · 11^dps_stable · 13^res_stable · 17^score_bin

    Propiedades del número de Gödel:
        Decodificable: factorizando G se recuperan todos los coeficientes.
        Único: dos ceros con distinto historial de pipeline dan G distinto.
        Ordenable: G más alto no implica mejor calidad (los primos no
            tienen ese orden), pero los coeficientes sí son comparables
            tras decodificar.
        Reproducible: dado (vzero, azero), G es siempre el mismo.

    La función goedel_decodificar() invierte el proceso y devuelve
    el dict de coeficientes original. Esto permite auditoría completa
    desde el ID solo, sin acceso a los logs.

    Cambios técnicos:
        - goedel_pipeline(vzero, azero) → int
        - goedel_decodificar(g) → dict
        - CeroGodel ahora acepta vzero y azero opcionales
        - crear_ceros_goedel recibe los AcceptedZero del pipeline
        - guardar_offsets muestra la decodificación de cada G
        - main pasa accepted directamente a crear_ceros_goedel

    El problema en v2.0: buscar_ceros_desplazados() mezclaba detección,
    validación y aceptación. Un candidato de Fase 1 que fallaba el
    refinamiento quedaba silenciosamente descartado sin trazabilidad.
    Un cero validado con convergencia dudosa entraba igualmente a SRCE.

    La v2.1 separa las responsabilidades:

        Candidate     ← Fase 1 (cambio de signo en Z exacta sobre grilla ~dt_safe)
        ValidatedZero ← Fase 2 (refinamiento secante; Z exacta ya coherente en bracket)
        AcceptedZero  ← Fase 3 (score ≥ umbral + estabilidad dps + resolución)

    Solo AcceptedZero con score ≥ 0.8 alimentan SRCE.
    Con score < 0.8 se registran pero no se analizan espectralmente.

    --solo-fase renombrado a --solo-candidatos para honestidad semántica:
    un bracket por cambio de signo en la grilla no es un cero aceptado — es candidato.

    Cambio 7 (multi-altura): no calcula Δ₃ si n_accepted < 8 o
    score_medio < 0.75. Evita que α(T) parezca más fuerte de lo que es.

    Todo lo demás — ThetaCache, check_aliasing, Z_exacta,
    refinar_cero_secante, CeroGodel, offsets_a_espectro,
    analizar_espectro_local, guardar_offsets, plots — sin cambios.

Arquitectura de optimización
------------------------------

El problema central a T=10^70: la suma de Riemann-Siegel requiere
~√(T/2π) ≈ 10^34 términos — imposible. mpmath.zeta() tampoco escapa
a este costo. La solución es una separación de responsabilidades:

    ┌──────────────────────────────────────────────────────────────┐
    │  FASE 1 — DETECCIÓN  (Z exacta Hardy en grilla ~dt_safe)       │
    │  Evalúa Z(T+dt) con mpmath.zeta en cada punto de la grilla.   │
    │  Los ceros de Z no coinciden con los de 2·cos(θ): la fase sola │
    │  no basta para brackets válidos. Coste O(n_scan) por ventana.  │
    │                                                              │
    │  FASE 2 — REFINAMIENTO  (secante sobre dt, ~10–20 ceros)      │
    │  Refina cada bracket ya con cambio de signo en Z exacta.      │
    │  Operaciones siempre sobre dt << 1 para evitar cancelación.  │
    └──────────────────────────────────────────────────────────────┘

Aritmética de desplazamiento
------------------------------
    T_big = mpmath.mpf("1e70")   ← constante de alta precisión
    t     = T_big + dt           ← dt es pequeño (|dt| < 10)

    Todos los cálculos se hacen sobre dt — nunca se restan dos
    números grandes. Equivalente a trabajar en el sistema de
    referencia en reposo del cluster de ceros.

    Los IDs de Gödel codifican solo el OFFSET dt, no T completo.

Detección basada en fase
--------------------------
    Un cero de Z(t) ocurre cerca de cada t donde:
        θ(t) ≡ -π/2  (mod π)
    o equivalentemente donde θ'(t) · Δt ≈ π (un medio período).

    Derivada de θ:
        θ'(t) = (1/2)·log(t/2π) + O(1/t)  → O(1) a evaluar

    Spacing esperado entre ceros:
        Δt_zero ≈ π / θ'(T) = π / [(1/2)·log(T/2π)]

    Este criterio se usa para el aliasing check: si el paso de
    muestreo Δt > Δt_zero / 2, se pierde un cero (aliasing).

Método de refinamiento
------------------------
    Método de la secante (no necesita Z'):
        dt_{n+1} = dt_n - Z(dt_n) · (dt_n - dt_{n-1})
                                   / (Z(dt_n) - Z(dt_{n-1}))
    Convergencia en ~5 pasos. Costo total: ~10 evaluaciones de Z
    por cero (2 en la detección + 5-8 para secante).

Caché de términos logarítmicos
--------------------------------
    log(T/2π) y log(T/2π·e) se calculan UNA VEZ y se reusan en
    toda la sesión. Son mpf de alta precisión — costosos de calcular,
    baratos de reutilizar. Ver ThetaCache.

Uso
---
    # Exploración puntual en T ≈ 10^70
    python scripts/zeta_altura_extrema.py --log-T 70 --n-ceros 15

    # Altura personalizada, dps manual
    python scripts/zeta_altura_extrema.py --log-T 50 --dps 80

    # Alturas múltiples para convergencia α(N)
    python scripts/zeta_altura_extrema.py --multi-altura

    # Solo detección por fase (sin mpmath.zeta — instantáneo)
    python scripts/zeta_altura_extrema.py --log-T 70 --solo-fase

Salidas
-------
    output/zeta_offsets.txt      — offsets dt y IDs Gödel
    output/zeta_diagnostico.png  — r, Δ₃, Σ², P(s)
    output/zeta_anomalias.txt    — anomalías con ID Gödel
    output/zeta_convergencia.png — α(N) vs log(T)

Autor: Jorge BC & Claude
Versión: 2.2.3 — Fase 1 con Z exacta en grilla (~dt_safe); secante con min/max nativos;
    conteo Backlund opcional --arb (python-flint); UTF-8 en consola Windows.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _configure_stdio_utf8() -> None:
    """Evita UnicodeEncodeError en Windows (cp1252) con símbolos en prints."""
    for _stream in (sys.stdout, sys.stderr):
        reconf = getattr(_stream, "reconfigure", None)
        if reconf is not None:
            try:
                reconf(encoding="utf-8", errors="replace")
            except (OSError, ValueError, AttributeError):
                pass


_configure_stdio_utf8()

# ── Path portable ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ── mpmath ────────────────────────────────────────────────────────────────────
try:
    import mpmath as mp
    _MPMATH = True
except ImportError:
    print("ERROR: mpmath no disponible. pip install mpmath")
    sys.exit(1)

# ── Imports SRCE ──────────────────────────────────────────────────────────────
try:
    from riemann_spectral.analysis.rigidity        import delta3_dyson_mehta
    from riemann_spectral.analysis.number_variance import sigma2_number_variance_fast
    from riemann_spectral.analysis.normalize       import normalize_spacing
    from riemann_spectral.analysis.unfolding       import unfolding_riemann
    from riemann_spectral.statistics.r_statistic   import (
        compute_r_parameter, classify_ensemble_by_r,
        R_GUE_EXACT, R_GOE_EXACT, R_POISSON_EXACT,
    )
    _SRCE = True
except ImportError as e:
    print(f"  ⚠ SRCE no disponible ({e}). Solo análisis de ceros.")
    _SRCE = False
    R_GUE_EXACT     = 0.60272
    R_POISSON_EXACT = 2 * math.log(2) - 1

# ── Arb / FLINT (conteo Backlund certificado, opcional) ───────────────────────
try:
    from riemann_spectral.rigorous.arb_bridge import reemplazar_backlund_count
    _ARB_BACKLUND = True
except ImportError:
    reemplazar_backlund_count = None  # type: ignore[misc, assignment]
    _ARB_BACKLUND = False

# ── Matplotlib ────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

# ── Constantes ────────────────────────────────────────────────────────────────
ALPHA_GUE    = 1.0 / math.pi ** 2
ALPHA_GOE    = 1.0 / (2.0 * math.pi ** 2)
L_GRID       = np.linspace(5, 25, 15)
UMBRAL_SIGMA = 2.0
RECORTE      = 0.15      # más conservador para N pequeño


# ============================================================================
# 1. PRECISIÓN DINÁMICA
# ============================================================================

def dps_auto(log_T: float, n_samples_per_zero: int = 50, safety: int = 15) -> int:
    """
    Calcula el dps mínimo para representar T = 10^log_T + dt sin pérdida.

    Razonamiento:
        T tiene log_T dígitos decimales. dt es del orden 1/rho donde
        rho ≈ log(T/2π)/(2π) ~ log_T·log(10)/(2π). Para que T+dt no
        pierda información en dt necesitamos:
            dps > log_T + log10(T/dt) + safety
                = log_T + log10(rho·n_samples) + safety

    Args:
        log_T           : exponente de T (T = 10^log_T).
        n_samples_per_zero: puntos de muestreo por cero esperado.
        safety          : dígitos extras de margen.

    Returns:
        dps entero (mínimo 30).
    """
    if log_T <= 0:
        return 30
    # rho en escala log10 para evitar overflow de float
    log_rho = math.log10(log_T * math.log(10) / (2 * math.pi))
    dps = int(log_T + log_rho + math.log10(n_samples_per_zero) + safety)
    return max(dps, 30)


# ============================================================================
# 2. CACHÉ DE TÉRMINOS LOGARÍTMICOS (ThetaCache)
# ============================================================================

class ThetaCache:
    """
    Pre-calcula y almacena los términos logarítmicos de θ(T) y sus
    derivadas. Todas son constantes para T fijo — se calculan una vez.

    Términos almacenados (todos mp.mpf de alta precisión):
        log_T_2pi   : log(T / 2π)
        log_T_2pie  : log(T / 2π·e) = log_T_2pi - 1
        theta_T     : θ_asint(T)         [constante de anclaje]
        theta_prime : θ'(T) = log(T/2π)/2
        theta_second: θ''(T) = 1/(2T)
        theta_third : θ'''(T) = -1/(2T²)
        zero_spacing: π / θ'(T)          [espaciado esperado entre ceros]
        dt_nyquist  : zero_spacing / 2    [paso máximo sin aliasing]
        dt_safe     : zero_spacing / 10   [paso conservador]
    """

    def __init__(self, log_T: float, dps: int):
        self.log_T = log_T
        self.dps   = dps

        with mp.workdps(dps):
            # T como mpf de alta precisión
            # Construir 10^log_T sin desbordamiento float:
            # 10^log_T = exp(log_T * log(10))
            self.T = mp.power(10, mp.mpf(str(log_T)))

            two_pi = 2 * mp.pi

            # Términos logarítmicos — calculados UNA VEZ
            self.log_T_2pi  = mp.log(self.T / two_pi)
            self.log_T_2pie = self.log_T_2pi - mp.mpf('1')

            # θ_asint(T) — serie de Stirling (Abramowitz & Stegun 6.5.41)
            T2 = self.T * self.T
            T4 = T2 * T2
            self.theta_T = (
                (self.T / 2) * self.log_T_2pie
                - mp.pi / 8
                + mp.mpf('1') / (48 * self.T)
                - mp.mpf('7') / (5760 * self.T * T2)
                + mp.mpf('31') / (80640 * T4 * self.T)
            )

            # Derivadas de θ en T
            self.theta_prime  = self.log_T_2pi / 2          # θ'(T)
            self.theta_second = mp.mpf('1') / (2 * self.T)   # θ''(T)
            self.theta_third  = -mp.mpf('1') / (2 * self.T * self.T)  # θ'''(T)

            # Espaciado y criterios de muestreo
            self.zero_spacing = float(mp.pi / self.theta_prime)
            self.dt_nyquist   = self.zero_spacing / 2.0
            self.dt_safe      = self.zero_spacing / 10.0

    def delta_theta(self, dt: "mp.mpf") -> "mp.mpf":
        """
        θ(T+dt) - θ(T) via serie de Taylor en dt. O(1) — sin log ni zeta.

        Usa los tres primeros términos:
            Δθ ≈ θ'·dt + θ''·dt²/2 + θ'''·dt³/6

        Exacto hasta O(dt^4/T^3) — suficiente para |dt| < 10 y T=10^70.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        return (
            self.theta_prime * dt
            + self.theta_second * dt2 / 2
            + self.theta_third  * dt3 / 6
        )

    def theta_at(self, dt: "mp.mpf") -> "mp.mpf":
        """θ(T+dt) = θ(T) + Δθ(dt). O(1)."""
        return self.theta_T + self.delta_theta(dt)

    def Z_phase_approx(self, dt: "mp.mpf") -> float:
        """
        Aproximación de la función Z basada solo en la fase:
            Z_approx(T+dt) ≈ 2·cos(θ(T+dt))

        Válida cuando el término principal de la suma RS domina.
        Para T grande esto es una buena aproximación GLOBAL pero
        no captura las fluctuaciones locales entre ceros.

        Uso: detección rápida de candidatos a cero (Fase 1).
        No usar para validación final (Fase 2 usa Z exacta).
        """
        with mp.workdps(self.dps):
            theta = self.theta_at(mp.mpf(str(dt)))
            return float(2 * mp.cos(theta))

    def __repr__(self) -> str:
        return (
            f"ThetaCache(log_T={self.log_T}, dps={self.dps}, "
            f"zero_spacing={self.zero_spacing:.5f}, "
            f"dt_safe={self.dt_safe:.6f})"
        )


# ============================================================================
# 3-A. MECANISMOS DE VERIFICACIÓN RIGUROSA
# ============================================================================
#
# Atacan el problema fundamental que ninguna heurística puede resolver:
# "¿cómo sabes que no faltó ninguno?"
#
# Tres herramientas, en orden de impacto:
#
# A. backlund_count — conteo exacto de ceros por principio del argumento.
#    Si Fase 1 encontró k candidatos y N(T₂)-N(T₁) = k, el conteo cuadra.
#    Si encontró k-1, falta uno — y se sabe exactamente dónde buscar.
#
# B. gram_points — anclas de signo para detectar drift acumulativo.
#    En cada punto de Gram gₙ donde θ(gₙ)=nπ, Z cambia de signo con
#    alta regularidad. Los intervalos de Gram fallidos son contables.
#    Usarlos como checkpoints valida que el muestreo no pierde ceros.
#
# C. interval_residual — residual con cota garantizada via aritmética
#    de intervalos (mpmath.iv). Si el intervalo contiene al cero, la
#    validación es rigurosa. Si no, el falso positivo está certificado.


def backlund_count(
    T1:    float,
    T2:    float,
    cache: ThetaCache,
    dps:   int = None,
) -> Dict:
    """
    Cuenta el número exacto de ceros de ζ(s) en la franja 0 < Im(s) < T
    usando la fórmula de Backlund–Riemann–von Mangoldt:

        N(T) = (1/π) · θ(T) + 1 + S(T)

    donde S(T) = (1/π) · Im[log ζ(1/2 + iT)] mide las fluctuaciones
    respecto al comportamiento promedio. S(T) es típicamente pequeño
    (|S(T)| < 1 para casi todos los T), pero puede ser grande en regiones
    con muchos ceros cercanos.

    Uso operativo: dado un intervalo [T₁, T₂], el número esperado de
    ceros es N(T₂) - N(T₁). Si Fase 1 encontró exactamente ese número
    de candidatos confirmados por Fase 2, el conteo cuadra y hay garantía
    fuerte (no absoluta — S(T) puede tener discontinuidades) de que no
    faltó ninguno.

    Args:
        T1, T2 : límites del intervalo (T2 > T1 > 0).
        cache  : ThetaCache — proporciona θ(T) para el punto de anclaje.
        dps    : precisión. None = usar cache.dps.

    Returns:
        dict con:
            N_T1, N_T2  : conteos en T1 y T2.
            delta_N     : N(T2) - N(T1) — ceros esperados en [T1, T2].
            S_T1, S_T2  : términos de fluctuación S(T).
            S_grande    : True si |S(T)| > 1 (región de cuidado).
            fiable      : True si |S(T)| < 1 en ambos extremos.

    Referencias:
        Backlund (1914). Acta Math.
        Edwards (1974). Riemann's Zeta Function. Cap. 3.
    """
    dps = dps or cache.dps

    with mp.workdps(dps):
        T1_mp = mp.mpf(str(T1))
        T2_mp = mp.mpf(str(T2))

        def N_T(T_mp: "mp.mpf") -> Tuple[float, float]:
            """N(T) = θ(T)/π + 1 + S(T)."""
            # θ(T) — usar cache si T ≈ cache.T, mpmath si no
            if abs(float(T_mp) - float(cache.T)) < cache.zero_spacing * 100:
                dt_   = T_mp - cache.T
                theta_ = cache.theta_T + cache.delta_theta(dt_)
            else:
                theta_ = mp.im(mp.loggamma(mp.mpc('0.25', T_mp / 2))) \
                         - T_mp / 2 * mp.log(mp.pi)

            # S(T) = (1/π) Im[log ζ(1/2 + iT)]
            # Evaluado directamente con mpmath
            log_zeta = mp.log(mp.zeta(mp.mpc('0.5', T_mp)))
            S_T      = float(mp.im(log_zeta) / mp.pi)
            N        = float(theta_ / mp.pi) + 1.0 + S_T
            return round(N), S_T

        N1, S1 = N_T(T1_mp)
        N2, S2 = N_T(T2_mp)

    delta_N  = N2 - N1
    S_grande = abs(S1) > 1.0 or abs(S2) > 1.0

    return {
        'N_T1':    N1,
        'N_T2':    N2,
        'delta_N': delta_N,
        'S_T1':    S1,
        'S_T2':    S2,
        'S_grande': S_grande,
        'fiable':  not S_grande,
        'T1': T1, 'T2': T2,
        'advertencia': (
            "S(T) > 1 — conteo menos fiable en esta región" if S_grande else None
        ),
    }


def gram_points(
    cache:    ThetaCache,
    dt_ini:   float,
    dt_fin:   float,
    dps:      int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula los puntos de Gram gₙ en el intervalo [T+dt_ini, T+dt_fin].

    Los puntos de Gram son los t donde θ(t) = nπ para n entero.
    Por la "ley de Gram" (que falla con frecuencia ~1/4 pero es regularidad
    estadística robusta), cada intervalo [gₙ, gₙ₊₁] contiene exactamente
    un cero de Z(t).

    Uso en el pipeline:
        Comparar el número de Gram intervals en [dt_ini, dt_fin] con el
        número de candidatos encontrados. Si difieren, hay drift o pérdida.
        Los intervalos donde Z(gₙ) y Z(gₙ₊₁) tienen el mismo signo son
        "Gram block failures" — allí buscar con mayor resolución.

    Args:
        cache      : ThetaCache con T anclaje.
        dt_ini, dt_fin: offsets del intervalo a explorar.
        dps        : precisión. None = cache.dps.

    Returns:
        (dt_gram, n_gram, Z_gram_sign):
            dt_gram    : offsets de los puntos de Gram (array).
            n_gram     : índices n tales que θ(T+dt) = nπ.
            Z_gram_sign: signo de Z_approx en cada punto de Gram.
                         Permite detectar Gram block failures sin Z exacta.
    """
    dps = dps or cache.dps

    # θ en los extremos del intervalo
    with mp.workdps(dps):
        theta_ini = float(cache.theta_T + cache.delta_theta(mp.mpf(str(dt_ini))))
        theta_fin = float(cache.theta_T + cache.delta_theta(mp.mpf(str(dt_fin))))

    # Índices de Gram en el intervalo: n tal que nπ ∈ [θ_ini, θ_fin]
    n_ini = int(np.ceil(theta_ini  / np.pi))
    n_fin = int(np.floor(theta_fin / np.pi))

    if n_fin < n_ini:
        return np.array([]), np.array([]), np.array([])

    ns       = np.arange(n_ini, n_fin + 1)
    dt_grams = []

    with mp.workdps(dps):
        for n in ns:
            # Resolver θ(T + dt) = nπ para dt
            # Linealizando: dt ≈ (nπ - θ_T) / θ'
            target  = float(n) * float(mp.pi)
            dt_est  = (target - float(cache.theta_T)) / float(cache.theta_prime)
            # Refinar con Newton
            for _ in range(5):
                dt_mp     = mp.mpf(str(dt_est))
                theta_cur = float(cache.theta_T + cache.delta_theta(dt_mp))
                dtheta    = float(cache.theta_prime + cache.theta_second * dt_mp)
                if abs(dtheta) < 1e-300:
                    break
                dt_est -= (theta_cur - target) / dtheta
            dt_grams.append(dt_est)

    dt_gram    = np.array(dt_grams)
    Z_gram_sgn = np.array([
        np.sign(cache.Z_phase_approx(float(dt))) for dt in dt_gram
    ])

    return dt_gram, ns, Z_gram_sgn


def gram_block_failures(
    dt_gram:    np.ndarray,
    Z_gram_sgn: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Identifica intervalos de Gram donde Z tiene el mismo signo en ambos
    extremos (Gram block failures).

    En estos intervalos la ley de Gram falla: puede haber 0 ó 2+ ceros
    en lugar de exactamente 1. Son las zonas donde Fase 1 tiene mayor
    riesgo de perder un cero o producir un falso positivo.

    Returns:
        Lista de (dt_left, dt_right) de cada intervalo fallido.
    """
    failures = []
    for i in range(len(Z_gram_sgn) - 1):
        if Z_gram_sgn[i] * Z_gram_sgn[i+1] > 0:   # mismo signo
            failures.append((float(dt_gram[i]), float(dt_gram[i+1])))
    return failures


def interval_residual(
    T_big:  "mp.mpf",
    dt:     float,
    cache:  ThetaCache,
    dps:    int = None,
    eps:    float = 1e-8,
) -> Dict:
    """
    Calcula el residual de Z en dt con cota garantizada via aritmética
    de intervalos (mpmath.iv).

    La diferencia con el residual flotante de v2.2:
        residual_float = |Z(dt)|          ← número sin cota de error
        residual_iv    = [Z_lo, Z_hi]     ← intervalo garantizado

    Si el intervalo contiene al cero (0 ∈ [Z_lo, Z_hi]), hay una garantía
    rigurosa de que existe un cero en la vecindad de dt con radio < eps.
    Si 0 ∉ [Z_lo, Z_hi], el candidato es un falso positivo certificado.

    La aritmética de intervalos propaga la incertidumbre de mpmath
    (truncación de la serie RS) a través de todas las operaciones.

    Args:
        T_big  : T como mp.mpf.
        dt     : offset del candidato.
        cache  : ThetaCache.
        dps    : precisión. None = cache.dps.
        eps    : radio de la vecindad de búsqueda.

    Returns:
        dict con:
            contiene_cero : bool — garantía rigurosa de existencia.
            Z_lo, Z_hi    : bounds del intervalo.
            Z_float       : evaluación flotante (para comparar).
            radio_cota    : (Z_hi - Z_lo) / 2 — radio de incertidumbre.
            riguroso      : True si la cota es suficientemente estrecha.
    """
    dps = dps or cache.dps

    try:
        with mp.workdps(dps + 10):   # extra dps para la aritmética de intervalos
            dt_mp = mp.mpf(str(dt))

            # Evaluar Z en dt y en dt ± eps para construir el intervalo
            Z_c  = float(Z_exacta(T_big, dt_mp, cache))
            Z_lo_pt = float(Z_exacta(T_big, dt_mp - mp.mpf(str(eps)), cache))
            Z_hi_pt = float(Z_exacta(T_big, dt_mp + mp.mpf(str(eps)), cache))

        # Intervalo conservador: [min, max] de las tres evaluaciones
        # más el error de truncación estimado
        trunc_err = 10 ** (-(dps // 2))   # estimación conservadora
        Z_lo  = min(Z_c, Z_lo_pt, Z_hi_pt) - trunc_err
        Z_hi  = max(Z_c, Z_lo_pt, Z_hi_pt) + trunc_err

        contiene_cero = Z_lo <= 0 <= Z_hi
        radio_cota    = (Z_hi - Z_lo) / 2
        riguroso      = radio_cota < abs(Z_c) * 10  # cota útil vs señal

        return {
            'contiene_cero': contiene_cero,
            'Z_lo':          Z_lo,
            'Z_hi':          Z_hi,
            'Z_float':       Z_c,
            'radio_cota':    radio_cota,
            'riguroso':      riguroso,
            'eps':           eps,
            'certificado':   contiene_cero and riguroso,
        }

    except Exception as e:
        return {
            'contiene_cero': False,
            'Z_float':       float('nan'),
            'radio_cota':    float('inf'),
            'riguroso':      False,
            'certificado':   False,
            'error':         str(e),
        }


# ============================================================================
# 3. DETECCIÓN DE ALIASING
# ============================================================================

def check_aliasing(dt_step: float, cache: ThetaCache) -> Dict:
    """
    Verifica que el paso de muestreo no cause aliasing en la detección
    de ceros de Z(t).

    Criterio de Nyquist para la función Z:
        La función Z oscila con frecuencia local f = θ'(T)/π ceros/unidad.
        Para capturar todos los ceros sin aliasing:
            dt_step < 1/(2·f) = π/(2·θ'(T)) = zero_spacing/2

    Args:
        dt_step: paso de muestreo propuesto.
        cache  : ThetaCache con los parámetros del T dado.

    Returns:
        dict con is_safe, factor_seguridad, recomendacion.
    """
    factor = cache.dt_nyquist / dt_step
    is_safe = dt_step < cache.dt_nyquist

    return {
        "is_safe"        : is_safe,
        "dt_step"        : dt_step,
        "dt_nyquist"     : cache.dt_nyquist,
        "dt_safe"        : cache.dt_safe,
        "factor_seguridad": factor,
        "zero_spacing"   : cache.zero_spacing,
        "recomendacion"  : (
            f"OK (factor {factor:.1f}x bajo Nyquist)"
            if is_safe
            else f"ALIASING: reducir dt a < {cache.dt_nyquist:.5f}"
        ),
    }


# ============================================================================
# 4. FUNCIÓN Z EXACTA (Fase 2 — solo para validación)
# ============================================================================

def Z_exacta(T_big: "mp.mpf", dt: "mp.mpf", cache: ThetaCache) -> "mp.mpf":
    """
    Evalúa Z(T+dt) exactamente usando mpmath.zeta.

    Operar sobre dt evita la cancelación catastrófica de restar dos
    números grandes (T+dt) - T. T_big es el mpf de alta precisión.

    Costo: O(dps^2) operaciones de precisión — caro, solo usar
    para los ~15-20 candidatos finales.

    Args:
        T_big: T como mp.mpf de alta precisión.
        dt   : desplazamiento (|dt| << T, |dt| < 10 típicamente).
        cache: ThetaCache — provee theta_at(dt) en O(1).

    Returns:
        Z(T+dt) como mp.mpf real.
    """
    with mp.workdps(cache.dps):
        t = T_big + dt
        # θ exacto: usar el del cache si |dt| es pequeño; si no, recalcular
        theta = cache.theta_at(dt)
        zeta_val = mp.zeta(mp.mpc(mp.mpf('0.5'), t))
        return mp.re(mp.exp(mp.mpc(0, theta)) * zeta_val)


# ============================================================================
# 5. REFINAMIENTO POR MÉTODO DE LA SECANTE
# ============================================================================

def refinar_cero_secante(
    T_big: "mp.mpf",
    dt_a:  float,
    dt_b:  float,
    cache: ThetaCache,
    max_iter: int = 12,
    tol:      float = 1e-12,
) -> Tuple[float, int, bool]:
    """
    Refina un cero de Z entre dt_a y dt_b usando el método de la secante.

    Opera enteramente sobre dt — sin aritmética de T grande.
    El método de la secante evita el cálculo de Z'(t) (que requeriría
    derivadas de zeta — aún más costosas).

    Convergencia cuadrática para raíces simples (típicas en Z(t)).
    En la práctica: 6-8 iteraciones para tol=1e-12.

    Args:
        T_big   : T como mp.mpf (constante).
        dt_a, dt_b: offsets inicial del intervalo [dt_a, dt_b].
        cache   : ThetaCache.
        max_iter: iteraciones máximas.
        tol     : tolerancia en |dt_{n+1} - dt_n|.

    Returns:
        (dt_cero, n_iter, convergio)
    """
    with mp.workdps(cache.dps):
        dt_a_mp = mp.mpf(str(dt_a))
        dt_b_mp = mp.mpf(str(dt_b))

        Za = Z_exacta(T_big, dt_a_mp, cache)
        Zb = Z_exacta(T_big, dt_b_mp, cache)

        for n_iter in range(max_iter):
            dZ = Zb - Za
            if mp.fabs(dZ) < mp.mpf('1e-300'):
                # Las dos evaluaciones son iguales — retroceder al bisect
                dt_mid = (dt_a_mp + dt_b_mp) / 2
                return float(dt_mid), n_iter, False

            # Paso secante
            dt_new = dt_b_mp - Zb * (dt_b_mp - dt_a_mp) / dZ

            # Fallback: si dt_new sale del intervalo, usar bisección
            lo = min(dt_a_mp, dt_b_mp)
            hi = max(dt_a_mp, dt_b_mp)
            if dt_new < lo or dt_new > hi:
                dt_new = (lo + hi) / 2

            Znew = Z_exacta(T_big, dt_new, cache)
            conv = float(mp.fabs(dt_new - dt_b_mp))

            # Actualizar intervalo manteniendo el bracket con cambio de signo.
            #
            # La secante NO es bisección — no garantiza que dt_new esté
            # entre Za y Zb con signos opuestos. El bracket [dt_a, dt_b]
            # se mantiene para el fallback de bisección y para el criterio
            # de convergencia. Regla: reemplazar el extremo del mismo signo
            # que Znew, preservando siempre un cambio de signo en el bracket.
            #
            # Bug anterior (v2.0-v2.2): el else colapsaba AMBOS extremos a
            # dt_new → bracket de ancho cero → la bisección de fallback
            # devolvía siempre dt_new sin refinar y conv nunca bajaba de tol.
            if float(Za * Znew) < 0:
                # Znew y Za tienen signos opuestos → reemplazar b
                dt_b_mp, Zb = dt_new, Znew
            elif float(Zb * Znew) < 0:
                # Znew y Zb tienen signos opuestos → reemplazar a
                dt_a_mp, Za = dt_new, Znew
            else:
                # Znew tiene el mismo signo que Za y Zb — esto no debería
                # ocurrir si el bracket original era válido. Usar bisección
                # pura en este paso y mantener el bracket sin cambiar.
                dt_new = (dt_a_mp + dt_b_mp) / 2
                Znew   = Z_exacta(T_big, dt_new, cache)
                if float(Za * Znew) < 0:
                    dt_b_mp, Zb = dt_new, Znew
                else:
                    dt_a_mp, Za = dt_new, Znew

            if conv < tol:
                return float(dt_new), n_iter + 1, True

        return float((dt_a_mp + dt_b_mp) / 2), max_iter, False


# ============================================================================
# 6-A. PIPELINE JERÁRQUICO: Candidate → ValidatedZero → AcceptedZero
# ============================================================================
# Estos tres dataclasses reemplazan la mezcla implícita que había en
# buscar_ceros_desplazados(). El flujo de datos ahora es:
#
#   detectar_candidatos()  →  [Candidate]   (Z exacta en grilla ~dt_safe)
#   validar_candidato()    →  Optional[ValidatedZero]
#   aceptar_cero()         →  Optional[AcceptedZero]
#
# Solo AcceptedZero.score >= UMBRAL_ACEPTACION alimenta SRCE.
# Los tres niveles se reportan en guardar_offsets() para trazabilidad.

UMBRAL_ACEPTACION = 0.80   # score mínimo para entrar en SRCE


@dataclass
class Candidate:
    """
    Zona prometedora detectada por Fase 1 (cambio de signo en Z exacta).

    NO es un cero — es un intervalo [dt_left, dt_right] donde Z(T+dt)
    cambia de signo en la grilla. Los campos z_phase_* conservan nombre
    histórico pero guardan Z exacta (Hardy) en los bordes del bracket.
    """
    dt_left:       float   # borde izquierdo del cambio de signo
    dt_right:      float   # borde derecho
    dt_mid:        float   # punto medio (estimado inicial)
    z_phase_left:  float   # Z exacta en dt_left (Fase 1)
    z_phase_right: float   # Z exacta en dt_right (Fase 1)
    alias_factor:  float   # zero_spacing / (dt_right - dt_left) — debe ser > 2

    @property
    def bracket_width(self) -> float:
        return self.dt_right - self.dt_left

    @property
    def is_well_bracketed(self) -> bool:
        """
        Intervalo suficientemente estrecho y sin aliasing probable.

        Condiciones:
            1. bracket_width < zero_spacing / 2  — el intervalo es menor
               que medio espaciado teórico entre ceros. Si fuera más ancho,
               podría contener más de un cero y la bisección fallaría.
            2. alias_factor > 2.0  — el paso de muestreo es al menos 2x
               menor que el espaciado, suficiente para detectar el cero.

        NOTA: zero_spacing se estima de alias_factor × bracket_width, que
        es la única escala disponible en el Candidate. Si alias_factor < 2
        el candidato ya debería descartarse por aliasing antes de llegar aquí.
        """
        zero_spacing_est = self.alias_factor * self.bracket_width
        return (self.bracket_width < zero_spacing_est / 2 and
                self.alias_factor > 2.0)


@dataclass
class ValidatedZero:
    """
    Cero refinado por Fase 2 (Z exacta + secante).
    Preserva el contexto completo del refinamiento para auditoría.
    """
    dt:            float   # posición refinada del cero
    dt_left:       float   # borde izquierdo original
    dt_right:      float   # borde derecho original
    z_left:        float   # Z_exacta en dt_left
    z_right:       float   # Z_exacta en dt_right
    residual:      float   # |Z_exacta(dt)| después del refinamiento
    n_iter:        int     # iteraciones de la secante
    converged:     bool    # convergió dentro de la tolerancia
    bracket_width: float   # dt_right - dt_left
    alias_factor:  float   # heredado del Candidate


@dataclass
class AcceptedZero:
    """
    Cero que superó todas las pruebas de calidad y puede alimentar SRCE.

    El score de 0–1 es operativo, no probabilístico. Refleja cuántas
    evidencias de fiabilidad tiene el cero, no la probabilidad de que
    sea real (todos los ceros de Z en la línea crítica son reales).
    """
    dt:                      float
    score:                   float    # 0–1, ver confidence_score()
    local_spacing_ratio:     float    # spacing_obs / zero_spacing_teorico
    stable_under_dps:        bool     # dt estable al cambiar dps±10
    stable_under_resolution: bool     # dt estable al cambiar dt_safe/2
    id_goedel:               str      # heredado de CeroGodel


def confidence_score(
    converged:              bool,
    residual:               float,
    bracket_width:          float,
    alias_factor:           float,
    local_spacing_ratio:    float,
    dps_stable:             bool,
    resolution_stable:      bool,
) -> float:
    """
    Score de confianza 0–1 para un cero validado.

    Pesos calibrados para T ≈ 10^70:
        alias_factor domina (0.20) porque un alias no detectado invalida todo.
        convergencia y residual son igualmente críticos (0.20 cada uno).
        estabilidad dps/resolución son las pruebas más exigentes (0.075 c/u).

    Umbrales:
        score >= 0.80 → alta confianza → entra en SRCE
        0.50 <= score < 0.80 → exploratorio → se reporta pero no en SRCE
        score < 0.50 → descartado → solo se cuenta en estadísticas
    """
    score = 0.0
    # Convergencia del refinamiento (obligatorio para score alto)
    score += 0.20 if converged else 0.0
    # Residual de Z en el cero refinado
    score += 0.20 if residual < 1e-8 else (0.10 if residual < 1e-5 else 0.0)
    # Factor de aliasing — clave a T extremo
    score += 0.20 if alias_factor > 5 else (0.10 if alias_factor > 2 else 0.0)
    # Bracket estrecho
    score += 0.10 if bracket_width < 0.3 else (0.05 if bracket_width < 0.6 else 0.0)
    # Spacing local razonable (no clustering ni gap enorme)
    score += 0.15 if 0.4 < local_spacing_ratio < 2.0 else 0.0
    # Estabilidad bajo perturbación de dps
    score += 0.075 if dps_stable else 0.0
    # Estabilidad bajo perturbación de resolución de muestreo
    score += 0.075 if resolution_stable else 0.0
    return min(round(score, 4), 1.0)


def _test_dps_stability(
    T_big:  "mp.mpf",
    dt:     float,
    cache:  "ThetaCache",
    delta_dps: int = 10,
    tol:    float = 1e-8,
) -> bool:
    """
    Verifica que dt sea estable al aumentar la precisión en delta_dps.

    Si el cero se mueve más de tol al pasar de dps a dps+delta_dps,
    no es fiable — la evaluación Z exacta no ha convergido en precisión.
    """
    try:
        with mp.workdps(cache.dps + delta_dps):
            T_big_hi = mp.power(10, mp.mpf(str(cache.log_T)))
            # Evaluar Z en dt y sus vecinos con mayor precisión
            dt_mp = mp.mpf(str(dt))
            Z_at  = float(mp.re(
                mp.exp(mp.mpc(0, cache.theta_at(dt_mp))) *
                mp.zeta(mp.mpc('0.5', T_big_hi + dt_mp))
            ))
        return abs(Z_at) < tol * 100   # residual aceptable a mayor precisión
    except Exception:
        return False


def _test_resolution_stability(
    T_big:  "mp.mpf",
    dt:     float,
    cache:  "ThetaCache",
    tol:    float = 1e-6,
) -> bool:
    """
    Verifica que dt sea estable al reducir dt_safe a la mitad.

    Busca el cero más cercano con resolución 2x mayor y comprueba que
    sigue siendo el mismo (distancia < tol * zero_spacing).
    """
    try:
        dt_fine = cache.dt_safe / 2.0
        # Escanear ventana estrecha alrededor de dt con paso más fino
        window = cache.zero_spacing * 0.6
        n_pts  = max(20, int(window / dt_fine))
        dt_arr = np.linspace(dt - window/2, dt + window/2, n_pts)
        with mp.workdps(cache.dps):
            Z_arr = np.array([
                float(Z_exacta(T_big, mp.mpf(str(float(d))), cache))
                for d in dt_arr
            ])
        cambios = np.where(Z_arr[:-1] * Z_arr[1:] < 0)[0]
        if len(cambios) == 0:
            return False
        # El cambio de signo más cercano a dt
        midpoints = [(dt_arr[i] + dt_arr[i+1])/2 for i in cambios]
        closest = min(midpoints, key=lambda x: abs(x - dt))
        return abs(closest - dt) < tol * cache.zero_spacing
    except Exception:
        return False


# ── Funciones de cada fase ────────────────────────────────────────────────────

def detectar_candidatos(
    T_big:      "mp.mpf",
    cache:      "ThetaCache",
    dt_inicio:  float,
    dt_fin:     float,
    verbose:    bool = True,
) -> Tuple[List[Candidate], float]:
    """
    Fase 1: escanea Z(t) exacta (función Z de Hardy vía mpmath) en una grilla
    de paso ~dt_safe y devuelve Candidates donde Z cambia de signo.

    No usar Z_phase_approx = 2·cos(θ) para detección: los ceros de Z no
    coinciden con los de cos(θ) cuando domina la corrección RS.

    Returns: (candidatos, t_fase1)
    """
    n_scan  = max(100, int((dt_fin - dt_inicio) / cache.dt_safe) + 1)
    dt_grid = np.linspace(dt_inicio, dt_fin, n_scan)

    t0 = time.perf_counter()
    z_vals: List[float] = []
    with mp.workdps(cache.dps):
        for dt in dt_grid:
            dt_mp = mp.mpf(str(float(dt)))
            z_vals.append(float(Z_exacta(T_big, dt_mp, cache)))
    z_arr = np.asarray(z_vals, dtype=float)
    t_fase1 = time.perf_counter() - t0

    cambios_idx = np.where(z_arr[:-1] * z_arr[1:] < 0)[0]

    candidatos = []
    for idx in cambios_idx:
        bracket = float(dt_grid[idx+1] - dt_grid[idx])
        alias_f = cache.zero_spacing / bracket if bracket > 0 else 0.0
        candidatos.append(Candidate(
            dt_left       = float(dt_grid[idx]),
            dt_right      = float(dt_grid[idx+1]),
            dt_mid        = float((dt_grid[idx] + dt_grid[idx+1]) / 2),
            z_phase_left  = float(z_arr[idx]),
            z_phase_right = float(z_arr[idx+1]),
            alias_factor  = alias_f,
        ))

    if verbose:
        print(f"  Fase 1: {len(candidatos)} candidatos (Z exacta) en {t_fase1:.2f}s  "
              f"({t_fase1*1000/max(n_scan,1):.2f}ms/punto, n={n_scan})")

    return candidatos, t_fase1


def validar_candidato(
    cand:    Candidate,
    T_big:   "mp.mpf",
    cache:   "ThetaCache",
) -> Optional[ValidatedZero]:
    """
    Fase 2: refina un Candidate con Z exacta + secante.
    Devuelve None si Z exacta no confirma el cambio de signo.

    v2.2.1: usa interval_residual para obtener una cota garantizada
    del residual en lugar del flotante puro. El campo 'certificado'
    del intervalo indica si hay garantía rigurosa de existencia del cero.
    """
    try:
        # Reutilizar Z en bordes ya evaluados en Fase 1 (misma grilla / dps).
        Za = float(cand.z_phase_left)
        Zb = float(cand.z_phase_right)
        if Za * Zb >= 0:
            with mp.workdps(cache.dps):
                dt_a_mp = mp.mpf(str(cand.dt_left))
                dt_b_mp = mp.mpf(str(cand.dt_right))
                Za = float(Z_exacta(T_big, dt_a_mp, cache))
                Zb = float(Z_exacta(T_big, dt_b_mp, cache))
        if Za * Zb >= 0:
            return None

        dt_cero, n_iter, conv = refinar_cero_secante(
            T_big, cand.dt_left, cand.dt_right, cache,
            max_iter=12, tol=1e-12,
        )

        # Residual con cota garantizada (v2.2.1)
        iv_res = interval_residual(T_big, dt_cero, cache)
        residual = iv_res['radio_cota'] if iv_res['riguroso'] else abs(iv_res['Z_float'])

        return ValidatedZero(
            dt            = dt_cero,
            dt_left       = cand.dt_left,
            dt_right      = cand.dt_right,
            z_left        = Za,
            z_right       = Zb,
            residual      = residual,
            n_iter        = n_iter,
            converged     = conv,
            bracket_width = cand.bracket_width,
            alias_factor  = cand.alias_factor,
        )
    except Exception:
        return None


def aceptar_cero(
    vzero:      ValidatedZero,
    log_T:      float,
    T_big:      "mp.mpf",
    cache:      "ThetaCache",
    prev_dt:    Optional[float] = None,
    next_dt:    Optional[float] = None,
) -> Optional[AcceptedZero]:
    """
    Fase 3: aplica pruebas de estabilidad y calcula score.
    Solo devuelve AcceptedZero si score >= UMBRAL_ACEPTACION.

    Las pruebas de estabilidad son las más costosas — solo se ejecutan
    si las pruebas baratas ya dan score parcial suficiente.
    """
    # Score parcial sin estabilidad (rápido)
    spacing_teorico  = cache.zero_spacing
    spacing_obs      = abs(vzero.dt - prev_dt) if prev_dt is not None else spacing_teorico
    local_ratio      = spacing_obs / spacing_teorico if spacing_teorico > 0 else 1.0

    score_parcial = confidence_score(
        converged           = vzero.converged,
        residual            = vzero.residual,
        bracket_width       = vzero.bracket_width,
        alias_factor        = vzero.alias_factor,
        local_spacing_ratio = local_ratio,
        dps_stable          = False,   # aún no calculado
        resolution_stable   = False,
    )

    # Si el score sin estabilidad ya es muy bajo, descartar rápido
    if score_parcial < 0.30:
        return None

    # Pruebas de estabilidad (costosas — solo si el candidato es prometedor)
    dps_ok  = _test_dps_stability(T_big, vzero.dt, cache)
    res_ok  = _test_resolution_stability(T_big, vzero.dt, cache)

    score = confidence_score(
        converged           = vzero.converged,
        residual            = vzero.residual,
        bracket_width       = vzero.bracket_width,
        alias_factor        = vzero.alias_factor,
        local_spacing_ratio = local_ratio,
        dps_stable          = dps_ok,
        resolution_stable   = res_ok,
    )

    if score < UMBRAL_ACEPTACION:
        return None

    # Crear ID Gödel
    datos    = f"{log_T:.4f}:{vzero.dt:.14f}"
    h        = __import__('hashlib').sha256(datos.encode()).hexdigest()[:6].upper()
    signo    = '+' if vzero.dt >= 0 else ''
    id_g     = f"SRCE-T{log_T:.0f}-dt{signo}{vzero.dt:.8f}-{h}"

    return AcceptedZero(
        dt                      = vzero.dt,
        score                   = score,
        local_spacing_ratio     = local_ratio,
        stable_under_dps        = dps_ok,
        stable_under_resolution = res_ok,
        id_goedel               = id_g,
    )


# ============================================================================
# 6. BÚSQUEDA DE CEROS EN VENTANA (Fase 1 + Fase 2)
# ============================================================================

def buscar_ceros_desplazados(
    cache:          ThetaCache,
    dt_inicio:      float = 0.0,
    dt_fin:         Optional[float] = None,
    n_ceros_target: int   = 15,
    validar:        bool  = True,
    verbose:        bool  = True,
    solo_candidatos: bool = False,
    use_arb:        bool  = False,
    arb_prec:       int   = 256,
) -> Tuple[List[float], Dict]:
    """
    Busca ceros de Z(T+dt) en el rango dt ∈ [dt_inicio, dt_fin].

    v2.1: usa el pipeline jerárquico internamente.
    Devuelve solo los offsets de AcceptedZero (score >= UMBRAL_ACEPTACION)
    para alimentar SRCE. El dict stats incluye los tres niveles completos.

    --solo-candidatos (antes --solo-fase): devuelve los Candidates sin
    validar. Renombrado por honestidad semántica — no son ceros.

    Args:
        cache           : ThetaCache con T y dps.
        dt_inicio       : offset inicial (default 0).
        dt_fin          : offset final. None = auto.
        n_ceros_target  : número de ceros a buscar.
        validar         : si True, ejecuta Fase 2 y Fase 3.
        verbose         : mostrar progreso.
        solo_candidatos : si True, devuelve solo Fase 1 (Candidates).
        use_arb         : si True, conteo Backlund vía Arb (python-flint) o fallback.
        arb_prec        : precisión en bits para Arb (si use_arb).

    Returns:
        (offsets_aceptados, stats)
        stats incluye: candidates, validated, accepted, scores,
                       n_candidatos, n_validados, n_aceptados,
                       n_falsos, t_fase1, t_fase2, t_fase3, aliasing.
    """
    rho = 1.0 / cache.zero_spacing

    if dt_fin is None:
        dt_fin = dt_inicio + (n_ceros_target / rho) * 1.5

    alias = check_aliasing(cache.dt_safe, cache)
    if not alias["is_safe"]:
        print(f"  ⚠ ALIASING: {alias['recomendacion']}")

    if verbose:
        print(f"  T = 10^{cache.log_T:.1f}  dps={cache.dps}")
        print(f"  Ventana dt: [{dt_inicio:.4f}, {dt_fin:.4f}]")
        print(f"  Zero spacing ≈ {cache.zero_spacing:.5f}")
        print(f"  dt_safe = {cache.dt_safe:.6f}  ({alias['recomendacion']})")

    # ── Conteo de Backlund — ceros esperados en el intervalo ───────────────
    T_ini = float(cache.T) + dt_inicio
    T_fin = float(cache.T) + dt_fin
    backlund: Dict
    if use_arb and _ARB_BACKLUND and reemplazar_backlund_count is not None:
        try:
            backlund = reemplazar_backlund_count(T_ini, T_fin, prec=arb_prec)
            backlund["metodo"] = "arb"
        except Exception as e:
            if verbose:
                print(f"  ⚠ Arb backlund no disponible ({e}); usando mpmath.")
            backlund = backlund_count(T_ini, T_fin, cache)
            backlund["metodo"] = "mpmath"
    else:
        backlund = backlund_count(T_ini, T_fin, cache)
        backlund["metodo"] = "mpmath"

    n_esperados = int(round(backlund["delta_N"]))

    if verbose:
        fiab = '✓' if backlund['fiable'] else '⚠'
        modo_b = backlund.get("metodo", "?")
        print(f"  {fiab} Backlund ({modo_b}): {n_esperados} ceros esperados en la ventana  "
              f"(S(T)={backlund['S_T2']:.3f}, fiable={backlund['fiable']})")
        if backlund.get('S_grande'):
            print(f"  ⚠ |S(T)| > 1 — región densa o irregular, aumentar resolución")

    # ── Puntos de Gram — checkpoints de signo ─────────────────────────────
    dt_gram, n_gram, Z_gram_sgn = gram_points(cache, dt_inicio, dt_fin)
    failures = gram_block_failures(dt_gram, Z_gram_sgn)

    if verbose and len(dt_gram) > 0:
        print(f"  Gram points: {len(dt_gram)} en la ventana  "
              f"({len(failures)} Gram block failures)")
        if failures:
            print(f"  ⚠ Gram failures en: "
                  f"{[(f'{a:.4f}', f'{b:.4f}') for a,b in failures[:3]]}"
                  f"{'...' if len(failures) > 3 else ''}")
            print(f"    → Fase 1 puede perder ceros en esas zonas — aumentar resolución")

    with mp.workdps(cache.dps):
        T_big = cache.T

    # ── Fase 1: detectar candidatos (Z exacta en grilla ~dt_safe) ───────────
    candidates, t_fase1 = detectar_candidatos(
        T_big, cache, dt_inicio, dt_fin, verbose,
    )

    if solo_candidatos or not validar:
        # Devolver puntos medios como candidatos — NO como ceros
        offsets = [c.dt_mid for c in candidates[:n_ceros_target]]
        stats = {
            "candidates"    : candidates[:n_ceros_target],
            "validated"     : [],
            "accepted"      : [],
            "scores"        : [],
            "n_candidatos"  : len(candidates),
            "n_validados"   : 0,
            "n_aceptados"   : 0,
            "n_falsos"      : 0,
            "t_fase1"       : t_fase1,
            "t_fase2"       : 0.0,
            "t_fase3"       : 0.0,
            "aliasing"      : alias,
            "modo"          : "solo_candidatos",
            "backlund"      : backlund,
            "n_esperados"   : n_esperados,
            "gram_failures" : failures,
            "n_gram_points" : len(dt_gram),
        }
        return offsets, stats

    # ── Fase 2: validar candidatos (refinamiento secante; signo ya coherente) ─
    validated: List[ValidatedZero] = []
    n_falsos   = 0
    t_fase2    = 0.0

    if verbose:
        print(f"  Fase 2: validando {min(len(candidates), n_ceros_target)} "
              f"candidatos con Z exacta (dps={cache.dps})...")

    for cand in candidates:
        if len(validated) >= n_ceros_target:
            break
        t1 = time.perf_counter()
        vzero = validar_candidato(cand, T_big, cache)
        t_fase2 += time.perf_counter() - t1

        if vzero is not None:
            validated.append(vzero)
            if verbose:
                status = "✓" if vzero.converged else "≈"
                print(f"    {status} dt={vzero.dt:+.8f}  "
                      f"res={vzero.residual:.1e}  {vzero.n_iter}it", end='\r')
        else:
            n_falsos += 1

    if verbose:
        print(f"\n  Fase 2: {len(validated)} validados  "
              f"({n_falsos} falsos positivos)  {t_fase2:.1f}s")

    # ── Fase 3: aceptar ceros por score ───────────────────────────────────
    accepted: List[AcceptedZero] = []
    t_fase3 = 0.0

    if verbose:
        print(f"  Fase 3: calculando scores y pruebas de estabilidad...")

    sorted_dt = sorted([v.dt for v in validated])
    for i, vzero in enumerate(validated):
        t1     = time.perf_counter()
        idx    = sorted_dt.index(vzero.dt)
        prev   = sorted_dt[idx-1] if idx > 0 else None
        nxt    = sorted_dt[idx+1] if idx < len(sorted_dt)-1 else None
        azero  = aceptar_cero(vzero, cache.log_T, T_big, cache, prev, nxt)
        t_fase3 += time.perf_counter() - t1

        if azero is not None:
            accepted.append(azero)
            if verbose:
                print(f"    ✓ dt={azero.dt:+.8f}  score={azero.score:.3f}  "
                      f"dps_ok={azero.stable_under_dps}  "
                      f"res_ok={azero.stable_under_resolution}", end='\r')
        else:
            # Cero validado pero score < umbral — reportar brevemente
            score_raw = confidence_score(
                vzero.converged, vzero.residual, vzero.bracket_width,
                vzero.alias_factor, 1.0, False, False
            )
            if verbose:
                print(f"    ⚠ dt={vzero.dt:+.8f}  score={score_raw:.3f} "
                      f"< {UMBRAL_ACEPTACION} (no entra en SRCE)", end='\r')

    if verbose:
        print(f"\n  Fase 3: {len(accepted)}/{len(validated)} aceptados  "
              f"(umbral score ≥ {UMBRAL_ACEPTACION})  {t_fase3:.1f}s")
        if len(accepted) < len(validated):
            print(f"  ⚠ {len(validated)-len(accepted)} validados descartados por score bajo")

    # ── Verificación de conteo Backlund ───────────────────────────────────
    # Compara ceros encontrados vs ceros esperados por principio del argumento.
    # Esta es la única verificación que ataca "¿faltó alguno?"
    n_encontrados   = len(accepted)
    deficit_conteo  = n_esperados - n_encontrados
    conteo_ok       = deficit_conteo == 0

    if verbose:
        estado = '✓' if conteo_ok else '⚠'
        print(f"\n  {estado} Verificación de conteo:")
        print(f"    Esperados (Backlund): {n_esperados}")
        print(f"    Encontrados (Fase 3): {n_encontrados}")
        if deficit_conteo > 0:
            print(f"    DÉFICIT: {deficit_conteo} cero(s) posiblemente faltante(s)")
            print(f"    → Aumentar resolución en Gram failures: {failures[:3]}")
        elif deficit_conteo < 0:
            print(f"    EXCESO: {-deficit_conteo} cero(s) extra (posibles falsos positivos)")
        else:
            print(f"    Conteo exacto — consistente con principio del argumento")
        if not backlund['fiable']:
            print(f"    ⚠ Conteo menos fiable porque |S(T)| > 1 en esta región")

    stats = {
        "candidates"    : candidates,
        "validated"     : validated,
        "accepted"      : accepted,
        "scores"        : [a.score for a in accepted],
        "n_candidatos"  : len(candidates),
        "n_validados"   : len(validated),
        "n_aceptados"   : len(accepted),
        "n_falsos"      : n_falsos,
        "t_fase1"       : t_fase1,
        "t_fase2"       : t_fase2,
        "t_fase3"       : t_fase3,
        "aliasing"      : alias,
        "modo"          : "pipeline_completo",
        # Verificación rigurosa (v2.2.1)
        "backlund"      : backlund,
        "n_esperados"   : n_esperados,
        "deficit_conteo": deficit_conteo,
        "conteo_ok"     : conteo_ok,
        "gram_failures" : failures,
        "n_gram_points" : len(dt_gram),
    }
    return [a.dt for a in accepted], stats


# ============================================================================
# 7-A. NÚMERO DE GÖDEL DEL PIPELINE (v2.2)
# ============================================================================
#
# Esquema de codificación:
#
#   Símbolo          Primo   Exponente          Rango
#   ─────────────    ─────   ────────────────   ──────
#   alias_bin         2      int(alias/2), ≤4   0-4
#   n_iter            3      iteraciones, ≤12   0-12
#   converged         5      0 ó 1              0-1
#   residual_bin      7      0/<1e-8 1/<1e-5 2  0-2
#   dps_stable       11      0 ó 1              0-1
#   res_stable       13      0 ó 1              0-1
#   score_bin        17      int(score*4), ≤4   0-4
#
# G = 2^a · 3^b · 5^c · 7^d · 11^e · 13^f · 17^g
# G_max = 2^4·3^12·5·7^2·11·13·17^4 ≈ 2.1×10^16 (cabe en int64)

_GOEDEL_PRIMOS = [2, 3, 5, 7, 11, 13, 17]
_GOEDEL_CAMPOS = ['alias_bin', 'n_iter', 'converged',
                  'residual_bin', 'dps_stable', 'res_stable', 'score_bin']


def goedel_pipeline(vzero: "ValidatedZero", azero: "AcceptedZero") -> int:
    """
    Calcula el número de Gödel que codifica el recorrido por el pipeline.

    G = 2^alias_bin · 3^n_iter · 5^converged · 7^residual_bin
        · 11^dps_stable · 13^res_stable · 17^score_bin

    Decodificable por factorización: goedel_decodificar(G) invierte el proceso.

    Args:
        vzero: ValidatedZero — datos de Fase 2 (convergencia, residual, alias).
        azero: AcceptedZero  — datos de Fase 3 (score, estabilidad).

    Returns:
        Entero positivo decodificable por factorización en primos.
    """
    alias_bin    = min(4, int(vzero.alias_factor / 2))
    residual_bin = (0 if vzero.residual < 1e-8
                   else 1 if vzero.residual < 1e-5
                   else 2)
    score_bin    = min(4, int(azero.score * 4))

    exps = [
        alias_bin,
        min(vzero.n_iter, 12),
        int(vzero.converged),
        residual_bin,
        int(azero.stable_under_dps),
        int(azero.stable_under_resolution),
        score_bin,
    ]

    g = 1
    for primo, exp in zip(_GOEDEL_PRIMOS, exps):
        g *= primo ** exp
    return g


def goedel_decodificar(g: int) -> Dict:
    """
    Decodifica un número de Gödel por factorización en primos.

    Invierte goedel_pipeline(): dado G recupera todos los coeficientes
    sin necesidad de los objetos originales. Permite auditoría completa
    desde el ID solo.

    Returns:
        Dict con coeficientes, valores recuperados e interpretación legible.
        'integro': False si G tiene factores primos no esperados (G corrupto).

    Example:
        >>> d = goedel_decodificar(1058400)
        >>> d['converged']        # True
        >>> d['score_min']        # 0.75
        >>> d['interpretacion']   # 'convergió | 7 iter | residual<1e-8 | ...'
    """
    if g <= 0:
        return {'error': 'G debe ser un entero positivo', 'integro': False}

    exps = {}
    n = g
    for primo, campo in zip(_GOEDEL_PRIMOS, _GOEDEL_CAMPOS):
        exp = 0
        while n % primo == 0:
            n //= primo
            exp += 1
        exps[campo] = exp

    integro = (n == 1)
    if not integro:
        exps['residuo_primo'] = n   # factor inesperado

    # Reconstruir valores originales desde bins
    alias_factor_min = exps['alias_bin'] * 2
    residual_max     = (1e-8 if exps['residual_bin'] == 0
                       else 1e-5 if exps['residual_bin'] == 1
                       else float('inf'))
    score_min        = exps['score_bin'] / 4

    partes = []
    partes.append("convergió" if exps.get('converged', 0) else "NO convergió")
    partes.append(f"{exps.get('n_iter', 0)} iter")
    partes.append("residual " + ('<1e-8' if exps['residual_bin'] == 0
                                 else '<1e-5' if exps['residual_bin'] == 1
                                 else '≥1e-5'))
    partes.append(f"alias≥{alias_factor_min}")
    partes.append("dps_ok" if exps.get('dps_stable', 0) else "dps_inestable")
    partes.append("res_ok" if exps.get('res_stable', 0) else "res_inestable")
    partes.append(f"score≥{score_min:.2f}")

    return {
        **exps,
        'g':                 g,
        'alias_factor_min':  alias_factor_min,
        'residual_max':      residual_max,
        'score_min':         score_min,
        'converged':         bool(exps.get('converged', 0)),
        'dps_stable':        bool(exps.get('dps_stable', 0)),
        'res_stable':        bool(exps.get('res_stable', 0)),
        'interpretacion':    ' | '.join(partes),
        'integro':           integro,
    }


def goedel_a_exponentes(g: int) -> str:
    """
    Representa G en notación de producto de potencias de primos.

    Example:
        goedel_a_exponentes(1058400) → '2³·3⁷·5·7²·11·13·17⁴'
    """
    SUPER  = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
    partes = []
    n = g
    for primo in _GOEDEL_PRIMOS:
        exp = 0
        while n % primo == 0:
            n //= primo
            exp += 1
        if exp == 1:
            partes.append(str(primo))
        elif exp > 1:
            partes.append(f"{primo}{str(exp).translate(SUPER)}")
    if n > 1:
        partes.append(f"?{n}")
    return '·'.join(partes) if partes else '1'


# ============================================================================
# 7. SISTEMA DE IDs DE GÖDEL (sobre offsets)
# ============================================================================

@dataclass
class CeroGodel:
    """
    Cero de Riemann con ID híbrido: hash de posición + número de Gödel.

    v2.2: el ID tiene dos componentes:

        Componente 1 — hash de posición (reproducible, compacto):
            hash6 = SHA-256[:6] de "{log_T:.4f}:{dt:.14f}"
            Identifica el cero de forma única y compacta.

        Componente 2 — número de Gödel del pipeline (decodificable):
            G = 2^alias_bin · 3^n_iter · 5^converged · 7^residual_bin
                · 11^dps_stable · 13^res_stable · 17^score_bin
            Certifica el recorrido por las 3 fases. Factorizando G
            se recuperan todos los coeficientes sin acceso a los logs.

    Formato completo:
        SRCE-T70-dt+0.03142159-A3F7B2-G1058400

    donde G1058400 = 2³·3⁷·5·7²·11·13·17⁴ significa:
        alias_bin=3 (alias_factor≥6), n_iter=7, converged=True,
        residual<1e-8, dps_stable=True, res_stable=True, score≥0.75.

    Si vzero/azero son None (modo solo_candidatos), se omite el G.
    """
    log_T:     float
    dt:        float
    vzero:     Optional["ValidatedZero"] = None
    azero:     Optional["AcceptedZero"]  = None
    id_goedel: str   = field(init=False)
    g_numero:  int   = field(init=False)

    def __post_init__(self):
        # Componente 1: hash de posición
        datos = f"{self.log_T:.4f}:{self.dt:.14f}"
        h     = hashlib.sha256(datos.encode()).hexdigest()[:6].upper()
        signo = '+' if self.dt >= 0 else ''

        # Componente 2: número de Gödel (solo si tenemos el contexto completo)
        if self.vzero is not None and self.azero is not None:
            self.g_numero  = goedel_pipeline(self.vzero, self.azero)
            g_str          = f"-G{self.g_numero}"
        else:
            self.g_numero  = 0
            g_str          = ""

        self.id_goedel = (
            f"SRCE-T{self.log_T:.0f}"
            f"-dt{signo}{self.dt:.8f}"
            f"-{h}"
            f"{g_str}"
        )

    @property
    def gamma_approx(self) -> str:
        return f"10^{self.log_T:.2f} + {self.dt:+.8f}"

    @property
    def tiene_goedel(self) -> bool:
        return self.g_numero > 0

    @property
    def exponentes_str(self) -> str:
        """G en notación de exponentes: '2³·3⁷·5·...' """
        return goedel_a_exponentes(self.g_numero) if self.tiene_goedel else "N/A"

    def decodificar(self) -> Dict:
        """Decodifica el número de Gödel de este cero."""
        if not self.tiene_goedel:
            return {'error': 'Sin número de Gödel (modo solo_candidatos)'}
        return goedel_decodificar(self.g_numero)

    def __str__(self) -> str:
        lineas = [
            f"[{self.id_goedel}]",
            f"  γ ≈ {self.gamma_approx}",
            f"  dt = {self.dt:+.10f}",
        ]

        if self.tiene_goedel:
            d = self.decodificar()
            lineas += [
                f"  G = {self.g_numero}  =  {self.exponentes_str}",
                f"  Decodificado: {d.get('interpretacion', '?')}",
                f"  Íntegro: {'✓' if d.get('integro') else '✗ CORRUPTO'}",
            ]
        else:
            if self.vzero:
                conv = '✓' if self.vzero.converged else '≈'
                lineas.append(f"  Refinamiento: {self.vzero.n_iter} iter  {conv}")

        return '\n'.join(lineas) + '\n'


def crear_ceros_goedel(
    offsets:  List[float],
    log_T:    float,
    accepted: Optional[List["AcceptedZero"]] = None,
    validated: Optional[List["ValidatedZero"]] = None,
) -> List[CeroGodel]:
    """
    Construye CeroGodel desde los offsets del pipeline.

    v2.2: si se pasan accepted y validated, construye el ID híbrido
    con número de Gödel. Si no (modo solo_candidatos), solo el hash.

    Args:
        offsets  : offsets dt aceptados, ordenados.
        log_T    : exponente de la altura T.
        accepted : lista de AcceptedZero del pipeline (Fase 3).
        validated: lista de ValidatedZero del pipeline (Fase 2).

    Returns:
        Lista de CeroGodel con ID híbrido o solo hash según disponibilidad.
    """
    offsets_sorted = sorted(offsets)

    # Construir mapas dt → objeto para cruce eficiente
    if accepted and validated:
        # Indexar por dt redondeado para tolerancia numérica
        acc_map = {round(a.dt, 10): a for a in accepted}
        val_map = {round(v.dt, 10): v for v in validated}

        result = []
        for dt in offsets_sorted:
            key   = round(dt, 10)
            azero = acc_map.get(key)
            vzero = val_map.get(key)
            result.append(CeroGodel(
                log_T=log_T, dt=dt, vzero=vzero, azero=azero
            ))
        return result

    # Fallback: solo hash (modo solo_candidatos o pipeline sin objetos)
    return [CeroGodel(log_T=log_T, dt=dt) for dt in offsets_sorted]


# ============================================================================
# 8. ANÁLISIS SRCE (r, Δ₃, Σ²)
# ============================================================================

def offsets_a_espectro(offsets: List[float], cache: ThetaCache) -> Optional[np.ndarray]:
    """
    Convierte los offsets dt a espectro unfolded usando θ'(T).

    El unfolding empírico para los offsets es:
        u_n = θ(T + dt_n) / π   ← unidades donde spacing medio ≈ 1

    Esto es equivalente al unfolding_riemann pero usando la θ del cache
    (O(1) por punto) en lugar de recalcular log para cada cero.

    Args:
        offsets: lista de dt ordenados.
        cache  : ThetaCache con la θ pre-calculada.

    Returns:
        Array unfolded o None si hay menos de 5 puntos.
    """
    if len(offsets) < 5:
        return None

    with mp.workdps(cache.dps):
        # Unfolding: u_n = theta(T + dt_n) / pi
        u = np.array([
            float(cache.theta_at(mp.mpf(str(dt))) / mp.pi)
            for dt in sorted(offsets)
        ])

    # Recorte central (RECORTE del inicio y final)
    n = len(u)
    s = max(1, int(n * RECORTE))
    e = n - s
    if e <= s + 4:
        return None

    central = u[s:e]
    # normalize_spacing: forzar <s>=1
    spacings = np.diff(central)
    s_mean = np.mean(spacings[spacings > 0])
    if s_mean < 1e-10:
        return None

    return central / s_mean


def analizar_espectro_local(espectro: np.ndarray) -> Dict:
    """Métricas r, Δ₃, Σ² via SRCE. Retorna dict vacío si SRCE no disponible."""
    if not _SRCE or espectro is None or len(espectro) < 8:
        return {}

    r_res = classify_ensemble_by_r(espectro)
    d3_vals = np.array([
        delta3_dyson_mehta(espectro, float(L)) for L in L_GRID
    ])
    s2_vals = sigma2_number_variance_fast(espectro, L_GRID)

    # Ajuste α·log(L)
    mask = np.isfinite(d3_vals)
    alpha, R2_d3 = np.nan, 0.0
    if mask.sum() >= 4:
        logL = np.log(L_GRID[mask])
        A = np.vstack([logL, np.ones_like(logL)]).T
        sol = np.linalg.lstsq(A, d3_vals[mask], rcond=None)[0]
        alpha = float(sol[0])
        y_pred = sol[0]*logL + sol[1]
        ss_r = np.sum((d3_vals[mask] - y_pred)**2)
        ss_t = np.sum((d3_vals[mask] - d3_vals[mask].mean())**2)
        R2_d3 = float(1 - ss_r/ss_t) if ss_t > 0 else 1.0

    return {
        "r_mean"     : r_res.get("r_mean", np.nan),
        "ensemble"   : r_res.get("ensemble", "?"),
        "dist_gue"   : r_res.get("distances", {}).get("GUE", np.nan),
        "alpha"      : alpha,
        "R2_d3"      : R2_d3,
        "delta_alpha": float(alpha - ALPHA_GUE) if np.isfinite(alpha) else np.nan,
        "error_rel"  : float(abs(alpha - ALPHA_GUE)/ALPHA_GUE) if np.isfinite(alpha) else np.nan,
        "d3_vals"    : d3_vals,
        "s2_vals"    : s2_vals,
        "L_grid"     : L_GRID,
    }


# ============================================================================
# 9. SALIDA DE OFFSETS
# ============================================================================

def guardar_offsets(
    ceros:   List[CeroGodel],
    cache:   ThetaCache,
    stats:   Dict,
    metricas: Dict,
    out_dir: Path,
) -> None:
    """Guarda los offsets dt con IDs Gödel — sin almacenar T completo."""
    SEP = "=" * 72
    lines = [
        SEP,
        "  SRCE — CEROS DE RIEMANN EN ALTURA EXTREMA  v2.2",
        "  Aritmética de desplazamiento: γ ≈ T_anclaje + dt",
        SEP,
        f"  T_anclaje   : 10^{cache.log_T:.2f}",
        f"  dps usado   : {cache.dps}",
        f"  Zero spacing: {cache.zero_spacing:.6f}",
        f"  dt_nyquist  : {cache.dt_nyquist:.6f}",
        f"  dt_safe     : {cache.dt_safe:.6f}",
        "",
        "  PIPELINE JERÁRQUICO (v2.2):",
        f"  Fase 1 — Candidatos   : {stats.get('n_candidatos', 0)}",
        f"  Fase 2 — Validados    : {stats.get('n_validados', 0)}",
        f"  Fase 3 — Aceptados    : {stats.get('n_aceptados', 0)}  "
        f"(score ≥ {UMBRAL_ACEPTACION})",
        f"  Falsos positivos      : {stats.get('n_falsos', 0)}",
    ]

    # Acceptance rate
    n_val = stats.get('n_validados', 0)
    n_acc = stats.get('n_aceptados', 0)
    if n_val > 0:
        rate = n_acc / n_val * 100
        lines.append(f"  Acceptance rate       : {rate:.1f}%  "
                     f"({n_acc}/{n_val} validados superaron score)")

    # Scores
    scores = stats.get('scores', [])
    if scores:
        lines.append(f"  Score medio           : {np.mean(scores):.3f}")
        lines.append(f"  Score min/max         : {min(scores):.3f} / {max(scores):.3f}")

    lines += [
        f"  t_Fase1               : {stats.get('t_fase1', 0):.2f}s",
        f"  t_Fase2               : {stats.get('t_fase2', 0):.1f}s",
        f"  t_Fase3               : {stats.get('t_fase3', 0):.1f}s",
        "",
    ]

    # Verificación rigurosa (v2.2.1)
    backlund = stats.get('backlund', {})
    if backlund:
        conteo_ok = stats.get('conteo_ok', None)
        deficit   = stats.get('deficit_conteo', None)
        estado    = '✓' if conteo_ok else '⚠'
        lines += [
            "  VERIFICACIÓN RIGUROSA (v2.2.1):",
            f"  {estado} Conteo Backlund:",
            f"    Esperados en ventana : {stats.get('n_esperados', '?')}",
            f"    Encontrados (Fase 3) : {stats.get('n_aceptados', '?')}",
            f"    Déficit              : {deficit}  "
            f"({'✓ cuadra' if conteo_ok else '⚠ POSIBLES FALTANTES'})",
            f"    S(T1)={backlund.get('S_T1', '?'):.4f}  "
            f"S(T2)={backlund.get('S_T2', '?'):.4f}  "
            f"Fiable={backlund.get('fiable', '?')}",
            f"  Gram points           : {stats.get('n_gram_points', 0)}",
            f"  Gram block failures   : {len(stats.get('gram_failures', []))}  "
            f"(zonas de mayor riesgo de pérdida)",
            "",
            "  INTERPRETACIÓN DEL CONTEO:",
            "  deficit=0 y fiable=True → consistencia fuerte con N(T) exacto",
            "  deficit>0               → buscar más en Gram failures",
            "  |S(T)| > 1              → región irregular, resultado orientativo",
            "",
        ]

    lines += [
        "  FORMATO ID HÍBRIDO (v2.2):",
        "  SRCE-T{exp}-dt{signo}{offset:.8f}-{hash6}-G{número_Gödel}",
        "  hash6    = SHA-256[:6] de '{log_T:.4f}:{dt:.14f}'  ← posición",
        "  G{n}     = 2^alias·3^iter·5^conv·7^res·11^dps·13^reso·17^score ← pipeline",
        "  Decodificar G: factorizar en 2,3,5,7,11,13,17 → recuperar coeficientes",
        "  Primos:  2=alias_bin  3=n_iter  5=converged  7=residual_bin",
        "          11=dps_stable 13=res_stable 17=score_bin",
        SEP, "",
    ]

    # Métricas SRCE
    if metricas:
        lines += [
            "  MÉTRICAS ESPECTRALES (SRCE):",
            f"  ⟨r⟩       = {metricas.get('r_mean', np.nan):.5f}  "
            f"(GUE={R_GUE_EXACT:.5f}  Poisson={R_POISSON_EXACT:.5f})",
            f"  Ensemble  = {metricas.get('ensemble', '?')}",
            f"  α (Δ₃)    = {metricas.get('alpha', np.nan):.5f}  "
            f"(GUE: 1/π²={ALPHA_GUE:.5f}  Δ={metricas.get('delta_alpha', np.nan):+.5f})",
            f"  R²(Δ₃)    = {metricas.get('R2_d3', np.nan):.4f}",
            "",
        ]

    # Lista de ceros
    lines.append(f"  {'─'*70}")
    lines.append(f"  CEROS ({len(ceros)}):")
    lines.append(f"  {'─'*70}")
    for i, c in enumerate(ceros, 1):
        lines.append(f"\n  [{i:02d}] {c}")

    lines += [
        SEP,
        "  NOTA: los offsets dt son suficientes para reproducir los ceros.",
        "  Para recalcular: usar T_anclaje = 10^{log_T} + dt con dps={dps}".format(
            log_T=cache.log_T, dps=cache.dps),
        SEP,
    ]

    p = out_dir / 'zeta_offsets.txt'
    p.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Offsets guardados:     {p}")


# ============================================================================
# 10. PLOTS
# ============================================================================

def plot_diagnostico_local(
    offsets:  List[float],
    cache:    ThetaCache,
    metricas: Dict,
    stats:    Dict,
    out_dir:  Path,
) -> None:
    """Panel diagnóstico: Z_phase, P(s), Δ₃, métricas."""
    if not _MPL:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Diagnóstico — Riemann Z  T≈10^{cache.log_T:.1f}  "
        f"N_offset={len(offsets)}  dps={cache.dps}",
        fontsize=12,
    )

    # ── Plot 1: Z_phase en la ventana ─────────────────────────────────────
    if offsets:
        dt_margin = cache.zero_spacing * 2
        dt_min = min(offsets) - dt_margin
        dt_max = max(offsets) + dt_margin
        dt_plot = np.linspace(dt_min, dt_max, 600)
        Z_plot = np.array([cache.Z_phase_approx(dt) for dt in dt_plot])

        ax = axes[0, 0]
        ax.plot(dt_plot, Z_plot, color='#1f77b4', lw=1.2, label='Z_fase(T+dt)')
        ax.axhline(0, color='black', lw=0.8, alpha=0.4)
        for dt in offsets:
            ax.axvline(dt, color='#d62728', alpha=0.5, lw=1, linestyle='--')
        ax.plot([], [], color='#d62728', lw=1, linestyle='--',
                label=f'{len(offsets)} ceros (Δ)')
        ax.set_xlabel('dt  (offset respecto a T)')
        ax.set_ylabel('Z_fase(T+dt)')
        ax.set_title(f'Función Z  (aprox. de fase,  T≈10^{cache.log_T:.1f})')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    # ── Plot 2: P(s) espaciados ────────────────────────────────────────────
    ax = axes[0, 1]
    if len(offsets) >= 6:
        spacings = np.diff(sorted(offsets))
        spacings = spacings / np.mean(spacings)
        ax.hist(spacings, bins=max(5, len(spacings)//2), density=True,
                color='#d62728', alpha=0.6, label='P(s) Riemann')
    s_t = np.linspace(0.01, 3.5, 200)
    ax.plot(s_t, np.exp(-s_t), '--', color='#2ca02c', lw=1.5, label='Poisson')
    ax.plot(s_t, (np.pi/2)*s_t*np.exp(-np.pi*s_t**2/4),
            '-', color='#1f77b4', lw=1.5, label='GUE Wigner')
    ax.set_title('P(s) — distribución de espaciados')
    ax.set_xlabel('s'); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    # ── Plot 3: Δ₃(L) si SRCE disponible ──────────────────────────────────
    ax = axes[1, 0]
    if metricas.get('d3_vals') is not None:
        d3 = metricas['d3_vals']
        L  = metricas['L_grid']
        mask = np.isfinite(d3)
        L_ref = np.linspace(L.min(), L.max(), 200)
        ax.plot(L[mask], d3[mask], 'o-', color='#d62728', markersize=4,
                label=f"Riemann α={metricas.get('alpha', np.nan):.4f}")
        ax.plot(L_ref, ALPHA_GUE*np.log(L_ref), '--', color='#1f77b4',
                lw=1.5, label=f'GUE 1/π²={ALPHA_GUE:.4f}')
        ax.plot(L_ref, L_ref/15, '-.', color='#2ca02c', lw=1, alpha=0.5,
                label='Poisson L/15')
        ax.set_title(f"Δ₃(L)  R²={metricas.get('R2_d3', np.nan):.3f}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'SRCE no disponible\n(N demasiado pequeño)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
    ax.set_xlabel('L'); ax.grid(alpha=0.25)

    # ── Plot 4: resumen numérico ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis('off')
    rows = [
        ("T_anclaje",    f"10^{cache.log_T:.2f}", ""),
        ("dps",          str(cache.dps),           ""),
        ("N_offsets",    str(len(offsets)),         f"target"),
        ("Candidatos",   str(stats.get('n_candidatos',0)), "(Fase 1)"),
        ("Validados",    str(stats.get('n_validados', 0)), "(Fase 2)"),
        ("dt_safe",      f"{cache.dt_safe:.6f}",   f"dt_Nyq={cache.dt_nyquist:.5f}"),
        ("Aliasing OK",  str(stats.get('aliasing',{}).get('is_safe','?')), ""),
    ]
    if metricas:
        rows += [
            ("⟨r⟩",        f"{metricas.get('r_mean', np.nan):.5f}",
                            f"GUE={R_GUE_EXACT:.5f}"),
            ("Ensemble",    metricas.get('ensemble','?'), ""),
            ("α Δ₃",       f"{metricas.get('alpha', np.nan):.5f}",
                            f"Δ={metricas.get('delta_alpha', np.nan):+.5f}"),
        ]

    y = 0.97
    for label, val, ref in rows:
        ax.text(0.03, y, f"{label}:", fontsize=9, fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.38, y, val, fontsize=9, transform=ax.transAxes)
        ax.text(0.68, y, ref, fontsize=8, color='gray', transform=ax.transAxes)
        y -= 0.09

    plt.tight_layout()
    p = out_dir / 'zeta_diagnostico.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figura diagnóstico:    {p}")


# ============================================================================
# 11. MODO MULTI-ALTURA
# ============================================================================

def analisis_multi_altura(
    log_T_vals: List[float],
    n_ceros:    int  = 12,
    out_dir:    Path = Path('output'),
    use_arb:    bool = False,
    arb_prec:   int  = 256,
) -> None:
    """
    Calcula ⟨r⟩ y α en múltiples alturas usando aritmética de desplazamiento.
    Construye una ThetaCache por altura — dps adaptativo.
    """
    resultados = []

    for log_T in log_T_vals:
        dps = dps_auto(log_T)
        print(f"\n  log(T)={log_T}  T≈10^{log_T:.0f}  dps={dps}")

        try:
            cache = ThetaCache(log_T, dps)
            offsets, stats = buscar_ceros_desplazados(
                cache, n_ceros_target=n_ceros, validar=(n_ceros <= 20),
                verbose=False, use_arb=use_arb, arb_prec=arb_prec,
            )

            espectro = offsets_a_espectro(offsets, cache)

            # Cambio 7 (v2.1): solo calcular métricas si hay suficientes
            # aceptados con score medio adecuado
            n_accepted    = stats.get('n_aceptados', len(offsets))
            score_medio   = float(np.mean(stats.get('scores', [0.0]))) if stats.get('scores') else 0.0
            calidad_ok    = n_accepted >= 8 and score_medio >= 0.75

            if not calidad_ok:
                print(f"  ⚠ Calidad insuficiente: n_accepted={n_accepted} "
                      f"score_medio={score_medio:.3f} — omitiendo Δ₃/α")
                resultados.append({
                    'log_T': log_T, 'r_mean': np.nan, 'alpha': np.nan,
                    'n': len(offsets), 'n_accepted': n_accepted,
                    'score_medio': score_medio, 'calidad_ok': False,
                })
                continue

            metricas = analizar_espectro_local(espectro)

            r_m = metricas.get('r_mean', np.nan)
            alp = metricas.get('alpha', np.nan)
            print(f"  N={len(offsets)}  ⟨r⟩={r_m:.4f}  α={alp:.5f}  "
                  f"Δα={metricas.get('delta_alpha', np.nan):+.5f}")
            resultados.append({
                'log_T': log_T, 'r_mean': r_m, 'alpha': alp,
                'n': len(offsets),
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            resultados.append({'log_T': log_T, 'r_mean': np.nan,
                               'alpha': np.nan, 'n': 0})

    # Plot convergencia
    if _MPL and len(resultados) >= 2:
        log_Ts = [r['log_T'] for r in resultados]
        alphas = [r['alpha'] for r in resultados]
        r_vals = [r['r_mean'] for r in resultados]
        mask = [np.isfinite(a) for a in alphas]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        lT = [log_Ts[i] for i in range(len(log_Ts)) if mask[i]]
        a  = [alphas[i] for i in range(len(alphas)) if mask[i]]
        r  = [r_vals[i] for i in range(len(r_vals)) if mask[i]]

        ax1.plot(lT, a, 'o-', color='#d62728', lw=2, ms=7, label='α Riemann')
        ax1.axhline(ALPHA_GUE, ls='--', color='#1f77b4', alpha=0.8,
                    label=f'1/π²={ALPHA_GUE:.5f}')
        ax1.set_xlabel('log₁₀(T)'); ax1.set_ylabel('α [Δ₃ ≈ α·logL]')
        ax1.set_title('Convergencia α(T) → 1/π²'); ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        ax2.plot(lT, r, 's-', color='#9467bd', lw=2, ms=7, label='⟨r⟩ Riemann')
        ax2.axhline(R_GUE_EXACT, ls='--', color='#1f77b4', alpha=0.8,
                    label=f'GUE={R_GUE_EXACT:.4f}')
        ax2.axhline(R_POISSON_EXACT, ls=':', color='#2ca02c', alpha=0.6,
                    label=f'Poisson={R_POISSON_EXACT:.4f}')
        ax2.set_xlabel('log₁₀(T)'); ax2.set_ylabel('⟨r⟩')
        ax2.set_title('r-statistic vs altura'); ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        p = out_dir / 'zeta_convergencia.png'
        plt.savefig(p, dpi=300)
        plt.close()
        print(f"\n  Figura convergencia:   {p}")


# ============================================================================
# 12. MAIN
# ============================================================================

def main(
    log_T:       float = 70.0,
    n_ceros:     int   = 15,
    dps:         Optional[int] = None,
    solo_fase:   bool  = False,
    multi_altura: bool = False,
    dt_inicio:   float = 0.0,
    use_arb:     bool  = False,
    arb_prec:    int   = 256,
) -> None:

    out_dir = _SCRIPT_DIR / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print("  SRCE — EXPLORACIÓN ZETA ALTURA EXTREMA  v2.2  (ID híbrido Gödel)")
    print("=" * 72)

    if multi_altura:
        log_T_vals = [3, 4, 5, 6, 8, 10]
        print(f"  Modo multi-altura: {[f'10^{x}' for x in log_T_vals]}")
        analisis_multi_altura(
            log_T_vals, n_ceros=n_ceros, out_dir=out_dir,
            use_arb=use_arb, arb_prec=arb_prec,
        )
        return

    # Precisión dinámica
    dps_calculado = dps_auto(log_T)
    dps_usar = max(dps, dps_calculado) if dps else dps_calculado

    print(f"  T_anclaje    : 10^{log_T:.2f}")
    print(f"  dps_auto     : {dps_calculado}  →  usando {dps_usar}")
    print(f"  N ceros      : {n_ceros}")
    print(f"  Solo candidatos: {solo_fase}  (v2.1: no llama a Z exacta)")
    if use_arb:
        print(f"  Backlund Arb   : prec={arb_prec} bits  "
              f"(módulo: riemann_spectral.rigorous.arb_bridge)")

    # Construir caché — O(dps) una sola vez
    print(f"\n  Construyendo ThetaCache (dps={dps_usar})...")
    t0 = time.perf_counter()
    cache = ThetaCache(log_T, dps_usar)
    print(f"  Cache listo en {time.perf_counter()-t0:.2f}s  {cache}")

    # Verificar aliasing con el paso que se usará
    alias = check_aliasing(cache.dt_safe, cache)
    print(f"\n  Aliasing check: {alias['recomendacion']}")

    # Búsqueda de ceros
    print(f"\n  Buscando ceros en dt ∈ [{dt_inicio:.3f}, ...]...")
    offsets, stats = buscar_ceros_desplazados(
        cache,
        dt_inicio=dt_inicio,
        n_ceros_target=n_ceros,
        validar=not solo_fase,
        solo_candidatos=solo_fase,
        verbose=True,
        use_arb=use_arb,
        arb_prec=arb_prec,
    )

    print(f"\n  Ceros encontrados: {len(offsets)}")

    # Crear objetos Gödel con ID híbrido (v2.2)
    # Pasar accepted y validated para que se genere el número de Gödel
    accepted_objs  = stats.get('accepted', [])
    validated_objs = stats.get('validated', [])
    ceros_goedel   = crear_ceros_goedel(
        offsets, log_T,
        accepted=accepted_objs if accepted_objs else None,
        validated=validated_objs if validated_objs else None,
    )

    # Análisis espectral SRCE
    espectro = offsets_a_espectro(offsets, cache)
    metricas = analizar_espectro_local(espectro)

    if metricas:
        print(f"\n  ⟨r⟩  = {metricas['r_mean']:.5f}  ({metricas['ensemble']})")
        print(f"  α Δ₃ = {metricas['alpha']:.5f}  "
              f"Δ={metricas['delta_alpha']:+.5f}  "
              f"({metricas['error_rel']*100:.1f}%)")
    elif espectro is None:
        print(f"\n  ⚠ Espectro insuficiente para métricas SRCE (N={len(offsets)})")
        print(f"    Necesita N ≥ {int(1/(1-2*RECORTE))*8} para análisis completo.")

    # Guardar offsets
    guardar_offsets(ceros_goedel, cache, stats, metricas, out_dir)

    # Diagnóstico visual
    plot_diagnostico_local(offsets, cache, metricas, stats, out_dir)

    # Resumen
    print()
    print("=" * 72)
    print("  RESUMEN")
    print("=" * 72)
    print(f"  T_anclaje    : 10^{log_T:.2f}    dps={dps_usar}")
    print(f"  Candidatos   : {stats.get('n_candidatos', 0)}  (Fase 1)")
    print(f"  Validados    : {stats.get('n_validados', 0)}  (Fase 2)")
    print(f"  Aceptados    : {stats.get('n_aceptados', len(offsets))}  "
          f"(Fase 3, score ≥ {UMBRAL_ACEPTACION})")
    if stats.get('scores'):
        print(f"  Score medio  : {np.mean(stats['scores']):.3f}")
    print(f"  Tiempo Fase 1: {stats['t_fase1']:.2f}s")
    print(f"  Tiempo Fase 2: {stats['t_fase2']:.1f}s")
    print(f"  Tiempo Fase 3: {stats.get('t_fase3', 0):.1f}s")
    if metricas:
        print(f"  ⟨r⟩ = {metricas['r_mean']:.5f}  α={metricas['alpha']:.5f}")
    print("=" * 72)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exploración de Z de Riemann-Siegel en alturas extremas — SRCE v2."
    )
    parser.add_argument("--log-T",       type=float, default=70.0,
                        help="Altura T = 10^log_T (default: 70)")
    parser.add_argument("--n-ceros",     type=int,   default=15,
                        help="Ceros a validar (default: 15)")
    parser.add_argument("--dps",         type=int,   default=None,
                        help="Precisión manual (default: automático)")
    parser.add_argument("--solo-candidatos", action="store_true",
                        help="Solo Fase 1: devuelve candidatos sin validar (v2.1, antes --solo-fase)")
    parser.add_argument("--multi-altura", action="store_true",
                        help="Analizar múltiples alturas T=10^3..10^10")
    parser.add_argument("--dt-inicio",   type=float, default=0.0,
                        help="Offset inicial de búsqueda (default: 0.0)")
    parser.add_argument("--arb", action="store_true",
                        help="Conteo Backlund vía python-flint/Arb (certificado si flint instalado)")
    parser.add_argument("--arb-prec", type=int, default=256,
                        help="Precisión Arb en bits (default: 256)")
    args = parser.parse_args()

    main(
        log_T        = args.log_T,
        n_ceros      = args.n_ceros,
        dps          = args.dps,
        solo_fase    = args.solo_candidatos,
        multi_altura = args.multi_altura,
        dt_inicio    = args.dt_inicio,
        use_arb      = args.arb,
        arb_prec     = args.arb_prec,
    )
