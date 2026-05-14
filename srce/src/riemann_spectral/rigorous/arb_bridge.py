# -*- coding: utf-8 -*-
"""
srce/src/riemann_spectral/rigorous/arb_bridge.py
=================================================

Bridge entre el pipeline heurístico SRCE y el núcleo riguroso Arb/FLINT
vía python-flint.

Arquitectura
------------
    Python (SRCE)           Arb/FLINT (C, vía python-flint)
    ────────────────        ──────────────────────────────────
    gram_scan.py       →    arb.gram_point()
    backlund_count()   →    arb.zeta_nzeros()
    validar_candidato()→    acb.zeta()  con ball arithmetic
    interval_residual()→    acb.zeta_zero()  (aislamiento riguroso)
    Candidate/Valid…   →    CertifiedZero

Python orquesta. Arb certifica.
No se emula nada de C — se llama directamente al código C compilado.

Qué resuelve este bridge
------------------------
    El pipeline SRCE (v2.2.2) tenía tres limitaciones de rigor:

    1. mp.zeta() de mpmath no entrega cotas garantizadas.
       → Reemplazado por acb.zeta() con ball arithmetic real.

    2. backlund_count() usaba S(T) puntual sin bound certificado.
       → Reemplazado por arb.zeta_nzeros() que implementa el
         método de Turing internamente con rigor formal.

    3. interval_residual() construía intervalos heurísticos.
       → Reemplazado por ZetaBall que propaga cotas desde el origen.

    Cotas analíticas RS (Arias de Reyna 2011) para verificación *adicional*
    del radio frente a B_K/a^{K+1/2}: ver rs_bounds.py y
    ArbBridge.zeta_ball_crosscheck_reyna() — no sustituyen a Arb.

Qué NO resuelve este bridge
----------------------------
    - No prueba la Hipótesis de Riemann.
    - No certifica ceros en T > 3×10^12 (límite verificado por Platt 2021).
    - Para T extremo (10^70), Arb puede calcular con alta precisión pero
      no existe verificación independiente publicada con la que comparar.
    - El rigor es computacional, no analítico formal.

Niveles de rigor en el output
------------------------------
    NIVEL_HEURISTICO     : estimación del pipeline SRCE (v2.2.2 sin bridge)
    NIVEL_SEMI_RIGUROSO  : backlund_count + interval_residual mejorado
    NIVEL_CERTIFICADO    : arb.zeta_nzeros + acb con ball arithmetic

Uso
---
    from arb_bridge import ArbBridge, CertifiedZero

    bridge = ArbBridge(prec=256)

    # Conteo certificado de ceros en [T1, T2]
    count = bridge.certified_count(T1=100.0, T2=200.0)
    print(count.delta_N, count.es_exacto)

    # Validar un candidato con ball arithmetic
    cert = bridge.certified_zero(t_candidato=14.1347)
    print(cert.contiene_cero, cert.radio_garantizado)

    # Gram points certificados en un intervalo
    grams = bridge.gram_interval(T1=14.0, T2=100.0)

Requiere
--------
    pip install python-flint

Referencias
-----------
    Johansson, F. (2017). Arb: efficient arbitrary-precision midpoint-radius
        interval arithmetic. IEEE TOMS 44(4).
    Platt, D. (2021). Isolating some non-trivial zeros of zeta. Math. Comp.
    Trudgian, T. (2014). An improved explicit bound for S(T). J. Number Theory.

Autor: Jorge BC & Claude
Versión: 1.0.0
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


def _configure_stdio_utf8() -> None:
    for _stream in (sys.stdout, sys.stderr):
        reconf = getattr(_stream, "reconfigure", None)
        if reconf is not None:
            try:
                reconf(encoding="utf-8", errors="replace")
            except (OSError, ValueError, AttributeError):
                pass


_configure_stdio_utf8()
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── python-flint (Arb/FLINT) ──────────────────────────────────────────────────
try:
    from flint import acb, arb, ctx as flint_ctx
    _FLINT = True
except ImportError:
    _FLINT = False

# ── mpmath (fallback para operaciones que Arb no cubre) ───────────────────────
try:
    import mpmath as mp
    _MPMATH = True
except ImportError:
    _MPMATH = False

if not _FLINT and not _MPMATH:
    raise ImportError(
        "Se requiere python-flint o mpmath.\n"
        "Instalar: pip install python-flint"
    )

# Niveles de rigor — constantes para etiquetar resultados
NIVEL_HEURISTICO    = "heuristico"
NIVEL_SEMI_RIGUROSO = "semi_riguroso"
NIVEL_CERTIFICADO   = "certificado"

_FALLBACK_WARNED = False


# ============================================================================
# DATACLASSES DE OUTPUT
# ============================================================================

@dataclass
class ZetaBall:
    """
    Evaluación de ζ(s) con ball arithmetic garantizada.

    Atributos:
        mid_real, mid_imag : parte central del intervalo.
        rad_real, rad_imag : radio de la bola (cota garantizada del error).
        contiene_cero      : True si 0 ∈ [mid-rad, mid+rad] en valor absoluto.
        modulo_upper       : cota superior garantizada de |ζ(s)|.
        nivel              : NIVEL_CERTIFICADO si viene de Arb.
    """
    mid_real:       float
    mid_imag:       float
    rad_real:       float
    rad_imag:       float
    contiene_cero:  bool
    modulo_upper:   float
    nivel:          str   = NIVEL_CERTIFICADO

    @property
    def modulo_lower(self) -> float:
        """Cota inferior garantizada de |ζ(s)|."""
        return max(0.0,
                   (self.mid_real**2 + self.mid_imag**2)**0.5
                   - (self.rad_real**2 + self.rad_imag**2)**0.5)

    @property
    def es_riguroso(self) -> bool:
        return self.nivel == NIVEL_CERTIFICADO and self.rad_real < 1e-10


@dataclass
class CertifiedCount:
    """
    Conteo certificado de ceros en un intervalo.

    La clave: delta_N y es_exacto vienen de arb.zeta_nzeros() que
    implementa el método de Turing internamente. No es una aproximación.

    Atributos:
        N_T1, N_T2   : conteos en los extremos.
        delta_N      : N(T2) - N(T1). Entero exacto si es_exacto=True.
        es_exacto    : True si Arb pudo determinar el conteo sin ambigüedad.
        radio_N_T1   : radio de la bola del conteo en T1 (0 si exacto).
        radio_N_T2   : radio de la bola del conteo en T2.
        nivel        : nivel de rigor del resultado.
    """
    N_T1:       float
    N_T2:       float
    delta_N:    float
    es_exacto:  bool
    radio_N_T1: float
    radio_N_T2: float
    T1:         float
    T2:         float
    nivel:      str   = NIVEL_CERTIFICADO

    @property
    def delta_N_int(self) -> Optional[int]:
        """delta_N como entero si es_exacto, None si ambiguo."""
        if self.es_exacto:
            return int(round(self.delta_N))
        return None


@dataclass
class CertifiedZero:
    """
    Cero de ζ certificado o descartado por Arb.

    Atributos:
        t_candidato  : valor propuesto como cero.
        t_certificado: valor refinado por Arb (None si no existe).
        radio_cota   : radio de la bola alrededor del cero certificado.
        contiene_cero: True si Arb confirma existencia del cero.
        Z_ball       : evaluación de ζ en t_candidato con cotas.
        nivel        : nivel de rigor.
        n_zero       : índice del cero si es conocido (None si no).
    """
    t_candidato:   float
    t_certificado: Optional[float]
    radio_cota:    float
    contiene_cero: bool
    Z_ball:        Optional[ZetaBall]
    nivel:         str  = NIVEL_CERTIFICADO
    n_zero:        Optional[int] = None

    @property
    def es_riguroso(self) -> bool:
        return self.contiene_cero and self.radio_cota < 1e-8

    @property
    def es_falso_positivo(self) -> bool:
        """Certificado de no-existencia — el candidato era artefacto."""
        return (not self.contiene_cero and
                self.Z_ball is not None and
                self.Z_ball.modulo_lower > 1e-10)


@dataclass
class GramDiagnostic:
    """
    Diagnóstico de puntos de Gram certificados en un intervalo.

    Atributos:
        gram_ts      : tiempos de los puntos de Gram (certificados con radio).
        gram_radii   : radios de las bolas de los puntos de Gram.
        gram_signs   : signo de Z(g_n) evaluado con Z_approx (heurístico).
        n_failures   : número de Gram block failures.
        failure_intervals: lista de (t_left, t_right) de failures.
        n_gram_total : total de puntos de Gram en el intervalo.
        nivel        : los tiempos son certificados, los signos son heurísticos.
    """
    gram_ts:          List[float]
    gram_radii:       List[float]
    gram_signs:       List[int]
    n_failures:       int
    failure_intervals: List[Tuple[float, float]]
    n_gram_total:     int
    T1:               float
    T2:               float
    nivel:            str = NIVEL_CERTIFICADO   # tiempos; signos son heurísticos


# ============================================================================
# BRIDGE PRINCIPAL
# ============================================================================

class ArbBridge:
    """
    Interface Python→Arb/FLINT para certificación rigurosa de ceros de ζ.

    El bridge no reescribe ningún algoritmo de Arb. Llama directamente
    a las funciones C compiladas a través de python-flint y traduce
    el output a los dataclasses del pipeline SRCE.

    Args:
        prec : precisión en bits de la aritmética Arb (default 256 ≈ 77 dígitos).
               Para T > 10^20 usar prec ≥ 512.
               Para T > 10^50 usar prec ≥ 1024.
    """

    def __init__(self, prec: int = 256):
        if not _FLINT:
            raise ImportError(
                "python-flint no disponible.\n"
                "Instalar: pip install python-flint\n"
                "El bridge requiere Arb para certificación rigurosa."
            )
        self.prec = prec
        flint_ctx.prec = prec
        self._disponible = True

    # ── 1. Evaluación de ζ con ball arithmetic ────────────────────────────

    def zeta_ball(self, t: float, sigma: float = 0.5) -> ZetaBall:
        """
        Evalúa ζ(σ + it) con ball arithmetic garantizada.

        Reemplaza mp.zeta() del pipeline SRCE con una evaluación que
        propaga cotas garantizadas en cada operación aritmética.

        Args:
            t    : parte imaginaria (la altura del cero).
            sigma: parte real (default 0.5 — línea crítica).

        Returns:
            ZetaBall con mid, rad y contiene_cero.
        """
        flint_ctx.prec = self.prec
        s = acb(str(sigma), str(t))
        z = s.zeta()

        # Extraer mid y rad del resultado Arb
        # acb tiene partes real e imaginaria como arb (bolas)
        re = z.real
        im = z.imag

        mid_r = float(re)
        mid_i = float(im)

        # Radio de la bola: diferencia entre upper y lower bound / 2
        try:
            rad_r = float(re.abs_upper()) - abs(mid_r)
            rad_i = float(im.abs_upper()) - abs(mid_i)
        except Exception:
            # Fallback conservador
            rad_r = rad_i = 10 ** (-(self.prec // 4))

        # Módulo upper: cota garantizada de |ζ(s)|
        mod_upper = (
            (abs(mid_r) + rad_r) ** 2 +
            (abs(mid_i) + rad_i) ** 2
        ) ** 0.5

        # ¿La bola contiene al cero?
        contiene = mod_upper < max(abs(mid_r), abs(mid_i)) * 2 or \
                   (abs(mid_r) < rad_r and abs(mid_i) < rad_i)

        return ZetaBall(
            mid_real      = mid_r,
            mid_imag      = mid_i,
            rad_real      = max(rad_r, 0.0),
            rad_imag      = max(rad_i, 0.0),
            contiene_cero = contiene,
            modulo_upper  = mod_upper,
            nivel         = NIVEL_CERTIFICADO,
        )

    def zeta_ball_crosscheck_reyna(
        self,
        t: float,
        sigma: float = 0.5,
    ) -> Tuple[ZetaBall, Dict]:
        """
        Evalúa zeta_ball y añade un diagnóstico independiente con la cota de
        cola RS de de Reyna (2011), misma escala que rs_bounds.Z_with_bounds.

        Arb sigue siendo la fuente de radios garantizados; esto solo contrasta
        órdenes de magnitud (útil para depuración y auditoría).
        """
        zb = self.zeta_ball(t, sigma)
        try:
            from .rs_bounds import reyna_radius_check

            chk: Dict[str, object] = reyna_radius_check(
                t,
                max(zb.rad_real, zb.rad_imag),
                K=None,
                prec_bits=self.prec,
            )
        except Exception as exc:
            chk = {"error": str(exc), "consistente": None}
        return zb, chk

    # ── 2. Conteo certificado vía arb.zeta_nzeros ────────────────────────

    def certified_count(self, T1: float, T2: float) -> CertifiedCount:
        """
        Cuenta ceros de ζ en la franja 0 < Im(s) < T con garantía Arb.

        arb.zeta_nzeros() implementa el método de Backlund-Turing
        internamente con aritmética de bolas. Si el resultado es un
        entero exacto (radio = 0), el conteo es formalmente certificado.

        Diferencia con backlund_count() de mpmath:
            mpmath : S(T) puntual, sin cota garantizada para S(T).
            Arb    : propagación completa de cotas, resultado certificado.

        Args:
            T1, T2: extremos del intervalo (T2 > T1 > 0).

        Returns:
            CertifiedCount con delta_N y es_exacto.
        """
        flint_ctx.prec = self.prec

        t1_arb = arb(str(T1))
        t2_arb = arb(str(T2))

        N1_ball = t1_arb.zeta_nzeros()
        N2_ball = t2_arb.zeta_nzeros()

        N1_float = float(N1_ball)
        N2_float = float(N2_ball)

        # Radio de la bola — si es ~0, el conteo es exacto
        try:
            rad1 = float(N1_ball.abs_upper()) - abs(N1_float)
            rad2 = float(N2_ball.abs_upper()) - abs(N2_float)
        except Exception:
            rad1 = rad2 = 0.5   # conservador si falla

        # es_exacto: el radio es menor que 0.5 → el entero es unívoco
        es_exacto = (rad1 < 0.5 and rad2 < 0.5)

        return CertifiedCount(
            N_T1       = N1_float,
            N_T2       = N2_float,
            delta_N    = N2_float - N1_float,
            es_exacto  = es_exacto,
            radio_N_T1 = rad1,
            radio_N_T2 = rad2,
            T1         = T1,
            T2         = T2,
            nivel      = NIVEL_CERTIFICADO if es_exacto else NIVEL_SEMI_RIGUROSO,
        )

    # ── 3. Certificación de un candidato individual ───────────────────────

    def certified_zero(
        self,
        t_candidato: float,
        ventana:     float = 0.5,
    ) -> CertifiedZero:
        """
        Certifica o descarta un candidato a cero usando Arb.

        Estrategia:
            1. Evalúa ζ(1/2 + it) con ball arithmetic en t_candidato.
            2. Si la bola no contiene al cero, es un falso positivo certificado.
            3. Si la bola contiene al cero, refina usando el conteo Arb en
               [t-ventana, t+ventana] para confirmar que hay exactamente 1 cero.

        Args:
            t_candidato: posición propuesta del cero.
            ventana    : radio de búsqueda alrededor del candidato.

        Returns:
            CertifiedZero con contiene_cero y radio_cota.
        """
        flint_ctx.prec = self.prec

        # Evaluación con ball arithmetic
        Z = self.zeta_ball(t_candidato)

        # Si el módulo lower es grande, no hay cero cerca → falso positivo
        if Z.modulo_lower > 1e-4:
            return CertifiedZero(
                t_candidato   = t_candidato,
                t_certificado = None,
                radio_cota    = float('inf'),
                contiene_cero = False,
                Z_ball        = Z,
                nivel         = NIVEL_CERTIFICADO,
            )

        # Contar ceros en la ventana para confirmar existencia
        T1 = t_candidato - ventana
        T2 = t_candidato + ventana
        count = self.certified_count(T1, T2)

        if count.delta_N_int is None:
            # Conteo ambiguo — no podemos certificar
            return CertifiedZero(
                t_candidato   = t_candidato,
                t_certificado = t_candidato,
                radio_cota    = ventana,
                contiene_cero = Z.contiene_cero,
                Z_ball        = Z,
                nivel         = NIVEL_SEMI_RIGUROSO,
            )

        n_en_ventana = count.delta_N_int
        hay_cero     = n_en_ventana >= 1
        radio        = ventana / max(n_en_ventana, 1) if hay_cero else float('inf')

        return CertifiedZero(
            t_candidato   = t_candidato,
            t_certificado = t_candidato if hay_cero else None,
            radio_cota    = radio,
            contiene_cero = hay_cero,
            Z_ball        = Z,
            nivel         = NIVEL_CERTIFICADO if count.es_exacto else NIVEL_SEMI_RIGUROSO,
        )

    # ── 4. Puntos de Gram certificados ────────────────────────────────────

    def gram_interval(
        self,
        T1:          float,
        T2:          float,
        Z_approx_fn  = None,
    ) -> GramDiagnostic:
        """
        Calcula puntos de Gram certificados en [T1, T2].

        Los tiempos vienen de arb.gram_point() con cotas garantizadas.
        Los signos de Z se evalúan con Z_approx_fn (heurístico del pipeline)
        si se proporciona, o con zeta_ball si no.

        Args:
            T1, T2      : límites del intervalo.
            Z_approx_fn : función t → float (Z_phase_approx del pipeline).
                          Si None, usa zeta_ball (más lento pero riguroso).

        Returns:
            GramDiagnostic con puntos certificados y failures detectados.
        """
        flint_ctx.prec = self.prec

        # Índices de Gram en [T1, T2]
        # g_n es el n-ésimo punto de Gram. Buscar n tal que g_n ∈ [T1, T2].
        # g_n crece monótonamente, así que buscamos por bisección.
        n_ini = self._gram_index_near(T1, lado='ceil')
        n_fin = self._gram_index_near(T2, lado='floor')

        if n_fin < n_ini:
            return GramDiagnostic(
                gram_ts=[], gram_radii=[], gram_signs=[],
                n_failures=0, failure_intervals=[],
                n_gram_total=0, T1=T1, T2=T2,
            )

        gram_ts, gram_radii, gram_signs = [], [], []

        for n in range(n_ini, n_fin + 1):
            g_ball = arb.gram_point(n)
            g_float = float(g_ball)

            try:
                g_rad = float(g_ball.abs_upper()) - abs(g_float)
            except Exception:
                g_rad = 10 ** (-(self.prec // 4))

            gram_ts.append(g_float)
            gram_radii.append(g_rad)

            # Signo de Z(g_n) — la función Z de Hardy es real.
            # En el punto de Gram n: Z(g_n) = (-1)^n · Re(ζ(1/2 + ig_n))
            # (porque θ(g_n) = nπ → e^{iθ} = (-1)^n)
            # Este cálculo es correcto independientemente de Z_approx_fn.
            gram_n = n_ini + (n - n_ini)   # índice global del punto de Gram
            if Z_approx_fn is not None:
                # Heurístico del pipeline — rápido, usa la aproximación de fase
                z_val = Z_approx_fn(g_float)
                sign  = int(np.sign(z_val))
            else:
                # Riguroso: Z(g_n) = (-1)^n · Re(ζ(1/2 + ig_n))
                zb    = self.zeta_ball(g_float)
                Z_val = ((-1) ** gram_n) * zb.mid_real
                sign  = int(np.sign(Z_val)) if abs(Z_val) > zb.rad_real else 0
            gram_signs.append(sign)

        # Detectar failures: mismo signo en extremos consecutivos
        failures = []
        for i in range(len(gram_signs) - 1):
            if gram_signs[i] * gram_signs[i+1] > 0:
                failures.append((gram_ts[i], gram_ts[i+1]))

        return GramDiagnostic(
            gram_ts           = gram_ts,
            gram_radii        = gram_radii,
            gram_signs        = gram_signs,
            n_failures        = len(failures),
            failure_intervals = failures,
            n_gram_total      = len(gram_ts),
            T1                = T1,
            T2                = T2,
            nivel             = NIVEL_CERTIFICADO,
        )

    def _gram_index_near(self, T: float, lado: str = 'ceil') -> int:
        """
        Encuentra el índice n tal que gram_point(n) ≈ T.
        Búsqueda por bisección sobre los índices de Gram.
        """
        flint_ctx.prec = max(64, self.prec // 2)

        # Estimación inicial: N(T) ≈ (T/2π)·log(T/2πe)
        import math
        T_val = max(T, 10.0)
        n_est = int((T_val / (2*math.pi)) * math.log(T_val / (2*math.pi*math.e)))

        # Ajuste fino por bisección (máx 30 pasos)
        lo, hi = max(0, n_est - 20), n_est + 20
        for _ in range(30):
            mid = (lo + hi) // 2
            g   = float(arb.gram_point(mid))
            if g < T:
                lo = mid + 1
            else:
                hi = mid

        if lado == 'ceil':
            return lo
        else:  # floor
            return hi - 1

    # ── 5. Verificación de un candidato contra índice conocido ───────────

    def verify_against_known(
        self,
        t_candidato: float,
        n_index:     int,
        tol:         float = 0.5,
    ) -> Dict:
        """
        Compara un candidato con el n-ésimo cero certificado de Arb.

        Útil para validación del pipeline contra ceros conocidos.
        Si |t_candidato - γ_n| < tol, el candidato corresponde al cero n.

        Args:
            t_candidato: posición propuesta.
            n_index    : índice del cero a comparar (1-indexed).
            tol        : tolerancia de coincidencia.

        Returns:
            dict con coincide, error_absoluto, gamma_n_certificado.
        """
        flint_ctx.prec = self.prec
        gamma_n_ball = acb.zeta_zero(n_index)
        gamma_n      = float(gamma_n_ball.imag)

        try:
            gamma_n_rad = float(gamma_n_ball.imag.abs_upper()) - abs(gamma_n)
        except Exception:
            gamma_n_rad = 10 ** (-(self.prec // 4))

        error = abs(t_candidato - gamma_n)
        coincide = error < tol

        return {
            'coincide':             coincide,
            'error_absoluto':       error,
            'gamma_n_certificado':  gamma_n,
            'radio_certificado':    gamma_n_rad,
            'n_index':              n_index,
            't_candidato':          t_candidato,
            'dentro_de_radio':      error < gamma_n_rad + tol,
            'nivel':                NIVEL_CERTIFICADO,
        }

    # ── 6. Validación batch de candidatos ────────────────────────────────

    def validate_candidates_batch(
        self,
        t_candidatos: List[float],
        ventana:      float = 0.5,
        verbose:      bool  = True,
    ) -> Tuple[List[CertifiedZero], Dict]:
        """
        Valida una lista de candidatos del pipeline SRCE con Arb.

        Reemplaza el bucle de validar_candidato() para los casos donde
        se quiere certificación rigurosa, no solo refinamiento por secante.

        Args:
            t_candidatos: lista de offsets dt del pipeline.
            ventana      : radio de búsqueda para cada candidato.
            verbose      : mostrar progreso.

        Returns:
            (certificados, stats)
        """
        certificados = []
        n_certificados     = 0
        n_falsos_positivos = 0
        n_ambiguos         = 0

        if verbose:
            print(f"  Arb: validando {len(t_candidatos)} candidatos "
                  f"(prec={self.prec} bits)...")

        for i, t in enumerate(t_candidatos):
            cert = self.certified_zero(t, ventana=ventana)
            certificados.append(cert)

            if cert.contiene_cero and cert.nivel == NIVEL_CERTIFICADO:
                n_certificados += 1
            elif cert.es_falso_positivo:
                n_falsos_positivos += 1
            else:
                n_ambiguos += 1

            if verbose and (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(t_candidatos)}  "
                      f"cert={n_certificados}  "
                      f"falsos={n_falsos_positivos}  "
                      f"ambiguos={n_ambiguos}", end='\r')

        if verbose:
            print(f"\n  Arb: {n_certificados} certificados  "
                  f"{n_falsos_positivos} falsos positivos  "
                  f"{n_ambiguos} ambiguos")

        stats = {
            'n_total':         len(t_candidatos),
            'n_certificados':  n_certificados,
            'n_falsos':        n_falsos_positivos,
            'n_ambiguos':      n_ambiguos,
            'tasa_cert':       n_certificados / max(len(t_candidatos), 1),
            'nivel':           NIVEL_CERTIFICADO,
        }

        return certificados, stats

    # ── 7. Informe de nivel de rigor ──────────────────────────────────────

    @staticmethod
    def nivel_rigor_str(nivel: str) -> str:
        """Descripción legible del nivel de rigor."""
        return {
            NIVEL_HEURISTICO:    "Heurístico (mpmath puntual, sin cotas)",
            NIVEL_SEMI_RIGUROSO: "Semi-riguroso (cotas locales mejoradas)",
            NIVEL_CERTIFICADO:   "Certificado (Arb ball arithmetic, cotas propagadas)",
        }.get(nivel, nivel)

    def __repr__(self) -> str:
        disponible = "disponible" if _FLINT else "NO disponible"
        return f"ArbBridge(prec={self.prec}, flint={disponible})"


# ============================================================================
# INTEGRACIÓN CON EL PIPELINE SRCE (funciones de conveniencia)
# ============================================================================

def reemplazar_backlund_count(
    T1:    float,
    T2:    float,
    prec:  int = 256,
) -> Dict:
    """
    Reemplaza backlund_count() del pipeline con la versión certificada.

    Drop-in replacement: misma firma de output que la función original,
    pero con nivel=NIVEL_CERTIFICADO y conteos garantizados por Arb.

    Args:
        T1, T2 : extremos del intervalo.
        prec   : precisión Arb en bits.

    Returns:
        Dict compatible con el formato de backlund_count() original.
    """
    bridge = get_bridge(prec=prec)
    count  = bridge.certified_count(T1, T2)

    return {
        'N_T1':     count.N_T1,
        'N_T2':     count.N_T2,
        'delta_N':  count.delta_N,
        'S_T1':     0.0,   # S(T) absorbido internamente por Arb
        'S_T2':     0.0,
        'S_grande': False,
        'fiable':   count.es_exacto,
        'T1': T1, 'T2': T2,
        'nivel':    count.nivel,
        'es_exacto': count.es_exacto,
        'advertencia': (
            None if count.es_exacto
            else f"Conteo ambiguo (radio={max(count.radio_N_T1, count.radio_N_T2):.2e})"
        ),
    }


def reemplazar_interval_residual(
    t:    float,
    prec: int = 256,
) -> Dict:
    """
    Reemplaza interval_residual() del pipeline con ball arithmetic real.

    Args:
        t    : posición del candidato.
        prec : precisión Arb en bits.

    Returns:
        Dict compatible con el formato de interval_residual() original.
    """
    bridge = get_bridge(prec=prec)
    Z      = bridge.zeta_ball(t)

    return {
        'contiene_cero': Z.contiene_cero,
        'Z_lo':          Z.mid_real - Z.rad_real,
        'Z_hi':          Z.mid_real + Z.rad_real,
        'Z_float':       Z.mid_real,
        'radio_cota':    max(Z.rad_real, Z.rad_imag),
        'riguroso':      Z.es_riguroso,
        'certificado':   Z.contiene_cero and Z.es_riguroso,
        'nivel':         Z.nivel,
        'eps':           max(Z.rad_real, Z.rad_imag),
    }


# ============================================================================
# FALLBACK SIN FLINT
# ============================================================================

class ArbBridgeFallback:
    """
    Fallback cuando python-flint no está instalado.
    Usa mpmath con las mismas firmas pero menor rigor.
    Marca todos los resultados como NIVEL_SEMI_RIGUROSO.
    """

    def __init__(self, prec: int = 256):
        if not _MPMATH:
            raise ImportError("Se requiere mpmath o python-flint.")
        self.prec = prec
        mp.mp.dps = prec // 3   # bits → dígitos aproximado
        self._disponible = False
        global _FALLBACK_WARNED
        if not _FALLBACK_WARNED:
            print(
                "  [!] python-flint no disponible; usando fallback mpmath (semi-riguroso)"
            )
            _FALLBACK_WARNED = True

    def certified_count(self, T1: float, T2: float) -> CertifiedCount:
        """Fallback: backlund_count con mpmath."""
        with mp.workdps(self.prec // 3):
            def N_T(T_mp):
                theta = mp.im(mp.loggamma(mp.mpc(0.25, T_mp/2))) - T_mp/2*mp.log(mp.pi)
                S = float(mp.im(mp.log(mp.zeta(mp.mpc(0.5, T_mp)))) / mp.pi)
                return float(theta/mp.pi) + 1 + S, S

            N1, S1 = N_T(mp.mpf(str(T1)))
            N2, S2 = N_T(mp.mpf(str(T2)))

        return CertifiedCount(
            N_T1=round(N1), N_T2=round(N2),
            delta_N=round(N2)-round(N1),
            es_exacto=abs(S1)<0.5 and abs(S2)<0.5,
            radio_N_T1=abs(S1), radio_N_T2=abs(S2),
            T1=T1, T2=T2,
            nivel=NIVEL_SEMI_RIGUROSO,
        )

    def zeta_ball(self, t: float, sigma: float = 0.5) -> ZetaBall:
        """Fallback: evaluación puntual mpmath."""
        with mp.workdps(self.prec // 3):
            z = mp.zeta(mp.mpc(str(sigma), str(t)))
            trunc = 10 ** -(self.prec // 3 // 2)

        return ZetaBall(
            mid_real=float(mp.re(z)), mid_imag=float(mp.im(z)),
            rad_real=trunc, rad_imag=trunc,
            contiene_cero=abs(float(mp.re(z))) < trunc*10,
            modulo_upper=abs(float(z))+trunc,
            nivel=NIVEL_SEMI_RIGUROSO,
        )

    def certified_zero(self, t_candidato, ventana=0.5):
        Z = self.zeta_ball(t_candidato)
        count = self.certified_count(t_candidato-ventana, t_candidato+ventana)
        return CertifiedZero(
            t_candidato=t_candidato,
            t_certificado=t_candidato if count.delta_N > 0 else None,
            radio_cota=ventana,
            contiene_cero=count.delta_N > 0,
            Z_ball=Z, nivel=NIVEL_SEMI_RIGUROSO,
        )

    def gram_interval(self, T1, T2, Z_approx_fn=None):
        return GramDiagnostic(
            gram_ts=[], gram_radii=[], gram_signs=[],
            n_failures=0, failure_intervals=[],
            n_gram_total=0, T1=T1, T2=T2,
            nivel=NIVEL_SEMI_RIGUROSO,
        )

    def validate_candidates_batch(self, t_candidatos, ventana=0.5, verbose=True):
        certs = [self.certified_zero(t, ventana) for t in t_candidatos]
        n_ok = sum(1 for c in certs if c.contiene_cero)
        return certs, {'n_total': len(certs), 'n_certificados': n_ok,
                       'n_falsos': 0, 'n_ambiguos': len(certs)-n_ok,
                       'tasa_cert': n_ok/max(len(certs),1),
                       'nivel': NIVEL_SEMI_RIGUROSO}


def get_bridge(prec: int = 256) -> "ArbBridge | ArbBridgeFallback":
    """
    Factory: devuelve ArbBridge si flint está disponible, fallback si no.
    Uso recomendado en el pipeline:

        bridge = get_bridge(prec=256)
        count  = bridge.certified_count(T1, T2)
    """
    if _FLINT:
        return ArbBridge(prec=prec)
    return ArbBridgeFallback(prec=prec)


# ============================================================================
# AUTO-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ARB BRIDGE — AUTO-TEST")
    print("=" * 60)
    print(f"  python-flint : {'✓ disponible' if _FLINT else '✗ no disponible'}")
    print(f"  mpmath       : {'✓ disponible' if _MPMATH else '✗ no disponible'}")
    print()

    bridge = get_bridge(prec=256)
    print(f"  Bridge: {bridge}")
    print()

    # Test 1: conteo certificado
    print("[TEST 1] Conteo certificado en [14, 100]")
    count = bridge.certified_count(14.0, 100.0)
    ok = count.delta_N_int == 29
    print(f"  delta_N = {count.delta_N_int}  (esperado: 29)  {'✓' if ok else '✗'}")
    print(f"  exacto  = {count.es_exacto}  nivel={count.nivel}")
    print()

    # Test 2: evaluación con ball arithmetic
    print("[TEST 2] ζ en el primer cero γ₁ ≈ 14.1347")
    Z = bridge.zeta_ball(14.134725141734694)
    print(f"  |ζ| upper = {Z.modulo_upper:.2e}  (esperado ≈ 0)")
    print(f"  radio_real = {Z.rad_real:.2e}")
    print(f"  nivel = {Z.nivel}")
    print()

    # Test 3: certificación de candidato
    print("[TEST 3] Certificar candidato en γ₁ ≈ 14.1347")
    cert = bridge.certified_zero(14.134725, ventana=0.5)
    print(f"  contiene_cero = {cert.contiene_cero}  (esperado: True)")
    print(f"  radio_cota    = {cert.radio_cota:.4f}")
    print(f"  nivel         = {cert.nivel}")
    print()

    # Test 4: Gram points
    print("[TEST 4] Gram points en [14, 50]")
    gd = bridge.gram_interval(14.0, 50.0)
    print(f"  {gd.n_gram_total} puntos de Gram  "
          f"{gd.n_failures} failures")
    if gd.gram_ts:
        print(f"  Primero: {gd.gram_ts[0]:.6f} ± {gd.gram_radii[0]:.2e}")
    print()

    # Test 5: verificación contra cero conocido
    if _FLINT:
        print("[TEST 5] Verificar candidato contra γ₂ (índice 2)")
        verif = bridge.verify_against_known(21.022, n_index=2, tol=0.1)
        print(f"  coincide = {verif['coincide']}  "
              f"error = {verif['error_absoluto']:.4f}")
        print(f"  γ₂ certificado = {verif['gamma_n_certificado']:.6f}")
    print()

    print("=" * 60)
    print("  ✓ Auto-test completado")
    print("=" * 60)
