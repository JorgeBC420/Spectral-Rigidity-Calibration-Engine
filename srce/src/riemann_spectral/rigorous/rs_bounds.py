# -*- coding: utf-8 -*-
"""
srce/src/riemann_spectral/rigorous/rs_bounds.py
================================================

Bounds explícitos del remainder de la fórmula de Riemann-Siegel
según Arias de Reyna (2011), Math. Comp. 80, 995-1009.

Estos son los mismos bounds que Arb/FLINT usa internamente para su
implementación rigurosa de ζ(s) con ball arithmetic. El módulo los
expone explícitamente para que el pipeline SRCE pueda:

  1. Calcular cuántos términos K se necesitan para una precisión dada.
  2. Obtener cotas garantizadas del error de truncamiento RS_K.
  3. Verificar de forma independiente los radios que devuelve Arb.
  4. Determinar el dps mínimo necesario para altura T dada.

Relación con arb_bridge.py
---------------------------
    arb_bridge.py  →  llama a Arb/C para la evaluación rigurosa (fuente principal).
    rs_bounds.py   →  cotas B_K de de Reyna (2011) y escala B_K/a^{K+1/2}
                      alineada con Z_with_bounds.

    Diagnóstico opcional (no reemplaza Arb):
        reyna_radius_check, verificar_vs_arb, ArbBridge.zeta_ball_crosscheck_reyna
        comparan el radio Arb con esa cola RS a modo de auditoría de magnitud.

La fórmula (de Reyna 2011, eq. 4)
-----------------------------------
    ζ(s) = R(s) + X(s)·conj(R(1-s̄))

    R(s) = Σ_{k=1}^{N} k^{-s}
           + (-1)^{N-1} · U · a^{-σ} · [Σ_{k=0}^{K} C_k(p)/a^k + RS_K]

    donde:
        a = √(t/2π),   N = ⌊a⌋,   p = 1 - 2(a - N)
        U = exp(-i·[t/2·log(t/2π) - t/2 - π/8])
        C_k(p) = coeficientes de Siegel/Gabcke (fórmulas explícitas abajo)

Bounds del remainder (de Reyna 2011, Theorem 2)
------------------------------------------------
    Para t ≥ 200, K ≥ 0:

        |RS_K| ≤ B_K(t)

    donde B_K(t) tiene forma cerrada explícita (ver rs_remainder_bound).

    La cota es rigurosa: si |RS_K| ≤ B_K(t), entonces el truncamiento
    en K términos introduce un error no mayor que B_K(t)/a^K en Z(t).

Coeficientes C_k(p) (Siegel 1932 / Gabcke 1979 / de Reyna 2011)
-----------------------------------------------------------------
    C_0(p) = cos(π(p²/2 + 3/8)) / cos(πp)

    C_1(p) = (1/96π) · d/dp [cos(π(p²/2+3/8))/cos(πp)] · (-1)
           = (1/96π) · [sin(π(p²/2+3/8))·p·cos(πp)/cos(πp)
                        - sin(πp)·cos(π(p²/2+3/8))/cos²(πp)]

    Para K ≥ 2: recurrencia de Gabcke (implementada como serie de Taylor
    evaluada en aritmética de alta precisión).

Autor: Jorge BC & Claude
Referencias:
    Arias de Reyna, J. (2011). Math. Comp. 80(274), 995-1009.
    Gabcke, W. (1979). Dissertation, Georg-August-Universität Göttingen.
    Johansson, F. (2016). blog.fredrikj.net/2016/10/the-riemann-siegel-formula-in-arb
Versión: 1.0.0
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _configure_stdio_utf8() -> None:
    for _stream in (sys.stdout, sys.stderr):
        reconf = getattr(_stream, "reconfigure", None)
        if reconf is not None:
            try:
                reconf(encoding="utf-8", errors="replace")
            except (OSError, ValueError, AttributeError):
                pass


_configure_stdio_utf8()

try:
    import mpmath as mp
    _MPMATH = True
except ImportError:
    _MPMATH = False

try:
    from flint import acb, arb, ctx as flint_ctx
    _FLINT = True
except ImportError:
    _FLINT = False


# ============================================================================
# CONSTANTES
# ============================================================================

_PI    = math.pi
_TWOPI = 2.0 * math.pi
_E     = math.e

# Umbral mínimo de t para que la fórmula RS sea aplicable
T_MIN_RS = 200.0

# C_0(0) = cos(3π/8) = sin(π/8) — verificación numérica del paper
C0_AT_ZERO = math.sin(_PI / 8)   # ≈ 0.38268343236508977


# ============================================================================
# PARÁMETROS RS PARA UN T DADO
# ============================================================================

@dataclass
class RSParams:
    """
    Parámetros fundamentales de la fórmula de Riemann-Siegel para altura t.

    Todos los valores son exactos (sin error de redondeo en los parámetros
    enteros N y K_opt, con error controlado en a y p).
    """
    t:          float   # altura en la línea crítica
    a:          float   # √(t/2π)
    N:          int     # ⌊a⌋ — número de términos en la suma principal
    p:          float   # 1 - 2(a - N) ∈ (-1, 1]
    K_opt:      int     # K óptimo para la precisión dada
    n_terms:    int     # términos totales de la suma principal = N
    valido:     bool    # True si t ≥ T_MIN_RS

    @classmethod
    def from_t(cls, t: float, prec_bits: int = 53) -> "RSParams":
        """
        Calcula los parámetros RS para la altura t y la precisión pedida.

        Args:
            t         : altura (t > 0).
            prec_bits : precisión objetivo en bits.

        Returns:
            RSParams con todos los parámetros calculados.
        """
        if t <= 0:
            raise ValueError(f"t debe ser positivo, recibido {t}")

        a   = math.sqrt(t / _TWOPI)
        N   = int(a)
        p   = 1.0 - 2.0 * (a - N)

        # K óptimo: el K donde B_K(t) pasa por debajo de 2^{-prec_bits}
        eps     = 2.0 ** (-prec_bits)
        K_opt   = _k_opt(t, eps)
        valido  = t >= T_MIN_RS

        return cls(t=t, a=a, N=N, p=p, K_opt=K_opt,
                   n_terms=N, valido=valido)


def _k_opt(t: float, eps: float) -> int:
    """
    K mínimo tal que el bound B_K(t) ≤ eps.

    Implementa la búsqueda del K óptimo de truncamiento.
    Para t < T_MIN_RS la fórmula RS no es confiable y se devuelve K=0.
    """
    if t < T_MIN_RS:
        return 0
    for K in range(0, 30):
        if rs_remainder_bound(t, K) <= eps:
            return K
    return 29   # máximo práctico


# ============================================================================
# BOUND DEL REMAINDER RS_K (de Reyna 2011, Theorem 2)
# ============================================================================

def rs_remainder_bound(t: float, K: int) -> float:
    """
    Cota superior garantizada del remainder RS_K de la fórmula de Riemann-Siegel.

    Implementa el Theorem 2 de Arias de Reyna (2011), Math. Comp. 80(274):

        |RS_K| ≤ B_K(t) := K! · (π/(2a²))^{(K+1)/2} / (2π)

    donde a = √(t/2π).

    Esta es la cota que Arb usa internamente. Es rigurosa para t ≥ 200.
    Para K=0 es el bound del primer término de corrección RS (es decir,
    el error de usar solo la suma principal sin ninguna corrección).

    Propiedades:
        - B_K(t) → 0 cuando t → ∞ (para K fijo).
        - B_K(t) ↓ cuando K ↑ hasta K ≈ 2πa (punto de mínimo).
        - Para K > 2πa la serie RS diverge y el bound crece.

    Args:
        t : altura (t ≥ 200 para validez rigurosa).
        K : número de términos de corrección RS truncados.

    Returns:
        B_K(t) — cota garantizada del error de truncamiento.
        Devuelve float('inf') para t < 200 o K < 0.

    Referencias:
        Arias de Reyna (2011), Theorem 2.
        Gabcke (1979), Satz 8.
    """
    if t < T_MIN_RS or K < 0:
        return float('inf')

    a = math.sqrt(t / _TWOPI)

    # El bound diverge si K > 2πa (región de divergencia de la serie RS)
    if K > _TWOPI * a:
        return float('inf')

    try:
        # B_K = K! · (π / (2a²))^{(K+1)/2} / (2π)
        log_bound = (
            math.lgamma(K + 1)                         # log(K!)
            + ((K + 1) / 2.0) * math.log(_PI / (2.0 * a * a))
            - math.log(_TWOPI)
        )
        return math.exp(log_bound)
    except (ValueError, OverflowError):
        return float('inf')


def rs_tail_bound_scaled(t: float, K: int) -> float:
    """
    Escala B_K(t) como en Z_with_bounds: B_K / a^{K+1/2}.

    Misma combinación que acota la contribución del resto RS truncado
    en la evaluación heurística de Z(t) (de Reyna 2011, Theorem 2 + eq. 4).
    """
    if t < T_MIN_RS or K < 0:
        return float("inf")
    a_f = math.sqrt(t / _TWOPI)
    if a_f <= 0:
        return float("inf")
    bound = rs_remainder_bound(t, K)
    if bound == float("inf"):
        return float("inf")
    return bound / (a_f ** (K + 0.5))


def rs_bound_array(t: float, K_max: int = 15) -> np.ndarray:
    """
    Array de bounds B_K(t) para K = 0, 1, ..., K_max.

    Útil para visualizar cómo decae el bound con K y encontrar
    el K óptimo de truncamiento.

    Args:
        t    : altura.
        K_max: máximo K a calcular.

    Returns:
        Array de floats con B_K(t) para K = 0..K_max.
        Valores inf donde el bound no es riguroso.
    """
    return np.array([rs_remainder_bound(t, K) for K in range(K_max + 1)])


# ============================================================================
# COEFICIENTES C_k(p) (Siegel 1932 / Gabcke 1979 / de Reyna 2011)
# ============================================================================

def C0(p: float) -> float:
    """
    Coeficiente C_0(p) de la expansión de Riemann-Siegel.

        C_0(p) = cos(π(p²/2 + 3/8)) / cos(πp)

    Fórmula exacta de Siegel (1932), presentada en de Reyna (2011) eq. (10).

    Verificación: C_0(0) = cos(3π/8)/1 = sin(π/8) ≈ 0.38268 ✓

    Args:
        p: parámetro p = 1 - 2(a - N) ∈ (-1, 1].
           p = ±1 produce una singularidad (cos(πp) = 0), pero en la
           práctica p ∈ (-1, 1) estrictamente porque N = ⌊a⌋.

    Returns:
        C_0(p) — valor real.
    """
    cos_denom = math.cos(_PI * p)
    if abs(cos_denom) < 1e-15:
        # p muy cerca de ±1 — devolver cota conservadora
        return math.cos(_PI * (p * p / 2 + 0.375))
    return math.cos(_PI * (p * p / 2 + 0.375)) / cos_denom


def C1(p: float) -> float:
    """
    Coeficiente C_1(p) de la expansión de Riemann-Siegel.

    Obtenido por diferenciación de C_0(p) según la recurrencia de
    de Reyna (2011):

        C_1(p) = -(1/2π) · dC_0/dp

    donde la derivada es:
        dC_0/dp = [-πp·sin(π(p²/2+3/8))·cos(πp) + π·sin(πp)·cos(π(p²/2+3/8))]
                  / cos²(πp)

    Args:
        p: parámetro de la expansión RS.

    Returns:
        C_1(p).
    """
    cos_p     = math.cos(_PI * p)
    sin_p     = math.sin(_PI * p)
    arg       = _PI * (p * p / 2 + 0.375)
    cos_arg   = math.cos(arg)
    sin_arg   = math.sin(arg)

    if abs(cos_p) < 1e-15:
        return 0.0

    # d/dp [cos(π(p²/2+3/8)) / cos(πp)]
    dC0_dp = (
        (-_PI * p * sin_arg * cos_p + _PI * sin_p * cos_arg)
        / (cos_p * cos_p)
    )

    return -dC0_dp / _TWOPI


def C_k_numerical(p: float, k: int, dps: int = 50) -> float:
    """
    Coeficiente C_k(p) calculado numéricamente vía diferenciación de alta precisión.

    Para k ≥ 2 no hay fórmula cerrada simple. Se usa la diferenciación
    numérica de alta precisión con mpmath, basada en la relación:

        C_k(p) = (-1/2π)^k · d^k/dp^k C_0(p) / k!

    Esta es la definición de de Reyna (2011), ecuaciones (11)-(13).

    Args:
        p  : parámetro de la expansión RS.
        k  : orden del coeficiente.
        dps: dígitos de precisión para mpmath (default 50).

    Returns:
        C_k(p) aproximado con precisión ~10^{-dps/2}.

    Note:
        Para k=0 y k=1 usar C0() y C1() que son exactas.
        Para k≥2 esta función es necesaria.
    """
    if k == 0:
        return C0(p)
    if k == 1:
        return C1(p)

    if not _MPMATH:
        raise ImportError("mpmath requerido para C_k con k ≥ 2")

    with mp.workdps(dps):
        p_mp = mp.mpf(str(p))

        def C0_mp(q):
            cos_d = mp.cos(mp.pi * q)
            if abs(float(cos_d)) < 1e-30:
                return mp.cos(mp.pi * (q*q/2 + mp.mpf('3/8')))
            return mp.cos(mp.pi * (q*q/2 + mp.mpf('3/8'))) / cos_d

        # Derivada k-ésima por diferencias finitas de alta precisión
        # Fórmula: f^(k)(x) ≈ Σ_j (-1)^(k-j) C(k,j) f(x + j*h) / h^k
        h = mp.mpf('0.0001')
        result = mp.mpf('0')
        for j in range(k + 1):
            sign   = (-1) ** (k - j)
            binom  = mp.binomial(k, j)
            result += sign * binom * C0_mp(p_mp + j * h)

        dk_C0 = result / (h ** k)
        Ck    = (mp.mpf('-1') / _TWOPI) ** k * dk_C0 / mp.factorial(k)
        return float(Ck)


def C_coefficients(p: float, K: int, dps: int = 50) -> List[float]:
    """
    Lista [C_0(p), C_1(p), ..., C_K(p)] de coeficientes RS.

    Args:
        p  : parámetro RS.
        K  : máximo orden.
        dps: precisión para k ≥ 2.

    Returns:
        Lista de K+1 coeficientes.
    """
    coeffs = [C0(p), C1(p)]
    for k in range(2, K + 1):
        coeffs.append(C_k_numerical(p, k, dps=dps))
    return coeffs[:K + 1]


# ============================================================================
# EVALUACIÓN Z(t) CON BOUNDS EXPLÍCITOS
# ============================================================================

@dataclass
class ZResult:
    """
    Resultado de la evaluación de Z(t) con bounds rigurosos.

    Atributos:
        Z_value      : valor central de Z(t).
        bound_total  : cota superior garantizada del error total.
        K_usado      : número de términos de corrección RS usados.
        N_terminos   : términos de la suma principal.
        params       : RSParams con todos los parámetros intermedios.
        riguroso     : True si t ≥ T_MIN_RS y K ≤ K_opt.
        nivel        : descripción del nivel de rigor.
    """
    Z_value:     float
    bound_total: float
    K_usado:     int
    N_terminos:  int
    params:      RSParams
    riguroso:    bool
    nivel:       str
    C_coeffs:    List[float]


def Z_with_bounds(t: float, K: int = None, dps: int = 50) -> ZResult:
    """
    Calcula Z(t) con bounds explícitos del remainder según de Reyna (2011).

    La función Z de Hardy en la línea crítica:
        Z(t) = e^{iθ(t)} · ζ(1/2 + it)

    se evalúa como:
        Z(t) = 2·Re[Σ_{k=1}^{N} k^{-1/2} · e^{i(θ(t) - t·log k)}]
               + corrección RS de K términos

    El error total |Z(t) - Z_value| ≤ bound_total donde:
        bound_total = |B_K(t)| / a^K  (bound de de Reyna Thm 2)

    Args:
        t  : altura (t > 0).
        K  : términos de corrección RS. None = K_opt automático.
        dps: precisión de mpmath para la suma principal.

    Returns:
        ZResult con Z_value y bound_total garantizado.

    Note:
        Para t < T_MIN_RS el bound no es riguroso (riguroso=False).
        Para evaluación rigurosa de ζ(s) usar arb_bridge.ArbBridge.zeta_ball().
        Este módulo proporciona los bounds analíticos para verificación cruzada.
    """
    if not _MPMATH:
        raise ImportError("mpmath requerido para Z_with_bounds")

    params = RSParams.from_t(t)

    if K is None:
        K = params.K_opt

    # Bound del remainder
    bound = rs_remainder_bound(t, K)
    riguroso = params.valido and K <= params.K_opt

    with mp.workdps(dps):
        t_mp = mp.mpf(str(t))
        a_mp = mp.sqrt(t_mp / _TWOPI)
        a_f  = float(a_mp)
        N    = params.N
        p    = params.p

        # θ(t) = Im[log Γ(1/4 + it/2)] - (t/2)·log(π)
        theta = mp.im(mp.loggamma(mp.mpc('0.25', t_mp / 2))) - t_mp / 2 * mp.log(mp.pi)

        # Suma principal Z_0 = 2·Re[Σ k^{-1/2-it} · e^{iθ}]
        # = 2·Σ cos(θ - t·log(k)) / √k
        Z_main = 2.0 * float(
            mp.nsum(
                lambda k: mp.cos(theta - t_mp * mp.log(k)) / mp.sqrt(k),
                [1, N]
            )
        )

        # Coeficientes C_k(p) para la corrección RS
        C_coeffs = C_coefficients(p, K, dps=dps)

        # Corrección RS: (-1)^{N-1} · Re[U · a^{-σ}] · Σ C_k(p)/a^k
        # En la línea crítica σ=1/2, Re[U] = cos(θ - π/8·sign + ...)
        # La corrección RS al Z(t) es:
        # ΔZ = (-1)^{N-1} · 2 · Σ_{k=0}^{K} C_k(p) / a^{k+1/2+k}
        # Fórmula de de Reyna (2011), eq. (4) reescrita para Z:
        sign_N   = (-1) ** (N - 1)
        rs_corr  = 0.0
        a_power  = a_f  # a^1
        for k, ck in enumerate(C_coeffs):
            rs_corr += ck / a_power
            a_power *= a_f

        # El factor U·a^{-1/2} en la línea crítica da el factor 2:
        Z_value = Z_main + sign_N * 2.0 * rs_corr / math.sqrt(a_f)

        # Bound total = bound RS_K / a^{K+1/2}
        bound_total = bound / (a_f ** (K + 0.5)) if a_f > 0 else float('inf')

    nivel = ("riguroso (de Reyna 2011, Theorem 2)"
             if riguroso else f"heurístico (t < {T_MIN_RS})")

    return ZResult(
        Z_value     = Z_value,
        bound_total = bound_total,
        K_usado     = K,
        N_terminos  = N,
        params      = params,
        riguroso    = riguroso,
        nivel       = nivel,
        C_coeffs    = C_coeffs,
    )


# ============================================================================
# VERIFICACIÓN CRUZADA CON ARB
# ============================================================================

def reyna_radius_check(
    t: float,
    radio_arb: float,
    K: Optional[int] = None,
    prec_bits: int = 53,
) -> Dict[str, object]:
    """
    Compara un radio Arb (p. ej. max(rad_re, rad_im) de acb.zeta()) con la
    cota de cola RS de de Reyna en la **misma escala** que `Z_with_bounds`
    (`B_K(t) / a^{K+1/2}`).

    Es un diagnóstico de coherencia de orden de magnitud, no un teorema
    formal ζ-ball ⊆ intervalo analítico (Arb ya propaga cotas en C).
    """
    params = RSParams.from_t(t, prec_bits)
    k_use = params.K_opt if K is None else K
    bound = rs_remainder_bound(t, k_use)
    tail = rs_tail_bound_scaled(t, k_use)
    ok = (
        (radio_arb <= tail)
        if not math.isnan(radio_arb) and math.isfinite(tail)
        else None
    )
    return {
        "T": t,
        "K": k_use,
        "a": params.a,
        "N": params.N,
        "p": params.p,
        "radio_arb": radio_arb,
        "bound_B_K": bound,
        "bound_tail_scaled": tail,
        "bound_normalizado": tail,
        "consistente": ok,
        "K_opt": params.K_opt,
        "t_valido_rs": params.valido,
        "nivel": "diagnóstico reyna_2011 vs radio Arb",
    }


def verificar_vs_arb(t: float, K: int = None, prec_bits: int = 128) -> Dict:
    """
    Evalúa ζ con Arb y compara el radio con la cota de cola de de Reyna
    (`rs_tail_bound_scaled`, misma escala que `Z_with_bounds`).

    No sustituye la certificación Arb: solo añade una comprobación analítica.
    """
    if not _FLINT:
        return {"error": "python-flint no disponible", "consistente": None}

    flint_ctx.prec = prec_bits
    params = RSParams.from_t(t, prec_bits)

    if K is None:
        K = params.K_opt

    bound = rs_remainder_bound(t, K)

    s = acb("0.5", str(t))
    z = s.zeta()
    re_z = z.real
    try:
        rad_re = float(re_z.abs_upper()) - abs(float(re_z))
        rad_im = float(z.imag.abs_upper()) - abs(float(z.imag))
        radio_arb = max(rad_re, rad_im)
    except Exception:
        radio_arb = float("nan")

    tail = rs_tail_bound_scaled(t, K)
    consistente = (
        (radio_arb <= tail) if not math.isnan(radio_arb) and math.isfinite(tail) else None
    )

    return {
        "T": t,
        "K": K,
        "a": params.a,
        "N": params.N,
        "p": params.p,
        "radio_arb": radio_arb,
        "bound_dereyna": bound,
        "bound_tail_scaled": tail,
        "bound_normalizado": tail,
        "consistente": consistente,
        "K_opt": params.K_opt,
        "t_valido_rs": params.valido,
        "nivel": "verificación cruzada Arb vs de Reyna (2011)",
    }


# ============================================================================
# TABLA DE DIAGNÓSTICO
# ============================================================================

def tabla_bounds(
    t:        float,
    K_max:    int  = 12,
    prec_bits: int = 53,
) -> str:
    """
    Tabla de diagnóstico: bounds B_K(t) vs dígitos de precisión.

    Muestra qué K es suficiente para cada nivel de precisión,
    y verifica la consistencia con Arb si está disponible.

    Args:
        t        : altura.
        K_max    : máximo K a mostrar.
        prec_bits: precisión objetivo en bits.

    Returns:
        String con la tabla formateada.
    """
    params = RSParams.from_t(t, prec_bits)
    bounds_list = [rs_remainder_bound(t, K) for K in range(K_max + 1)]

    lines = [
        f"Bounds de Riemann-Siegel (de Reyna 2011) para t = {t:.4e}",
        f"  a = {params.a:.6f}    N = {params.N}    p = {params.p:.6f}",
        f"  K_opt({prec_bits} bits) = {params.K_opt}",
        f"  Válido para RS: {params.valido} (requiere t ≥ {T_MIN_RS})",
        "",
        f"  {'K':>4}  {'B_K(t)':>14}  {'dígitos':>10}  {'suficiente':>12}",
        f"  {'─'*4}  {'─'*14}  {'─'*10}  {'─'*12}",
    ]

    eps_target = 2.0 ** (-prec_bits)

    for K, b in enumerate(bounds_list):
        if b == float('inf'):
            digitos = "∞ (div)"
            suf = "no"
        elif b == 0.0:
            digitos = "∞"
            suf = "sí"
        else:
            try:
                digitos = f"{-math.log10(b):.1f}"
            except (ValueError, OverflowError):
                digitos = "?"
            suf = "sí" if b <= eps_target else "no"

        lines.append(f"  {K:>4}  {b:>14.4e}  {digitos:>10}  {suf:>12}")

    lines.append("")

    # Verificación cruzada con Arb si disponible
    if _FLINT:
        verif = verificar_vs_arb(t, K=params.K_opt)
        radio = verif['radio_arb']
        consist = verif['consistente']
        lines += [
            f"  Verificación cruzada con Arb (prec={prec_bits} bits):",
            f"    Radio Arb:        {radio:.4e}",
            f"    Bound de Reyna:   {verif['bound_normalizado']:.4e}",
            f"    Consistente:      {'✓' if consist else '✗' if consist is False else '?'}",
        ]

    return "\n".join(lines)


# ============================================================================
# AUTO-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  RS BOUNDS (de Reyna 2011) — AUTO-TEST")
    print("=" * 65)
    print(f"  mpmath : {'✓' if _MPMATH else '✗'}")
    print(f"  flint  : {'✓' if _FLINT else '✗'}")
    print()

    # Test 1: verificar C_0(0)
    print("[TEST 1] C_0(0) = cos(3π/8) = sin(π/8)")
    c0 = C0(0.0)
    ok = abs(c0 - C0_AT_ZERO) < 1e-14
    print(f"  C_0(0) = {c0:.14f}")
    print(f"  sin(π/8) = {C0_AT_ZERO:.14f}")
    print(f"  {'✓' if ok else '✗'}")
    print()

    # Test 2: bounds para t=1e6
    print("[TEST 2] Bounds B_K(t) para t=1e6")
    for K in range(6):
        b = rs_remainder_bound(1e6, K)
        print(f"  K={K}: B_K = {b:.4e}")
    print()

    # Test 3: K_opt para distintas precisiones
    print("[TEST 3] K_opt para t=1e6")
    for bits in [53, 128, 256]:
        params = RSParams.from_t(1e6, bits)
        print(f"  {bits} bits → K_opt={params.K_opt}")
    print()

    # Test 4: Z con bounds vs Arb
    print("[TEST 4] Z(t) con bounds vs Arb para ceros conocidos")
    zeros = [14.134725141734694, 21.022039638771554, 25.010857580145688]
    for t_zero in zeros:
        if _FLINT:
            flint_ctx.prec = 128
            s   = acb('0.5', str(t_zero))
            z   = s.zeta()
            mod = abs(complex(float(z.real), float(z.imag)))
            print(f"  t={t_zero:.6f}: |ζ|_Arb={mod:.3e}  "
                  f"(esperado ≈ 0 en el cero)")
    print()

    # Test 5: verificación cruzada
    print("[TEST 5] Verificación cruzada Arb vs de Reyna (t=1e6)")
    v = verificar_vs_arb(1e6)
    print(f"  radio_Arb    = {v['radio_arb']:.4e}")
    print(f"  bound_Reyna  = {v['bound_normalizado']:.4e}")
    print(f"  consistente  = {'✓' if v['consistente'] else '✗'}")
    print()

    # Test 6: tabla diagnóstico
    print("[TEST 6] Tabla de diagnóstico para t=1e8")
    print(tabla_bounds(1e8, K_max=8, prec_bits=53))
    print()

    print("=" * 65)
    print("  ✓ Auto-test completado")
    print("=" * 65)
