"""
Analisis espectral: Jacobiano (Hessiano log-gas), gap, modo blando.
Depende de unfolding para normalizar; usa scipy para diagonalizacion.
"""

import logging
from typing import Dict, Any

import numpy as np
import scipy.linalg as la
from numba import jit, prange

from .unfolding import unfolding_riemann

logger = logging.getLogger(__name__)

EPSILON = 1e-10
MAX_VAL = 1e10


@jit(nopython=True, parallel=True, fastmath=True)
def calcular_jacobiano_kernel(gamma: np.ndarray) -> np.ndarray:
    """
    Jacobiano de la dinamica: J = -Hessiano(E_log_gas).
    J_kl = 2/(gamma_k - gamma_l)^2 (k != l), J_kk = -suma fila.
    """
    N = len(gamma)
    J = np.zeros((N, N))
    for k in prange(N):
        for l in range(k + 1, N):
            diff = gamma[k] - gamma[l]
            diff_abs = np.abs(diff)
            if diff_abs < EPSILON:
                val = MAX_VAL
            else:
                val = min(2.0 / (diff * diff), MAX_VAL)
            J[k, l] = val
            J[l, k] = val
    for k in range(N):
        J[k, k] = -np.sum(J[k, :])
    return J


@jit(nopython=True, fastmath=True)
def energia_log_gas(gamma: np.ndarray) -> float:
    """Energia Hamiltoniana E = -sum_{i<j} log|gamma_i - gamma_j|."""
    N = len(gamma)
    E = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            diff = gamma[i] - gamma[j]
            E -= np.log(np.abs(diff))
    return E


def analizar_espectro_completo(
    gamma: np.ndarray,
    sistema_label: str = "Anonimo",
) -> Dict[str, Any]:
    """
    Analisis espectral completo: unfolding, Jacobiano, eigh, gap, modo blando, energia.
    """
    if gamma is None or len(gamma) < 2:
        raise ValueError("gamma debe tener al menos 2 elementos")
    if not np.all(np.isfinite(gamma)):
        raise ValueError(f"gamma contiene valores invalidos en '{sistema_label}'")
    N = len(gamma)
    try:
        gamma_u = unfolding_riemann(gamma)
        if not np.all(np.isfinite(gamma_u)):
            logger.warning("%s: unfolding produjo NaN, usando gamma original", sistema_label)
            gamma_u = gamma
        J = calcular_jacobiano_kernel(gamma_u)
        if not np.all(np.isfinite(J)):
            raise RuntimeError(f"Jacobiano invalido para {sistema_label}")
        evals, evecs = la.eigh(J)
        idx_ordenado = np.argsort(evals)[::-1]
        evals_sorted = evals[idx_ordenado]
        evecs_reordenados = evecs[:, idx_ordenado]
        lambda_0 = float(evals_sorted[0])
        lambda_1 = float(evals_sorted[1]) if len(evals_sorted) > 1 else float(evals_sorted[0])
        gap = np.abs(lambda_1)
        if gap < 1e-15:
            gap = 1e-15
        v1 = evecs_reordenados[:, 1] if len(evals_sorted) > 1 else evecs_reordenados[:, 0]
        energia = energia_log_gas(gamma_u)
        cond_J = float(np.linalg.cond(J))
        if cond_J > 1e12:
            logger.warning("%s (N=%d): cond(J)=%.2e", sistema_label, N, cond_J)
        return {
            "N": N,
            "sistema": sistema_label,
            "gap": float(gap),
            "lambda_0": lambda_0,
            "lambda_1": lambda_1,
            "evals": evals_sorted,
            "v_modo_blando": v1,
            "energia": float(energia),
            "cond_J": cond_J,
            "gamma_original": gamma,
            "gamma_unfolded": gamma_u,
            "jacobiano": J,
        }
    except Exception as e:
        logger.error("Error en analizar_espectro_completo('%s', N=%s): %s", sistema_label, N, e)
        raise


def analizar_modo_blando(resultado: Dict[str, Any]) -> Dict[str, Any]:
    """Caracterizacion del vector propio del modo mas blando."""
    try:
        v1 = resultado["v_modo_blando"]
        N = len(v1)
        if N < 11:
            raise ValueError(f"Vector demasiado corto (N={N})")
        v1_max = np.max(np.abs(v1))
        if v1_max < 1e-10:
            raise ValueError("Vector propio nulo")
        v1_norm = v1 / v1_max
        tercio_inicio = np.mean(np.abs(v1_norm[: N // 3]))
        tercio_medio = np.mean(np.abs(v1_norm[N // 3 : 2 * N // 3]))
        tercio_final = np.mean(np.abs(v1_norm[2 * N // 3 :]))
        denom_loc = 2 * tercio_medio + 1e-10
        localizacion = (tercio_inicio + tercio_final) / denom_loc
        x = np.arange(N)
        sinusoide = np.sin(np.pi * x / N)
        try:
            corr_matrix = np.corrcoef(v1_norm, sinusoide)
            correlacion_sin = float(np.abs(corr_matrix[0, 1]))
            if np.isnan(correlacion_sin):
                correlacion_sin = 0.0
        except Exception:
            correlacion_sin = 0.0
        n_borde = min(5, N // 10)
        energia_borde = np.sum(v1_norm[:n_borde] ** 2) + np.sum(v1_norm[-n_borde:] ** 2)
        energia_centro = np.sum(v1_norm[n_borde:-n_borde] ** 2)
        ratio_borde = energia_borde / (energia_centro + 1e-10)
        return {
            "localizacion_index": float(localizacion),
            "correlacion_sinusoidal": correlacion_sin,
            "ratio_energia_borde": float(ratio_borde),
            "interpretacion": clasificar_modo_blando(localizacion, correlacion_sin, ratio_borde),
        }
    except Exception as e:
        logger.error("Error analizando modo blando: %s", e)
        return {
            "localizacion_index": np.nan,
            "correlacion_sinusoidal": np.nan,
            "ratio_energia_borde": np.nan,
            "interpretacion": "ERROR",
        }


def clasificar_modo_blando(loc: float, corr_sin: float, ratio_borde: float) -> str:
    """Clasificacion heuristica del modo blando."""
    if ratio_borde > 0.5:
        return "LOCALIZADO EN BORDES (artefacto)"
    if corr_sin > 0.8:
        return "SINUSOIDAL (elasticidad continua)"
    if loc < 0.3:
        return "MODO GLOBAL (ondulacion colectiva)"
    return "ESTRUCTURA COMPLEJA"
