"""
Generadores de espectros de referencia: Uniforme, GUE, Poisson.
Usados como baselines empíricos para comparación con Riemann.
Generación en lote de GUE en paralelo para aprovechar multi-núcleo.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg as la


def generar_uniforme(
    N: int,
    soporte: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Red uniforme (máxima rigidez). Soporte por defecto (0, N)."""
    if N < 2:
        raise ValueError(f"N debe ser >= 2, recibido: {N}")
    if soporte is None:
        soporte = (0.0, float(N))
    a, b = soporte
    if a >= b:
        raise ValueError(f"Soporte inválido: a={a} < b={b}")
    return np.linspace(a, b, N)


def generar_gue_normalizado(
    N: int,
    escala: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Espectro GUE: matriz Hermítica aleatoria, diagonalización, re-escalado."""
    if N < 2:
        raise ValueError(f"N debe ser >= 2, recibido: {N}")
    if escala <= 0:
        raise ValueError(f"escala debe ser > 0, recibido: {escala}")
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    H = (A + A.conj().T) / (2 * np.sqrt(N))
    evals = la.eigvalsh(H)
    media = np.mean(evals)
    desv = np.std(evals)
    if desv < 1e-10:
        desv = 1.0
    evals_escalado = escala * (evals - media) / desv + N / 2
    return np.sort(evals_escalado)


def generar_poisson(
    N: int,
    soporte: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Proceso de Poisson (mínima rigidez, sin correlaciones)."""
    if N < 2:
        raise ValueError(f"N debe ser >= 2, recibido: {N}")
    if soporte is None:
        soporte = (0.0, float(N))
    a, b = soporte
    if a >= b:
        raise ValueError(f"Soporte inválido: a={a} < b={b}")
    if seed is not None:
        np.random.seed(seed)
    return np.sort(np.random.uniform(a, b, N))


def _generar_gue_single(args: Tuple[int, float, Optional[int]]) -> np.ndarray:
    """Worker para generación paralela: (N, escala, seed) -> espectro."""
    N, escala, seed = args
    return generar_gue_normalizado(N, escala=escala, seed=seed)


def generar_gue_batch(
    N: int,
    num_realizaciones: int,
    escala: float = 1.0,
    seed_base: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Genera num_realizaciones espectros GUE de tamaño N en paralelo.
    Aprovecha múltiples núcleos (ProcessPoolExecutor) para diagonalizaciones.
    """
    if max_workers is None:
        max_workers = min(num_realizaciones, os.cpu_count() or 4)
    args_list = [
        (N, escala, (seed_base + i) if seed_base is not None else None)
        for i in range(num_realizaciones)
    ]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_generar_gue_single, a) for a in args_list]
        return [f.result() for f in futures]
