"""
Caché persistente de ceros de Riemann (mpmath).
Proveedor de datos para el pipeline de análisis.
"""

import pickle
from typing import Optional

import numpy as np

try:
    import mpmath as mp
    mp.mp.dps = 50
    _MPMATH_AVAILABLE = True
except ImportError:
    _MPMATH_AVAILABLE = False


class CacheZeros:
    """Caché persistente de ceros de Riemann."""

    def __init__(self, archivo: str = "cache_ceros_riemann.pkl"):
        self.archivo = archivo
        self.ceros: dict = {}
        self.cargar()

    def cargar(self) -> None:
        try:
            with open(self.archivo, "rb") as f:
                self.ceros = pickle.load(f)
        except FileNotFoundError:
            pass

    def guardar(self) -> None:
        with open(self.archivo, "wb") as f:
            pickle.dump(self.ceros, f)

    def obtener(self, N: int) -> np.ndarray:
        if not _MPMATH_AVAILABLE:
            raise RuntimeError("mpmath no disponible; instale mpmath para calcular ceros.")
        N_actual = len(self.ceros)
        if N <= N_actual:
            return np.array([self.ceros[i] for i in range(1, N + 1)], dtype=float)
        for n in range(N_actual + 1, N + 1):
            self.ceros[n] = float(mp.im(mp.zetazero(n)))
            if n % 1000 == 0:
                self.guardar()
        self.guardar()
        return np.array([self.ceros[i] for i in range(1, N + 1)], dtype=float)
