# -*- coding: utf-8 -*-
"""
src/riemann_spectral/engine/baseline_cache.py
=============================================
Caché persistente de distribuciones baseline (GUE, Poisson, GOE).

Problema que resuelve:
    BaselineFactory recalcula 100+ diagonalizaciones GUE en cada experimento.
    Para N=1000, cada diagonalización tarda ~0.5s → 50s de espera por run.
    Esta caché persiste los arrays de métricas baseline en disco (NumPy .npz).

Estrategia de invalidación:
    La clave de caché incluye (N, num_realizaciones, L, ensemble, seed).
    Si cualquier parámetro cambia, se recalcula automáticamente.

Uso integrado con BaselineFactory:
    from riemann_spectral.engine.baseline_cache import BaselineCache
    from riemann_spectral.engine.baseline_factory import BaselineFactory

    cache = BaselineCache(directorio="./cache_baselines")
    factory = BaselineFactory(num_realizaciones_gue=100, seed=42)

    # Primera vez: calcula y guarda. Siguientes veces: carga del disco.
    gue_vals, poi_vals = cache.get_or_compute(
        factory.baseline_delta3, N=500, L=2.0, label="delta3"
    )

Formato en disco:
    cache_baselines/
        baseline_delta3_N500_R100_L2.0_s42.npz
        baseline_d_min_N1000_R100_s42.npz
        ...
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np


class BaselineCache:
    """
    Caché de distribuciones baseline en disco (NumPy .npz).

    Args:
        directorio     : carpeta donde se guardan los archivos .npz.
                         Se crea automáticamente si no existe.
        verbose        : si True, imprime mensajes de hit/miss de caché.
    """

    def __init__(
        self,
        directorio: str = "./cache_baselines",
        verbose:    bool = True,
    ):
        self.dir     = Path(directorio)
        self.verbose = verbose
        self.dir.mkdir(parents=True, exist_ok=True)

    # ── API pública ───────────────────────────────────────────────────────────

    def get_or_compute(
        self,
        func:           Callable[[int], Tuple[np.ndarray, np.ndarray]],
        N:              int,
        label:          str,
        num_gue:        int            = 100,
        num_poisson:    int            = 100,
        L:              Optional[float] = None,
        seed:           Optional[int]  = None,
        forzar:         bool           = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve (gue_vals, poi_vals) desde caché o calculándolos.

        Args:
            func        : función de BaselineFactory que devuelve (gue_vals, poi_vals).
                          Ejemplo: factory.baseline_delta3
            N           : tamaño del espectro.
            label       : nombre de la métrica ('delta3', 'd_min', 'varianza_numero').
            num_gue     : número de realizaciones GUE usadas al generar el baseline.
            num_poisson : número de realizaciones Poisson.
            L           : ventana L (para métricas que dependen de L).
            seed        : semilla usada en BaselineFactory.
            forzar      : si True, recalcula aunque exista caché.

        Returns:
            (gue_vals, poi_vals): arrays de la distribución baseline.
        """
        clave  = self._clave(label, N, num_gue, num_poisson, L, seed)
        ruta   = self.dir / f"{clave}.npz"

        if not forzar and ruta.exists():
            return self._cargar(ruta)

        if self.verbose:
            print(f"  [cache MISS] Calculando baseline {label} N={N}...")

        gue_vals, poi_vals = func(N)
        self._guardar(ruta, gue_vals, poi_vals, label=label, N=N, L=L, seed=seed)

        if self.verbose:
            print(f"  [cache SAVE] {ruta.name}")

        return gue_vals, poi_vals

    def invalidar(self, label: Optional[str] = None, N: Optional[int] = None) -> int:
        """
        Elimina entradas de caché.

        Args:
            label : si se especifica, solo elimina entradas con ese label.
            N     : si se especifica, solo elimina entradas con ese N.

        Returns:
            Número de archivos eliminados.
        """
        patron   = f"baseline_{label or '*'}_N{N or '*'}*.npz"
        archivos = list(self.dir.glob(patron))
        for f in archivos:
            f.unlink()
        if self.verbose:
            print(f"  [cache] Invalidados {len(archivos)} archivos ({patron})")
        return len(archivos)

    def listar(self) -> Dict[str, dict]:
        """Lista todas las entradas de caché con sus metadatos."""
        resultado = {}
        for npz in sorted(self.dir.glob("*.npz")):
            try:
                data = np.load(npz, allow_pickle=True)
                meta = json.loads(str(data["meta"]))
                resultado[npz.name] = meta
            except Exception:
                resultado[npz.name] = {"error": "no se pudo leer metadata"}
        return resultado

    def stats(self) -> str:
        """Resumen legible del contenido de la caché."""
        entradas = self.listar()
        if not entradas:
            return "Caché vacía."
        lineas = [f"Caché en: {self.dir}  ({len(entradas)} entradas)\n"]
        for nombre, meta in entradas.items():
            if "error" in meta:
                lineas.append(f"  {nombre}: ERROR")
            else:
                lineas.append(
                    f"  {nombre}\n"
                    f"    label={meta.get('label')}, N={meta.get('N')}, "
                    f"L={meta.get('L')}, seed={meta.get('seed')}, "
                    f"gue_n={meta.get('gue_n')}, poi_n={meta.get('poi_n')}"
                )
        return "\n".join(lineas)

    # ── Métodos internos ──────────────────────────────────────────────────────

    @staticmethod
    def _clave(
        label:       str,
        N:           int,
        num_gue:     int,
        num_poisson: int,
        L:           Optional[float],
        seed:        Optional[int],
    ) -> str:
        """
        Genera un nombre de archivo determinista para los parámetros dados.
        Formato legible: baseline_{label}_N{N}_R{num_gue}_L{L}_s{seed}
        Si los parámetros son muy largos, se añade un hash corto.
        """
        L_str    = f"L{L:.2f}".replace(".", "p") if L is not None else "Lna"
        seed_str = f"s{seed}" if seed is not None else "srand"
        nombre   = f"baseline_{label}_N{N}_R{num_gue}x{num_poisson}_{L_str}_{seed_str}"
        # Limitar longitud y añadir hash si es necesario
        if len(nombre) > 80:
            h      = hashlib.md5(nombre.encode()).hexdigest()[:8]
            nombre = f"baseline_{label}_N{N}_{h}"
        return nombre

    @staticmethod
    def _guardar(
        ruta:      Path,
        gue_vals:  np.ndarray,
        poi_vals:  np.ndarray,
        **meta_kwargs,
    ) -> None:
        """Guarda arrays + metadata en formato .npz comprimido."""
        meta = json.dumps({k: v for k, v in meta_kwargs.items() if v is not None})
        # Añadir stats rápidos a la metadata
        meta_dict = json.loads(meta)
        meta_dict["gue_n"]    = len(gue_vals)
        meta_dict["poi_n"]    = len(poi_vals)
        meta_dict["gue_mean"] = float(np.nanmean(gue_vals))
        meta_dict["poi_mean"] = float(np.nanmean(poi_vals))
        np.savez_compressed(
            ruta,
            gue_vals=gue_vals,
            poi_vals=poi_vals,
            meta=np.array(json.dumps(meta_dict)),
        )

    @staticmethod
    def _cargar(ruta: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Carga arrays desde .npz."""
        data = np.load(ruta, allow_pickle=True)
        return data["gue_vals"], data["poi_vals"]


# ── Integración conveniente con BaselineFactory ───────────────────────────────

class CachedBaselineFactory:
    """
    Wrapper de BaselineFactory que usa BaselineCache de forma transparente.

    Uso:
        from riemann_spectral.engine.baseline_cache import CachedBaselineFactory

        factory = CachedBaselineFactory(
            num_realizaciones_gue=100,
            seed=42,
            cache_dir="./cache_baselines",
        )
        # Primera vez: calcula (lento). Siguientes: carga del disco (rápido).
        gue_vals, poi_vals = factory.baseline_delta3(N=500)
    """

    def __init__(
        self,
        num_realizaciones_gue:     int           = 100,
        num_realizaciones_poisson: int           = 100,
        L_varianza:                float         = 2.0,
        L_delta3:                  float         = 2.0,
        seed:                      Optional[int] = None,
        cache_dir:                 str           = "./cache_baselines",
        verbose:                   bool          = True,
    ):
        from .baseline_factory import BaselineFactory
        self._factory = BaselineFactory(
            num_realizaciones_gue     = num_realizaciones_gue,
            num_realizaciones_poisson = num_realizaciones_poisson,
            L_varianza                = L_varianza,
            L_delta3                  = L_delta3,
            seed                      = seed,
        )
        self._cache   = BaselineCache(directorio=cache_dir, verbose=verbose)
        self._num_gue = num_realizaciones_gue
        self._num_poi = num_realizaciones_poisson
        self._seed    = seed
        self.L_varianza = L_varianza
        self.L_delta3   = L_delta3

    def baseline_delta3(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._cache.get_or_compute(
            self._factory.baseline_delta3,
            N=N, label="delta3",
            num_gue=self._num_gue, num_poisson=self._num_poi,
            L=self.L_delta3, seed=self._seed,
        )

    def baseline_d_min(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._cache.get_or_compute(
            self._factory.baseline_d_min,
            N=N, label="d_min",
            num_gue=self._num_gue, num_poisson=self._num_poi,
            seed=self._seed,
        )

    def baseline_varianza_numero(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._cache.get_or_compute(
            self._factory.baseline_varianza_numero,
            N=N, label="varianza",
            num_gue=self._num_gue, num_poisson=self._num_poi,
            L=self.L_varianza, seed=self._seed,
        )

    def stats_cache(self) -> str:
        """Muestra el contenido actual de la caché."""
        return self._cache.stats()

    def invalidar_cache(self, label: Optional[str] = None, N: Optional[int] = None) -> int:
        """Elimina entradas de caché. Ver BaselineCache.invalidar()."""
        return self._cache.invalidar(label=label, N=N)
