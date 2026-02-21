# Arquitectura del Framework de Experimentación Espectral

## Resumen

El proyecto se organiza en **dos capas**:

1. **Legacy / scripts originales** (raíz): `solucionador_reimann.py`, `rigidez_espectral.py`, `TEST_SUITE.py`  
   - Siguen siendo el punto de entrada principal para el análisis de log-gas, flujo dinámico y protocolo de rigidez espectral.

2. **Paquete modular** `src/riemann_spectral/`: Data, Analysis, Storage, Engine.  
   - Pensado para experimentación reproducible, baselines empíricos, Z-scores y bitácora.

---

## Estructura de carpetas (paquete `src/riemann_spectral/`)

```
src/riemann_spectral/
├── __init__.py
├── data/              # Datos y generadores de referencia
│   ├── __init__.py
│   ├── zeros_cache.py   # CacheZeros (mpmath + pickle)
│   └── generators.py    # Uniforme, GUE, Poisson; generar_gue_batch (paralelo)
├── analysis/          # Métricas y unfolding
│   ├── __init__.py
│   ├── unfolding.py    # unfolding_riemann, N_T_approx
│   └── rigidity.py     # calcular_espaciados, espaciado_minimo, varianza_numero, delta3_dyson_mehta
├── storage/           # Persistencia
│   ├── __init__.py
│   └── bitacora.py     # Bitacora (SQLite + export JSON)
└── engine/            # Orquestación y detección
    ├── __init__.py
    ├── baseline_factory.py  # BaselineFactory (baselines empíricos GUE/Poisson)
    └── zscore_engine.py     # ZScoreEngine (Z-scores 5σ, anomalías)
```

---

## Responsabilidades

| Módulo   | Responsabilidad |
|----------|------------------|
| **Data** | Caché de ceros Riemann (mpmath), generación de espectros de referencia (Uniforme, GUE, Poisson). Generación en lote GUE en paralelo (`generar_gue_batch`) para aprovechar multi-núcleo. |
| **Analysis** | Unfolding (N(T)), espaciado mínimo, varianza del número, Delta_3 (Dyson–Mehta). Funciones núcleo con Numba `@jit(nopython=True, fastmath=True)` donde aplica. |
| **Storage** | Bitácora de hallazgos: SQLite (consultas, índices) y export JSON. |
| **Engine** | **BaselineFactory**: genera distribuciones empíricas de métricas bajo GUE y Poisson para comparación sin sesgo. **ZScoreEngine**: compara datos reales vs baselines y marca anomalías (umbral configurable, p. ej. 5σ). |

---

## Flujo típico

1. **Datos**: `CacheZeros.obtener(N)` o generadores GUE/Poisson.
2. **Unfolding**: `unfolding_riemann(gamma)` para normalizar densidad.
3. **Métricas**: `espaciado_minimo`, `varianza_numero`, `delta3_dyson_mehta` sobre espectro unfolded.
4. **Baselines**: `BaselineFactory.get_baseline_stats(N, metrica)` o `baseline_d_min` / `baseline_varianza_numero` / `baseline_delta3`.
5. **Detección**: `ZScoreEngine.evaluar(gamma, N)` → Z-scores vs GUE/Poisson y flag de anomalía.
6. **Persistencia**: `Bitacora.registrar(...)` para hallazgos; `exportar_json()` para volcado.

---

## Uso del paquete

Desde la raíz del proyecto (con `src` en `PYTHONPATH` o desde el directorio que contiene `src`):

```python
from riemann_spectral.data import CacheZeros, generar_gue_normalizado, generar_gue_batch
from riemann_spectral.analysis import unfolding_riemann, espaciado_minimo, varianza_numero, delta3_dyson_mehta
from riemann_spectral.engine import BaselineFactory, ZScoreEngine
from riemann_spectral.storage import Bitacora
```

Script de ejemplo: `run_experiment.py` (en la raíz).

---

## Optimización (Numba y paralelismo)

- **solucionador_reimann.py**: `velocidad_ceros_truncado` ya usa `@jit(nopython=True, parallel=True, fastmath=True)` y `prange`.  
- **rigidez_espectral.py**: `calcular_jacobiano_kernel` usa `@jit(..., parallel=True)` y `prange`; `unfolding_riemann`, `energia_log_gas` con `@jit(nopython=True, fastmath=True)`.  
- **src/riemann_spectral/analysis/rigidity.py**: `calcular_espaciados`, `espaciado_minimo`, `varianza_numero_impl`, `_delta3_recta` con `@jit(nopython=True, fastmath=True)`.  
- **GUE en lote**: `generar_gue_batch` usa `ProcessPoolExecutor` para varias realizaciones en paralelo (diagonalización con SciPy no es Numba, pero se paraleliza a nivel de realizaciones).

---

## Relación con los scripts legacy

- `solucionador_reimann.py` sigue usando su propio `CacheZeros` y `rigidez_espectral.ejecutar_analisis_completo`.  
- Opcionalmente se puede migrar a `from riemann_spectral.data import CacheZeros` y reutilizar `analysis.unfolding` / `analysis.rigidity` si se desea unificar.  
- La **lógica estadística** de comparación debe usar **BaselineFactory + ZScoreEngine** (baselines empíricos y Z-scores) en lugar de comparaciones ad hoc.
