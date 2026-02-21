# Test de Estres: Paralelismo en i7-1255U

## Objetivo

Comprobar que **Numba** (algebra lineal compleja / Jacobiano) y **ProcessPoolExecutor** (generacion GUE) trabajan en equipo sin saturar la maquina, aprovechando la arquitectura híbrida:

- **2 nucleos Performance (P-cores)**: ideal para el codigo Numba (Jacobiano, espaciados, unfolding) y para `scipy.linalg.eigh` (diagonalizacion). Windows suele asignar threads de NumPy/SciPy/Numba a estos nucleos cuando el trabajo es intensivo.
- **8 nucleos Efficiency (E-cores)**: ideales para muchas realizaciones GUE en paralelo (generacion de matrices + diagonalizacion por realizacion) via `generar_gue_batch` con `ProcessPoolExecutor`.

## Variables de entorno recomendadas

Antes de ejecutar el test de estres:

```powershell
# Limitar Numba/NumPy a 2 threads (evitar que invada todos los nucleos)
$env:NUMBA_NUM_THREADS = "2"
$env:OMP_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"

# Opcional: dejar que ProcessPoolExecutor use hasta 8 workers (E-cores)
# (por defecto generar_gue_batch ya usa min(num_realizaciones, cpu_count))
```

En CMD:

```cmd
set NUMBA_NUM_THREADS=2
set OMP_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
set MKL_NUM_THREADS=2
```

## Que hace cada componente

| Componente | Donde | Paralelismo | Nucleos tipicos |
|------------|--------|-------------|------------------|
| Jacobiano `calcular_jacobiano_kernel` | analysis/spectral.py | Numba `prange(N)` | P-cores (2 threads) |
| Espaciados / ecuacion espaciado / Delta3 | analysis/rigidity.py | Numba `@jit` (single thread por llamada) | P-cores |
| Diagonalizacion `scipy.linalg.eigh` | analysis/spectral.py | BLAS/LAPACK (respeta OMP_NUM_THREADS) | P-cores |
| Generacion GUE en lote | data/generators.py `generar_gue_batch` | ProcessPoolExecutor (N procesos) | E-cores (hasta 8 workers) |

Así, los 2 P-cores llevan la voz cantante en algebra lineal y los E-cores en la generacion de ruido GUE.

## Script de test de estres

Ejecutar desde la raiz (con `src` en PYTHONPATH o usando `main.py`):

```python
# test_estres.py (ejemplo minimo)
import os
os.environ["NUMBA_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

import sys
sys.path.insert(0, "src")
import numpy as np
from riemann_spectral.data import generar_gue_batch
from riemann_spectral.analysis import calcular_jacobiano_kernel, analizar_espectro_completo

# Estres: muchas realizaciones GUE en paralelo + analisis espectral (Jacobiano + eigh)
N = 500
num_realizaciones = 16  # ProcessPoolExecutor repartira en varios procesos
gammas = generar_gue_batch(N, num_realizaciones, seed_base=42, max_workers=8)
for i, gamma in enumerate(gammas):
    analizar_espectro_completo(gamma, f"GUE_{i}")
print("Test de estres completado")
```

- Ajustar `N` y `num_realizaciones` para que la PC trabaje fuerte unos segundos.
- Monitorear en Administrador de tareas: uso de 2 nucleos mas cargados (P) y varios nucleos con carga media (E) durante el batch GUE.

## Resumen

- **Import-safe**: Todo el framework se importa desde `riemann_spectral`; `main.py` verifica los imports.
- **P-cores**: Numba (2 hilos) + BLAS (2 hilos) para Jacobiano y eigh.
- **E-cores**: ProcessPoolExecutor en `generar_gue_batch` para generacion GUE masiva.
