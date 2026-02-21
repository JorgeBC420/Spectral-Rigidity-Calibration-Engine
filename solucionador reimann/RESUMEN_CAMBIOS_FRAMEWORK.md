# Resumen de cambios: Framework de Experimentación Espectral

## 1. Cómo se entendió la estructura actual

- **solucionador_reimann.py**: Caché de ceros (mpmath + pickle), análisis del espaciado mínimo (ecuación del espaciado, término singular/regular), integración del flujo Dyson truncado (`solve_ivp`), estudio vs N. Varias funciones con `@jit(nopython=True, parallel=True)` o `@jit(nopython=True, fastmath=True)` (Numba). Importa `rigidez_espectral.ejecutar_analisis_completo` para el Experimento 3.
- **rigidez_espectral.py**: Generadores Uniforme/GUE/Poisson, unfolding tipo N(T), Jacobiano (Hessiano del log-gas), gap espectral, modo blando, protocolo de escalamiento. Tiene **errores de indentación** en los cuerpos de `analizar_espectro_completo` y `analizar_modo_blando` (código al nivel del módulo) y docstrings de `generar_uniforme`/`generar_gue_normalizado` sin indentar. Además, **posible problema de codificación** (UTF-8) en el archivo que provoca `SyntaxError` en `py_compile`.
- **TEST_SUITE.py**: Suite de validación manual (no unittest), comprueba validación de entrada, protección numérica, manejo de excepciones, logging, estándares.
- **Faltaba**: Separación clara Data / Analysis / Storage / Engine; **BaselineFactory** como componente; **varianza del número** y **Delta_3 (Dyson–Mehta)**; **Z-Score Engine** con baselines empíricos (5σ); **persistencia** (SQLite + JSON); uso sistemático de baselines empíricos en lugar de comparaciones simples.

---

## 2. Cambios realizados

### 2.1 Estructura de carpetas (arquitectura profesional)

Se creó el paquete **`src/riemann_spectral/`** con cuatro submódulos:

| Carpeta    | Contenido |
|-----------|-----------|
| **data/** | `zeros_cache.py` (CacheZeros), `generators.py` (Uniforme, GUE, Poisson, `generar_gue_batch` en paralelo). |
| **analysis/** | `unfolding.py` (unfolding_riemann, N_T_approx), `rigidity.py` (espaciados, espaciado_minimo, varianza_numero, delta3_dyson_mehta) con Numba. |
| **storage/** | `bitacora.py` (Bitacora: SQLite + export JSON). |
| **engine/** | `baseline_factory.py` (BaselineFactory), `zscore_engine.py` (ZScoreEngine con umbral 5σ). |

Documentación: **ARQUITECTURA.md** en la raíz.

### 2.2 Optimización de rendimiento

- **Numba**: En `src/riemann_spectral/analysis/rigidity.py` todas las funciones núcleo usan `@jit(nopython=True, fastmath=True)` (calcular_espaciados, espaciado_minimo, varianza_numero_impl, _delta3_recta). En el código legacy, `solucionador_reimann` y `rigidez_espectral` ya tenían `@jit`/`@njit` y `parallel=True` en el Jacobiano y en la velocidad del flujo.
- **GUE en lote**: En `data/generators.py` se añadió **`generar_gue_batch(N, num_realizaciones, ...)`** que usa **`ProcessPoolExecutor`** para generar varias realizaciones GUE en paralelo (ideal para BaselineFactory y muchos N).

### 2.3 Lógica estadística (baselines empíricos)

- **BaselineFactory**: Genera realizaciones GUE y Poisson, calcula la métrica (d_min, varianza_numero, delta3) en cada una y devuelve arrays de valores (o estadísticos media/std). Así la comparación no es “un valor vs un número fijo” sino **vs la distribución empírica** del baseline.
- **ZScoreEngine**: Toma un espectro real (p. ej. Riemann), calcula las métricas, obtiene baselines vía BaselineFactory, calcula **Z-scores** frente a GUE y frente a Poisson, y marca **anomalía** si |Z| ≥ umbral (por defecto 5σ). Sustituye comparaciones simples por detección basada en baselines empíricos.

### 2.4 Módulos/funciones que faltaban y se añadieron

- **Varianza del número**: `varianza_numero(gamma_unfolded, L)` en `analysis/rigidity.py` (con implementación JIT interna).
- **Delta_3 (Dyson–Mehta)**: `delta3_dyson_mehta(gamma_unfolded, L)` en `analysis/rigidity.py`.
- **BaselineFactory**: `engine/baseline_factory.py` (baseline_d_min, baseline_varianza_numero, baseline_delta3, get_baseline_stats).
- **ZScoreEngine**: `engine/zscore_engine.py` (evaluar, hay_anomalia).
- **Bitácora**: `storage/bitacora.py` (SQLite: registrar, listar; exportar_json).

### 2.5 Persistencia

- **Bitacora**: Tabla `hallazgos` (timestamp, tipo, N, metrica, valor, z_score, baseline, extra JSON). Índices por `tipo` y `N`. Método **`exportar_json()`** para volcar a JSON.

### 2.6 Otros ajustes

- **requirements.txt**: Se eliminó `sqlite3` (es parte de la biblioteca estándar).
- **run_experiment.py**: Script de ejemplo que usa CacheZeros, unfolding, métricas, BaselineFactory, ZScoreEngine y Bitacora; registra anomalías y exporta JSON.

---

## 3. Pendiente / recomendaciones

1. **rigidez_espectral.py**: Corregir **indentación** de los cuerpos de `analizar_espectro_completo` y `analizar_modo_blando` (todo el bloque que está al nivel del módulo debe ir indentado dentro de la función). Comprobar **codificación del archivo** (UTF-8 sin BOM) y corregir el carácter que provoca el error en posición 66 (byte 0xf3) para que `py_compile` no falle.
2. **Integración opcional**: En `solucionador_reimann.py` se puede sustituir la comparación directa con GUE/Poisson por `ZScoreEngine.evaluar(...)` y `Bitacora.registrar(...)` cuando se quiera bitácora y detección 5σ.
3. **Tests**: Añadir `tests/` con unittest o pytest que importen `riemann_spectral` y comprueben data, analysis, engine y storage (por ejemplo métricas conocidas sobre espectro uniforme o GUE con seed fija).

---

## 4. Cómo ejecutar el nuevo framework

- Desde la raíz, con `src` en el path:
  ```bash
  set PYTHONPATH=src
  python run_experiment.py
  ```
- O desde `src`:
  ```bash
  cd src
  python -c "from riemann_spectral.data import ...; from riemann_spectral.engine import ..."
  ```

Los hallazgos se guardan en `bitacora_riemann.db` y el export en `bitacora_export.json` (por defecto en la raíz del proyecto).
