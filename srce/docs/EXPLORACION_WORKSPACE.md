# Exploración exhaustiva del workspace "solucionador reimann"

## 1. Estructura de carpetas y archivos

```
solucionador reimann/
├── Python (módulos y script)
│   ├── solucionador_reimann.py    [MAIN] Programa principal
│   ├── rigidez_espectral.py       [MODULE] Protocolo de rigidez espectral
│   └── TEST_SUITE.py              [TEST] Suite de validación (no unittest)
├── Config / proyecto
│   ├── requirements.txt
│   └── solucionador reimann.pyproj
├── Documentación
│   ├── readme.md
│   ├── QUICKSTART.md
│   ├── INDICE_COMPLETO.md
│   ├── CHANGELOG.md
│   ├── RESUMEN_IMPLEMENTACION.md
│   ├── README_PROTOCOLO_RIGIDEZ.md
│   └── AUDITORIA_SEGURIDAD.md
└── Data/Output (runtime)
    ├── cache_ceros_riemann.pkl    (persistencia actual: pickle)
    ├── analisis_espaciado_vs_N.png
    ├── evolucion_dmin.png
    └── rigidez_espectral_protocolo.png
```

**No existen**: carpeta `tests/` (solo `TEST_SUITE.py` en raíz), carpeta `config/` dedicada, ni módulos separados por responsabilidad (Data, Analysis, Storage, Engine).

---

## 2. Módulos Python: responsabilidades y librerías

### 2.1 `solucionador_reimann.py` (MAIN)

| Responsabilidad | Implementación |
|-----------------|----------------|
| **Data** | `CacheZeros`: ingesta de ceros vía `mpmath.zetazero(n)`, persistencia en **pickle** (`cache_ceros_riemann.pkl`). |
| **Engine** | Flujo Dyson truncado: `velocidad_ceros_truncado`, `integrar_flujo` (scipy `solve_ivp`), `sistema_dinamico`. |
| **Analysis** | Espaciado: `calcular_espaciados`, `espaciado_minimo`, `ecuacion_espaciado_minimo_correcta`, `descomponer_termino_regular`, `analizar_espaciado_puntual`, `estudiar_espaciado_vs_N`, `monitorear_coalescencia`. |

**Numba/njit**:  
- `@jit(nopython=True, parallel=True, fastmath=True)`: `velocidad_ceros_truncado`  
- `@jit(nopython=True, fastmath=True)`: `calcular_espaciados`, `espaciado_minimo`, `ecuacion_espaciado_minimo_correcta`, `descomponer_termino_regular`

**mpmath**: `mp.im(mp.zetazero(n))` para ceros (con `mp.mp.dps = 50`).

**scipy**: `scipy.integrate.solve_ivp` (RK45) para integración del flujo.

---

### 2.2 `rigidez_espectral.py` (MODULE)

| Responsabilidad | Implementación |
|-----------------|----------------|
| **Data** | Generadores de benchmarks: `generar_uniforme`, `generar_gue_normalizado`, `generar_poisson`. No ingesta de ceros reales (recibe `gamma` desde fuera). |
| **Analysis** | `analizar_espectro_completo`, `analizar_modo_blando`, `clasificar_modo_blando`, `ajustar_exponente_critico`; protocolo: `ejecutar_protocolo_escalamiento`, `ejecutar_analisis_completo`. |
| **Engine** | Jacobiano del log-gas: `calcular_jacobiano_kernel`, `energia_log_gas`; unfolding: `unfolding_riemann`. |
| **Storage** | Ninguno (no persiste resultados en DB ni JSON). |

**Numba/njit**:  
- `@jit(nopython=True, fastmath=True)`: `unfolding_riemann`, `energia_log_gas`  
- `@jit(nopython=True, parallel=True, fastmath=True)`: `calcular_jacobiano_kernel`

**scipy**: `scipy.linalg.la.eigvalsh`, `la.eigh` (diagonalización).

**mpmath**: Importado pero no usado en el código del módulo (solo en docstrings/comentarios).

---

### 2.3 `TEST_SUITE.py`

Script de validación que lista casos de prueba, versiones esperadas y cobertura de funciones; no ejecuta tests automatizados (no usa `unittest`/`pytest`). No usa numba/mpmath/scipy en lógica (solo importa scipy/numba para comprobar versiones).

---

## 3. Búsqueda de conceptos clave

| Concepto | ¿Existe? | Dónde / Notas |
|----------|----------|----------------|
| **BaselineFactory** | No (solo en docs) | `readme.md` menciona "Motor de Baselines (BaselineFactory)" como diseño; en código no hay clase ni función con ese nombre. Equivalente funcional: generadores en `rigidez_espectral.py` (`generar_gue_normalizado`, `generar_poisson`, `generar_uniforme`). |
| **Unfolding** | Sí | `rigidez_espectral.py`: `unfolding_riemann(gamma)` con fórmula tipo N(T); JIT. Usado en `analizar_espectro_completo`; fallback a `gamma` si hay NaN. |
| **N(T)** | Sí (solo fórmula) | Fórmula Riemann–von Mangoldt en docstring y comentario: `N(T) = (T/2π) log(T/2πe) + O(1)`. Implementación: `unfolding_riemann` usa `(gamma/(2π))*(log(gamma/(2π)) - 1)` (transformación para densidad local ~1). |
| **Espaciado mínimo** | Sí | `solucionador_reimann.py`: `espaciado_minimo(gamma)` → `(d_min, idx)`; `calcular_espaciados`; análisis puntual y estudio vs N. |
| **Varianza del número** | No | No hay cálculo de varianza del número de ceros en un intervalo (Number Variance). |
| **Delta_3 (Dyson–Mehta)** | No | No hay estadística Δ₃(L) ni referencia explícita a Dyson–Mehta en código. |
| **Z-scores** | No (solo en docs) | `readme.md` menciona "Z-Score Engine" y detección 5σ; no hay implementación de Z-scores en el código. |
| **Persistencia SQLite** | No | `readme.md` y docs hablan de "Bitácora en SQLite"; no hay uso de SQLite. `requirements.txt` incluye `sqlite3` (innecesario: es stdlib). |
| **Persistencia JSON** | No | No hay guardado/carga de resultados en JSON. Persistencia actual: solo **pickle** para el caché de ceros. |

---

## 4. Resumen según framework de experimentación espectral (Riemann vs GUE)

### 4.1 Lo que existe

- **Datos**: Ceros vía mpmath; caché en pickle. Benchmarks: Uniforme, GUE (matriz aleatoria + `eigvalsh`), Poisson.
- **Unfolding**: `unfolding_riemann` con fórmula N(T); espaciado medio normalizado localmente.
- **Rigidez local**: Jacobiano del log-gas, gap espectral, modo blando; comparación Riemann vs Uniforme/GUE/Poisson.
- **Espaciado mínimo**: Cálculo y análisis puntual (ecuación del espaciado, descomposición singular/regular); estudio vs N; integración dinámica del flujo truncado y monitoreo de d_min(t).
- **Stack numérico**: Numba (jit/njit, parallel en kernel Jacobiano y velocidad), scipy (integrate, linalg), mpmath (solo ceros).
- **Tests**: `TEST_SUITE.py` como checklist de validación; no estadística espectral (NNSD, Δ₃, etc.).

### 4.2 Lo que falta (para un framework espectral completo)

| Componente | Estado |
|------------|--------|
| **BaselineFactory** | Solo en documentación; en código son funciones sueltas (GUE/Poisson/Uniforme). |
| **Estadísticas de espaciados** | Sin NNSD (distribución del espaciado al siguiente cero), sin P(s). |
| **Delta_3 (Dyson–Mehta)** | No implementado. |
| **Varianza del número** | No implementada. |
| **Z-Score Engine** | Mencionado en readme; no implementado. |
| **Bitácora SQLite** | Mencionada; no implementada. |
| **Persistencia de resultados** | Solo caché de ceros (pickle). No export de métricas/experimentos a JSON/SQLite. |
| **GOE** | readme menciona GUE y GOE; solo GUE está implementado. |
| **Datasets externos (alta altura)** | readme menciona "datasets externos"; solo mpmath en código. |
| **Carpeta tests/** | No hay; un único script de validación en raíz. |
| **Config explícito** | No hay carpeta `config/` ni archivos de configuración separados. |

---

## 5. Rutas de archivos relevantes

| Tema | Archivo(s) |
|------|------------|
| Unfolding, N(T) | `rigidez_espectral.py` (líneas 185–195, 297–302) |
| Espaciado mínimo, ecuación del espaciado | `solucionador_reimann.py` (líneas 108–224, 255–341, 347–416) |
| GUE, benchmarks | `rigidez_espectral.py` (líneas 86–139, 143–178) |
| Caché ceros, persistencia | `solucionador_reimann.py` (líneas 44–78) |
| Jacobiano, gap espectral | `rigidez_espectral.py` (líneas 197–241, 268–358) |
| Numba JIT | `rigidez_espectral.py` (184–245); `solucionador_reimann.py` (86–189) |
| Diseño BaselineFactory / Z-Score / SQLite | `readme.md` |
| Dependencias | `requirements.txt` |

---

## 6. Conclusión breve

El workspace implementa un **núcleo sólido** de experimentación: unfolding tipo N(T), espaciado mínimo con análisis puntual y dinámico, y protocolo de rigidez espectral (Jacobiano, gap, modo blando) comparando Riemann con Uniforme, GUE y Poisson, con Numba y scipy. **Faltan** estadísticas estándar de teoría de matrices aleatorias (NNSD, Δ₃, varianza del número), el motor de baselines como clase (BaselineFactory), Z-scores, y cualquier persistencia de experimentos (SQLite/JSON). La persistencia actual es solo el caché de ceros en pickle.
