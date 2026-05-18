# Auditoría completa — Spectral Rigidity Calibration Engine (SRCE)

**Fecha:** 2026-05-18  
**Entrada auditada:** `Spectral-Rigidity-Calibration-Engine-main.zip`  
**Contexto visual:** capturas del dashboard con N≈500 ceros; `K(t)` fallaba con `name 'spectral_form_factor_mehta' is not defined`.

**Actualización repo (post-fusión, 2026-05-18):** los fixes K(t)/imports de esta auditoría están en `main`. `zeta_altura_extrema.py` en el repo es **v2.2.3** (Fase 1 con Z exacta en grilla; secante con `min`/`max` nativos — corrige rechazo masivo en Fase 2). La sección §7 sobre “Fase 1 basada en fase” describe el zip auditado, no la versión fusionada.

---

## 1. Resultado ejecutivo

El proyecto tiene una arquitectura razonable para exploración RMT/Riemann: separación entre generación de ensembles, unfolding, estadísticas espectrales, dashboard y módulos rigurosos opcionales. Las estadísticas principales observadas en capturas son coherentes con la hipótesis Montgomery–Odlyzko: Riemann aparece cercano a GUE en `r-statistic`, `R₂(s)` y rigidez/varianza de número, mientras Poisson queda bien separado.

La falla visible en `K(t)` no era matemática sino de dependencia/importación: el dashboard llamaba a `spectral_form_factor_mehta(...)`, pero no lo importaba desde `spectral_form_factor.py`. Se corrigió en `dashboard.py` y se sincronizó `analysis/__init__.py` para exportar los símbolos nuevos.

Estado de entrega corregida:

- `dashboard.py`: import agregado para `spectral_form_factor_mehta`.
- `dashboard.py`: fórmula mostrada de `K(t)` corregida de `1/N` a `1/N²`, consistente con la implementación Mehta del propio módulo.
- `src/riemann_spectral/analysis/__init__.py`: actualizado para exportar `pair_correlation`, `spectral_form_factor_mehta`, `spectral_form_factor_teorico`, `r_statistic` y constantes RMT.
- `validate_imports.py`: corregido para que `CACHE` sea validado como atributo, no como función callable.
- `requirements-rigorous.txt`: agregado para instalar explícitamente el backend opcional `python-flint` usado por `arb_bridge.py`.
- `PATCH_KT_IMPORT.diff`: incluido con el diff mínimo de los cambios.

---

## 2. Auditoría del error `K(t)`

### Hallazgo

En las capturas, la pestaña `K(t)` muestra repetidamente:

```text
K(t): error en Poisson: name 'spectral_form_factor_mehta' is not defined
K(t): error en GUE: name 'spectral_form_factor_mehta' is not defined
K(t): error en GOE: name 'spectral_form_factor_mehta' is not defined
K(t): error en Riemann: name 'spectral_form_factor_mehta' is not defined
```

### Causa raíz

En `dashboard.py`, el bloque de importación tenía:

```python
from src.riemann_spectral.analysis.spectral_form_factor import (
    spectral_form_factor,
    spectral_form_factor_teorico,
    r_statistic,
    r_distribucion_teorica,
    ...
)
```

pero más abajo llamaba:

```python
t_obs, K_obs = spectral_form_factor_mehta(...)
```

Por eso `K(t)` fallaba solo en tiempo de renderizado de esa pestaña.

### Corrección aplicada

```python
from src.riemann_spectral.analysis.spectral_form_factor import (
    spectral_form_factor,
    spectral_form_factor_mehta,
    spectral_form_factor_teorico,
    r_statistic,
    r_distribucion_teorica,
    ...
)
```

También se corrigió la fórmula visual del dashboard:

```python
K(t) = \frac{1}{N^2}\left|\sum_n e^{2\pi i\,t\,\gamma_n}\right|^2
```

La implementación `spectral_form_factor_mehta` ya usa `1/(N*N)`, por lo que la interfaz quedó alineada con el código.

### Recomendación adicional sobre `K(t)`

Aunque el error queda corregido, conviene agregar una nota metodológica: una sola realización de `K(t)` es ruidosa, y `t=0` produce un pico finito grande por autocorrelación. Para visualización didáctica sería mejor:

1. iniciar el grid en un `t_min > 0`, por ejemplo `t_min = 1/N` o `1/(2N)`;
2. mostrar `K_connected(t)` o explicar que el punto `t=0` no debe compararse directamente con el ramp teórico;
3. promediar varias realizaciones para Poisson/GUE/GOE cuando el objetivo sea comparar forma global.

---

## 3. Auditoría de las capturas del dashboard

### 3.1 Unfolding comparado

Los tres métodos (`KDE`, `Spline`, `Polinomial`) producen espectros unfolded monótonos. En la tabla de métricas para Poisson se observa:

- `<s>` ≈ 1 en los tres métodos.
- `σ(s)` ≈ 1, esperado para Poisson.
- `<r>` ≈ 0.3807, cercano al valor teórico Poisson ≈ 0.3863.
- los tres métodos clasifican como Poisson.

Auditoría: esta sección funciona y es útil. La adición de métricas por método evita que el usuario confíe solo en la forma visual del unfolding. El recorte de extremos es correcto porque los bordes de un unfolding empírico son los más inestables.

Riesgo: con `Polinomio grado 7` puede haber sobreajuste local y oscilaciones de borde si se usa con pocos puntos. El dashboard debería mostrar una advertencia si `grado >= N/20` o si la derivada local del unfolding se vuelve negativa.

### 3.2 P(s)

La comparación `Poisson / GUE / GOE / Riemann` es coherente:

- Poisson decae desde `s≈0`, sin repulsión de niveles.
- GUE/GOE/Riemann muestran supresión cerca de `s=0` y pico alrededor de `s≈1`.
- Riemann se aproxima visualmente a GUE, aunque con N≈500 hay ruido muestral.

Riesgo: `P(s)` depende fuertemente del unfolding y del binning. Debe presentarse como evidencia visual, no como diagnóstico principal.

### 3.3 P(s) por método de unfolding

La sección es buena porque muestra sensibilidad metodológica. Para Poisson, los métodos coinciden razonablemente, aunque la curva KDE/Spline/Polinomial mantiene ruido en la cola.

Mejora recomendada: agregar una métrica de distancia entre distribuciones, por ejemplo KS/AD o distancia L² frente a las curvas teóricas, junto con bootstrap por resampling de espaciados.

### 3.4 r-statistic

La sección es robusta porque no requiere unfolding. En la captura:

- Poisson observado: `<r> ≈ 0.3871`, clasificación Poisson.
- GOE observado: `<r> ≈ 0.5323`, clasificación GOE.
- GUE observado: `<r> ≈ 0.6109`, clasificación GUE.
- Riemann observado: `<r> ≈ 0.6191`, clasificación GUE.

Auditoría: es la sección más confiable del dashboard para una primera clasificación, porque elimina el sesgo de unfolding. La desviación Riemann > GUE puede ser ruido de N≈500, efecto de selección de ceros bajos o sensibilidad a ventana.

Mejora recomendada: reportar intervalo de confianza por bootstrap y p-value por simulación Monte Carlo para `|<r>_obs - <r>_GUE|`.

### 3.5 R₂(s)

La captura muestra `χ²/dof Riemann vs GUE ≈ 0.03`, demasiado bueno visualmente para N≈500 si se interpreta literalmente. Puede ser correcto como métrica suavizada, pero debe tratarse con cautela.

Riesgo: una normalización/suavizado muy fuerte puede subestimar la incertidumbre. Recomendación: mostrar bandas Monte Carlo GUE/GOE/Poisson y no solo una curva central.

### 3.6 Σ²(L)

La varianza de número separa bien Poisson de GUE/GOE/Riemann. Riemann queda muy por debajo de la línea Poisson, compatible con rigidez espectral.

Riesgo: para N≈500 y ventanas hasta L≈30, el número de ventanas independientes disminuye al crecer L. Debe agregarse una columna/tooltip con `n_windows(L)`.

### 3.7 K(t)

Error corregido. Después del fix, la sección debe renderizar. La interpretación, sin embargo, necesita cautela: `K(t)` empírico sin ensemble averaging puede verse ruidoso. Si la curva se normaliza por la cola, el punto `t=0` queda fuera de escala y no debe interpretarse como falla del modelo.

---

## 4. Auditoría de arquitectura del repositorio

### Fortalezas

- Separación clara entre `dashboard.py`, módulos `analysis/`, generadores `data/`, motor `engine/`, rigurosidad `rigorous/` y scripts reproducibles.
- `r_statistic` se usa como métrica robusta y no dependiente de unfolding.
- La inclusión de `arb_bridge.py` y `rs_bounds.py` marca una ruta correcta hacia certificación matemática.
- El sistema de documentación es amplio: arquitectura, metodología, validación, changelog y auditorías previas.
- `spectral_form_factor.py` documenta explícitamente dos convenciones: coherente y Mehta.

### Riesgos

- Hay duplicación histórica: `analysis__init__.py` contenía exports más recientes que `analysis/__init__.py`, pero el paquete real cargaba el archivo incompleto. Ya se sincronizó.
- Algunos scripts son exploratorios y pesados; no todos deberían ejecutarse en CI estándar.
- Varias excepciones se capturan de forma amplia (`except Exception`) y pueden ocultar errores de modelo como errores de visualización.
- La versión declarada en cabeceras/documentación no siempre coincide entre módulos.
- `requirements.txt` fija `numpy<2.0`; esto es razonable para entornos Python 3.9–3.12, pero puede ser conflictivo con Python 3.13. Conviene fijar oficialmente Python 3.11/3.12 para despliegue Streamlit.

---

## 5. Auditoría de `arb_bridge.py`

### Lo bueno

`arb_bridge.py` está bien orientado: introduce una capa opcional basada en `python-flint`/Arb para conteo certificado (`zeta_nzeros`), evaluación con ball arithmetic, puntos de Gram y verificación cruzada contra cotas de Riemann–Siegel.

Esto es exactamente el tipo de módulo que puede convertir una exploración numérica en una auditoría reproducible.

### Riesgos

- `python-flint` no está en `requirements.txt`; solo aparece como extra en `setup.py`. Se agregó `requirements-rigorous.txt` para instalarlo explícitamente cuando se quiera usar `--arb`.
- El módulo debe mantener dos niveles separados: “certificado por Arb” y “fallback mpmath”. En reportes no debe mezclarse el lenguaje de certificación si `flint` no está activo.
- Para alturas extremas, convertir `T` a `float` invalida el sentido de alta precisión. Hay rutas en `zeta_altura_extrema.py` que hacen `T_ini = float(cache.T) + dt_inicio`; esto es aceptable solo para diagnóstico grueso, no para certificado.

### Recomendación estructural

Crear una clase `IntervalCertificate` que almacene:

```python
@dataclass
class IntervalCertificate:
    T_anchor: str
    dt_left: str
    dt_right: str
    method: Literal["arb", "mpmath", "phase-only"]
    precision_bits: int
    N_left: Optional[int]
    N_right: Optional[int]
    delta_N: Optional[int]
    z_left_interval: Optional[str]
    z_right_interval: Optional[str]
    sign_change_certified: bool
    tail_bound: Optional[str]
    accepted: bool
    failure_reason: Optional[str]
```

Todo cero aceptado debería apuntar a un certificado de intervalo. Sin eso, los IDs Gödel son trazabilidad operativa, pero no certificación matemática.

---

## 6. Auditoría de `rs_bounds.py`

### Lo bueno

El módulo intenta separar las cotas explícitas de la fórmula de Riemann–Siegel del puente Arb. Esta separación es correcta: `rs_bounds.py` debe ser el “libro de contabilidad analítico” y `arb_bridge.py` el backend de evaluación certificada.

### Riesgos

- Las cotas de cola no deben presentarse como prueba completa de ceros por sí solas; son una pieza de control del error.
- La comparación con radios Arb debe etiquetarse como verificación de consistencia/magnitud, no como sustituto de conteo certificado.
- Hay que evitar que `float` entre en caminos que pretenden ser rigurosos. Para T≈10^70, `float` destruye la información del offset `dt`.

### Recomendación estructural

Separar tres APIs:

1. `rs_parameters(t: DecimalLike | mp.mpf | arb) -> RSParameters`
2. `rs_tail_bound(params, K) -> BoundCertificate`
3. `z_interval_with_tail(t, K, backend="arb") -> ZIntervalCertificate`

Así se evita mezclar “valor central aproximado” con “cota certificada”.

---

## 7. Auditoría de `scripts/zeta_altura_extrema.py`

### Diagnóstico general

El archivo es ambicioso y útil como laboratorio, pero todavía es más heurístico que riguroso. Tiene fases, scoring, IDs Gödel, aliasing checks y tests de estabilidad, pero el núcleo de Fase 1 usa aproximación de fase y el score de aceptación es operacional. Eso no es malo si se etiqueta como exploración; sí sería peligroso si se presenta como verificación/certificación.

### Hallazgos principales

1. **Fase 1 basada en fase:** buena para proponer candidatos, no para afirmar ceros.
2. **Fase 2 con `mpmath.zeta`:** útil en alturas moderadas; en alturas extremas puede ser impracticable o depender de algoritmos internos no certificados intervalarmente.
3. **Fase 3 score:** transparente, pero los pesos son heurísticos. El score debe llamarse `quality_score`, no `confidence` si se quiere evitar lectura probabilística.
4. **Backlund con fallback:** correcto como diagnóstico, pero si no hay Arb no debe marcarse como certificado.
5. **Uso de `float(cache.T)`:** crítico para T extremo. Debe evitarse en todo camino que pretenda conteo fino en `T+dt`.
6. **IDs Gödel:** buenos para auditoría del pipeline, pero no sustituyen un certificado de intervalo.

---

## 8. Estructuras para que `zeta_altura_extrema.py` no sea solo heurístico

### 8.1 Separar el pipeline en cuatro capas

Propuesta:

```text
Layer A — Candidate generation
    Entrada: T_anchor, ventana dt, resolución
    Salida: CandidateInterval[]
    Garantía: ninguna; solo cobertura de muestreo.

Layer B — Analytic enclosure
    Entrada: CandidateInterval
    Salida: ZInterval con cota de cola RS
    Garantía: intervalo numérico con error explícito.

Layer C — Certified counting
    Entrada: [T+dt_left, T+dt_right]
    Salida: N(T2)-N(T1), exact flag
    Garantía: conteo certificado si backend=Arb.

Layer D — Spectral analysis
    Entrada: accepted certified zeros
    Salida: r, P(s), Σ², Δ₃, R₂, K(t)
    Garantía: estadística condicionada a certificados.
```

La capa D nunca debería consumir candidatos de A directamente salvo en modo `--exploratory`.

### 8.2 Tipos de datos recomendados

```python
@dataclass(frozen=True)
class CandidateInterval:
    T_anchor: str
    dt_left: str
    dt_right: str
    source: Literal["phase-grid", "gram", "external"]
    alias_factor: float
    grid_step: str

@dataclass(frozen=True)
class ZInterval:
    midpoint: str
    radius: str
    sign: Literal["positive", "negative", "contains_zero", "unknown"]
    backend: Literal["arb", "rs_tail", "mpmath"]
    precision: int

@dataclass(frozen=True)
class ZeroCertificate:
    zero_id: str
    bracket: CandidateInterval
    z_left: ZInterval
    z_right: ZInterval
    nzeros_delta: int
    method: Literal["arb-zeta_nzeros", "turing", "gram+rs"]
    certified: bool
    notes: list[str]
```

### 8.3 Cambiar aceptación por política explícita

En lugar de un score único, usar política:

```python
class AcceptanceLevel(Enum):
    REJECTED = "rejected"
    EXPLORATORY = "exploratory"
    NUMERICALLY_STABLE = "numerically_stable"
    CERTIFIED_INTERVAL = "certified_interval"
    CERTIFIED_COUNTED = "certified_counted"
```

Así el dashboard puede decir “estos resultados son exploratorios” o “estos resultados tienen conteo certificado”.

### 8.4 Sustituir pesos heurísticos por invariantes verificables

Criterios duros sugeridos:

- `alias_factor >= 4` para no perder ceros por muestreo.
- El intervalo `[dt_left, dt_right]` debe tener signos opuestos certificados o `nzeros_delta == 1`.
- `ZInterval` en ambos extremos no debe contener cero si se usa signo.
- `N(T2)-N(T1) == 1` para certificar un cero aislado.
- La anchura final debe ser menor que una fracción del spacing esperado, por ejemplo `width < 0.05 * zero_spacing`.
- La evaluación debe ser estable al aumentar bits/dps y al subdividir el intervalo.

El score puede quedar como métrica secundaria visual, no como puerta principal.

### 8.5 Registrar una bitácora reproducible

Cada corrida debería exportar `certificates.jsonl`:

```json
{"zero_id":"...","T_anchor":"1e70","dt_left":"...","dt_right":"...","backend":"arb","precision_bits":256,"delta_N":1,"certified":true}
```

Esto permite que otro equipo reproduzca o audite sin depender del estado del dashboard.

### 8.6 Modo recomendado de CLI

```bash
python scripts/zeta_altura_extrema.py --log-T 70 --n-ceros 15 --mode exploratory
python scripts/zeta_altura_extrema.py --log-T 70 --n-ceros 15 --mode certified --arb --arb-prec 512
python scripts/zeta_altura_extrema.py --verify certificates.jsonl --arb --arb-prec 1024
```

---

## 9. Recomendaciones de testing

### Tests mínimos que faltan

1. `test_dashboard_imports_k_t`: verificar que todo símbolo usado en la pestaña `K(t)` esté importado.
2. `test_sff_mehta_smoke`: `spectral_form_factor_mehta(np.arange(20))` devuelve arrays finitos.
3. `test_analysis_exports`: `from riemann_spectral.analysis import spectral_form_factor_mehta` funciona.
4. `test_validate_imports_cache`: `CACHE` se valida como atributo.
5. `test_extreme_no_float_certified_path`: en modo certificado, ninguna ruta convierte `T_anchor` extremo a `float`.
6. `test_certificate_schema_roundtrip`: `ZeroCertificate` serializa/deserializa sin pérdida.

### Separar suites

- `pytest -m smoke`: imports y funciones rápidas.
- `pytest -m statistical`: simulaciones RMT medianas.
- `pytest -m rigorous`: requiere `python-flint` y puede tardar.
- `pytest -m slow`: altura extrema y plots pesados.

---

## 10. Recomendaciones de dashboard

1. Agregar etiquetas de confiabilidad por pestaña: “robusto”, “sensible a unfolding”, “requiere promedio”, “exploratorio”.
2. Agregar bandas Monte Carlo para GUE/GOE/Poisson en R₂, Σ² y K(t).
3. Para `K(t)`, ocultar o anotar `t=0`.
4. Agregar exportación JSON de métricas visibles.
5. En `Unfolding comparado`, avisar sobre sobreajuste cuando el grado polinomial es alto.
6. En `Σ²(L)`, mostrar cuántas ventanas contribuyen a cada L.
7. En `P(s)`, reportar distancia KS/AD frente a Poisson/GUE/GOE.
8. En `r-statistic`, agregar bootstrap de `<r>`.

---

## 11. Verificaciones realizadas en esta auditoría

Se realizaron verificaciones locales sobre el paquete corregido:

- `python -m compileall -q .` pasó sin errores de sintaxis.
- Import smoke test de `riemann_spectral.rigorous.arb_bridge`, `riemann_spectral.rigorous.rs_bounds` y `riemann_spectral.analysis.spectral_form_factor` pasó.
- Export smoke test de `from riemann_spectral.analysis import spectral_form_factor_mehta` pasó después del fix.
- `spectral_form_factor_mehta(...)` devuelve arrays finitos en un test rápido.
- `pytest` completo no terminó dentro del límite de tiempo disponible en el entorno; alcanzó a iniciar y ejecutar parte de `test_delta3.py`. Esto sugiere que la suite contiene tests pesados y debería separarse en marcas `smoke/slow`.
- `validate_imports.py` falló en este contenedor porque `streamlit` no está instalado aquí; además, antes del fix marcaba `CACHE` incorrectamente como “no callable”. Ese falso negativo fue corregido. En un entorno con `requirements.txt` instalado, debería pasar esa parte.

---

## 12. Prioridades de implementación

### Prioridad alta

1. Fix `K(t)` ya aplicado.
2. Sincronizar exports de `analysis/__init__.py` ya aplicado.
3. Separar `CACHE` como atributo en `validate_imports.py` ya aplicado.
4. Agregar tests smoke para imports y K(t).
5. Evitar `float(T)` en caminos certificados de altura extrema.

### Prioridad media

1. Crear `ZeroCertificate` y `certificates.jsonl`.
2. Añadir bandas Monte Carlo al dashboard.
3. Separar tests lentos.
4. Renombrar `confidence_score` a `quality_score` o `operational_score`.

### Prioridad baja

1. Refactor visual del dashboard.
2. Exportación de reportes PDF/HTML.
3. Cache persistente de ensembles y métricas Monte Carlo.

---

## 13. Conclusión

El proyecto es científicamente prometedor como laboratorio RMT/Riemann y el dashboard ya comunica bien las diferencias Poisson/GOE/GUE/Riemann. El error de `K(t)` era un fallo de importación, no un fallo conceptual del módulo de factor de forma. La parte más delicada es `zeta_altura_extrema.py`: debe distinguir claramente entre exploración heurística y certificación. La ruta correcta es pasar de “score de confianza” a “certificados de intervalo + conteo certificado + bitácora reproducible”.
