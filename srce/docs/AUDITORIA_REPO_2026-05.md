# Auditoría del repositorio — mayo 2026

**Alcance:** estado del código en `srce/`, tests, documentación, dependencias opcionales y alineación con el mensaje público del proyecto ([Spectral-Rigidity-Calibration-Engine](https://github.com/JorgeBC420/Spectral-Rigidity-Calibration-Engine)).

## 1. Propósito del proyecto (verificado)

El núcleo sigue siendo **calibración y auditoría de rigidez espectral (Δ₃, Dyson–Mehta)** con validación cruzada Poisson–GUE, unfolding estructurado y tests reproducibles. Los scripts sobre ceros de ζ y el módulo `rigorous/` son **capas exploratorias o de certificación numérica auxiliar**; no sustituyen una prueba de la hipótesis de Riemann.

## 2. Estructura relevante

| Ruta | Rol |
|------|-----|
| `srce/src/riemann_spectral/analysis/` | Δ₃, unfolding, normalización, correlación de pares, etc. |
| `srce/src/riemann_spectral/statistics/` | Estadísticos (p. ej. parámetro \(r\)). |
| `srce/src/riemann_spectral/rigorous/` | **Opcional:** puente `python-flint`/Arb (`arb_bridge.py`) y cotas RS de de Reyna (`rs_bounds.py`) para diagnóstico y conteo certificado cuando flint está instalado. |
| `srce/scripts/` | Experimentos reproducibles (RMT, ζ, convergencia Δ₃, altura extrema). |
| `srce/imported/files_zip_2026/` | Referencias importadas; **no** pisan `src/`. |
| `srce/TEST_SUITE_ASSERTIONS.py`, `test_*.py` | Suite pytest (56 tests al momento de esta auditoría). |

## 3. Tests y calidad

- **pytest:** 56 tests pasando desde `srce/` con `PYTHONPATH` apuntando a `srce/src` (convención del proyecto).
- **Advertencia conocida:** `pytest.ini` declara `testpaths = tests` pero los tests viven en la raíz de `srce/`; pytest hace fallback recursivo. Recomendación opcional: ajustar `testpaths` o mover tests a `srce/tests/`.

## 4. Dependencias

- **Obligatorias:** ver `srce/setup.py` (numpy, scipy, mpmath, numba, streamlit, etc.).
- **Opcional rigurosa:** `pip install ".[rigorous]"` o `python-flint` para Arb; sin flint, el bridge usa fallback mpmath donde aplica.

## 5. Scripts de altura / ζ (`zeta_altura_extrema.py`)

- Pipeline en tres fases (candidatos → validación mpmath → aceptación por score).
- Verificaciones añadidas: conteo tipo Backlund (mpmath o `--arb`), puntos de Gram, residual intervalar en Fase 2.
- **Limitación explícita:** Fase 1 por fase aproximada es heurística; el conteo y la validación mitigan pero no garantizan completitud absoluta sin verificación externa (p. ej. ceros tabulados a \(T\) moderado).
- **Windows:** salida UTF-8 reconfigurada al inicio del script para evitar errores de consola.

## 6. Módulo `rigorous/`

- `arb_bridge.py`: conteo certificado vía `zeta_nzeros` cuando flint está disponible; evaluación en bolas; puntos de Gram; cruz opcional con de Reyna (`zeta_ball_crosscheck_reyna`).
- `rs_bounds.py`: cotas \(B_K\) tipo de Reyna (2011) y utilidades de diagnóstico; comparación con radios Arb documentada como **orden de magnitud / auditoría**, no reemplazo del certificado Arb.

## 7. Riesgos y deuda técnica

| Ítem | Severidad | Nota |
|------|-----------|------|
| `pytest.ini` vs ubicación real de tests | Baja | Confunde a CI nuevos; corregir cuando convenga. |
| Dependencia pesada (streamlit, plotly) en install base | Media | Aceptable para dashboard; considerar extras `viz` en el futuro. |
| Scripts ζ a altura extrema | — | Coste mpmath alto; uso documentado como exploratorio. |

## 8. Conclusión

El repositorio **cumple** el rol declarado en GitHub: motor de calibración espectral con RMT y trazabilidad razonable. Las extensiones recientes **refuerzan auditoría y reproducibilidad** sin contradecir el alcance científico. Se recomienda mantener el README raíz y `srce/docs/readme.md` sincronizados con scripts y dependencias opcionales tras cada cambio mayor.
