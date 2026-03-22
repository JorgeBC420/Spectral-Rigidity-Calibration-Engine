# Integración de archivos desde los ZIP de Descargas

Revisión de **`files.zip`**, **`files1.zip`** y **`file2s.zip`** (rutas típicas `Downloads`).

## Resumen

| ZIP | Contenido | Estado en SRCE |
|-----|-----------|-----------------|
| **file2s.zip** | `normalize.py`, `number_variance.py`, `CORRECCIONES_APLICADAS.md` | **Parcialmente integrado.** Las correcciones clave (Σ² con `mean_N`, ordenar espectro en `normalize`) están aplicadas en `analysis/`. La versión “solo `spectrum/s_mean` sin restar” **no** se copió tal cual: se mantiene `(x-x₀)/⟨s⟩` tras ordenar, equivalente en espaciados al flujo SRCE + `conftest`. |
| **files1.zip** | `test_theoretical_validation.py`, `METHODOLOGY.md`, `normalize.py`, `number_variance.py` | **Tests:** añadido `test_theoretical_validation.py` en `srce/` con imports corregidos (`generar_*` del proyecto), `R_GUE_EXACT` numérico (la fórmula `(27/4)-6√3` del zip es incorrecta). **Docs:** no copiados por defecto; fusionar `METHODOLOGY.md` manualmente si hace falta. |
| **files.zip** | `THEORY.md`, `VALIDATION.md`, `ARCHITECTURE.md`, `explicit_formula.py`, etc. | **No fusionado automáticamente.** El repo ya tiene `srce/THEORY.md` y documentación propia; conviene comparar a mano antes de sobrescribir. |

## Dónde quedó documentada la auditoría

- Copia de referencia: **`CORRECCIONES_APLICADAS.md`** puede guardarse en el repo si se desea (no se añadió en esta pasada para no duplicar `THEORY.md` / `CERTIFICACION.md`).
- **`rigidity.py`**: el zip sugiere cambiar el filtro en `_delta3_recta`; eso **no** se tocó (política previa: no modificar núcleo sin revisión aparte).

## Dashboard

Pestañas nuevas en **Validación RMT**:

- **⟨r⟩ Atas (modular)** — `statistics.compute_r_parameter` / `classify_ensemble_by_r`.
- **K(τ) coherente** — `spectral_form_factor_coherent.spectral_form_factor`.

## Comandos

```bash
cd srce
set PYTHONPATH=src
pytest test_delta3.py test_theoretical_validation.py -v
```
