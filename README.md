# Spectral Rigidity Calibration Engine

Motor de **análisis y calibración de rigidez espectral** (estadística Δ₃ de Dyson–Mehta, unfolding, validación cruzada tipo RMT) para procesos Poisson, ensambles GUE y secuencias deterministas (incluido uso exploratorio con ceros de ζ).

**Repositorio:** [github.com/JorgeBC420/Spectral-Rigidity-Calibration-Engine](https://github.com/JorgeBC420/Spectral-Rigidity-Calibration-Engine)

El proyecto **no** pretende demostrar la hipótesis de Riemann ni sustituir verificación analítica formal. Prioriza consistencia matemática, tests reproducibles y documentación del pipeline numérico.

## Inicio rápido

```bash
cd srce
pip install -e .
set PYTHONPATH=%CD%\src
python -m pytest -q
```

En Linux o macOS:

```bash
cd srce
pip install -e .
export PYTHONPATH="$(pwd)/src"
python -m pytest -q
```

**Opcional (evaluación / conteo con Arb vía python-flint):**

```bash
pip install -e ".[rigorous]"
```

## Contenido principal

- **`srce/src/riemann_spectral/`** — Núcleo: `analysis/` (rigidez Δ₃, unfolding, normalización), `statistics/`, `engine/`, `data/`, etc.
- **`srce/src/riemann_spectral/rigorous/`** — Puente opcional a **python-flint** (Arb): conteo de ceros certificado cuando está instalado; `rs_bounds.py` con cotas tipo de Reyna (2011) para diagnóstico cruzado (no reemplaza a Arb).
- **`srce/scripts/`** — Experimentos y utilidades (`rmt_validation.py`, `riemann_empirical.py`, `zeta_altura_extrema.py`, análisis Δ₃ y pendientes, etc.).
- **`srce/docs/`** — Índice en [srce/docs/readme.md](srce/docs/readme.md): teoría, validación RMT, arquitectura, auditorías, notas de integración.

## Documentación destacada

| Documento | Contenido |
|-----------|-----------|
| [srce/docs/THEORY.md](srce/docs/THEORY.md) | Régimen de rigidez en ventanas finitas y referencias teóricas |
| [srce/docs/VALIDATION_RMT.md](srce/docs/VALIDATION_RMT.md) | Validación numérica RMT |
| [srce/docs/QUICKSTART.md](srce/docs/QUICKSTART.md) | Guía de arranque |
| [srce/docs/AUDITORIA_REPO_2026-05.md](srce/docs/AUDITORIA_REPO_2026-05.md) | Auditoría de estado del repo (mayo 2026) |

## Scripts sobre ζ (exploratorio)

`srce/scripts/zeta_altura_extrema.py` implementa detección por fase, validación con Z exacta (mpmath), puntuación y comprobaciones tipo Backlund / Gram. Flags útiles: `--solo-candidatos`, `--arb`, `--arb-prec`, `--multi-altura`. Las salidas por defecto van a `srce/scripts/output/`.

## Licencia y contribución

Ver el repositorio en GitHub para licencia e historial de commits. Para cambios sustantivos en métricas o unfolding, actualizar tests y la documentación en `srce/docs/` en el mismo cambio.
