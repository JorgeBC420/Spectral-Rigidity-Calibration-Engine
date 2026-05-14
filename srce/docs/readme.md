# Documentación SRCE

Índice de la documentación del **Spectral Rigidity Calibration Engine** (carpeta `srce/docs/`).

| Documento | Contenido |
|-----------|-----------|
| **THEORY.md** | Régimen de rigidez en ventanas finitas vs límites asintóticos; referencias `PENDIENTE_*` |
| **VALIDATION_RMT.md** | Resultados de referencia del script `scripts/rmt_validation.py` |
| **VALIDATION.md** | Resultados de tests y verificación teórica (suite pytest) |
| **METHODOLOGY.md** | Fundamentos metodológicos (desde zip de integración) |
| **ARCHITECTURE.md** | Diseño del sistema (inglés) |
| **ARQUITECTURA.md** | Arquitectura (español, proyecto) |
| **INDICE_COMPLETO.md** | Índice navegable del material |
| **QUICKSTART.md** | Inicio rápido |
| **CHANGELOG.md** | Historial de cambios |
| **AUDITORIA_REPO_2026-05.md** | Auditoría de estado del repo, tests y módulo `rigorous/` |
| Otros `*.md` | Auditorías, integración, guías de ferias, certificación, etc. |

---

## Paquete `riemann_spectral.rigorous` (opcional)

- **`arb_bridge.py`** — Conteo certificado con `python-flint`/Arb cuando está instalado; fallback controlado con mpmath. Ver `extras_require["rigorous"]` en `setup.py`.
- **`rs_bounds.py`** — Cotas del remainder RS (Arias de Reyna, 2011) y utilidades de diagnóstico frente a radios Arb (auditoría de magnitud).

---

**Scripts relacionados**

- `scripts/rmt_validation.py` — validación RMT numérica
- `scripts/reproduce_figures.py` — figuras de referencia
- `scripts/riemann_delta3_analysis.py` — ceros de ζ (exploratorio; Δ₃ simplificado)
- `scripts/riemann_delta3_slope_analysis.py` — ceros de ζ con `delta3_dyson_mehta` + pendiente local
- `scripts/zeta_altura_extrema.py` — Riemann-Siegel en alturas extremas (exploratorio; flags `--solo-candidatos`, `--arb`, `--arb-prec`, `--multi-altura`)
- `scripts/riemann_empirical.py` — RMT sobre ceros reales de ζ
- `scripts/delta3_convergence.py` — convergencia α(N) para Δ₃

**Ejecución de tests:** desde `srce/`, definir `PYTHONPATH=.../srce/src` y ejecutar `python -m pytest -q`.

**Importaciones de referencia (no sustituyen `src/`):** ver `imported/files_zip_2026/README.md`.

**Apuntes de sesión:** `APUNTES_SESION_IMPORT_2026.md`
