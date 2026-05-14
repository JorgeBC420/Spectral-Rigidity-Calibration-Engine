# Notas de entrega — 2026-05-11

Resumen de cambios integrados en esta línea de trabajo (ver commits en `main`).

## Documentación y auditoría

- README raíz reescrito: inicio rápido, enlaces al repo GitHub, tabla de docs, alcance científico explícito.
- `srce/docs/AUDITORIA_REPO_2026-05.md`: auditoría de estructura, tests, `rigorous/`, scripts ζ y deuda técnica menor.
- `srce/docs/readme.md`: índice actualizado con `rigorous/`, flags de `zeta_altura_extrema.py` y recordatorio de `PYTHONPATH` para pytest.

## Código y paquetes

- `riemann_spectral/rigorous/`: `arb_bridge.py`, `rs_bounds.py` (de Reyna 2011), integración opcional con `python-flint`.
- `scripts/zeta_altura_extrema.py`: pipeline v2.2.x con verificaciones Backlund/Gram, UTF-8 en Windows, opción `--arb`.
- `setup.py`: extra `rigorous` para `python-flint`.
- Ajustes en tests/assertions, scripts empíricos y material importado bajo `imported/files_zip_2026/` (referencia, no núcleo).

## Verificación

- Suite pytest: 56 tests en verde (ejecutar desde `srce/` con `PYTHONPATH=srce/src`).
