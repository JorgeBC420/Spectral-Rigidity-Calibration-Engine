# Apuntes de sesión — importación desde Downloads / `files.zip`

**Fecha:** abril 2026  

## Archivos integrados

### Scripts (`srce/scripts/`)

| Archivo | Origen | Notas |
|---------|--------|--------|
| `zeta_altura_extrema.py` | Downloads | Exploración Z de Riemann-Siegel en alturas extremas; requiere `mpmath`; importa SRCE (`delta3_dyson_mehta`, `unfolding_riemann`, etc.). |
| `riemann_empirical.py` | Downloads | Validación empírica RMT sobre ceros de ζ; salidas en `scripts/output/`. |
| `delta3_convergence.py` | Downloads | Convergencia α(N) para Δ₃ ~ α log L (GUE/GOE/Poisson/Riemann). |

### Referencia sin sobrescribir el núcleo

- **`srce/imported/files_zip_2026/`** — copia de `files.zip`: `ensemble_classifier.py`, `rigidity.py`, `r_statistic.py`, `unfolding.py`.  
  **Motivo:** no reemplazar el árbol auditado en `src/riemann_spectral/` sin revisión explícita.

### `pair_correlation.py` (Downloads)

- Comparación binaria con `src/riemann_spectral/analysis/pair_correlation.py`: **sin diferencias**.  
- No se duplicó en el repo.

## Tests

- Tras cambios, ejecutar desde `srce/` con el entorno habitual:  
  `python -m pytest -q`  
  (si hace falta: `PYTHONPATH=src` o equivalente en Windows).

## Recordatorios

- Los nuevos scripts escriben en **`scripts/output/`** (ya en `.gitignore` donde aplica).
- `zeta_altura_extrema.py` es pesado y depende de rutas `scripts/` → `src/`; ejecutar desde la raíz del paquete `srce/` o revisar argumentos CLI del propio script.
