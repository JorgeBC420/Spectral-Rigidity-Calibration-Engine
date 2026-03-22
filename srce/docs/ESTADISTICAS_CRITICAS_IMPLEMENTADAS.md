# Estadísticas críticas implementadas

**Versión:** 2.1.0 (integración en el árbol SRCE)  
**Autor original del paquete zip:** Jorge BC & Claude

---

## Resumen

Se integraron los archivos de `new test.zip` **sin romper** el dashboard ni la API existente de `pair_correlation.py` / `spectral_form_factor.py`:

| Archivo zip | Ubicación en SRCE |
|-------------|-------------------|
| `r_statistic.py` | `src/riemann_spectral/statistics/r_statistic.py` |
| `pair_correlation.py` (teorías + histograma Numba) | Añadidos a `analysis/pair_correlation.py` (aliases + `pair_correlation_histogram_numba`) |
| `spectral_form_factor.py` (K(τ) dip–ramp–plateau) | `src/riemann_spectral/analysis/spectral_form_factor_coherent.py` (convive con `spectral_form_factor.py`) |
| `ESTADISTICAS_CRITICAS_IMPLEMENTADAS.md` | Este documento (`srce/ESTADISTICAS_CRITICAS_IMPLEMENTADAS.md`) |

---

## Imports recomendados

```python
# r-parameter (Atas exactos Poisson/GOE; GUE como constante de alta precisión)
from riemann_spectral.statistics import (
    compute_r_parameter,
    classify_ensemble_by_r,
    R_GUE_EXACT,
    R_GOE_EXACT,
    R_POISSON_EXACT,
)

# g(r) — API histórica SRCE + aliases Montgomery
from riemann_spectral.analysis.pair_correlation import (
    pair_correlation_fast,
    r2_teorica_gue,
    pair_correlation_gue,  # alias de r2_teorica_gue
    pair_correlation_histogram_numba,
)

# K(τ) coherente |∑ exp(iτEⱼ)|²/N
from riemann_spectral.analysis.spectral_form_factor_coherent import (
    spectral_form_factor,
    extract_ramp_slope,
)

# K(t) del dashboard (Fourier / Mehta) — sin cambios
from riemann_spectral.analysis.spectral_form_factor import (
    spectral_form_factor as spectral_form_factor_mehta,
    r_statistic,
)
```

---

## Notas técnicas

1. **`statistics.r_statistic`**: el zip declaraba una fórmula cerrada errónea para ⟨r⟩ GUE (`27/4 - 6√3` es negativa). Se sustituyó `R_GUE_EXACT` por el valor estándar **0.60272166211556** (literatura / Atas).

2. **`pair_correlation`**: el módulo SRCE ya define `pair_correlation()` y `pair_correlation_fast()`. El histograma del zip se expone como **`pair_correlation_histogram_numba`** para evitar colisión de nombres.

3. **`spectral_form_factor`**: existen **dos** convenciones:
   - `spectral_form_factor.py` — K(t) usado en el dashboard (transformada de R₂, normalización cola).
   - `spectral_form_factor_coherent.py` — K(τ) = |∑ exp(iτEⱼ)|²/N (literatura quantum chaos).

4. Paquete **`riemann_spectral.statistics`** está disponible vía `from riemann_spectral import statistics`.

---

## Auto-tests (opcional)

Desde `srce` con `PYTHONPATH=src`:

```bash
python src/riemann_spectral/statistics/r_statistic.py
python src/riemann_spectral/analysis/spectral_form_factor_coherent.py
python src/riemann_spectral/analysis/pair_correlation.py
```

---

## Checklist de integración

- [x] `statistics/` creado con `__init__.py`
- [x] `r_statistic.py` integrado; constante GUE corregida
- [x] `pair_correlation` ampliado (aliases + Numba)
- [x] `spectral_form_factor_coherent.py` añadido
- [x] `riemann_spectral.__init__` exporta `statistics`
- [ ] Integrar al dashboard (opcional)
