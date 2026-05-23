# Certificación, niveles de aceptación y bitácora JSONL

SRCE distingue **rigor estadístico** de **certificación formal** en alturas extremas.

## `AcceptanceLevel`

| Valor | Uso |
|--------|-----|
| `exploratorio` | Fase aproximada, score bajo, o sin Arb |
| `semi_riguroso` | Z refinada con mpmath; Backlund mpmath |
| `certificado` | Arb/python-flint con `es_exacto` y score ≥ umbral |
| `diagnostico` | Solo candidato Fase 1 / sin aceptación |

El dashboard y `zeta_altura_extrema.py` deben mostrar esta etiqueta junto al score operativo.

## Alturas sin `float(T)`

Para `log_T > 12` (`FLOAT_SAFE_LOG_T`), **no** usar `float(cache.T)` en caminos que pretendan certificar.
Usar `window_im_mpf`, `t_im_from_offset` e `im_str_for_backend` (`riemann_spectral.certification.heights`).

## `certificates.jsonl`

Cada línea es un JSON (certificado o `run_summary`). Generado por defecto en:

`scripts/output/certificates.jsonl`

CLI:

```bash
python scripts/zeta_altura_extrema.py --log-T 3 --n-ceros 8 --cert-log output/mi_bitacora.jsonl
```

## Datasets externos

`riemann_spectral.io.external_loader` y `analytics.rmt_pipeline.run_rmt_audit` — pestaña **Dataset externo** en el dashboard.

Pendiente (roadmap): PDF, bandas Monte Carlo en UI, refactor `ui/` completo.
