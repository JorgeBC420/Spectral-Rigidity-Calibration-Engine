# SRCE Platform — hoja de ruta

El núcleo matemático permanece en `src/riemann_spectral/` (`analysis/`, `engine/`, `rigorous/`, `statistics/`).

Esta carpeta documenta la evolución hacia una plataforma modular **sin romper** el dashboard actual.

## Implementado (2026-05)

| Capa | Ubicación | Estado |
|------|-----------|--------|
| Certificación | `certification/` | `AcceptanceLevel`, `ZeroCertificate`, `certificates.jsonl`, alturas mpf |
| IO externo | `io/external_loader.py` | CSV, TSV, TXT, JSON, XLSX |
| Analytics | `analytics/rmt_pipeline.py` | Unfolding + r + Δ₃ + clasificador + disclaimer |
| Tests smoke | `test_certification_smoke.py`, `test_platform_io.py` | pytest |
| Dashboard | `tab` Dataset externo | Carga + auditoría básica |

## Pendiente (prioridad auditoría)

1. Bandas Monte Carlo GUE/GOE/Poisson en plots (requiere pipeline certificado estable).
2. Export PDF con metodología automática.
3. Cache persistente Streamlit multi-sesión.
4. Refactor visual completo `ui/` / `plots/` (no urgente).
5. `IntervalCertificate` formal end-to-end en Arb para todo `zeta_altura_extrema`.
6. Integración APIs astronómicas / cosmológicas.

## Mapa objetivo (futuro)

```
platform/
  ui/          → wrappers Streamlit (delegan a dashboard.py)
  core/        → orquestación de sesiones
  analytics/   → ✅ rmt_pipeline.py
  io/          → ✅ external_loader (paquete riemann_spectral.io)
  validation/  → tests + disclaimers
  exports/     → PDF/CSV/JSON (stub)
  models/      → referencias GUE/GOE/Poisson (usa engine + statistics)
```
