# Validación RMT — SRCE

Resultados de una corrida de referencia del script **`scripts/rmt_validation.py`** (solo lectura del código en `src/`), sin modificar el núcleo.

**Reproducibilidad:** `RNG_SEED = 20250323`, `N = 2000`, **10 realizaciones** por ensemble donde aplica, **L ∈ [5, 50]** con **25 puntos**, `pytest` y dependencias según `requirements.txt`.

---

## Tabla resumen (métricas obligatorias)

| Métrica | Poisson | GOE | GUE | Estado |
|---------|---------|-----|-----|--------|
| Δ₃ vs L/15 (error rel. medio) | 0.0180 | — | — | OK (&lt; 10 %) |
| dΔ₃/d(log L) OLS *a* | — | 0.0972 | 0.0500 | GUE ∈ [0.045, 0.055]: OK |
| Media ∂Δ₃/∂(log L) (gradiente) | — | 0.0965 | 0.0504 | — |
| ⟨Δ₃⟩ GOE &lt; ⟨Δ₃⟩ GUE (media global) | — | no | — | revisar en esta muestra |
| Σ²: tendencia | dΣ²/dL ≈ 0.941 | — | dΣ²/d log L ≈ 0.1244 | Poisson ~ lineal; GUE ~ log |
| P(s) KS | 0.0119 | 0.0219 | 0.0388 | *p* altos Poisson/GOE |
| P(s) L² | 0.0288 | 0.0614 | 0.0623 | — |
| ⟨*r*⟩ (media 10 real.) | 0.3856 | 0.5312 | 0.5974 | |
| \|⟨*r*⟩ − teórico\| | 0.0007 | 0.0047 | 0.0053 | OK |
| SFF (dip cualitativo) | sí | sí | sí | |

**Diagnóstico automático del script:** *CONSISTENTE con RMT (régimen finito)* (5/6 checks heurísticos).

---

## Régimen finito

En ventanas **acotadas** de *L* (p. ej. [5, 50]) y con **espectros finitos** (matrices *N* × *N*, bulk truncado, unfolding numérico), las estadísticas **no** coinciden con los coeficientes asintóticos de RMT en *L → ∞*.

En particular, el **ajuste OLS** de Δ₃(*L*) frente a log *L* devuelve una **pendiente efectiva** del orden **0.05** para GUE en el pipeline SRCE, coherente con la calibración documentada en `THEORY.md` (`PENDIENTE_GUE_REFERENCIA` ≈ 0.05), **no** con el uso directo de 1/π² ≈ 0.101 como pendiente de un ajuste log en esa ventana.

---

## Δ₃ y el coeficiente 1/π²

En el límite **L → ∞** (tras unfolding a densidad 1), se cita con frecuencia

\[
\Delta_3(L) \sim \frac{1}{\pi^2}\log L + C + o(1) \quad \text{(GUE)}.
\]

En **L finito** no se debe **identificar** la curva observada con la recta (1/π²) log *L* punto a punto: hay constantes *C*, correcciones de borde y efectos de discretización. Por tanto:

- **No** es un fallo del núcleo que la pendiente OLS en [5, 50] sea ≈ 0.05 y no ≈ 0.101.
- La **derivada local** dΔ₃/d(log *L*) puede fluctuar a lo largo de la malla de *L*; lo relevante para comparar con el motor es el **comportamiento conjunto** en el mismo protocolo que el clasificador (GUE/GOE/Poisson).

---

## Cómo re-ejecutar

```bash
cd srce
python -u scripts/rmt_validation.py
```

---

## Referencias en el repo

- Marco teórico (ventanas finitas vs asintótico): `docs/THEORY.md`
- Tests de validación teórica: `test_theoretical_validation.py` (raíz `srce/`)

---

*Última actualización: alineada con el script `scripts/rmt_validation.py`.*
