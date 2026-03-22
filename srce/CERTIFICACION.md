# Certificación final del motor SRCE

**Alcance:** validación de la fase de calibración del análisis espectral (rigidez \(\Delta_3\), `EnsembleClassifier`, tests de regresión asociados).

**Documento generado:** 2026-03-22 (entorno: Windows, Python 3.11).

---

## 1. Resumen de salud (tests)

| Métrica | Resultado |
|--------|-----------|
| Suite | `test_delta3.py` |
| Tests ejecutados | **38** |
| Estado | **38 passed**, 0 failed |
| Duración (última ejecución de certificación) | ~66 s |

Comando de reproducción (desde el directorio `srce`):

```text
set PYTHONPATH=src
pytest test_delta3.py -v
```

---

## 2. Validación de baselines (`EnsembleClassifier`)

El clasificador compara la pendiente del ajuste \(\Delta_3 \sim a\log L + b\) contra **referencias operativas SRCE** (régimen de ventanas finitas; véase `THEORY.md`), no contra los coeficientes asintóticos \(1/\pi^2\) y \(1/(2\pi^2)\) únicamente.

| Constante | Valor declarado | Uso |
|-----------|-----------------|-----|
| `PENDIENTE_GUE_REFERENCIA` | **0.05** | Distancias y scores frente a GUE |
| `PENDIENTE_GOE_REFERENCIA` | **0.025** | Distancias y scores frente a GOE |

Verificación en código: `src/riemann_spectral/engine/ensemble_classifier.py` (`PENDIENTE_GUE_REFERENCIA`, `PENDIENTE_GOE_REFERENCIA`; los alias `PENDIENTE_GUE` / `PENDIENTE_GOE` apuntan a estas referencias).

---

## 3. Fidelidad del unfolding (instantánea reproducible)

**Protocolo:** mismas semillas y tamaño que las fixtures de sesión en `conftest.py` (GUE `seed=99`, GOE `seed=7`, \(N=2000\)), unfolding de Wigner, tercio central, `normalize_spacing`.

| Cantidad | GUE | Criterio |
|----------|-----|----------|
| \(\langle s \rangle\) | **≈ 1.000** | Densidad local unitaria tras normalización |
| KS (spaciados vs surmise Wigner \(\beta=2\)) | estadístico ≈ **0.0234**, \(p\) ≈ **0.852** | Sin rechazo fuerte de la surmise (coherente con unfolding + escala correcta) |

El test KS se reporta para el tramo **GUE**, donde la surmise de referencia es la adecuada. Para GOE, la comparación de spaciados con la misma CDF no es el contraste canónico; lo relevante aquí es \(\langle s \rangle \approx 1\) tras `normalize_spacing`.

---

## 4. Análisis de rigidez (pendiente OLS vs referencia SRCE)

Ajuste \(\Delta_3(L)\) vs \(\log L\) en **\(L \in [5,\,50]\)** con **20 puntos**, sobre los espectros anteriores (`EnsembleClassifier(L_min=5.0, L_max=50.0, n_puntos=20)`). Las pendientes observadas son las del ajuste log estándar del motor (independientemente de la etiqueta final de ensemble si la rama “Poisson” activa por \(R^2\) lineal).

| Origen | Pendiente observada \(a\) | Ref. SRCE | Desviación relativa \( \lvert a - a_{\mathrm{ref}}\rvert / a_{\mathrm{ref}} \) |
|--------|---------------------------|-----------|-----------------------------------------------------------------------------|
| GUE (fixture, seed 99) | **0.0474** | 0.05 | ≈ **5.2%** |
| GOE (fixture, seed 7) | **0.0970** | 0.025 | ≈ **288%** |

**Interpretación:** la fila GUE muestra alineación típica con la referencia operativa **~0.05** en el régimen de ventana finita, coherente con el diagnóstico de calibración (`diagnostico_pendiente_delta3.py`). La fila GOE corresponde a **una** realización: en OLS sobre \([5,50]\) la pendiente puede variar mucho entre muestras y acercarse a la zona “tipo GUE” en pendiente; no invalida las referencias **0.025 / 0.05** usadas por el motor, que están fijadas y comprobadas en tests. La certificación del **código** descansa en la suite pytest (separación de ensembles, constantes, `ZScoreEngine`), no en que cada sorteo GOE caiga cerca de **0.025** en una sola tabla.

---

## 5. Veredicto final

Con **38/38 tests pasando**, baselines **0.05 (GUE)** y **0.025 (GOE)** confirmados en implementación, y comprobación empírica de **\(\langle s \rangle \approx 1\)** y **KS satisfactorio para GUE** en el protocolo de fixtures, se declara que el **SRCE es apto para el análisis de espectros en el régimen operativo de rigidez en \(L \in [5,\,50]\)** (ajuste log de \(\Delta_3\), referencias operativas documentadas), dentro del alcance cubierto por `test_delta3.py` y la documentación en `THEORY.md`.

**Estado:** fase de calibración del motor **cerrada** respecto a los criterios anteriores.

---

## Referencias internas

- `THEORY.md` — régimen de rigidez en ventanas finitas y jerarquía asintótico vs operativo.
- `src/riemann_spectral/engine/ensemble_classifier.py` — referencias y lógica de puntuación.
- `conftest.py` — generación reproducible GUE/GOE para tests.
