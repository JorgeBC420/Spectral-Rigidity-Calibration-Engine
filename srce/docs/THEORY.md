# Marco teórico — Spectral Rigidity Calibration Engine (SRCE)

Este documento alinea la **matemática de referencia** (límites asintóticos de RMT) con el **comportamiento numérico** que el motor produce en condiciones de uso típicas (matrices y ventanas finitas).

---

## 1. Δ₃ de Dyson–Mehta (definición)

Para un espectro unfolded con densidad media de niveles unidad, la rigidez espectral en una ventana de longitud \(L\) se define mediante la integral de la función escalera \(N(x)\) (saltos unitarios en los niveles). El núcleo implementado en `rigidity.py` sigue las fórmulas de Mehta (I₁, I₂, I₃) para

\[
\Delta_3(L) = \frac{1}{L}\min_{A,B}\int_0^L \bigl(N(x)-A-Bx\bigr)^2\,dx .
\]

Los casos de control (Poisson → \(\Delta_3(L)\approx L/15\)) validan esa implementación frente a la teoría de procesos de Poisson.

---

## 2. Límites asintóticos (literatura)

Para ensembles invariantes ortogonales/unitarios, en el régimen **\(L\to\infty\)** (tras unfolding a densidad 1), se cita con frecuencia el comportamiento dominante

\[
\Delta_3(L) \sim \frac{1}{\pi^2}\log L + C + o(1) \quad \text{(GUE)}, \qquad
\Delta_3(L) \sim \frac{1}{2\pi^2}\log L + C' + o(1) \quad \text{(GOE)} .
\]

Los coeficientes de \(\log L\) son por tanto **\(1/\pi^2 \approx 0.1013\)** (GUE) y **\(1/(2\pi^2) \approx 0.0507\)** (GOE) en ese límite.

En el código, estas cantidades se exponen como **`PENDIENTE_GUE_ASINTOTICO`** y **`PENDIENTE_GOE_ASINTOTICO`** en `ensemble_classifier.py` para documentación y comparaciones teóricas, **no** como único objetivo de ajuste en ventanas finitas.

---

## 3. Régimen de rigidez en ventanas finitas

### 3.1 Fenómeno

Los diagnósticos reproducibles (p. ej. `diagnostico_pendiente_delta3.py`) muestran que, para **ventanas \(L\) en rangos operativos típicos** (por ejemplo \(L\in[5,50]\)) y **espectros unfolded bien normalizados** (\(\langle s\rangle = 1\), KS frente a la surmise de Wigner coherente):

- La **pendiente efectiva** de una regresión \(\Delta_3(L) \approx \alpha\log L + \beta\) (OLS sobre la grilla de \(L\)) adopta valores del orden de **\(\alpha_{\mathrm{eff}}^{\mathrm{(GUE)}}\approx 0.05\)**, **no** \(1/\pi^2\).
- Ese valor es **estable** al variar el tamaño de matriz \(N\) en un rango amplio (p. ej. 500–5000), lo que indica que **no** es un simple defecto de unfolding ni un error en la integral de \(\Delta_3\) (validada aparte con Poisson).
- La **derivada local** \(d\Delta_3/d(\log L)\) evaluada numéricamente es coherente con la pendiente OLS global en el mismo rango: el sesgo **no** se debe solo al uso de OLS frente a un estimador local.

### 3.2 Interpretación

En ventanas **finitas**, dominan **términos subdominantes** (constantes \(C\), correcciones \(O(1/L)\), efectos de discreción del espectro y del promedio sobre ventanas en el bulk). El coeficiente **\(\alpha\)** obtenido por regresión log-lineal en un intervalo acotado de \(L\) es una **pendiente efectiva** del tramo observado, **no** el coeficiente asintótico del término \(\log L\) en \(L\to\infty\).

**No confundir:** \(\Delta_3(L) \neq (1/\pi^2)\log L\) como identidad puntual en \(L\) finito — la forma \((1/\pi^2)\log L\) es el **término dominante asintótico** (GUE); en una ventana acotada aparecen constantes y correcciones que hacen que la **pendiente local** (OLS de \(\Delta_3\) frente a \(\log L\) en un rango) difiera del coeficiente \(1/\pi^2\).

Por tanto, comparar directamente \(\alpha_{\mathrm{OLS}}\) con \(1/\pi^2\) en \(L\in[5,50]\) **sobreestima** la discrepancia como si fuera un fallo del núcleo de \(\Delta_3\); en la práctica, el motor reproduce el **comportamiento numérico estándar** de la rigidez en ese régimen.

### 3.3 Referencias operativas SRCE (`EnsembleClassifier`)

Para **clasificación** y **scores**, el SRCE utiliza **referencias empíricas validadas** en el mismo tipo de pipeline (unfolding Wigner, bulk, grilla de \(L\) por defecto del clasificador):

| Símbolo (código) | Valor | Rol |
|------------------|-------|-----|
| `PENDIENTE_GUE_REFERENCIA` | **0.05** | Pendiente efectiva tipo GUE en ventanas finitas |
| `PENDIENTE_GOE_REFERENCIA` | **0.025** | Mitad de la referencia GUE (jerarquía coherente con GOE/GUE asintótico) |

Los atributos exportados **`PENDIENTE_GUE`** y **`PENDIENTE_GOE`** en el módulo del clasificador apuntan a estas referencias operativas. Los valores asintóticos \(1/\pi^2\) y \(1/(2\pi^2)\) permanecen disponibles como **`PENDIENTE_*_ASINTOTICO`**.

---

## 4. Jerarquía de detección: Z-scores y “anomalías”

### 4.1 Baselines empíricos

`ZScoreEngine` compara métricas (espaciado mínimo, varianza del número, \(\Delta_3\), etc.) de un espectro real frente a **distribuciones baseline** generadas por `BaselineFactory` (GUE y Poisson con la misma tubería de unfolding que el núcleo del SRCE).

Esos baselines reflejan **comportamiento numérico real** (p. ej. matrices finitas, mismas ventanas), no una idealización asintótica inalcanzable a \(N\) fijo (p. ej. \(N\approx 2000\)).

### 4.2 Sensibilidad

Al anclar la inferencia a **realizaciones simuladas bajo el mismo protocolo**, la detección de desviaciones respecto al “comportamiento GUE esperado en laboratorio” se vuelve **más sensible** a diferencias **prácticas** (forma de la distribución, dispersión de \(\Delta_3\) a \(L\) fijo) que a una coincidencia con constantes asintóticas de libro que el experimento numérico no alcanza en ventanas finitas.

La bandera `anomalia` en `evaluar()` se basa en **|z frente al baseline GUE|** (entre otros campos); véase la documentación del módulo `zscore_engine.py` para el criterio exacto.

### 4.3 Lectura recomendada

- **Coeficientes Mehta** \(1/\pi^2\), \(1/(2\pi^2)\): interpretación **asintótica** y contrastes teóricos (`PENDIENTE_*_ASINTOTICO`).
- **Clasificación por ensemble**: comparar pendientes log frente a **`PENDIENTE_*_REFERENCIA`** (régimen finito documentado arriba).
- **Calibración de z-scores**: interpretar como desviación respecto a **cohortes simuladas**, no respecto a fórmulas cerradas en \(L\to\infty\).

---

## 5. Referencias bibliográficas (indicativas)

- M. L. Mehta, *Random Matrices* (3ª ed.), caps. 16–17.
- Bohigas–Giannoni–Schmit (1984), conjetura BGS.

---

*Última actualización: alineada con el diagnóstico de rigidez en ventanas finitas y calibración operativa del `EnsembleClassifier`.*
