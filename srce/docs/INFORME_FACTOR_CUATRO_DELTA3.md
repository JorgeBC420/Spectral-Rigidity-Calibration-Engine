# Origen del factor 4 y corrección estructural (Delta₃)

## Resumen

- **Parche 0.25 eliminado**: no se usa ningún factor empírico en `_delta3_recta`.
- **Test Poisson**: espectro Poisson con densidad 1 da **Δ₃(L) ≈ L/15** (ratio observado/teórico ≈ 1.06). La integral y la convención (1/L)·min ∫(N−A−Bx)² dx son correctas.
- **Conclusión sobre el factor 4**: El factor 4 que se compensaba con 0.25 **no** venía de un error en la integral ni en el ancho L de la ventana. Venía del **unfolding de GUE**: usar el “tercio central” con posiciones 0, 1, 2, … (rango) **elimina las fluctuaciones locales** del espectro real y no es un unfolding por densidad. Con unfolding correcto (CDF del semicírculo de Wigner), la pendiente en L sigue siendo mayor que 1/π² en tests con N moderado; no se ha introducido ningún factor de calibración.

---

## 1. Auditoría analítica de la integral

### Definición implementada

- **Ventana**: [y₀, y₀+L] en coordenadas unfolded (longitud **L**).
- **Escalera**: N(x) = número de niveles en la ventana con posición **≤ x**, con salto **1** en cada nivel.
- **Valor por ventana**: (1/L) · min_{A,B} ∫₀^L (N(x) − A − Bx)² dx.

### Momentos I₁, I₂, I₃

Con posiciones en la ventana 0 ≤ x₁ ≤ … ≤ xₖ ≤ L:

- **I₁ = ∫ N dx** = k·L − ∑ⱼ xⱼ  
- **I₂ = ∫ x·N dx** = (1/2)(k·L² − ∑ⱼ xⱼ²)  
- **I₃ = ∫ N² dx** = k²·L − ∑ⱼ (2j−1)·xⱼ  

(Índice j = 1,…,k; en código 0-based: (2(j+1)−1)·xs[j].)  
Comprobado con un caso explícito (k=2, x₁=L/3, x₂=2L/3) que I₃ coincide con el valor manual. No hay factor 2 erróneo en L ni paso de conteo distinto de 1.

### Minimización y valor de la integral

- A y B por ecuaciones normales: A·L + B·L²/2 = I₁, A·L²/2 + B·L³/3 = I₂.
- Valor mínimo: ∫(N−A−Bx)² = I₃ − 2A·I₁ − 2B·I₂ + A²·L + A·B·L² + B²·L³/3.
- Se usa **fastmath=False** en `_delta3_recta` para no alterar el redondeo.

No se ha encontrado error algebraico que justifique un factor 4 en la fórmula de la integral.

---

## 2. Test de Poisson (obligatorio)

- **Generador**: posiciones = cumsum(Exp(1)) hasta una longitud dada → proceso de Poisson de densidad 1 (espacio unfolded por construcción).
- **Sin factor 0.25**: se usa directamente (1/L)·integral.
- **Resultado**: Δ₃(L)/L observado ≈ 0.0705, teórico 1/15 ≈ 0.0667, ratio ≈ **1.06**.
- **Interpretación**: Si el resultado hubiera sido **4·(L/15)**, el error sería una constante de integración global (factor 4 en la fórmula de la varianza/integral). Como el resultado es **≈ L/15**, la implementación de la integral y la normalización por L son correctas; el error que se compensaba con 0.25 **no** está en la integral ni en la ventana, sino en el **proceso de unfolding de GUE**.

---

## 3. Refactorización del unfolding (GUE)

- **Antes**: `unfolding_tercio_central` asignaba al tercio central las posiciones 0, 1, …, M−1 (rango). Eso **no** usa la densidad local ni los valores reales de los autovalores y **pierde** las fluctuaciones que caracterizan al GUE.
- **Ahora**: 
  - **Unfolding por densidad teórica**: en `unfolding.py` está `unfolding_wigner_gue(evals_raw, sigma)`. Usa la **CDF del semicírculo de Wigner** F(x) = ½ + (1/(4π))(x√(4−x²) + 4·arcsin(x/2)) para autovalores en [-2,2], y define uᵢ = N·F(eᵢ) (espaciado medio 1).
  - En `validate_delta3.py`: `generar_gue_raw(N)` devuelve autovalores **sin** reescalado posterior; `unfolding_wigner(evals_raw)` aplica esa CDF; `tercio_central(u)` toma el tercio central y re-centra a 0 para evitar bordes.
- Así se usan los **valores reales** de los autovalores escalados por la densidad teórica y se preservan las fluctuaciones locales en el bulk.

---

## 4. Eliminación del parche y estado actual

- **Eliminado**: `NORMALIZACION_DELTA3 = 0.25` y cualquier multiplicación por ese factor en `_delta3_recta`.
- **Código**: solo `sum_d3 += integral / L` (sin factor empírico).
- **Poisson**: converge a **L/15** en el test.
- **GUE**: con unfolding Wigner y tercio central, la pendiente en L (ajuste a·log L + b) en tests con N=800–2000 sigue siendo mayor que 1/π² (del orden de ~3× en nuestras corridas). No se ha introducido ningún factor de calibración; posibles causas que quedan abiertas:
  - Efectos de **tamaño finito** (N no suficientemente grande).
  - Diferencia de **convención** en la literatura (p. ej. definición con 1/(4L) en algún texto) que no hemos impuesto en código.

---

## 5. Explicación física/matemática del factor 4

- **Por qué parecía necesario el 0.25**: Con el “unfolding” por rango (0, 1, 2, …) en el tercio central, el espectro que ve Δ₃ es **casi determinista** (espaciado casi constante). Eso **subestima** las fluctuaciones respecto al ajuste lineal y hace que Δ₃(L) sea **menor** de lo que debería para un GUE real. Al comparar con la teoría (1/π²)·ln L, la pendiente observada con ese pseudo-unfolding era **mayor** que 1/π², y se “corregía” con un factor 0.25. Es decir, el factor 4 era una **compensación** a un unfolding incorrecto, no a un error en la integral.
- **Corrección estructural**:
  1. **Integral**: se mantiene (1/L)·min ∫(N−A−Bx)² dx, con N(x) de salto 1 y ventana de longitud L; sin factor adicional.
  2. **Poisson**: confirma que esa convención da **L/15**.
  3. **GUE**: se deja de usar el tercio central por rango y se usa **unfolding por CDF del semicírculo** sobre autovalores raw, y luego tercio central en espacio unfolded. No se añade ningún factor empírico; si en el futuro se identifica una convención 1/(4L) en la literatura, la corrección debería documentarse como tal convención, no como “parche”.

---

## Archivos modificados

- **`src/riemann_spectral/analysis/rigidity.py`**: eliminado el parche 0.25; docstring actualizado; sin `NORMALIZACION_DELTA3`.
- **`src/riemann_spectral/analysis/unfolding.py`**: añadido `unfolding_wigner_gue(evals_raw, sigma)`; comentario en `unfolding_tercio_central`.
- **`validate_delta3.py`**: generador Poisson densidad 1, GUE raw, unfolding Wigner, tercio central; tests Poisson y GUE sin factor empírico.
