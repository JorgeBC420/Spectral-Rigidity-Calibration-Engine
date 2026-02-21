# 🔬 Riemann Spectral Analysis Framework (Project: Riemann Spec analysis)

Este framework es un laboratorio de experimentación numérica diseñado para el análisis de la **Hipótesis de Riemann (RH)** y la **Rigidez Espectral**. El sistema modela los ceros no triviales de la función Zeta como un **Gas Logarítmico (log-gas)** en equilibrio térmico, comparando su estabilidad y estructura con el **Gaussian Unitary Ensemble (GUE)**.

## 🏗️ Arquitectura del Sistema

El proyecto se divide en 4 dimensiones operativas:

### 1. Universo A: Adquisición y Unfolding
- **Data Provider:** Ingesta de ceros desde `mpmath` (baja altura) o datasets externos (alta altura).
- **Unfolding Engine:** Normalización de ceros mediante la función de conteo $N(T) \approx \frac{T}{2\pi} \log \frac{T}{2\pi e}$ para obtener un espaciado medio ($\Delta = 1$).

### 2. Universo B: Análisis de Rigidez (Hessiano)
- **Hessian Engine:** Cálculo del Jacobiano del potencial de interacción.
- **Spectral Gap:** Medición del autovalor mínimo ($\lambda_{min}$) para determinar la estabilidad estructural frente a perturbaciones.

### 3. Universo C: Motor de Baselines (BaselineFactory)
- Generación de matrices aleatorias **GUE** y **GOE**.
- Simulación de **Procesos de Poisson** para detectar transiciones entre orden y caos.

### 4. Universo D: Detector de Anomalías
- **Z-Score Engine:** Detección de desviaciones de alta confianza ($5\sigma$).
- **Bitácora:** Registro persistente en SQLite para el seguimiento de hallazgos a diferentes alturas críticas.

## 🚀 Optimizaciones Hardware (Intel i7-1255U)
- **Paralelismo:** Implementación de `Numba` con `@njit(parallel=True)` para distribuir el cálculo de fuerzas inter-partícula en los 10 núcleos del CPU.
- **Ahorro de Memoria:** Uso de `subset_by_index` en álgebra lineal para evitar la diagonalización completa de matrices densas.

## 🧪 Objetivos Experimentales
1. **Validación de la Conjetura de Montgomery:** Confirmar si la estadística de pares se mantiene en alturas extremas.
2. **Búsqueda de Puntos de Ruptura:** Identificar bloques de ceros con anomalías en la rigidez que sugieran fallos en la línea crítica.
3. **Fortalecimiento de CounterCore:** Utilizar la estructura determinista de los ceros como base para algoritmos de ciberseguridad y validación de primos.

---
*Nota: Este proyecto es una herramienta científica. La resolución de problemas del milenio requiere verificación analítica adicional.*