# METHODOLOGY — Fundamentos Matemáticos del SRCE

**Spectral Rigidity Calibration Engine - Metodología Matemática**

Versión: 2.0.0  
Fecha: Marzo 2026  
Autores: Jorge BC

---

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Generación de Ensembles](#generación-de-ensembles)
3. [Unfolding Espectral](#unfolding-espectral)
4. [Estadísticas Implementadas](#estadísticas-implementadas)
5. [Pipeline de Análisis](#pipeline-de-análisis)
6. [Referencias](#referencias)

---

## Introducción

Este documento describe la **metodología matemática completa** implementada en el Spectral Rigidity Calibration Engine (SRCE), incluyendo todas las fórmulas exactas, procedimientos de validación y referencias académicas.

### Objetivo

Analizar estadísticas espectrales de sistemas cuánticos y conectarlas con predicciones de Random Matrix Theory (RMT).

### Ensembles Implementados

- **Poisson**: Espectros no correlacionados (sistemas integrables)
- **GOE (β=1)**: Gaussian Orthogonal Ensemble (sistemas caóticos con simetría tiempo-reversa)
- **GUE (β=2)**: Gaussian Unitary Ensemble (sistemas caóticos sin simetría tiempo-reversa)

---

## Generación de Ensembles

### 1. Poisson Ensemble

**Método:** Proceso de Poisson con densidad unitaria

```python
spacings = rng.exponential(1.0, size=N)
spectrum = np.cumsum(spacings)
```

**Distribución de spacings:**
```
P(s) = exp(-s)
```

**Propiedades:**
- Sin correlación entre niveles
- Modelo de sistemas integrables
- Referencia para comparación

---

### 2. GOE (Gaussian Orthogonal Ensemble)

**Construcción:** Matriz real simétrica aleatoria

```python
A = rng.standard_normal((N, N))
H = (A + A.T) / (2 * np.sqrt(N))
eigenvalues = eigvalsh(H)
```

**Normalización:**
- Factor `1/(2√N)` produce radio del semicírculo ≈ 2.0
- Compatible con `unfolding_wigner_gue` que usa CDF en [-2, 2]
- Resultado: ⟨s⟩ ≈ 1 después del unfolding

**Distribución de spacings (Wigner surmise):**
```
P(s) = (π/2) s exp(-πs²/4)
```

**Propiedades:**
- Repulsión lineal: P(s) ~ s para s → 0
- Aplicable a sistemas con simetría tiempo-reversa
- β = 1 (índice de Dyson)

**Referencias:**
- Mehta, M.L. (2004). *Random Matrices*, Cap. 1-2
- Haake, F. (2010). *Quantum Signatures of Chaos*, Cap. 3

---

### 3. GUE (Gaussian Unitary Ensemble)

**Construcción:** Matriz compleja hermitiana aleatoria

```python
A_real = rng.standard_normal((N, N))
A_imag = rng.standard_normal((N, N))
A = A_real + 1j * A_imag
H = (A + A.conj().T) / (2 * np.sqrt(N))
eigenvalues = eigvalsh(H)
```

**Normalización:**
- Factor `1/(2√N)` consistente con GOE
- Radio del semicírculo ≈ 2.0
- Compatible con unfolding estándar

**Distribución de spacings (Wigner surmise):**
```
P(s) = (32/π²) s² exp(-4s²/π)
```

**Propiedades:**
- Repulsión cuadrática: P(s) ~ s² para s → 0
- Aplicable a sistemas sin simetría tiempo-reversa
- β = 2 (índice de Dyson)

**Referencias:**
- Mehta, M.L. (2004). *Random Matrices*, Cap. 3
- Forrester, P.J. (2010). *Log-Gases and Random Matrices*

---

## Unfolding Espectral

### Propósito

Transformar el espectro {Eᵢ} a escala "unfolded" {ξᵢ} con densidad promedio = 1.

### Método: Semicírculo de Wigner

**Para GOE/GUE normalizado con radio R = 2:**

```
ξᵢ = N(Eᵢ)
```

donde N(E) es la CDF del semicírculo de Wigner:

```
ρ(E) = (1/2π) √(4 - E²)  para |E| ≤ 2

N(E) = ∫₋₂^E ρ(x) dx
```

**Implementación:**

```python
def unfolding_wigner_gue(eigenvalues):
    """
    Unfolding usando CDF del semicírculo con radio = 2.
    
    Compatible con normalización H = (A + A†)/(2√N)
    que produce eigenvalues en [-2, 2].
    """
    # CDF analítica del semicírculo
    # (implementación en src/riemann_spectral/analysis/unfolding.py)
    ...
```

**Verificación:**

Después del unfolding, el spacing medio debe ser:
```
⟨s⟩ = ⟨ξᵢ₊₁ - ξᵢ⟩ ≈ 1
```

**Referencias:**
- Wigner, E.P. (1955). *Characteristic Vectors of Bordered Matrices*
- Mehta, M.L. (2004). *Random Matrices*, Sec. 6.3

---

## Estadísticas Implementadas

### 1. Spacing Distribution P(s)

**Definición:**

Distribución de spacings normalizados:
```
s = (ξᵢ₊₁ - ξᵢ) / ⟨spacing⟩
```

**Predicciones teóricas:**

| Ensemble | P(s) | Comportamiento s→0 |
|----------|------|-------------------|
| **Poisson** | exp(-s) | P(0) = 1 |
| **GOE** | (π/2)s exp(-πs²/4) | P(s) ~ s |
| **GUE** | (32/π²)s² exp(-4s²/π) | P(s) ~ s² |

**Implementación:**

```python
spacings = np.diff(spectrum)
s_normalized = spacings / np.mean(spacings)
hist, bins = np.histogram(s_normalized, bins=50, density=True)
```

**Referencias:**
- Mehta, M.L. (2004). *Random Matrices*, Cap. 15
- Guhr et al. (1998). *Random-matrix theories in quantum physics*

---

### 2. r-Parameter

**Definición:**

```
rᵢ = min(sᵢ, sᵢ₊₁) / max(sᵢ, sᵢ₊₁)

⟨r⟩ = (1/N) ∑ᵢ rᵢ
```

**Valores exactos (Atas et al., 2013):**

| Ensemble | ⟨r⟩ | Expresión analítica |
|----------|-----|---------------------|
| **Poisson** | 0.38629436... | 2ln(2) - 1 |
| **GOE** | 0.53589838... | 4 - 2√3 |
| **GUE** | 0.60272166... | Constante numérica |

**Propiedades:**
- ✅ Independiente del unfolding
- ✅ Robusto a errores de normalización
- ✅ Clasificador directo de ensembles

**Implementación:**

```python
spacings = np.diff(spectrum)
r_vals = np.minimum(spacings[:-1], spacings[1:]) / \
         np.maximum(spacings[:-1], spacings[1:])
r_mean = np.mean(r_vals)
```

**Referencias:**
- Oganesyan & Huse (2007). Phys. Rev. B 75, 155111
- Atas et al. (2013). Phys. Rev. Lett. 110, 084101

---

### 3. Pair Correlation g(r)

**Definición:**

Función de correlación de pares:
```
g(r) = ⟨∑ᵢ≠ⱼ δ(r - |Eᵢ - Eⱼ|)⟩ / ρ²ₚₒᵢₛₛₒₙ
```

**Predicción GUE (Montgomery-Odlyzko):**

```
g(r) = 1 - [sin(πr)/(πr)]²
```

**Propiedades:**
- Para r → 0: g(r) ~ r² (repulsión cuadrática)
- Para r → ∞: g(r) → 1 (sin correlación)
- Oscilaciones decrecientes

**Implementación:**

```python
# Calcular todas las distancias
distances = []
for i in range(N):
    for j in range(i+1, N):
        distances.append(spectrum[j] - spectrum[i])

# Histograma normalizado
g_r = histogram(distances) / (N(N-1)/2)
```

**Referencias:**
- Montgomery, H.L. (1973). *Pair correlation of zeros*
- Odlyzko, A.M. (1987). *On the distribution of spacings*

---

### 4. Number Variance Σ²(L)

**Definición:**

Varianza del número de niveles en intervalos de longitud L:

```
N(x, L) = #{eigenvalues en [x, x+L]}

Σ²(L) = ⟨(N(x,L) - ⟨N(x,L)⟩)²⟩
```

**Nota crítica sobre el estimador:**

```python
# CORRECTO:
mean_N = np.mean(n_L)
sigma2 = np.mean((n_L - mean_N)**2)

# INCORRECTO (solo válido para N→∞):
sigma2 = np.mean((n_L - L)**2)
```

**Predicciones teóricas:**

| Ensemble | Σ²(L) para L grande |
|----------|---------------------|
| **Poisson** | L |
| **GOE** | (2/π²)[log(2πL) + γ + 1 - π²/8] |
| **GUE** | (1/π²)[log(2πL) + γ + 1] |

donde γ ≈ 0.5772... es la constante de Euler-Mascheroni.

**Orden universal:**
```
Σ²ₚₒᵢₛₛₒₙ(L) > Σ²ₐₒₑ(L) > Σ²ₐᵤₑ(L)  para todo L
```

**Implementación:**

```python
def sigma2_number_variance(spectrum, L):
    n_windows = len(spectrum) - int(L)
    counts = []
    
    for i in range(n_windows):
        left = spectrum[i]
        right = left + L
        count = np.sum((spectrum >= left) & (spectrum < right))
        counts.append(count)
    
    mean_N = np.mean(counts)  # ← CRÍTICO
    return np.mean((counts - mean_N)**2)
```

**Referencias:**
- Dyson, F.J. & Mehta, M.L. (1963). *Statistical theory of energy levels*
- Mehta, M.L. (2004). *Random Matrices*, Sec. 17.2

---

### 5. Dyson-Mehta Δ₃ Statistic

**Definición:**

Rigidez espectral - mide desviación de la escalera espectral de una recta:

```
N(E) = número acumulado de eigenvalues ≤ E

Δ₃(L) = (1/L) min_{A,B} ∫₀^L [N(E) - AE - B]² dE
```

**Solución analítica (Mehta, 2004):**

Minimizando sobre A, B:

```
Δ₃(L) = (1/L³) [I₃ - 2AI₁ - 2BI₂ + A²L³ + ABL⁴ + B²L⁵/3]
```

donde:
```
I₁ = ∫₀^L E·N(E) dE
I₂ = ∫₀^L N(E) dE  
I₃ = ∫₀^L N(E)² dE
```

**Predicciones asintóticas:**

| Ensemble | Δ₃(L) para L grande |
|----------|---------------------|
| **Poisson** | L/15 |
| **GOE** | (1/2π²) log L + constante |
| **GUE** | (1/π²) log L + constante |

**Coeficientes exactos:**
```
ΔGOE ≈ 0.05066 log L
ΔGUE ≈ 0.10132 log L
```

**Implementación:**

Ver `src/riemann_spectral/analysis/rigidity.py` para implementación completa con optimización Numba.

**Referencias:**
- Dyson, F.J. & Mehta, M.L. (1963). *Statistical theory of the energy levels*
- Mehta, M.L. (2004). *Random Matrices*, Sec. 17.3

---

### 6. Spectral Form Factor K(τ)

**Definición:**

```
K(τ) = |∑ⱼ exp(iτEⱼ)|² / N
```

**Regímenes universales (GUE):**

| Régimen | τ | K(τ) | Significado |
|---------|---|------|-------------|
| **Dip** | τ ≪ 1 | τ² | Repulsión cuadrática |
| **Ramp** | 1 < τ < N | τ | Rigidez espectral |
| **Plateau** | τ ≫ N | 1 | Saturación (N finito) |

**Implementación:**

```python
@njit
def spectral_form_factor(spectrum, tau_grid):
    K = np.zeros(len(tau_grid))
    N = len(spectrum)
    
    for i, tau in enumerate(tau_grid):
        z_sum = np.sum(np.exp(1j * tau * spectrum))
        K[i] = np.abs(z_sum)**2 / N
    
    return K
```

**Referencias:**
- Haake, F. (2010). *Quantum Signatures of Chaos*, Cap. 4
- Cotler et al. (2017). *Black holes and random matrices*

---

## Pipeline de Análisis

### Flujo Completo

```
1. Generar ensemble (GOE/GUE/Poisson)
   ↓
2. Calcular eigenvalues
   ↓
3. Unfolding (Wigner semicircle)
   ↓
4. Extraer bulk (percentiles 10-90)
   ↓
5. Normalizar spacing (salvaguarda)
   ↓
6. Calcular estadísticas:
   - P(s)
   - ⟨r⟩
   - g(r)
   - Σ²(L)
   - Δ₃(L)
   - K(τ)
   ↓
7. Clasificar ensemble
   ↓
8. Comparar con teoría
```

### Normalización Post-Unfolding

**Salvaguarda implementada:**

```python
def normalize_spacing(spectrum):
    """
    Fuerza ⟨s⟩ = 1 después del unfolding.
    
    Esto corrige pequeñas desviaciones causadas por:
    - Efectos de borde
    - Truncamiento del bulk
    - Errores numéricos en el unfolding
    """
    spectrum = np.sort(spectrum)
    s_mean = np.mean(np.diff(spectrum))
    return spectrum / s_mean
```

**Aplicación:**

```python
# Después del unfolding
unfolded = unfolding_wigner_gue(eigenvalues)

# Extraer bulk (evitar bordes)
n = len(unfolded)
bulk = unfolded[n//3 : 2*n//3]

# Forzar normalización
normalized = normalize_spacing(bulk)

# Verificar
assert abs(np.mean(np.diff(normalized)) - 1.0) < 1e-10
```

---

## Validación Numérica

### Criterios de Aceptación

| Estadística | Error aceptable |
|-------------|-----------------|
| P(s) vs teoría | < 15% (error L1) |
| ⟨r⟩ vs exacto | < 3% |
| Δ₃ slope vs teórico | < 10% |
| Σ² ordering | Poisson > GOE > GUE |

### Tamaños Recomendados

| Objetivo | N mínimo |
|----------|----------|
| Validación básica | 1000 |
| Producción | 2000 |
| Convergencia | 5000 |

### Convergencia Estadística

Error escala como:
```
Error ~ 1/√N
```

**Verificación:**

```python
N_vals = [500, 1000, 2000, 5000]
errors = [compute_error(N) for N in N_vals]

# Debe cumplir
assert np.allclose(errors * np.sqrt(N_vals), constant, rtol=0.2)
```

---

## Referencias

### Textos Fundamentales

1. **Mehta, M.L.** (2004). *Random Matrices* (3rd ed.). Academic Press.
   - Referencia definitiva en RMT

2. **Haake, F.** (2010). *Quantum Signatures of Chaos* (3rd ed.). Springer.
   - Aplicaciones a quantum chaos

3. **Forrester, P.J.** (2010). *Log-Gases and Random Matrices*. Princeton.
   - Teoría moderna de ensembles

### Papers Clave

4. **Dyson, F.J. & Mehta, M.L.** (1963). *Statistical Theory of the Energy Levels of Complex Systems*. J. Math. Phys. 4, 701-712.

5. **Montgomery, H.L.** (1973). *The pair correlation of zeros of the zeta function*. Proc. Sympos. Pure Math. 24, 181-193.

6. **Oganesyan & Huse** (2007). *Localization of interacting fermions at high temperature*. Phys. Rev. B 75, 155111.

7. **Atas et al.** (2013). *Distribution of the Ratio of Consecutive Level Spacings*. Phys. Rev. Lett. 110, 084101.

### Implementaciones

8. **Odlyzko, A.M.** (1987). *On the distribution of spacings between zeros of the zeta function*. Mathematics of Computation 48(177), 273-308.

---

## Apéndice: Constantes Numéricas

```python
# Euler-Mascheroni
EULER_GAMMA = 0.5772156649015329

# r-parameter exactos
R_POISSON = 2 * np.log(2) - 1          # 0.38629436...
R_GOE = 4 - 2 * np.sqrt(3)             # 0.53589838...
R_GUE = 0.60272166211556               # valor numérico estándar

# Δ₃ coeficientes
DELTA3_GOE_COEFF = 1 / (2 * np.pi**2)  # 0.05066059...
DELTA3_GUE_COEFF = 1 / np.pi**2        # 0.10132118...
```

---

**Última actualización:** Marzo 2026  
**Versión:** 2.0.0  
**Mantenedor:** Jorge BC  
**Licencia:** MIT
