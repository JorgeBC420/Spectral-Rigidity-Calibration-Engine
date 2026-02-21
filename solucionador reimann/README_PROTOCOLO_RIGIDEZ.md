# Protocolo de Rigidez Espectral

## Visión General

Este módulo implementa un **Protocolo de Validación Metrológica Riguroso** para analizar la estructura dinámica de los ceros de Riemann mediante comparación sistemática con benchmarks estadísticos conocidos.

### Problema Fundamental

El análisis puntual (Experimento 2) muestra que en la configuración inicial, el término singular `4/d_i` domina sobre el término regular `R_i`, sugiriendo una tendencia repulsiva local. **Pero esto NO responde la pregunta crítica:**

> ¿Es esta repulsión específica de los ceros de Riemann, o es un comportamiento universal de cualquier sistema de partículas con interacción logarítmica confinada?

### Solución: Protocolo de Rigidez Espectral

Comparamos tres sistemas de control con propiedades conocidas:

1. **Riemann**: Los ceros reales de la función ?(s)
2. **Uniforme**: Red cristalina perfecta (máxima rigidez posible)
3. **GUE**: Espectro de Gaussian Unitary Ensemble (máxima flexibilidad)
4. **Poisson**: Proceso de Poisson puro (sin correlaciones)

## Marco Teórico

### El Jacobiano como Medida de Estabilidad

Para un sistema dinámico `?? = -?E` (donde E es la energía log-gas):

```
E = -?_{i<j} log|?_i - ?_j|
```

El Jacobiano es el **negativo del Hessiano**:

```
J = -?²E
```

Con elementos:
```
J_kl = 2 / (?_k - ?_l)²   para k ? l
J_kk = -?_{j?k} J_kj      (suma de fila)
```

**Propiedades Fundamentales:**
- **J es simétrica**: Diagonalizable en base ortonormal
- **J ? 0 (semidefinida negativa)**: Por definición, pues es derivada de una energía convexa
- **?? = 0**: Autovalor de traslación (degeneración de Galileo)
- **?? = gap espectral**: Primer autovalor no nulo (modo más blando)

### El Gap Espectral como Observable

El gap espectral `?? = |??|` mide la **velocidad fundamental de relajación** del sistema hacia su equilibrio estadístico.

En sistemas de partículas con repulsión:
- **?? grande**: Sistema "rígido" (resiste perturbaciones)
- **?? pequeño**: Sistema "blando" (fluctúa fácilmente)

#### Escalamiento Universal

Para partículas con repulsión logarítmica en un dominio de tamaño ~N:

```
?? ~ N^(-?)
```

El exponente ? caractériza la clase universal:
- **? = 2**: Hidrodinámica universal de gases
- **? < 2**: Rigidez anómala (sistema "congelado" localmente)
- **? > 2**: Fluidez anómala (sistema "caótico")

### El Modo Blando como Sonda Estructural

El vector propio `v?` asociado a `??` revela la **forma de la fluctuación más probable**:

```
Sinusoidal:     v? ~ sin(?i/N)
                ? Elasticidad continua auténtica
                ? El truncamiento es benigno

Global:         v? distribuido uniformemente
                ? Onda colectiva del bulto
                ? Física genuina de bulk

Localizado:     v? concentrado en bordes
                ? Artefacto de condiciones de contorno
                ? Experimento invalida para ese N
```

## Estructura del Módulo

### `rigidez_espectral.py`

#### 1. Generadores de Benchmarks

```python
generar_uniforme(N)           # Red cristalina: máxima rigidez
generar_gue_normalizado(N)    # GUE: máxima flexibilidad
generar_poisson(N)            # Poisson: sin correlaciones
```

#### 2. Cálculos Espectrales (JIT para velocidad)

```python
@jit(nopython=True)
calcular_jacobiano_kernel(gamma)   # J = -?²E

@jit(nopython=True)
energia_log_gas(gamma)              # E = -? log|?_i - ?_j|

unfolding_riemann(gamma)            # Normalizar densidad local
```

#### 3. Análisis Espectral

```python
analizar_espectro_completo(gamma)
    ? {'gap', 'lambda_0', 'lambda_1', 'v_modo_blando', 'energia', ...}

analizar_modo_blando(resultado)
    ? {'localizacion_index', 'correlacion_sinusoidal', 'ratio_energia_borde', ...}

ajustar_exponente_critico(N_values, gaps)
    ? {'exponente': ?, 'prefactor': C, 'R2': bondad_ajuste}
```

#### 4. Protocolo de Escalamiento

```python
ejecutar_protocolo_escalamiento(N_values, cache_obtener)
    For each N in N_values:
        Para cada sistema (Riemann, Uniforme, GUE, Poisson):
            Calcular Jacobiano
            Extraer gap y modo blando
            Analizar localización y periodicidad
    ? Datos completos de escalamiento
```

## Uso

### Integración en `solucionador_reimann.py`

El programa principal ejecuta automáticamente:

```python
# EXPERIMENTO 3
if RIGIDEZ_DISPONIBLE:
    datos_rigidez = ejecutar_analisis_completo(
        CACHE.obtener,
        N_values=[100, 200, 500, 1000, 2000]
    )
```

### Ejecución Standalone

```python
from rigidez_espectral import ejecutar_analisis_completo
from solucionador_reimann import CACHE

# Ejecutar protocolo
datos = ejecutar_analisis_completo(
    CACHE.obtener,
    N_values=[50, 100, 200, 500, 1000, 2000],
    verbose=True
)

# Acceso a resultados
gap_riemann = [r['gap'] for r in datos['espectra']['riemann']]
modo_blando_riemann = datos['modos_blandos']['riemann']
```

## Interpretación de Resultados

### Panel 1: Gap vs N (Escala Lineal)

Muestra la evolución bruta del gap espectral. Búsqueda visual de discontinuidades o cambios de régimen.

**Señal Importante**: Si Riemann se desvía significativamente de Uniforme/GUE, hay estructura especial.

### Panel 2: Gap vs N (Escala Log-Log)

Ajuste del exponente crítico `?? ~ N^(-?)`.

**Interpretación**:
- Riemann con `? ? 2`: Comportamiento universal (no especial)
- Riemann con `? < 2`: Rigidez anómala (especial)
- Curvatura en log-log: Transición de régimen (muy raro)

### Panel 3: Energía vs N

Evolución de `E = -? log|?_i - ?_j|`.

**Lecturas**:
- E crece lentamente con N: Sistema en equilibrio
- E crece rápidamente: Ceros siendo "comprimidos"
- E diverge: Indicativo de coalescencia (no esperado)

### Panel 4-6: Modo Blando

Gráficos del vector propio normalizado vs posición.

**Criterios**:
1. **Correlación con sin(?i/N)** > 0.8 ? Sinusoidal ? Auténtico
2. **Energía en bordes / energía en centro** < 0.3 ? Genuino ? Válido
3. **Índice de localización** < 0.3 ? Global ? Modo colectivo

### Panel 7: Índice de Localización

Mide qué fracción de la energía del modo está concentrada en los bordes.

**Interpretación**:
- Fluctúa < 0.3: Modos globales auténticos
- Crece > 0.5: Artefacto de borde (invalida N)
- Intermedio: Mezcla de efectos

### Panel 8: Correlación Sinusoidal

Correlación del modo observado con `sin(?i/N)`.

**Interpretación**:
- ? > 0.8: Sistema se comporta como cuerda elástica
- ? < 0.5: Estructura compleja (no continua)
- Cambio abrupto: Transición de régimen físico

### Panel 9: Resumen Estadístico

Diagnóstico en tiempo real de validez del experimento.

## Diagnostics Automáticos

El protocolo incluye validaciones:

1. **Número de condición de J**: Si cond(J) > 10^12, resultados numéricos sospechosos
2. **Desviación de ?? de cero**: Si |??| > 10^(-8), truncamiento deficiente
3. **Proporcionalidad ?? ~ N^(-2)**: Si R² < 0.95 en ajuste, hay efectos no universales

## Interpretación Conceptual

### ¿Qué revelamos?

El protocolo responde preguntas en cascada:

1. **¿Riemann es especial?** ? Comparar ? con Uniforme/GUE
2. **¿Cómo es especial?** ? Analizar modo blando y estructura local
3. **¿Es artefacto?** ? Verificar localización y estabilidad numérica
4. **¿Qué implica?** ? Conectar con teoría de matrices aleatorias

### ¿Qué NO revelamos?

- **NO probamos RH**: El protocolo es condicional al truncamiento
- **NO eliminamos el error de truncamiento**: Solo lo medimos
- **NO predecimos comportamiento dinámico**: Es análisis puntual del Jacobiano
- **NO hacemos conclusiones globales**: Solo observaciones locales a N fijo

## Referencias Teóricas

1. **Dyson's Circular Law**: Espectro de matrices aleatorias
2. **Pólya-Schur Theorem**: Conexión con funciones de frecuencia
3. **Log-Gas Physics**: Modelo Coulomb 2D reducido a 1D
4. **Newman's Deformation**: Interpolación entre ?(s) y versiones "perturbadas"

## Limitaciones Explícitas

1. **Truncamiento**: Los primeros N ceros, no el límite ?
2. **Numeración**: Errores de condicionamiento para N > 5000
3. **Comparación**: GUE no es exactamente el límite termodinámico
4. **Estacionariedad**: Análisis en t=0, no dinámico

## Mejoras Futuras

1. **Comparación con ?-ensemble**: Generalizar GUE
2. **Análisis de tres puntos**: Estadística de nivel superior
3. **Simulación dinámica del Jacobiano**: Integración de flujo de Riemann-Siegel
4. **Correcciones de borde**: Separar bulk de interface effects
