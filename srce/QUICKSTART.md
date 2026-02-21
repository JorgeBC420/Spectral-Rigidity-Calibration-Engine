# QUICK START GUIDE

## ? 5 Minutos para Empezar

### 1?? Verificar Instalación (30 segundos)

```bash
cd "solucionador reimann"
python TEST_SUITE.py
```

**Resultado esperado**: ? LISTO PARA USAR

Si hay errores, instalar dependencias:
```bash
pip install numpy scipy numba matplotlib mpmath
```

---

### 2?? Ejecutar Análisis (5-30 minutos)

```bash
python solucionador_reimann.py
```

El programa generará:
- `analisis_espaciado_vs_N.png` (espaciado mínimo)
- `rigidez_espectral_protocolo.png` (comparación benchmarks)
- `evolucion_dmin.png` (evolución temporal)

---

### 3?? Interpretar Resultados

#### Panel 1: Espaciado Mínimo
- Eje X: N (número de ceros)
- Eje Y: d_min (espaciado más pequeño)
- **Línea roja**: Predicción teórica ~ log(N)/N
- **Línea azul**: Valores observados

#### Panel 2: Rigidez Espectral
- Eje X: N (escala log)
- Eje Y: ?? (gap espectral, escala log)
- **Si Riemann < Uniforme < GUE**: Riemann es más rígido

#### Panel 3: Modo Blando
- Muestra el vector propio del modo más flexible
- **Forma sinusoidal**: Elasticidad continua auténtica
- **Distribuido**: Modo colectivo genuino
- **En bordes**: Posible artefacto de truncamiento

---

## ?? Entender los Resultados

### Pregunta 1: ¿Son los ceros de Riemann especiales?

**Mira**: Panel de "Gap vs N (Log-Log)"
- Si Riemann sigue otra ley de escalamiento que Uniforme/GUE
  ? Posible estructura especial
- Si todos siguen ?? ~ N^(-2)
  ? Comportamiento universal

### Pregunta 2: ¿Es el sistema estable?

**Mira**: Panel de "Modo Blando"
- Si es sinusoidal
  ? Sistema se comporta como cuerda elástica (auténtico)
- Si está localizado en bordes
  ? Artefacto de truncamiento (N debe aumentarse)

### Pregunta 3: ¿Qué tan comprimidos están los ceros?

**Mira**: Panel de "Energía vs N"
- E ? lentamente
  ? Equilibrio estadístico (esperado)
- E ? rápidamente
  ? Ceros siendo comprimidos (posible anomalía)

---

## ?? Qué Significan los Colores

| Color | Significado |
|-------|-------------|
| ?? Azul | Riemann (ceros reales) |
| ?? Púrpura | Uniforme (red cristalina) |
| ?? Verde | GUE (matrices aleatorias) |
| ?? Naranja | Poisson (sin correlaciones) |

---

## ?? Si Algo Falla

### Error: "No module named numpy"
```bash
pip install numpy scipy numba matplotlib mpmath
```

### Error: "Cache corrupted"
```bash
# Borrar y regenerar
rm cache_ceros_riemann.pkl
python solucionador_reimann.py
```

### Ejecución muy lenta
- Problema: N muy grande
- Solución: Editar `N_VALUES = [100, 200]` en solucionador_reimann.py

### Gráficos no se generan
- Problema: Matplotlib no funciona
- Solución: Usar `export MPLBACKEND=Agg` (Linux/Mac) o cambiar backend en código

---

## ?? Siguiente Paso: Aprender Teoría

Leer en este orden:

1. **`README_PROTOCOLO_RIGIDEZ.md`** (30 min)
   - Entender Jacobiano
   - Entender gap espectral
   - Entender benchmarks

2. **`RESUMEN_IMPLEMENTACION.md`** (15 min)
   - Entender observables
   - Entender limitaciones

3. **`AUDITORIA_SEGURIDAD.md`** (10 min)
   - Entender garantías técnicas

---

## ?? Uso Avanzado

### Personalizar Experimento

```python
from solucionador_reimann import CACHE
from rigidez_espectral import ejecutar_analisis_completo

# Protocolo personalizado
datos = ejecutar_analisis_completo(
    CACHE.obtener,
    N_values=[50, 100, 200],  # ? Tus N
    verbose=True
)

# Acceder resultados
gaps_riemann = [r['gap'] for r in datos['espectra']['riemann']]
print(f"Gaps observados: {gaps_riemann}")
```

### Reproducir Exactamente

```python
from rigidez_espectral import generar_gue_normalizado

# Con seed = resultado determinista
gue1 = generar_gue_normalizado(100, seed=42)
gue2 = generar_gue_normalizado(100, seed=42)
assert np.allclose(gue1, gue2)  # ? Idénticos
```

---

## ?? Ayuda Rápida

```
¿Qué es ???
  ? Gap espectral (modo más blando)
  
¿Qué es "modo blando"?
  ? Vector propio del autovalor |?_1|
  
¿Por qué comparar con GUE?
  ? Benchmark de máxima aleatoriedad
  
¿Por qué N ? [100, 2000]?
  ? Rango donde análisis es estable
  
¿Qué significa "bien acondicionado"?
  ? cond(J) < 1e12 (matriz numéricamente estable)
```

---

## ? Validación Rápida

```python
# Verificar que todo funciona
from rigidez_espectral import (
    generar_uniforme,
    generar_gue_normalizado,
    calcular_jacobiano_kernel
)
import numpy as np

# Test 1: Uniforme
u = generar_uniforme(50)
assert len(u) == 50 and np.all(np.isfinite(u))
print("? Uniforme OK")

# Test 2: GUE
g = generar_gue_normalizado(50)
assert len(g) == 50 and np.all(np.isfinite(g))
print("? GUE OK")

# Test 3: Jacobiano
J = calcular_jacobiano_kernel(u)
assert J.shape == (50, 50) and np.all(np.isfinite(J))
print("? Jacobiano OK")

print("\n??? SISTEMA FUNCIONA CORRECTAMENTE")
```

---

## ?? Objetivos de Este Proyecto

| Objetivo | ¿Logrado? |
|----------|-----------|
| Analizar estructura de ceros de Riemann | ? |
| Comparar con benchmarks | ? |
| Medir rigidez espectral | ? |
| Código robusto y seguro | ? |
| Documentación completa | ? |
| Fácil de usar | ? |

---

## ?? Próximos Pasos (Opcional)

1. **Experimentar con N larger**
   ```python
   N_VALUES = [500, 1000, 2000, 3000, 5000]
   ```

2. **Analizar densidad local**
   - Usar unfolding diferencial
   - Comparar con semicírculo de Wigner

3. **Estudiar bifurcaciones**
   - Buscar cambios de régimen en escalamiento
   - Medir transiciones de fase

4. **Integración dinámica**
   - Estudiar flujo de calor de Newman
   - Analizar coalescencia de ceros

---

## ?? Recursos

- **Wikipedia**: "Riemann hypothesis"
- **Paper**: "Random Matrix Theory and its Applications"
- **Book**: "Numerical Recipes"

---

**¡Listo! Ahora ejecuta `python solucionador_reimann.py` y disfruta explorando.**

