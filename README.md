🔬 Spectral Rigidity Calibration Engine

(formerly Riemann Spectral Analysis Framework)

Este proyecto es un motor de análisis de rigidez espectral (Δ₃ de Dyson–Mehta) diseñado para la validación numérica de estadística espectral en distintos modelos:

Procesos de Poisson (desorden total)

Ensambles aleatorios tipo GUE

Secuencias deterministas como los ceros no triviales de la función zeta

El framework nació como una herramienta exploratoria orientada al estudio numérico de la Hipótesis de Riemann.
Durante su desarrollo evolucionó hacia algo más fundamental y metodológicamente sólido:

Un entorno de calibración y auditoría de métricas espectrales basado en Random Matrix Theory (RMT).

🧠 Enfoque Científico Actual

El proyecto no intenta “resolver” la Hipótesis de Riemann.

Su objetivo es más preciso:

Implementar correctamente la estadística Δ₃(L).

Validar su normalización mediante casos control (Poisson).

Calibrar el unfolding para GUE usando la CDF exacta del semicírculo de Wigner.

Proveer una infraestructura reproducible para comparar espectros.

La prioridad es consistencia matemática, no resultados espectaculares.

📘 Marco teórico y calibración

- **[srce/THEORY.md](srce/THEORY.md)** — Régimen de rigidez en **ventanas finitas** vs límites asintóticos de Mehta; referencias operativas del `EnsembleClassifier` (\(\alpha_{\mathrm{eff}}\) para GUE/GOE); jerarquía de **Z-scores** frente a baselines empíricos (detección sensible al comportamiento numérico real, no solo a idealizaciones \(L\to\infty\)).

🏗️ Arquitectura del Sistema
1️⃣ Núcleo de Rigidez (analysis/rigidity.py)

Implementación directa de:

Δ
3
(
𝐿
)
=
1
𝐿
min
⁡
𝐴
,
𝐵
∫
𝑥
0
𝑥
0
+
𝐿
(
𝑁
(
𝑥
)
−
𝐴
−
𝐵
𝑥
)
2
𝑑
𝑥
Δ
3
	​

(L)=
L
1
	​

A,B
min
	​

∫
x
0
	​

x
0
	​

+L
	​

(N(x)−A−Bx)
2
dx

Características:

N(x) como función escalera con saltos unitarios.

Ventanas reales [yᵢ, yᵢ + L].

Sin factores empíricos.

Auditoría algebraica completa (ver INFORME_FACTOR_CUATRO_DELTA3.md).

Validación Poisson → Δ₃(L) ≈ L/15.

No hay normalizaciones ocultas.

2️⃣ Unfolding Engine (analysis/unfolding.py)

Para GUE:

Uso de la CDF exacta del semicírculo de Wigner:

𝐹
(
𝑥
)
=
1
2
+
1
4
𝜋
(
𝑥
4
−
𝑥
2
+
4
arcsin
⁡
(
𝑥
/
2
)
)
F(x)=
2
1
	​

+
4π
1
	​

(x
4−x
2
	​

+4arcsin(x/2))

Transformación:

𝑢
𝑖
=
𝑁
⋅
𝐹
(
𝑒
𝑖
)
u
i
	​

=N⋅F(e
i
	​

)

Corte del tercio central en espacio unfolded para evitar efectos de borde.

Se eliminó el unfolding por rango, que producía distorsiones estructurales.

3️⃣ Baselines (Poisson y GUE)

Poisson densidad 1:
posiciones = cumsum(Exp(1))
Resultado consistente con teoría:

Δ
3
(
𝐿
)
≈
𝐿
15
Δ
3
	​

(L)≈
15
L
	​


GUE raw → unfolding Wigner → Δ₃(L)
Pendiente comparada con:

1
𝜋
2
log
⁡
𝐿
π
2
1
	​

logL

El sistema permite estudiar convergencia en tamaño finito.

4️⃣ Validación y Diagnóstico

Incluye:

Tests sin factores de calibración artificial.

Eliminación documentada del parche 0.25.

Comparación estructural entre Poisson y GUE.

Registro reproducible de experimentos.

📊 Qué es hoy el proyecto

Formalmente es:

Un framework de calibración de rigidez espectral basado en Random Matrix Theory con validación cruzada Poisson–GUE.

No es un “Riemann solver”.
No es una prueba numérica de RH.
No es un sistema criptográfico.

Es instrumentación matemática.

Y eso tiene valor real.

🚧 Qué NO hace

No prueba la Hipótesis de Riemann.

No detecta “rupturas” de la línea crítica.

No reemplaza verificación analítica.

No garantiza aplicaciones criptográficas.

🔎 Valor Científico

El valor actual del proyecto está en:

Implementación auditada de Δ₃.

Corrección estructural del unfolding.

Validación contra modelos teóricos conocidos.

Infraestructura reproducible para experimentación en RMT.

Muchos errores en estudios numéricos provienen precisamente de:

normalizaciones incorrectas,

unfolding defectuoso,

factores empíricos ocultos.

Este proyecto documenta y corrige esos puntos.

🧭 Origen del Proyecto

El proyecto comenzó como una exploración numérica motivada por la Hipótesis de Riemann.

En el proceso se transformó en algo más general:

Un motor de calibración espectral que puede aplicarse a cualquier sistema donde la estadística de autovalores o niveles energéticos sea relevante.

La evolución no fue una renuncia.
Fue una depuración metodológica.

📌 Estado Actual

Integral Δ₃ auditada.

Parche empírico eliminado.

Poisson validado.

Unfolding GUE estructuralmente correcto.

Pendiente GUE en estudio para tamaños finitos.
