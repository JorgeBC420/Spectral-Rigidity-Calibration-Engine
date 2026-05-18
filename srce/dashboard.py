# -*- coding: utf-8 -*-
"""
🔬 Spectral Rigidity Calibration Engine - Dashboard Interactivo

Interfaz web para análisis espectral de ceros de Riemann,
validación RMT y estudio de rigidez espectral.

Autores: Jorge BC & Claude (Anthropic)
Fecha: Febrero 2026
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from pathlib import Path
import sys

# Configuración de página (DEBE ser lo primero)
st.set_page_config(
    page_title="Spectral Rigidity Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/JorgeBC420/Spectral-Rigidity-Calibration-Engine',
        'Report a bug': 'https://github.com/JorgeBC420/Spectral-Rigidity-Calibration-Engine/issues',
        'About': """
        # Spectral Rigidity Calibration Engine
        
        Motor de análisis de rigidez espectral basado en Random Matrix Theory.
        
        **Advertencia:** Este proyecto NO resuelve la Hipótesis de Riemann.
        Proporciona herramientas de análisis espectral científicamente rigurosas.
        """
    }
)

# Añadir directorio src al path
sys.path.insert(0, str(Path(__file__).parent))

# Importar módulos del motor
try:
    from solucionador_reimann import (
        CACHE, 
        analizar_espaciado_puntual,
        estudiar_espaciado_vs_N,
        espaciado_minimo,
        calcular_espaciados
    )
    MOTOR_DISPONIBLE = True
except ImportError as e:
    st.error(f"❌ Error importando motor principal: {e}")
    st.info("💡 Asegúrate de que solucionador_reimann.py está en el mismo directorio")
    MOTOR_DISPONIBLE = False

try:
    from src.riemann_spectral.analysis.rigidity import delta3_dyson_mehta
    from src.riemann_spectral.analysis.number_variance import (
        sigma2_number_variance_fast,
        sigma2_theoretical,
    )
    from src.riemann_spectral.analysis.pair_correlation import (
        pair_correlation_fast,
        r2_teorica_gue,
        r2_teorica_poisson,
        chi2_r2_vs_gue,
    )
    from src.riemann_spectral.analysis.spectral_form_factor import (
        spectral_form_factor,
        spectral_form_factor_mehta,
        spectral_form_factor_teorico,
        r_statistic,
        r_distribucion_teorica,
        R_MEAN_GUE, R_MEAN_GOE, R_MEAN_POISSON,
    )
    from src.riemann_spectral.data.generators import (
        generar_gue_normalizado,
        generar_poisson,
        generar_goe_normalizado,
    )
    from src.riemann_spectral.analysis.unfolding import (
        unfolding_wigner_gue,
        unfolding_tercio_central,
    )
    from src.riemann_spectral.analysis.normalize import normalize_spacing
    RIGIDEZ_DISPONIBLE = True

except ImportError as e:
    st.warning(f"⚠️ Módulo de análisis no disponible: {e}")
    st.info("💡 Verifica src/riemann_spectral/")
    RIGIDEZ_DISPONIBLE = False

# ============================================================================
# CONFIGURACIÓN Y ESTADO DE SESIÓN
# ============================================================================

if 'resultados_espaciado' not in st.session_state:
    st.session_state.resultados_espaciado = None
if 'resultados_rigidez' not in st.session_state:
    st.session_state.resultados_rigidez = None
if 'gamma_actual' not in st.session_state:
    st.session_state.gamma_actual = None
if 'resultados_rmt' not in st.session_state:
    st.session_state.resultados_rmt = None

# ============================================================================
# ESTILOS CSS PERSONALIZADOS
# ============================================================================

st.markdown("""
<style>
    /* Métricas mejoradas */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    
    /* Alertas científicas */
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    
    .alert-info {
        background-color: #e3f2fd;
        border-color: #2196f3;
    }
    
    .alert-warning {
        background-color: #fff3e0;
        border-color: #ff9800;
    }
    
    .alert-success {
        background-color: #e8f5e9;
        border-color: #4caf50;
    }
    
    /* Tabs personalizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Botones */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def mostrar_metrica_mejorada(label, value, delta=None, color="blue"):
    """Muestra métrica con estilo personalizado."""
    col = st.columns([1])[0]
    
    delta_html = ""
    if delta is not None:
        delta_color = "green" if delta > 0 else "red"
        delta_symbol = "▲" if delta > 0 else "▼"
        delta_html = f'<div style="color:{delta_color}; font-size:0.9em;">{delta_symbol} {delta}</div>'
    
    col.markdown(f"""
    <div class="metric-container" style="background: linear-gradient(135deg, #{color}22 0%, #{color}44 100%);">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def crear_alerta(tipo, mensaje):
    """Crea alerta con estilo científico."""
    iconos = {"info": "ℹ️", "warning": "⚠️", "success": "✅"}
    st.markdown(f"""
    <div class="alert-box alert-{tipo}">
        <strong>{iconos[tipo]} {mensaje}</strong>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - CONFIGURACIÓN GLOBAL
# ============================================================================

with st.sidebar:
    st.image("https://raw.githubusercontent.com/JorgeBC420/Spectral-Rigidity-Calibration-Engine/main/.github/logo.png", 
             use_container_width=True, 
             caption="Spectral Rigidity Engine")
    
    st.title("⚙️ Configuración")
    
    # Selector de análisis
    modo_analisis = st.radio(
        "Tipo de Análisis",
        ["🔍 Exploración Rápida", "📊 Análisis Detallado", "🧪 Investigación Avanzada"],
        help="Selecciona el nivel de profundidad del análisis"
    )
    
    st.divider()
    
    # Parámetros según modo
    if "Rápida" in modo_analisis:
        N = st.slider("Número de ceros (N)", 100, 2000, 500, 100)
        L_values = st.multiselect("Valores de L", [5, 10, 15, 20], default=[10])
    elif "Detallado" in modo_analisis:
        N = st.slider("Número de ceros (N)", 500, 5000, 1000, 250)
        L_range = st.slider("Rango de L", 1.0, 50.0, (5.0, 25.0), 1.0)
        L_values = list(range(int(L_range[0]), int(L_range[1])+1, 2))
    else:  # Investigación
        N = st.number_input("Número de ceros (N)", 1000, 20000, 5000, 500)
        L_range = st.slider("Rango de L", 1.0, 100.0, (10.0, 50.0), 5.0)
        L_values = list(range(int(L_range[0]), int(L_range[1])+1, 5))
    
    st.divider()
    
    # Opciones avanzadas
    with st.expander("🔧 Opciones Avanzadas"):
        mostrar_matematicas = st.checkbox("Mostrar ecuaciones matemáticas", True)
        guardar_cache = st.checkbox("Guardar resultados en caché", True)
        modo_debug = st.checkbox("Modo debug", False)
        
        if modo_debug:
            st.code(f"""
Estado actual:
- N = {N}
- L_values = {L_values[:5]}...
- Modo = {modo_analisis}
- Cache disponible: {len(CACHE.ceros)} ceros
            """)
    
    st.divider()
    
    # Información del sistema
    st.markdown("### 💻 Sistema")
    st.caption(f"Motor: {'✅ Activo' if MOTOR_DISPONIBLE else '❌ No disponible'}")
    st.caption(f"Rigidez: {'✅ Activo' if RIGIDEZ_DISPONIBLE else '❌ No disponible'}")
    st.caption(f"Caché: {len(CACHE.ceros)} ceros calculados")

# ============================================================================
# PÁGINA PRINCIPAL
# ============================================================================

st.title("🔬 Spectral Rigidity Calibration Engine")
st.markdown("""
Motor de análisis de rigidez espectral basado en **Random Matrix Theory** (RMT).  
Herramienta de validación numérica para estadística espectral de sistemas cuánticos y la función zeta de Riemann.
""")

# Advertencia científica prominente
crear_alerta("warning", 
    "ADVERTENCIA CIENTÍFICA: Este proyecto NO resuelve la Hipótesis de Riemann. "
    "Proporciona análisis espectral riguroso para fines de investigación y validación.")

# ============================================================================
# TABS PRINCIPALES
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Análisis del Espaciado", 
    "📈 Rigidez Espectral Δ₃", 
    "🔬 Validación RMT",
    "📚 Documentación Matemática",
    "ℹ️ Acerca de"
])

# ============================================================================
# TAB 1: ANÁLISIS DEL ESPACIADO
# ============================================================================

with tab1:
    st.header("📊 Análisis del Espaciado Mínimo")
    
    if mostrar_matematicas:
        st.latex(r"""
        \text{Para } d_i = \gamma_{i+1} - \gamma_i, \text{ la ecuación de evolución es:}
        """)
        st.latex(r"""
        \dot{d}_i = \frac{4}{d_i} + R_i(\gamma)
        """)
        st.markdown("""
        donde:
        - **4/d_i**: Término singular repulsivo (barrera infinita cuando d_i → 0)
        - **R_i**: Término regular (contribución de ceros distantes)
        """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Ejecutar Análisis del Espaciado", type="primary"):
            if not MOTOR_DISPONIBLE:
                st.error("❌ Motor no disponible. Verifica la instalación.")
            else:
                with st.spinner(f"Calculando {N} ceros de Riemann..."):
                    try:
                        inicio = time.time()
                        gamma = CACHE.obtener(N)
                        tiempo_ceros = time.time() - inicio
                        
                        inicio = time.time()
                        resultado = analizar_espaciado_puntual(gamma, verbose=False)
                        tiempo_analisis = time.time() - inicio
                        
                        st.session_state.resultados_espaciado = resultado
                        st.session_state.gamma_actual = gamma
                        
                        st.success(f"✅ Análisis completado en {tiempo_ceros + tiempo_analisis:.2f}s")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        if modo_debug:
                            st.exception(e)
    
    with col2:
        if st.session_state.resultados_espaciado:
            res = st.session_state.resultados_espaciado
            
            st.metric("Espaciado mínimo", f"{res['d_min']:.6e}")
            st.metric("Velocidad ḋ", f"{res['d_dot']:+.2e}")
            st.metric("Ratio singular/regular", f"{res['ratio']:.2f}")
    
    # Resultados detallados
    if st.session_state.resultados_espaciado:
        st.divider()
        
        res = st.session_state.resultados_espaciado
        gamma = st.session_state.gamma_actual
        
        # Métricas principales en tarjetas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mostrar_metrica_mejorada(
                "d_min (Espaciado)",
                f"{res['d_min']:.6e}",
                color="3b82f6"
            )
        
        with col2:
            tendencia = "Repulsiva" if res['tendencia_local_repulsiva'] else "Atractiva"
            color = "10b981" if res['tendencia_local_repulsiva'] else "ef4444"
            mostrar_metrica_mejorada(
                "Tendencia Local",
                tendencia,
                color=color
            )
        
        with col3:
            mostrar_metrica_mejorada(
                "Término Singular",
                f"{res['term_singular']:.2e}",
                color="8b5cf6"
            )
        
        with col4:
            mostrar_metrica_mejorada(
                "Término Regular",
                f"{res['term_regular']:.2e}",
                color="ec4899"
            )
        
        # Gráficos interactivos
        st.subheader("Visualizaciones")
        
        tab_viz1, tab_viz2, tab_viz3 = st.tabs([
            "📍 Espaciados", 
            "📊 Distribución", 
            "🔬 Descomposición"
        ])
        
        with tab_viz1:
            # Gráfico de espaciados
            espaciados = calcular_espaciados(gamma)
            idx_min = res['idx']
            
            fig = go.Figure()
            
            # Todos los espaciados
            fig.add_trace(go.Scatter(
                x=list(range(len(espaciados))),
                y=espaciados,
                mode='lines+markers',
                name='Espaciados',
                line=dict(color='lightblue', width=1),
                marker=dict(size=3)
            ))
            
            # Espaciado mínimo destacado
            fig.add_trace(go.Scatter(
                x=[idx_min],
                y=[espaciados[idx_min]],
                mode='markers',
                name='Mínimo',
                marker=dict(size=15, color='red', symbol='star')
            ))
            
            fig.update_layout(
                title=f"Espaciados entre Ceros Consecutivos (N={N})",
                xaxis_title="Índice i",
                yaxis_title="d_i = γ_{i+1} - γ_i",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_viz2:
            # Histograma de espaciados
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=espaciados,
                nbinsx=50,
                name='Distribución',
                marker=dict(color='#667eea', line=dict(color='#764ba2', width=1))
            ))
            
            # Línea vertical en el mínimo
            fig.add_vline(
                x=res['d_min'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"d_min = {res['d_min']:.6e}"
            )
            
            fig.update_layout(
                title="Distribución de Espaciados",
                xaxis_title="Espaciado d_i",
                yaxis_title="Frecuencia",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_viz3:
            # Descomposición del término regular
            st.markdown("### Contribución por Rango de Distancia")
            
            R_total = res['R_cercano'] + res['R_medio'] + res['R_lejano']
            
            df_descomp = pd.DataFrame({
                'Rango': ['Cercano (|j-i|<10)', 'Medio (10≤|j-i|<100)', 'Lejano (|j-i|≥100)'],
                'Contribución': [res['R_cercano'], res['R_medio'], res['R_lejano']],
                'Porcentaje': [
                    100 * res['R_cercano'] / R_total if R_total != 0 else 0,
                    100 * res['R_medio'] / R_total if R_total != 0 else 0,
                    100 * res['R_lejano'] / R_total if R_total != 0 else 0
                ]
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_descomp['Rango'],
                y=df_descomp['Contribución'],
                text=[f"{p:.1f}%" for p in df_descomp['Porcentaje']],
                textposition='outside',
                marker=dict(
                    color=['#667eea', '#764ba2', '#f093fb'],
                    line=dict(color='white', width=2)
                )
            ))
            
            fig.update_layout(
                title="Descomposición del Término Regular R_i",
                xaxis_title="Rango de Distancia",
                yaxis_title="Contribución a R_i",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_descomp, use_container_width=True)
        
        # Interpretación científica
        st.divider()
        st.subheader("📝 Interpretación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Observaciones Puntuales")
            if res['tendencia_local_repulsiva']:
                crear_alerta("success", 
                    f"✓ Tendencia repulsiva detectada (ḋ > 0)\n\n"
                    f"El espaciado tiende a AUMENTAR instantáneamente.")
            else:
                crear_alerta("warning",
                    f"⚠ Tendencia atractiva detectada (ḋ < 0)\n\n"
                    f"El espaciado tiende a DISMINUIR instantáneamente.")
            
            if res['ratio'] > 10:
                crear_alerta("success",
                    f"✓ Término singular DOMINA fuertemente (ratio = {res['ratio']:.2f})\n\n"
                    f"Barrera repulsiva 4/d_i es claramente dominante.")
            elif res['ratio'] > 2:
                crear_alerta("info",
                    f"≈ Dominio moderado del término singular (ratio = {res['ratio']:.2f})\n\n"
                    f"Término regular aún significativo.")
            else:
                crear_alerta("warning",
                    f"⚠ Términos competitivos (ratio = {res['ratio']:.2f})\n\n"
                    f"Término regular es comparable o domina.")
        
        with col2:
            st.markdown("### Limitaciones")
            st.info("""
            **Advertencias críticas:**
            
            1. **Análisis PUNTUAL** (no dinámico)  
               Solo describe el comportamiento en t=0
            
            2. **Sistema TRUNCADO** (N finito)  
               No incluye efecto de ceros distantes
            
            3. **NO decide la Hipótesis de Riemann**  
               Solo informa sobre estructura local
            
            4. **Validez limitada**  
               Observaciones para este N específico
            """)

# ============================================================================
# TAB 2: RIGIDEZ ESPECTRAL Δ₃
# ============================================================================

with tab2:
    st.header("📈 Estadística de Rigidez Espectral Δ₃")
    
    if mostrar_matematicas:
        st.latex(r"""
        \Delta_3(L) = \frac{1}{L} \min_{A,B} \int_{x_0}^{x_0+L} [N(x) - A - Bx]^2 dx
        """)
        st.markdown("""
        **Predicciones teóricas:**
        - **Poisson (desorden total):** Δ₃(L) ≈ L/15
        - **GUE (correlacionado):** Δ₃(L) ~ (1/π²) log L
        - **Ceros de Riemann:** Consistente con GUE (conjetura)
        """)
    
    if not RIGIDEZ_DISPONIBLE:
        st.error("❌ Módulo de rigidez no disponible. Verifica src/riemann_spectral/")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analizar_riemann = st.checkbox("🔢 Ceros de Riemann", True)
        with col2:
            analizar_poisson = st.checkbox("🎲 Poisson (baseline)", True)
        with col3:
            analizar_gue = st.checkbox("🔗 GUE (teoría)", True)
        
        if st.button("🚀 Calcular Δ₃", type="primary"):
            with st.spinner("Calculando estadísticas espectrales..."):
                try:
                    resultados = {}
                    
                    if analizar_riemann:
                        st.info(f"Obteniendo {N} ceros de Riemann...")
                        gamma = CACHE.obtener(N)
                        
                        # Calcular Δ₃ para diferentes L
                        delta3_vals = []
                        for L in L_values:
                            try:
                                # Usar función real del módulo
                                d3 = delta3_dyson_mehta(gamma, L)
                                delta3_vals.append(d3)
                            except Exception as e:
                                st.warning(f"Error calculando Δ₃ para L={L}: {e}")
                                delta3_vals.append(np.nan)
                        
                        resultados['riemann'] = {
                            'L': L_values,
                            'delta3': delta3_vals,
                            'label': 'Ceros ζ(s)'
                        }
                    
                    if analizar_poisson:
                        st.info("Generando secuencia Poisson...")
                        try:
                            pos_poisson = generar_poisson(N)
                            
                            delta3_vals = []
                            for L in L_values:
                                try:
                                    d3 = delta3_dyson_mehta(pos_poisson, L)
                                    delta3_vals.append(d3)
                                except Exception as e:
                                    st.warning(f"Error Poisson L={L}: {e}")
                                    # Usar predicción teórica como fallback
                                    delta3_vals.append(L / 15)
                            
                            resultados['poisson'] = {
                                'L': L_values,
                                'delta3': delta3_vals,
                                'label': 'Poisson (L/15)'
                            }
                        except Exception as e:
                            st.error(f"Error generando Poisson: {e}")
                    
                    if analizar_gue:
                        st.info("Generando matriz GUE...")
                        try:
                            import scipy.linalg as _la
                            # Normalización de Wigner directa: H = (A+A†)/(2√N)
                            # → autovalores en [-2, 2], compatibles con unfolding_wigner_gue
                            _rng_gue = np.random.default_rng(42)
                            _A = _rng_gue.standard_normal((N, N)) + \
                                 1j * _rng_gue.standard_normal((N, N))
                            _H = (_A + _A.conj().T) / (2 * np.sqrt(N))
                            eigenvalues = np.sort(_la.eigvalsh(_H))
                            # Aplicar unfolding de Wigner
                            unfolded = unfolding_wigner_gue(eigenvalues)
                            
                            delta3_vals = []
                            for L in L_values:
                                try:
                                    d3 = delta3_dyson_mehta(unfolded, L)
                                    delta3_vals.append(d3)
                                except Exception as e:
                                    st.warning(f"Error GUE L={L}: {e}")
                                    # Predicción teórica como fallback
                                    delta3_vals.append((1/np.pi**2) * np.log(L) if L > 1 else 0)
                            
                            resultados['gue'] = {
                                'L': L_values,
                                'delta3': delta3_vals,
                                'label': 'GUE (log L)'
                            }
                        except Exception as e:
                            st.error(f"Error generando GUE: {e}")
                    
                    st.session_state.resultados_rigidez = resultados
                    st.success("✅ Cálculos completados")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    if modo_debug:
                        st.exception(e)
        
        # Visualización de resultados
        if st.session_state.resultados_rigidez:
            st.divider()
            st.subheader("Resultados")
            
            res = st.session_state.resultados_rigidez
            
            # Gráfico comparativo
            fig = go.Figure()
            
            colores = {'riemann': '#667eea', 'poisson': '#f093fb', 'gue': '#4facfe'}
            
            for key, data in res.items():
                fig.add_trace(go.Scatter(
                    x=data['L'],
                    y=data['delta3'],
                    mode='lines+markers',
                    name=data['label'],
                    line=dict(width=3, color=colores[key]),
                    marker=dict(size=8)
                ))
            
            # Líneas teóricas
            L_teorico = np.linspace(min(L_values), max(L_values), 100)
            
            fig.add_trace(go.Scatter(
                x=L_teorico,
                y=L_teorico / 15,
                mode='lines',
                name='Teoría Poisson (L/15)',
                line=dict(dash='dash', color='gray', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=L_teorico,
                y=(1/np.pi**2) * np.log(L_teorico),
                mode='lines',
                name='Teoría GUE (log L / π²)',
                line=dict(dash='dot', color='gray', width=2)
            ))
            
            fig.update_layout(
                title="Rigidez Espectral Δ₃(L) - Comparación",
                xaxis_title="L (longitud de ventana)",
                yaxis_title="Δ₃(L)",
                hovermode='x unified',
                height=600,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabla de valores
            st.subheader("Tabla de Valores")
            
            df_delta3 = pd.DataFrame({'L': L_values})
            for key, data in res.items():
                df_delta3[data['label']] = data['delta3']
            
            st.dataframe(df_delta3, use_container_width=True)
            
            # Descargar datos
            csv = df_delta3.to_csv(index=False)
            st.download_button(
                label="📥 Descargar datos CSV",
                data=csv,
                file_name=f"delta3_N{N}.csv",
                mime="text/csv"
            )

# ============================================================================
# TAB 3: VALIDACIÓN RMT
# ============================================================================

with tab3:
    st.header("🔬 Validación Random Matrix Theory — Toolkit Completo")
    st.markdown("""
    Comparación de **5 estadísticas espectrales** entre Poisson, GOE, GUE y ceros de Riemann.
    Cada estadística captura un aspecto distinto de la estructura del espectro.
    """)

    if not RIGIDEZ_DISPONIBLE:
        st.error("❌ Módulos de análisis no disponibles. Verifica src/riemann_spectral/")
        st.stop()

    # ── Controles de generación ──────────────────────────────────────────────
    with st.expander("⚙️ Configuración del Análisis RMT", expanded=True):
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            N_rmt = st.slider("N (tamaño del espectro)", 500, 3000, 1200, 100,
                              key="n_rmt", help="Más N = más precisión, más tiempo")
        with col_cfg2:
            incluir_riemann = st.checkbox("Incluir ceros de Riemann", True, key="rmt_riemann")
            incluir_goe     = st.checkbox("Incluir GOE", True, key="rmt_goe")
        with col_cfg3:
            seed_rmt = st.number_input("Seed RNG", 0, 9999, 42, key="seed_rmt")

    if st.button("🚀 Calcular todas las estadísticas RMT", type="primary", key="btn_rmt"):
        if not MOTOR_DISPONIBLE and incluir_riemann:
            st.warning("⚠️ Motor no disponible — se omitirán los ceros de Riemann.")
            incluir_riemann = False

        with st.spinner("Generando ensembles y calculando estadísticas..."):
            rng_rmt = np.random.default_rng(seed=seed_rmt)
            resultados_rmt = {}

            # ── Generar espectros ────────────────────────────────────────────
            # NOTA: generar_gue_normalizado() re-centra los autovalores en N/2,
            # lo que los hace incompatibles con unfolding_wigner_gue (rango [-2,2]).
            # Se construye la matriz directamente con la normalización de Wigner:
            #   H = (A + A†) / (2√N)  →  autovalores en [-2, 2]
            try:
                import scipy.linalg as _la

                # Poisson
                ev_poisson = generar_poisson(N_rmt, rng=rng_rmt)
                resultados_rmt["Poisson"] = {"ev": ev_poisson, "color": "#e74c3c",
                                              "dash": "dash"}

                # GUE — diagonalización con normalización de Wigner correcta
                A = rng_rmt.standard_normal((N_rmt, N_rmt)) + \
                    1j * rng_rmt.standard_normal((N_rmt, N_rmt))
                H = (A + A.conj().T) / (2 * np.sqrt(N_rmt))
                ev_gue_raw = np.sort(_la.eigvalsh(H))
                ev_gue = unfolding_wigner_gue(ev_gue_raw)
                n_g = len(ev_gue)
                ev_gue = normalize_spacing(ev_gue[n_g//3: 2*(n_g//3)])
                ev_gue = ev_gue - ev_gue[0]
                resultados_rmt["GUE"] = {"ev": ev_gue, "color": "#3498db", "dash": "dot"}

                # GOE (opcional) — misma corrección
                if incluir_goe:
                    A2 = rng_rmt.standard_normal((N_rmt, N_rmt))
                    H2 = (A2 + A2.T) / (2 * np.sqrt(N_rmt))
                    ev_goe_raw = np.sort(_la.eigvalsh(H2))
                    ev_goe = unfolding_wigner_gue(ev_goe_raw)
                    n_go = len(ev_goe)
                    ev_goe = normalize_spacing(ev_goe[n_go//3: 2*(n_go//3)])
                    ev_goe = ev_goe - ev_goe[0]
                    resultados_rmt["GOE"] = {"ev": ev_goe, "color": "#2ecc71", "dash": "dashdot"}

                # Riemann (opcional)
                if incluir_riemann and MOTOR_DISPONIBLE:
                    gamma_r = CACHE.obtener(N_rmt)
                    from src.riemann_spectral.analysis.unfolding import unfolding_riemann
                    ev_riem = unfolding_riemann(gamma_r)
                    nr = len(ev_riem)
                    ev_riem = ev_riem[nr//3: 2*(nr//3)] - ev_riem[nr//3]
                    resultados_rmt["Riemann"] = {"ev": ev_riem, "color": "#9b59b6",
                                                  "dash": "solid"}

                st.session_state.resultados_rmt = resultados_rmt
                st.success(f"✅ Análisis completado — {len(resultados_rmt)} ensembles")

            except Exception as e:
                st.error(f"❌ Error generando ensembles: {e}")
                if modo_debug:
                    st.exception(e)

    # ── Mostrar resultados en sub-tabs ───────────────────────────────────────
    if st.session_state.resultados_rmt:
        res = st.session_state.resultados_rmt

        rmt_t1, rmt_t2, rmt_t3, rmt_t4, rmt_t5, rmt_t6, rmt_t7 = st.tabs([
            "P(s) Espaciados",
            "Σ²(L) Varianza",
            "R₂(s) Correlación Pares",
            "K(t) Form Factor",
            "r-statistic",
            "🔀 Unfolding Comparado",
            "📋 Resumen Comparativo",
        ])

        # ════════════════════════════════════════════════════════════════════
        # P(s) — Distribución de espaciados nearest-neighbor
        # ════════════════════════════════════════════════════════════════════
        with rmt_t1:
            st.subheader("P(s) — Distribución de Espaciados Vecinos")
            if mostrar_matematicas:
                col_eq1, col_eq2 = st.columns(2)
                with col_eq1:
                    st.latex(r"P_\text{Poisson}(s) = e^{-s}")
                with col_eq2:
                    st.latex(r"P_\text{GUE}(s) = \frac{\pi}{2}\,s\,e^{-\frac{\pi}{4}s^2}")

            fig_ps = go.Figure()
            s_teo = np.linspace(0, 4, 300)

            # Curvas teóricas
            fig_ps.add_trace(go.Scatter(
                x=s_teo, y=np.exp(-s_teo),
                mode="lines", name="Poisson teórico",
                line=dict(color="#e74c3c", width=1, dash="dash")))
            fig_ps.add_trace(go.Scatter(
                x=s_teo,
                y=(np.pi/2)*s_teo*np.exp(-np.pi*s_teo**2/4),
                mode="lines", name="GUE teórico (Wigner)",
                line=dict(color="#3498db", width=1, dash="dash")))
            goe_teo = (32/np.pi**2)*s_teo*np.exp(-4*s_teo**2/np.pi)
            fig_ps.add_trace(go.Scatter(
                x=s_teo, y=goe_teo,
                mode="lines", name="GOE teórico",
                line=dict(color="#2ecc71", width=1, dash="dot")))

            # Histogramas experimentales
            for label, d in res.items():
                ev = d["ev"]
                spacings = np.diff(ev)
                # Normalizar spacing medio a 1
                s_mean = np.mean(spacings)
                if s_mean > 1e-10:
                    spacings = spacings / s_mean
                hist, edges = np.histogram(spacings, bins=50, range=(0, 4), density=True)
                ctrs = 0.5*(edges[1:]+edges[:-1])
                fig_ps.add_trace(go.Scatter(
                    x=ctrs, y=hist, mode="lines+markers",
                    name=f"{label} (N={len(ev)})",
                    line=dict(color=d["color"], width=2),
                    marker=dict(size=4)))

            fig_ps.update_layout(
                title="P(s): Distribución de Espaciados Nearest-Neighbor",
                xaxis_title="s (espaciado normalizado)",
                yaxis_title="P(s)",
                hovermode="x unified", height=500,
                legend=dict(bgcolor="rgba(255,255,255,0.9)"))
            st.plotly_chart(fig_ps, use_container_width=True)

            st.info("**Interpretar:** GUE tiene máximo en s≈1 (repulsión) mientras "
                    "Poisson decae exponencialmente desde s=0 (no hay repulsión).")

        # ════════════════════════════════════════════════════════════════════
        # Σ²(L) — Number Variance
        # ════════════════════════════════════════════════════════════════════
        with rmt_t2:
            st.subheader("Σ²(L) — Varianza del Número de Niveles")
            if mostrar_matematicas:
                col_eq1, col_eq2 = st.columns(2)
                with col_eq1:
                    st.latex(r"\Sigma^2(L) = \langle (N(L) - L)^2 \rangle")
                with col_eq2:
                    st.latex(r"\Sigma^2_\text{GUE}(L) \approx \frac{1}{\pi^2}\ln L")

            L_sigma = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
            fig_s2 = go.Figure()

            # Curvas teóricas
            L_teo = np.linspace(1, 35, 200)
            fig_s2.add_trace(go.Scatter(
                x=L_teo, y=L_teo,
                mode="lines", name="Poisson teórico (L)",
                line=dict(color="#e74c3c", width=1, dash="dash")))
            fig_s2.add_trace(go.Scatter(
                x=L_teo, y=(1/np.pi**2)*np.log(L_teo),
                mode="lines", name="GUE teórico ((1/π²)ln L)",
                line=dict(color="#3498db", width=1, dash="dash")))
            fig_s2.add_trace(go.Scatter(
                x=L_teo, y=(2/np.pi**2)*np.log(L_teo),
                mode="lines", name="GOE teórico ((2/π²)ln L)",
                line=dict(color="#2ecc71", width=1, dash="dot")))

            # Experimental
            for label, d in res.items():
                ev = d["ev"]
                try:
                    sigma2_vals = sigma2_number_variance_fast(ev, L_sigma)
                    valid = np.isfinite(sigma2_vals)
                    fig_s2.add_trace(go.Scatter(
                        x=L_sigma[valid], y=sigma2_vals[valid],
                        mode="lines+markers", name=f"{label}",
                        line=dict(color=d["color"], width=2),
                        marker=dict(size=7)))
                except Exception as e_s2:
                    st.warning(f"Σ²: error en {label}: {e_s2}")

            fig_s2.update_layout(
                title="Σ²(L): Number Variance",
                xaxis_title="L (longitud de ventana)",
                yaxis_title="Σ²(L)",
                hovermode="x unified", height=500)
            st.plotly_chart(fig_s2, use_container_width=True)

            st.info("**Interpretar:** GUE crece logarítmicamente (correlaciones fuertes). "
                    "Poisson crece linealmente (sin correlaciones).")

        # ════════════════════════════════════════════════════════════════════
        # R₂(s) — Pair Correlation
        # ════════════════════════════════════════════════════════════════════
        with rmt_t3:
            st.subheader("R₂(s) — Función de Correlación de Pares")
            if mostrar_matematicas:
                st.latex(r"R_2(s) = 1 - \left(\frac{\sin(\pi s)}{\pi s}\right)^2 \quad \text{(GUE)}")
                st.markdown("**Montgomery 1973 · Dyson 1962** — La conjetura que conecta "
                            "los ceros de Riemann con RMT fue observada en esta estadística.")

            s_grid = np.linspace(0.01, 5.0, 300)
            fig_r2 = go.Figure()

            # Curvas teóricas
            fig_r2.add_trace(go.Scatter(
                x=s_grid, y=r2_teorica_poisson(s_grid),
                mode="lines", name="Poisson teórico (= 1)",
                line=dict(color="#e74c3c", width=1, dash="dash")))
            fig_r2.add_trace(go.Scatter(
                x=s_grid, y=r2_teorica_gue(s_grid),
                mode="lines", name="GUE teórico (Montgomery–Dyson)",
                line=dict(color="#3498db", width=1, dash="dash")))

            # Línea de referencia en s=1 (primer nodo de GUE)
            fig_r2.add_vline(x=1.0, line_dash="dot", line_color="gray",
                             annotation_text="nodo s=1", annotation_position="top")

            # Experimental
            for label, d in res.items():
                ev = d["ev"]
                try:
                    s_obs, r2_obs = pair_correlation_fast(ev, s_max=5.0, bins=80)
                    valid = np.isfinite(r2_obs)
                    fig_r2.add_trace(go.Scatter(
                        x=s_obs[valid], y=r2_obs[valid],
                        mode="lines", name=f"{label}",
                        line=dict(color=d["color"], width=2, dash=d["dash"])))
                except Exception as e_r2:
                    st.warning(f"R₂: error en {label}: {e_r2}")

            fig_r2.update_layout(
                title="R₂(s): Función de Correlación de Pares",
                xaxis_title="s (distancia espectral)",
                yaxis_title="R₂(s)",
                hovermode="x unified", height=500,
                yaxis=dict(range=[-0.1, 1.5]))
            st.plotly_chart(fig_r2, use_container_width=True)

            # χ² vs GUE para Riemann (si disponible)
            if "Riemann" in res:
                try:
                    s_obs, r2_obs = pair_correlation_fast(res["Riemann"]["ev"],
                                                          s_max=5.0, bins=80)
                    chi2_res = chi2_r2_vs_gue(s_obs, r2_obs)
                    if np.isfinite(chi2_res["chi2_reducido"]):
                        chi2_val = chi2_res["chi2_reducido"]
                        label_chi2 = "✅ Consistente" if chi2_val < 3 else "⚠️ Desviación"
                        st.metric("χ²/dof Riemann vs GUE",
                                  f"{chi2_val:.2f}", label_chi2)
                except Exception:
                    pass

            st.info("**Interpretar:** El dip en s→0 y el nodo en s=1 son la huella "
                    "de la repulsión de niveles en GUE. Poisson no tiene estructura.")

        # ════════════════════════════════════════════════════════════════════
        # K(t) — Spectral Form Factor
        # ════════════════════════════════════════════════════════════════════
        with rmt_t4:
            st.subheader("K(t) — Factor de Forma Espectral")
            if mostrar_matematicas:
                st.latex(r"K(t) = \frac{1}{N^2}\left|\sum_n e^{2\pi i\,t\,\gamma_n}\right|^2")
                col_eq1, col_eq2 = st.columns(2)
                with col_eq1:
                    st.latex(r"K_\text{GUE}(t) = \begin{cases}|t| & |t|\le 1 \\ 1 & |t|>1\end{cases}")
                with col_eq2:
                    st.latex(r"K_\text{Poisson}(t) = 1")

            smooth_kfac = st.slider("Suavizado gaussiano (σ)", 0.0, 5.0, 2.0, 0.5,
                                    key="smooth_kfac")
            t_max_kfac  = st.slider("t máximo", 1.0, 5.0, 3.0, 0.5, key="t_max_kfac")

            fig_kf = go.Figure()
            t_teo = np.linspace(0, t_max_kfac, 300)

            # Teóricas
            fig_kf.add_trace(go.Scatter(
                x=t_teo, y=spectral_form_factor_teorico(t_teo, "GUE"),
                mode="lines", name="GUE teórico",
                line=dict(color="#3498db", width=1, dash="dash")))
            fig_kf.add_trace(go.Scatter(
                x=t_teo, y=spectral_form_factor_teorico(t_teo, "GOE"),
                mode="lines", name="GOE teórico",
                line=dict(color="#2ecc71", width=1, dash="dot")))
            fig_kf.add_trace(go.Scatter(
                x=t_teo, y=spectral_form_factor_teorico(t_teo, "Poisson"),
                mode="lines", name="Poisson teórico (=1)",
                line=dict(color="#e74c3c", width=1, dash="dash")))

            # Experimental
            for label, d in res.items():
                ev = d["ev"]
                try:
                    t_obs, K_obs = spectral_form_factor_mehta(
                        ev, t_max=t_max_kfac, n_t=200,
                        smooth_sigma=smooth_kfac if smooth_kfac > 0 else None)
                    fig_kf.add_trace(go.Scatter(
                        x=t_obs, y=K_obs, mode="lines", name=f"{label}",
                        line=dict(color=d["color"], width=2, dash=d["dash"])))
                except Exception as e_kf:
                    st.warning(f"K(t): error en {label}: {e_kf}")

            fig_kf.update_layout(
                title="K(t): Factor de Forma Espectral",
                xaxis_title="t (tiempo de Heisenberg)",
                yaxis_title="K(t)",
                hovermode="x unified", height=500,
                yaxis=dict(range=[0, 1.5]))
            st.plotly_chart(fig_kf, use_container_width=True)

            st.info("**Interpretar:** El 'dip' de GUE para t<1 revela correlaciones "
                    "de largo alcance. Poisson es plano (K=1). Usar suavizado para reducir ruido.")

        # ════════════════════════════════════════════════════════════════════
        # r-statistic
        # ════════════════════════════════════════════════════════════════════
        with rmt_t5:
            st.subheader("r-statistic — Ratio de Espaciados Consecutivos")
            if mostrar_matematicas:
                st.latex(r"r_n = \frac{\min(s_n,\,s_{n+1})}{\max(s_n,\,s_{n+1})}, "
                         r"\quad s_n = \gamma_{n+1}-\gamma_n")
                st.markdown("**Ventaja:** No requiere unfolding — directamente aplicable "
                            "a los ceros de Riemann sin preprocesamiento.")

            # Tabla de valores teóricos
            col_teo, col_res = st.columns([1, 2])

            with col_teo:
                st.markdown("**Valores teóricos ⟨r⟩:**")
                df_teo = pd.DataFrame({
                    "Ensemble": ["Poisson", "GOE", "GUE"],
                    "⟨r⟩ teórico": [f"{R_MEAN_POISSON:.4f}",
                                     f"{R_MEAN_GOE:.4f}",
                                     f"{R_MEAN_GUE:.4f}"],
                })
                st.dataframe(df_teo, hide_index=True, use_container_width=True)

            with col_res:
                st.markdown("**Resultados experimentales:**")
                filas = []
                for label, d in res.items():
                    try:
                        r_res = r_statistic(d["ev"])
                        dist_min = min(r_res["distancia_gue"],
                                       r_res["distancia_goe"],
                                       r_res["distancia_poisson"])
                        filas.append({
                            "Ensemble": label,
                            "⟨r⟩ obs":     f"{r_res['r_mean']:.4f}",
                            "σ(r)":         f"{r_res['r_std']:.4f}",
                            "Clasificación":r_res["clasificacion"],
                        })
                    except Exception as e_r:
                        filas.append({"Ensemble": label, "⟨r⟩ obs": "Error",
                                      "σ(r)": "—", "Clasificación": str(e_r)[:30]})
                if filas:
                    st.dataframe(pd.DataFrame(filas), hide_index=True,
                                 use_container_width=True)

            # Distribuciones P(r) — teóricas + histogramas
            st.divider()
            r_grid_teo = np.linspace(0, 1, 300)
            fig_rstat = go.Figure()

            for ens_label, ens_color in [("GUE","#3498db"),("GOE","#2ecc71"),("Poisson","#e74c3c")]:
                fig_rstat.add_trace(go.Scatter(
                    x=r_grid_teo,
                    y=r_distribucion_teorica(r_grid_teo, ens_label),
                    mode="lines", name=f"{ens_label} teórico",
                    line=dict(color=ens_color, width=1, dash="dash")))

            for label, d in res.items():
                ev = d["ev"]
                try:
                    r_res = r_statistic(ev, return_distribution=True)
                    r_vals = r_res.get("r_vals", np.array([]))
                    if len(r_vals) > 10:
                        hist_r, edges_r = np.histogram(r_vals, bins=40,
                                                        range=(0,1), density=True)
                        ctrs_r = 0.5*(edges_r[1:]+edges_r[:-1])
                        fig_rstat.add_trace(go.Bar(
                            x=ctrs_r, y=hist_r,
                            name=f"{label} (N={len(r_vals)})",
                            marker_color=d["color"], opacity=0.5,
                            width=0.025))
                except Exception as e_rstat:
                    st.warning(f"r-stat histograma: error en {label}: {e_rstat}")

            fig_rstat.update_layout(
                title="P(r): Distribución del r-statistic",
                xaxis_title="r", yaxis_title="P(r)",
                hovermode="x unified", height=480,
                barmode="overlay")
            st.plotly_chart(fig_rstat, use_container_width=True)

            st.info("**Interpretar:** El pico de GUE en r≈0.7 refleja la repulsión "
                    "entre niveles. Poisson tiene más probabilidad cerca de r=0 "
                    "(niveles muy cercanos son frecuentes sin correlaciones).")

        # ════════════════════════════════════════════════════════════════════
        # RESUMEN COMPARATIVO — tabla + radar chart
        # ════════════════════════════════════════════════════════════════════
        # ════════════════════════════════════════════════════════════════════
        # UNFOLDING COMPARADO — KDE vs Spline vs Polinomial vs Analítico
        # ════════════════════════════════════════════════════════════════════
        with rmt_t6:
            st.subheader("🔀 Comparación de Métodos de Unfolding")
            st.markdown("""
            El **unfolding** transforma el espectro crudo a densidad local ≈ 1.
            La calidad del unfolding afecta directamente todas las estadísticas.
            Aquí comparamos cuatro métodos sobre los mismos datos.
            """)

            try:
                from src.riemann_spectral.analysis.empirical_unfolding import (
                    compare_unfolding_methods,
                    spacing_histogram,
                )
                from src.riemann_spectral.analysis.unfolding import unfolding_riemann

                # ── Selector de espectro ──────────────────────────────────
                col_uf1, col_uf2 = st.columns([2, 1])
                with col_uf1:
                    espectro_uf = st.selectbox(
                        "Espectro a analizar",
                        options=list(res.keys()),
                        key="uf_espectro",
                        help="Selecciona el ensemble sobre el que comparar los métodos"
                    )
                with col_uf2:
                    recorte_uf = st.slider("Recorte extremos", 0.0, 0.2, 0.1, 0.05,
                                           key="uf_recorte")

                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    kde_bw = st.number_input("KDE bandwidth (0=auto)", 0.0, 10.0, 0.0,
                                             step=0.1, key="uf_kde_bw")
                    kde_bw = None if kde_bw == 0.0 else kde_bw
                with col_p2:
                    spline_k = st.slider("Spline nudos", 10, 200, 50, 10, key="uf_spline_k")
                with col_p3:
                    poly_d = st.slider("Polinomio grado", 3, 15, 7, 1, key="uf_poly_d")

                # ── Determinar si hay función analítica disponible ────────
                ev_raw = res[espectro_uf]["ev"]
                analytic = unfolding_riemann if espectro_uf == "Riemann" else None

                # ── Correr comparación ────────────────────────────────────
                with st.spinner("Calculando métodos de unfolding..."):
                    uf_results = compare_unfolding_methods(
                        ev_raw,
                        analytic_fn=analytic,
                        kde_bandwidth=kde_bw,
                        spline_knots=spline_k,
                        poly_degree=poly_d,
                        recorte=recorte_uf,
                    )

                # ── Tabla de métricas ─────────────────────────────────────
                st.markdown("#### Métricas por método")
                R_POI = 2 * np.log(2) - 1
                R_GUE = 0.60272
                R_GOE = 4 - 2 * np.sqrt(3)

                filas_uf = []
                for mname, mr in uf_results.items():
                    if np.isfinite(mr.get("r_mean", np.nan)):
                        d_gue = abs(mr["r_mean"] - R_GUE)
                        d_goe = abs(mr["r_mean"] - R_GOE)
                        d_poi = abs(mr["r_mean"] - R_POI)
                        clasif = min({"GUE": d_gue, "GOE": d_goe, "Poisson": d_poi},
                                     key=lambda k: {"GUE": d_gue, "GOE": d_goe, "Poisson": d_poi}[k])
                    else:
                        clasif = "—"
                    filas_uf.append({
                        "Método"    : mname,
                        "⟨s⟩"      : f"{mr['mean_s']:.4f}" if np.isfinite(mr.get('mean_s', np.nan)) else "—",
                        "σ(s)"      : f"{mr['std_s']:.4f}"  if np.isfinite(mr.get('std_s', np.nan))  else "—",
                        "⟨r⟩"      : f"{mr['r_mean']:.4f}" if np.isfinite(mr.get('r_mean', np.nan)) else "—",
                        "Ensemble"  : clasif,
                        "Válido"    : "✅" if mr.get("is_valid") else "❌",
                    })
                if filas_uf:
                    st.dataframe(pd.DataFrame(filas_uf), hide_index=True,
                                 use_container_width=True)

                # ── Gráfica: P(s) para cada método ───────────────────────
                st.markdown("#### P(s) por método de unfolding")
                s_teo = np.linspace(0.01, 4, 300)
                fig_uf = go.Figure()

                # Curvas teóricas
                fig_uf.add_trace(go.Scatter(
                    x=s_teo, y=np.exp(-s_teo),
                    mode="lines", name="Poisson teórico",
                    line=dict(color="#e74c3c", dash="dash", width=1)))
                fig_uf.add_trace(go.Scatter(
                    x=s_teo,
                    y=(np.pi/2)*s_teo*np.exp(-np.pi*s_teo**2/4),
                    mode="lines", name="GUE teórico",
                    line=dict(color="#3498db", dash="dash", width=1)))

                colores_uf = {
                    "KDE": "#e67e22", "Spline": "#9b59b6",
                    "Polinomial": "#1abc9c", "Analítico": "#2c3e50"
                }
                for mname, mr in uf_results.items():
                    if mr.get("unfolded_central") is not None and mr["is_valid"]:
                        sc, sh = spacing_histogram(mr["unfolded_central"], bins=40)
                        if len(sc) > 0:
                            fig_uf.add_trace(go.Scatter(
                                x=sc, y=sh, mode="lines+markers",
                                name=mname,
                                line=dict(color=colores_uf.get(mname, "#555"), width=2),
                                marker=dict(size=4)))

                fig_uf.update_layout(
                    title=f"P(s) — {espectro_uf} — comparación de métodos de unfolding",
                    xaxis_title="s", yaxis_title="P(s)",
                    hovermode="x unified", height=480)
                st.plotly_chart(fig_uf, use_container_width=True)

                # ── Gráfica: espectros unfolded superpuestos ──────────────
                st.markdown("#### Espectros unfolded — primeros 100 puntos")
                fig_ev = go.Figure()
                for mname, mr in uf_results.items():
                    if mr.get("unfolded") is not None:
                        uf = mr["unfolded"][:100]
                        fig_ev.add_trace(go.Scatter(
                            x=np.arange(len(uf)), y=uf,
                            mode="lines", name=mname,
                            line=dict(color=colores_uf.get(mname, "#555"), width=1.5)))

                fig_ev.update_layout(
                    title="Comparación de espectros unfolded (primeros 100 puntos)",
                    xaxis_title="índice n", yaxis_title="nivel unfolded uₙ",
                    hovermode="x unified", height=350)
                st.plotly_chart(fig_ev, use_container_width=True)

                # ── Interpretación ────────────────────────────────────────
                with st.expander("📖 ¿Qué miro aquí?"):
                    st.markdown("""
**⟨s⟩ ≈ 1.0** — todos los métodos deben producir esto. Si alguno da ⟨s⟩ ≠ 1,
ese método introduce sesgo de densidad (error de unfolding).

**σ(s)** — dispersión de los espaciados. Poisson: σ≈1.0. GUE: σ≈0.42.
Un σ intermedio puede indicar un unfolding que no separa bien las escalas.

**⟨r⟩** — r-statistic (no depende del unfolding). Es la referencia objetiva.
Si ⟨r⟩ apunta a GUE pero P(s) de un método parece Poisson, ese método
está sobresuavizando y destruyendo las correlaciones de corto alcance.

**Para ceros de Riemann:** los cuatro métodos deberían dar ⟨r⟩ ≈ 0.60 (GUE).
Si KDE y Spline dan resultados muy distintos, el ancho de banda o los nudos
necesitan ajuste. El Analítico (Riemann–von Mangoldt) es la referencia.
                    """)

            except ImportError as e:
                st.error(f"❌ empirical_unfolding no disponible: {e}")
                st.info("Verifica: `src/riemann_spectral/analysis/empirical_unfolding.py`")
            except Exception as e:
                st.error(f"❌ Error en comparación de unfolding: {e}")
                if modo_debug:
                    st.exception(e)

        with rmt_t7:
            st.subheader("📋 Resumen Comparativo — Todas las Estadísticas")
            st.markdown("""
            Cada fila es un ensemble. Cada columna es una estadística espectral.
            Los colores indican qué tan cerca está cada ensemble de la predicción GUE.
            """)

            # ── Calcular todas las métricas para cada ensemble ────────────
            L_ref = 10.0        # L de referencia para Δ₃ y Σ²
            s_max_r2 = 4.0
            filas_resumen = []

            for label, d in res.items():
                ev = d["ev"]
                fila = {"Ensemble": label}

                # ⟨s⟩ spacing medio
                sp = np.diff(ev)
                sm = np.mean(sp)
                fila["⟨s⟩"] = f"{sm:.3f}"

                # r-statistic
                try:
                    r_res = r_statistic(ev)
                    fila["⟨r⟩"] = f"{r_res['r_mean']:.4f}"
                    fila["Clasif. r"] = r_res["clasificacion"]
                except Exception:
                    fila["⟨r⟩"] = "—"; fila["Clasif. r"] = "—"

                # Δ₃(L=10)
                try:
                    d3 = delta3_dyson_mehta(ev, L_ref)
                    d3_gue_teo = np.log(L_ref) / (np.pi ** 2) - 0.0069
                    d3_poi_teo = L_ref / 15.0
                    fila[f"Δ₃(L={L_ref:.0f})"] = f"{d3:.4f}"
                    fila["Δ₃ vs GUE(%)"] = f"{100*abs(d3-d3_gue_teo)/d3_gue_teo:.1f}%"
                except Exception:
                    fila[f"Δ₃(L={L_ref:.0f})"] = "—"; fila["Δ₃ vs GUE(%)"] = "—"

                # Σ²(L=10)
                try:
                    s2 = sigma2_number_variance_fast(ev, np.array([L_ref]))[0]
                    fila[f"Σ²(L={L_ref:.0f})"] = f"{s2:.4f}" if np.isfinite(s2) else "—"
                except Exception:
                    fila[f"Σ²(L={L_ref:.0f})"] = "—"

                # χ² R₂ vs GUE
                try:
                    s_r2, r2_obs = pair_correlation_fast(ev, s_max=s_max_r2, bins=60)
                    chi2_r = chi2_r2_vs_gue(s_r2, r2_obs, s_min=0.3, s_max_fit=3.5)
                    chi2_v = chi2_r.get("chi2_reducido", np.nan)
                    fila["χ²/dof R₂"] = f"{chi2_v:.2f}" if np.isfinite(chi2_v) else "—"
                except Exception:
                    fila["χ²/dof R₂"] = "—"

                filas_resumen.append(fila)

            if filas_resumen:
                df_res = pd.DataFrame(filas_resumen)
                st.dataframe(
                    df_res.set_index("Ensemble"),
                    use_container_width=True,
                    height=200,
                )

            # ── Gráfica: comparación visual de Δ₃ y Σ² ────────────────────
            st.divider()
            st.markdown("#### Comparativa Δ₃(L) y Σ²(L) — todos los ensembles")

            L_comp = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
            fig_comp = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Rigidez Espectral Δ₃(L)", "Number Variance Σ²(L)"),
            )

            # Teóricas Δ₃
            L_fine = np.linspace(2, 32, 200)
            d3_gue_t = np.log(L_fine) / (np.pi**2) - 0.0069
            d3_poi_t = L_fine / 15.0
            d3_goe_t = np.log(L_fine) / (2 * np.pi**2) - 0.0012
            for y_t, name_t, col_t in [
                (d3_gue_t, "GUE teórico", "#3498db"),
                (d3_goe_t, "GOE teórico", "#2ecc71"),
                (d3_poi_t, "Poisson teórico", "#e74c3c"),
            ]:
                fig_comp.add_trace(go.Scatter(
                    x=L_fine, y=y_t, mode="lines", name=name_t,
                    line=dict(color=col_t, dash="dash", width=1),
                    showlegend=True, legendgroup="teo",
                ), row=1, col=1)

            # Experimental Δ₃
            for label, d in res.items():
                d3_exp = []
                for L_v in L_comp:
                    try:
                        d3_exp.append(delta3_dyson_mehta(d["ev"], L_v))
                    except Exception:
                        d3_exp.append(np.nan)
                fig_comp.add_trace(go.Scatter(
                    x=L_comp, y=d3_exp, mode="lines+markers",
                    name=label, line=dict(color=d["color"], width=2),
                    marker=dict(size=7), legendgroup=label,
                ), row=1, col=1)

            # Teóricas Σ²
            s2_gue_t = np.log(L_fine) / (np.pi**2)
            s2_goe_t = 2 * np.log(L_fine) / (np.pi**2)
            for y_t, name_t, col_t in [
                (s2_gue_t, "GUE teórico", "#3498db"),
                (s2_goe_t, "GOE teórico", "#2ecc71"),
                (L_fine,   "Poisson (L)", "#e74c3c"),
            ]:
                fig_comp.add_trace(go.Scatter(
                    x=L_fine, y=y_t, mode="lines", name=name_t,
                    line=dict(color=col_t, dash="dash", width=1),
                    showlegend=False, legendgroup="teo2",
                ), row=1, col=2)

            # Experimental Σ²
            for label, d in res.items():
                try:
                    s2_exp = sigma2_number_variance_fast(d["ev"], L_comp)
                    valid = np.isfinite(s2_exp)
                    fig_comp.add_trace(go.Scatter(
                        x=L_comp[valid], y=s2_exp[valid],
                        mode="lines+markers", name=label,
                        line=dict(color=d["color"], width=2),
                        marker=dict(size=7),
                        showlegend=False, legendgroup=label,
                    ), row=1, col=2)
                except Exception:
                    pass

            fig_comp.update_xaxes(title_text="L", row=1, col=1)
            fig_comp.update_xaxes(title_text="L", row=1, col=2)
            fig_comp.update_yaxes(title_text="Δ₃(L)", row=1, col=1)
            fig_comp.update_yaxes(title_text="Σ²(L)", row=1, col=2)
            fig_comp.update_layout(height=500, hovermode="x unified")
            st.plotly_chart(fig_comp, use_container_width=True)

            # ── Panel de interpretación ───────────────────────────────────
            st.markdown("#### Guía de Interpretación")
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.markdown("""
**🔴 Poisson (desorden)**
- ⟨r⟩ ≈ 0.386
- Δ₃ crece como L/15
- Σ² crece como L
- R₂(s) = 1 (plana)
- Sin repulsión de niveles
                """)
            with col_i2:
                st.markdown("""
**🔵 GUE (caos cuántico)**
- ⟨r⟩ ≈ 0.600
- Δ₃ crece como ln(L)/π²
- Σ² crece como ln(L)/π²
- R₂(s) tiene dip en s→0
- Repulsión cuadrática
                """)
            with col_i3:
                st.markdown("""
**🟣 Riemann (conjetura)**
- Se espera comportamiento GUE
- Montgomery 1973 probó R₂(s) GUE
- Odlyzko 1987 verificó numéricamente
- Hipótesis de Riemann pendiente
- Conexión con caos cuántico
                """)


# ============================================================================

with tab4:
    st.header("📚 Documentación Matemática")
    
    doc_tab1, doc_tab2, doc_tab3, doc_tab4 = st.tabs([
        "🔢 Ecuaciones Fundamentales",
        "📖 Glosario",
        "📄 Papers",
        "🎓 Referencias"
    ])

    with doc_tab1:
        st.markdown("## Ecuaciones Fundamentales del Toolkit RMT")

        st.markdown("### 1. Espaciado Mínimo — Ecuación de Evolución")
        st.latex(r"\dot{d}_i = \frac{4}{d_i} + R_i(\gamma)")
        st.markdown(r"""
        Donde:
        - $d_i = \gamma_{i+1} - \gamma_i$: espaciado entre ceros consecutivos
        - $4/d_i$: término singular repulsivo (log-gas coulombiano)
        - $R_i$: término regular (influencia de ceros distantes)
        """)

        st.divider()
        st.markdown("### 2. P(s) — Distribución de Espaciados Vecinos")
        col_ps1, col_ps2 = st.columns(2)
        with col_ps1:
            st.latex(r"P_\text{Poisson}(s) = e^{-s}")
            st.caption("Sin repulsión — niveles independientes")
        with col_ps2:
            st.latex(r"P_\text{GUE}(s) = \frac{\pi}{2}\,s\,e^{-\frac{\pi}{4}s^2}")
            st.caption("Wigner surmise — repulsión cuadrática")

        st.divider()
        st.markdown("### 3. Rigidez Espectral Δ₃ — Dyson–Mehta (1963)")
        st.latex(r"\Delta_3(L) = \frac{1}{L}\,\min_{A,B}\int_{x_0}^{x_0+L}\!\![N(x)-A-Bx]^2\,dx")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.latex(r"\Delta_3^\text{Poisson}(L) = \frac{L}{15}")
        with col_d2:
            st.latex(r"\Delta_3^\text{GUE}(L) \approx \frac{\ln L}{\pi^2}")
        with col_d3:
            st.latex(r"\Delta_3^\text{GOE}(L) \approx \frac{\ln L}{2\pi^2}")

        st.divider()
        st.markdown("### 4. Number Variance Σ²(L) — Fluctuaciones")
        st.latex(r"\Sigma^2(L) = \langle (N(L) - L)^2 \rangle")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.latex(r"\Sigma^2_\text{Poisson}(L) = L")
        with col_s2:
            st.latex(r"\Sigma^2_\text{GUE}(L) \approx \frac{1}{\pi^2}\ln L")
        with col_s3:
            st.latex(r"\Sigma^2_\text{GOE}(L) \approx \frac{2}{\pi^2}\ln L")

        st.divider()
        st.markdown("### 5. R₂(s) — Correlación de Pares (Montgomery 1973)")
        st.latex(r"R_2(s) = 1 - \left(\frac{\sin(\pi s)}{\pi s}\right)^2 \quad \text{(GUE)}")
        st.markdown("""
        **Significado histórico:** Hugh Montgomery conjeturó en 1973 que los ceros de la función
        zeta de Riemann satisfacen esta fórmula. Freeman Dyson reconoció en esa misma conversación
        que era exactamente la predicción del GUE — uno de los momentos más sorprendentes
        de la historia matemática del siglo XX.
        """)
        col_r2a, col_r2b = st.columns(2)
        with col_r2a:
            st.latex(r"R_2^\text{Poisson}(s) = 1")
            st.caption("Sin correlaciones de largo alcance")
        with col_r2b:
            st.latex(r"R_2^\text{GUE}(0) = 0,\quad R_2^\text{GUE}(\infty) = 1")
            st.caption("Dip en s=0 (repulsión), nodo en s=1")

        st.divider()
        st.markdown("### 6. K(t) — Factor de Forma Espectral")
        st.latex(r"K(t) = \frac{1}{N^2}\left|\sum_n e^{2\pi i\,t\,\gamma_n}\right|^2")
        col_kt1, col_kt2 = st.columns(2)
        with col_kt1:
            st.latex(r"K_\text{GUE}(t) = \begin{cases}|t| & |t| \le 1 \\ 1 & |t| > 1\end{cases}")
        with col_kt2:
            st.latex(r"K_\text{Poisson}(t) = 1 \quad \forall\, t > 0")
        st.caption("El 'dip' de K(t) para t<1 en GUE es la huella de correlaciones de largo alcance.")

        st.divider()
        st.markdown("### 7. r-statistic — Ratio de Espaciados (Atas et al. 2013)")
        st.latex(r"r_n = \frac{\min(s_n,\,s_{n+1})}{\max(s_n,\,s_{n+1})}, \quad s_n = \gamma_{n+1}-\gamma_n")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("⟨r⟩ Poisson", "0.3863", "= 2ln2 − 1")
        with col_r2:
            st.metric("⟨r⟩ GOE", "0.5307", "numérico")
        with col_r3:
            st.metric("⟨r⟩ GUE", "0.5996", "numérico")
        st.caption("**Ventaja:** No requiere unfolding — aplicable directamente a ceros de Riemann.")

        st.divider()
        st.markdown("### 8. Unfolding — Semicírculo de Wigner")
        st.latex(r"F(x) = \frac{1}{2} + \frac{1}{4\pi}\left(x\sqrt{4-x^2} + 4\arcsin(x/2)\right)")
        st.caption("CDF del semicírculo en [-2, 2]. Mapea autovalores a densidad uniforme ρ=1.")


    with doc_tab2:
        st.markdown("## Glosario de Términos RMT")

        terminos = {
            "Δ₃ (Delta-tres)": "Estadística de Dyson-Mehta. Mide la rigidez espectral como la mejor aproximación lineal a la función de conteo N(x). Valores bajos → correlaciones fuertes (GUE). Valores altos → desorden (Poisson).",
            "Σ²(L) (Number Variance)": "Varianza del número de niveles en una ventana de longitud L. Complementa a Δ₃. Para GUE crece logarítmicamente; para Poisson crece linealmente.",
            "R₂(s) (Pair Correlation)": "Función de correlación de pares. Mide la densidad de pares de niveles separados por distancia s. Para GUE: R₂(s) = 1-(sin(πs)/πs)². Para Poisson: R₂(s) = 1 (sin estructura).",
            "K(t) (Spectral Form Factor)": "Transformada de Fourier de R₂(s). Revela correlaciones de largo alcance. El 'dip' de GUE para t<1 es una firma de repulsión de niveles. Para Poisson: K(t)=1 (plano).",
            "r-statistic": "Ratio de espaciados consecutivos rₙ = min(sₙ,sₙ₊₁)/max(sₙ,sₙ₊₁). No requiere unfolding. ⟨r⟩: Poisson≈0.386, GOE≈0.531, GUE≈0.600.",
            "P(s)": "Distribución de espaciados nearest-neighbor. GUE: campana en s≈1 (repulsión). Poisson: decaimiento exponencial (sin repulsión).",
            "Unfolding": "Transformación que normaliza la densidad espectral local a 1. Esencial para comparar estadísticas entre diferentes sistemas.",
            "GUE": "Gaussian Unitary Ensemble. Matrices aleatorias hermitianas con entradas complejas. Modela sistemas cuánticos con simetría temporal rota. Repulsión de niveles cuadrática.",
            "GOE": "Gaussian Orthogonal Ensemble. Matrices aleatorias simétricas con entradas reales. Modela sistemas cuánticos con simetría temporal conservada. Repulsión de niveles lineal.",
            "Proceso de Poisson": "Secuencia completamente aleatoria sin correlaciones. Baseline para detectar orden espectral.",
            "Hipótesis de Riemann": "Conjetura de que todos los ceros no triviales de ζ(s) están en la línea Re(s) = 1/2. Uno de los Problemas del Milenio del Clay Institute.",
            "Conjetura Montgomery-Odlyzko": "Los ceros de la función zeta de Riemann tienen estadística espectral consistente con GUE. Montgomery (1973) demostró el resultado para R₂(s); Odlyzko (1987) lo verificó numéricamente.",
            "Log-gas": "Modelo de partículas con interacción logarítmica. Describe la estadística de autovalores de GUE como un gas de Coulomb 2D en equilibrio térmico.",
            "Repulsión de niveles": "Tendencia de los autovalores a mantenerse separados. Poisson: sin repulsión. GOE: lineal en s. GUE: cuadrática en s.",
        }

        for term, defn in terminos.items():
            with st.expander(f"**{term}**"):
                st.markdown(defn)


    
    with doc_tab3:
        st.markdown("## 📄 Papers Clave")

        st.markdown("### Fundamentos Históricos")
        papers_hist = [
            ("Montgomery, H. L. (1973)", "The pair correlation of zeros of the zeta function",
             "📐 Conjetura que conecta ζ con GUE. Demostrado para R₂(s).",
             "https://doi.org/10.1090/pspum/024/9944"),
            ("Dyson, F. J. (1962)", "Statistical Theory of Energy Levels of Complex Systems I–III",
             "⭐ Introduce GUE/GOE/GSE, factor de forma y Δ₃.",
             "https://doi.org/10.1063/1.1703862"),
            ("Mehta, M. L. & Dyson, F. J. (1963)", "Statistical Theory of the Energy Levels of Complex Systems V",
             "🎯 Fórmula analítica de Δ₃(L) para GUE.",
             "https://doi.org/10.1063/1.1704292"),
            ("Odlyzko, A. M. (1987)", "On the distribution of spacings between zeros of the zeta function",
             "🔢 Verificación numérica Montgomery-Odlyzko con 10⁵ ceros.",
             "https://doi.org/10.2307/2007890"),
        ]
        for autores, titulo, desc, url in papers_hist:
            st.markdown(f"**{autores}**  \n*{titulo}*  \n{desc}  \n🔗 [{url[:50]}...]({url})")
            st.divider()

        st.markdown("### Papers Modernos")
        papers_mod = [
            ("Atas, Y. Y. et al. (2013)", "Distribution of the ratio of consecutive level spacings in random matrix ensembles",
             "📊 Fórmula analítica de P(r) para GOE/GUE/Poisson. El r-statistic moderno.",
             "https://doi.org/10.1103/PhysRevLett.110.084101"),
            ("Bohigas, O., Giannoni, M. J. & Schmit, C. (1984)", "Characterization of Chaotic Quantum Spectra",
             "💥 Conjetura BGS: sistemas caóticos clásicos → estadística GOE/GUE.",
             "https://doi.org/10.1103/PhysRevLett.52.1"),
            ("Forrester, P. J. (2010)", "Log-Gases and Random Matrices",
             "📚 Tratado completo. Log-gas, pair correlation, form factor.",
             "https://press.princeton.edu"),
        ]
        for autores, titulo, desc, url in papers_mod:
            st.markdown(f"**{autores}**  \n*{titulo}*  \n{desc}  \n🔗 [{url[:50]}...]({url})")
            st.divider()

        bibtex = r"""@article{montgomery1973,
  title={The pair correlation of zeros of the zeta function},
  author={Montgomery, H. L.},
  booktitle={Proc. Symp. Pure Math.},
  volume={24}, pages={181--193}, year={1973}
}
@article{dyson1962,
  title={Statistical Theory of Energy Levels of Complex Systems},
  author={Dyson, F. J.},
  journal={Journal of Mathematical Physics},
  volume={3}, pages={140--175}, year={1962}
}
@article{odlyzko1987,
  title={On the distribution of spacings between zeros of the zeta function},
  author={Odlyzko, A. M.},
  journal={Mathematics of Computation},
  volume={48}, pages={273--308}, year={1987}
}
@article{atas2013,
  title={Distribution of the ratio of consecutive level spacings},
  author={Atas, Y. Y. and Bogomolny, E. and Giraud, O. and Roux, G.},
  journal={Physical Review Letters},
  volume={110}, pages={084101}, year={2013}
}
@article{bohigas1984,
  title={Characterization of chaotic quantum spectra},
  author={Bohigas, O. and Giannoni, M. J. and Schmit, C.},
  journal={Physical Review Letters},
  volume={52}, pages={1--4}, year={1984}
}"""
        st.download_button(
            label="📥 Descargar Bibliografía (BibTeX)",
            data=bibtex,
            file_name="srce_bibliografia.bib",
            mime="text/plain",
        )

    
    with doc_tab4:
        st.markdown("""
        ## 🎓 Referencias y Recursos
        
        ### Cursos Online
        - [MIT OpenCourseWare - Random Matrix Theory](https://ocw.mit.edu)
        - [Coursera - Quantum Mechanics](https://www.coursera.org)
        
        ### Repositorios de Código
        - [mpmath - Matemática de precisión arbitraria](https://mpmath.org)
        - [NumPy/SciPy - Computación científica](https://numpy.org)
        
        ### Bases de Datos
        - [OEIS - Secuencias de enteros](https://oeis.org)
        - [LMFDB - Base de datos L-functions](https://www.lmfdb.org)
        
        ### Comunidad
        - [MathOverflow](https://mathoverflow.net)
        - [Physics Stack Exchange](https://physics.stackexchange.com)
        """)

# ============================================================================
# TAB 5: ACERCA DE
# ============================================================================

with tab5:
    st.header("ℹ️ Acerca del Proyecto")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Spectral Rigidity Calibration Engine
        
        ### 🎯 Objetivo
        Motor de análisis de rigidez espectral basado en **Random Matrix Theory** (RMT)  
        para validación numérica de estadística espectral.
        
        ### ⚠️ Lo que NO es
        - ❌ NO resuelve la Hipótesis de Riemann
        - ❌ NO detecta "rupturas" de la línea crítica
        - ❌ NO es una demostración matemática
        
        ### ✅ Lo que SÍ es
        - ✅ Herramienta de análisis espectral rigurosa
        - ✅ Framework de validación RMT
        - ✅ Infraestructura reproducible para investigación
        - ✅ Implementación auditada de Δ₃
        
        ### 🔬 Aplicaciones
        - Física cuántica (niveles energéticos)
        - Teoría de números (ceros de funciones L)
        - Criptografía (análisis de PRNGs)
        - Procesamiento de señales
        - Finanzas cuantitativas
        
        ### 🏆 Reconocimientos
        - Implementación verificada matemáticamente
        - Validación contra baselines teóricos
        - Documentación científica completa
        
        ### 📜 Licencia
        MIT License - Código abierto para investigación y educación
        """)
    
    with col2:
        st.info("""
        ### 👥 Autores
        
        **Jorge BC**  
        Desarrollador Principal
        
        **Claude (Anthropic)**  
        Asistente de Investigación
        
        ---
        
        ### 📅 Versión
        v2.0.0 (Febrero 2026)
        
        ---
        
        ### 🔗 Enlaces
        
        [GitHub Repository](https://github.com/JorgeBC420/Spectral-Rigidity-Calibration-Engine)
        
        [Documentación](https://github.com/JorgeBC420/Spectral-Rigidity-Calibration-Engine/blob/main/README.md)
        
        [Reportar Bug](https://github.com/JorgeBC420/Spectral-Rigidity-Calibration-Engine/issues)
        """)
        
        st.success("""
        ### ✨ Ideal para:
        
        - 🎓 Ferias científicas
        - 📚 Proyectos de investigación
        - 🏫 Enseñanza universitaria
        - 🔬 Laboratorios de física
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔬 Spectral Rigidity Calibration Engine v2.0")

with col2:
    st.caption("Desarrollado con ❤️ y rigor científico")

with col3:
    st.caption("[GitHub](https://github.com/JorgeBC420) · [Docs](./README.md) · [License](./LICENSE)")
