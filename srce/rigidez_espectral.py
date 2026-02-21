"""
Wrapper de compatibilidad: redirige a riemann_spectral.
La logica vive en src/riemann_spectral/. Ejecutar desde la raiz del proyecto.
"""

import sys
import os
import logging

# Anadir src al path para imports del paquete
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Re-exportar desde el paquete
from riemann_spectral.data.generators import (
    generar_uniforme,
    generar_gue_normalizado,
    generar_poisson,
)
from riemann_spectral.analysis.unfolding import unfolding_riemann
from riemann_spectral.analysis.spectral import (
    calcular_jacobiano_kernel,
    energia_log_gas,
    analizar_espectro_completo,
    analizar_modo_blando,
    clasificar_modo_blando,
)
from riemann_spectral.engine.protocolo_rigidez import (
    ejecutar_protocolo_escalamiento,
    ajustar_exponente_critico,
    visualizar_protocolo_completo,
    ejecutar_analisis_completo,
)

__all__ = [
    "generar_uniforme",
    "generar_gue_normalizado",
    "generar_poisson",
    "unfolding_riemann",
    "calcular_jacobiano_kernel",
    "energia_log_gas",
    "analizar_espectro_completo",
    "analizar_modo_blando",
    "clasificar_modo_blando",
    "ejecutar_protocolo_escalamiento",
    "ajustar_exponente_critico",
    "visualizar_protocolo_completo",
    "ejecutar_analisis_completo",
]
