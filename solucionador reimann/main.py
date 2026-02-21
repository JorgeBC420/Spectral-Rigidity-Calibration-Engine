"""
Orquestador principal del framework de experimentacion espectral.
Importa todo desde riemann_spectral. Ejecutar desde la raiz con PYTHONPATH=src.

Uso:
  set PYTHONPATH=src
  python main.py

O desde la raiz (main.py anade src al path):
  python main.py
"""

import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Check import-safe: todos los modulos del framework
def _check_imports():
    from riemann_spectral import data, analysis, storage, engine
    from riemann_spectral.data import CacheZeros, generar_uniforme, generar_gue_normalizado, generar_poisson
    from riemann_spectral.analysis import (
        unfolding_riemann,
        N_T_approx,
        calcular_espaciados,
        espaciado_minimo,
        varianza_numero,
        delta3_dyson_mehta,
        ecuacion_espaciado_minimo_correcta,
        descomponer_termino_regular,
        calcular_jacobiano_kernel,
        energia_log_gas,
        analizar_espectro_completo,
        analizar_modo_blando,
        clasificar_modo_blando,
    )
    from riemann_spectral.storage import Bitacora
    from riemann_spectral.engine import (
        BaselineFactory,
        ZScoreEngine,
        ejecutar_protocolo_escalamiento,
        ejecutar_analisis_completo,
    )
    return True


if __name__ == "__main__":
    print("Riemann Spectral Framework - Orquestador")
    print("=" * 50)
    try:
        _check_imports()
        print("OK: Imports correctos (entorno import-safe)")
    except Exception as e:
        print("ERROR en imports:", e)
        sys.exit(1)
    print()
    print("Para ejecutar experimento completo:")
    print("  python run_experiment.py   # Z-scores + bitacora")
    print("  python solucionador_reimann.py   # Log-gas + flujo + protocolo rigidez")
    print("=" * 50)
