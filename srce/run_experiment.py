"""
Script de experimento usando el framework riemann_spectral.
Ejemplo: cargar ceros, evaluar Z-scores vs GUE/Poisson, registrar en bitácora.
Ejecutar desde la raíz del proyecto: python run_experiment.py
"""

import sys
import os

# Añadir raíz y src al path para imports
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

def main():
    from riemann_spectral.data import CacheZeros, generar_gue_normalizado, generar_poisson
    from riemann_spectral.analysis import unfolding_riemann, espaciado_minimo, varianza_numero, delta3_dyson_mehta
    from riemann_spectral.engine import BaselineFactory, ZScoreEngine
    from riemann_spectral.storage import Bitacora

    print("Framework de Experimentación Espectral - Experimento de ejemplo")
    print("=" * 60)

    # Datos: ceros (requiere mpmath)
    try:
        cache = CacheZeros()
        N = 200
        gamma = cache.obtener(N)
        print(f"Ceros Riemann: N = {N}")
    except Exception as e:
        print(f"Sin mpmath/ceros, usando GUE de prueba: {e}")
        gamma = generar_gue_normalizado(200, seed=42)
        N = 200

    # Unfolding y métricas
    u = unfolding_riemann(gamma)
    d_min, idx = espaciado_minimo(u)
    var_n = varianza_numero(u, L=2.0)
    d3 = delta3_dyson_mehta(u, L=2.0)
    print(f"  d_min = {d_min:.6e}, varianza_numero(L=2) = {var_n:.6f}, Delta_3(2) = {d3:.6f}")

    # Baselines y Z-scores (menos realizaciones para ejemplo rápido)
    factory = BaselineFactory(num_realizaciones_gue=20, num_realizaciones_poisson=20, seed=123)
    engine = ZScoreEngine(baseline_factory=factory, sigma_umbral=5.0)
    resultados = engine.evaluar(gamma, N)
    print("\nZ-Scores vs baselines empíricos (GUE / Poisson):")
    for met, r in resultados.items():
        print(f"  {met}: valor={r['valor']:.6f}, z_gue={r['z_gue']:.2f}, z_poisson={r['z_poisson']:.2f}, anomalía={r['anomalia']}")

    # Bitácora
    bitacora = Bitacora(db_path=os.path.join(ROOT, "bitacora_riemann.db"))
    if engine.hay_anomalia(resultados):
        for met, r in resultados.items():
            if r.get("anomalia"):
                bitacora.registrar(
                    tipo="anomalia_zscore",
                    N=N,
                    metrica=met,
                    valor=r["valor"],
                    z_score=max(abs(r["z_gue"]), abs(r["z_poisson"])),
                    baseline="GUE/Poisson",
                    extra={"z_gue": r["z_gue"], "z_poisson": r["z_poisson"]},
                )
        print("\nHallazgos registrados en bitácora.")
    else:
        bitacora.registrar(tipo="experimento_ok", N=N, extra=resultados)
        print("\nSin anomalías 5-sigma; registro de experimento OK en bitácora.")

    path_json = bitacora.exportar_json(os.path.join(ROOT, "bitacora_export.json"))
    print(f"Export JSON: {path_json}")
    print("=" * 60)

if __name__ == "__main__":
    main()
