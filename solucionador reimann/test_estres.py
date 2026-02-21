"""
Test de estres: Numba (Jacobiano) + ProcessPoolExecutor (GUE batch).
Configurar NUMBA_NUM_THREADS=2 y OMP_NUM_THREADS=2 para que 2 P-cores
llevan el algebra lineal y los E-cores el batch GUE.
Ejecutar desde la raiz: python test_estres.py
"""

import os
import sys

# Configurar antes de importar numpy/numba
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def main():
    from riemann_spectral.data import generar_gue_batch
    from riemann_spectral.analysis import analizar_espectro_completo

    N = 400
    num_realizaciones = 12
    max_workers = 8
    print(f"Test de estres: N={N}, realizaciones={num_realizaciones}, max_workers={max_workers}")
    print("Generando GUE en paralelo...")
    gammas = generar_gue_batch(N, num_realizaciones, seed_base=123, max_workers=max_workers)
    print("Analizando espectro (Jacobiano + eigh) para cada realizacion...")
    for i, gamma in enumerate(gammas):
        analizar_espectro_completo(gamma, f"GUE_{i}")
    print("Test de estres completado.")

if __name__ == "__main__":
    main()
