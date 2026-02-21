"""
TEST_SUITE_ASSERTIONS.py — Verificación real de Delta_3 (Dyson–Mehta)
=====================================================================

3 assertions científicamente fundamentadas que FALLAN si la implementación
está rota. Ejecutar con:

    cd "solucionador reimann"
    python TEST_SUITE_ASSERTIONS.py

Cada test imprime PASS / FAIL con valores observados vs esperados.
Los umbrales son amplios (tolerancia del 20–30%) para ser robustos ante
varianza estadística con pocas realizaciones, pero suficientemente estrechos
para detectar bugs reales.

Fundamento teórico:
    - Poisson  : Delta_3(L) = L/15            (exacto, derivado analíticamente)
    - GUE      : Delta_3(L) ~ (1/pi^2)*log(L) (límite termodinámico, bulk)
    - Separación: Delta_3_Poisson >> Delta_3_GUE para L suficientemente grande
"""

import sys
import os

# Asegurar que src/ está en el path
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import scipy.linalg as la

from riemann_spectral.analysis.rigidity import delta3_dyson_mehta
from riemann_spectral.analysis.unfolding import unfolding_wigner_gue

# ── Colores para output ──────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

_resultados = []


def _report(nombre: str, passed: bool, detalle: str) -> None:
    estado = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"\n  [{estado}] {nombre}")
    print(f"         {detalle}")
    _resultados.append((nombre, passed))


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — Poisson: Delta_3(L) = L/15
# ════════════════════════════════════════════════════════════════════════════
#
# Para un proceso de Poisson homogéneo de densidad 1, la rigidez de
# Dyson–Mehta es exactamente Delta_3(L) = L/15.
# (Mehta, "Random Matrices", 3ª ed., Cap. 16)
#
# Procedimiento:
#   - Generamos Poisson de densidad 1 via espaciados Exp(1): los ceros son
#     posiciones = cumsum(Exp(1)), por construcción ya unfolded con densidad 1.
#   - Promediamos sobre 80 realizaciones para reducir varianza.
#   - Verificamos en 4 valores de L.
#   - Tolerancia: 25% relativo (la varianza muestral de Delta_3 Poisson ~ L^2/N).
# ════════════════════════════════════════════════════════════════════════════

def test_poisson_delta3_formula():
    print(f"\n{'─'*70}")
    print("TEST 1 — Poisson: Delta_3(L) debe ser ≈ L/15")
    print(f"{'─'*70}")

    LONGITUD    = 3000.0   # rango total del proceso Poisson
    N_REAL      = 80       # realizaciones para promediar
    L_VALORES   = [5.0, 10.0, 20.0, 30.0]
    TOLERANCIA  = 0.25     # 25% relativo máximo admitido

    rng = np.random.default_rng(seed=42)
    resultados_L = {L: [] for L in L_VALORES}

    for _ in range(N_REAL):
        # Poisson densidad 1: posiciones = cumsum de Exp(1)
        n_est = int(LONGITUD * 1.3)
        espaciados = rng.exponential(1.0, size=n_est)
        pos = np.cumsum(espaciados)
        pos = pos[pos <= LONGITUD]

        # Tomar segmento central para evitar efectos de borde
        n = len(pos)
        start = n // 3
        end   = 2 * (n // 3)
        u = pos[start:end] - pos[start]
        if len(u) < max(L_VALORES) + 20:
            continue

        for L in L_VALORES:
            d3 = delta3_dyson_mehta(u, L)
            if np.isfinite(d3):
                resultados_L[L].append(d3)

    todos_ok = True
    for L in L_VALORES:
        vals = resultados_L[L]
        if len(vals) < 10:
            _report(f"  L={L:.0f}", False, f"Datos insuficientes: {len(vals)} realizaciones")
            todos_ok = False
            continue

        observado = float(np.mean(vals))
        teorico   = L / 15.0
        error_rel = abs(observado - teorico) / teorico

        ok = error_rel <= TOLERANCIA
        todos_ok = todos_ok and ok
        print(f"    L={L:5.1f}: Delta_3 obs={observado:.5f}  teórico={teorico:.5f}  "
              f"error_rel={100*error_rel:.1f}%  {'✓' if ok else '✗'}")

        if not ok:
            assert False, (
                f"Test 1 FALLÓ en L={L}: Delta_3={observado:.5f} "
                f"pero teórico={teorico:.5f} (error {100*error_rel:.1f}% > {100*TOLERANCIA:.0f}%)"
            )

    _report(
        "Poisson: Delta_3(L) ≈ L/15",
        todos_ok,
        f"Promedio sobre {N_REAL} realizaciones, L ∈ {L_VALORES}, tolerancia {100*TOLERANCIA:.0f}%",
    )
    return todos_ok


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — GUE: Delta_3(L) < Poisson (separación de ensembles)
# ════════════════════════════════════════════════════════════════════════════
#
# La rigidez de Dyson–Mehta para GUE satisface:
#   Delta_3^GUE(L)  ~  (1/pi^2) * log(L)    (logarítmica, "rígido")
#   Delta_3^Poi(L)  =  L/15                  (lineal, "blando")
#
# Para L suficientemente grande (L >= 10), Poisson debe ser claramente
# mayor que GUE. Esta es la firma más importante de la correlación GUE.
#
# Procedimiento:
#   - GUE: genera H = (A+A†)/(2√N), extrae autovalores, unfolding Wigner,
#     tercio central, Delta_3.
#   - Poisson: mismo tamaño efectivo, Delta_3 directo.
#   - Assertion: media_GUE < media_Poisson con margen del 30%.
# ════════════════════════════════════════════════════════════════════════════

def _generar_gue_unfolded(N: int, seed: int) -> np.ndarray:
    """GUE → unfolding Wigner → tercio central → recentrado."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = (A + A.conj().T) / (2 * np.sqrt(N))
    evals = np.sort(la.eigvalsh(H))
    u = unfolding_wigner_gue(evals)
    n = len(u)
    start = n // 3
    end   = 2 * (n // 3)
    central = u[start:end]
    return central - central[0]


def test_gue_menor_que_poisson():
    print(f"\n{'─'*70}")
    print("TEST 2 — GUE vs Poisson: Delta_3^GUE << Delta_3^Poisson")
    print(f"{'─'*70}")

    N_GUE    = 600   # tamaño de matriz GUE (tercio central ≈ 200 puntos)
    N_REAL   = 40    # realizaciones por ensemble
    L        = 15.0  # ventana donde la separación es clara
    MARGEN   = 0.30  # GUE debe ser al menos 30% menor que Poisson

    rng = np.random.default_rng(seed=99)

    # Baseline GUE
    d3_gue = []
    for i in range(N_REAL):
        u = _generar_gue_unfolded(N_GUE, seed=i)
        if len(u) < L + 10:
            continue
        val = delta3_dyson_mehta(u, L)
        if np.isfinite(val):
            d3_gue.append(val)

    # Baseline Poisson
    d3_poi = []
    LONGITUD_POI = float(N_GUE // 3 + 50)
    for i in range(N_REAL):
        espaciados = rng.exponential(1.0, size=int(LONGITUD_POI * 1.3))
        pos = np.cumsum(espaciados)
        pos = pos[pos <= LONGITUD_POI] - 0.0
        n = len(pos)
        start = n // 3
        end   = 2 * (n // 3)
        u = pos[start:end] - pos[start] if end > start else np.empty(0)
        if len(u) < L + 10:
            continue
        val = delta3_dyson_mehta(u, L)
        if np.isfinite(val):
            d3_poi.append(val)

    mean_gue = float(np.mean(d3_gue)) if d3_gue else np.nan
    mean_poi = float(np.mean(d3_poi)) if d3_poi else np.nan
    teorico_poi = L / 15.0
    teorico_gue = np.log(L) / (np.pi ** 2)

    print(f"    L = {L}")
    print(f"    Delta_3 GUE     obs={mean_gue:.5f}  teórico≈{teorico_gue:.5f}  (log(L)/pi^2)")
    print(f"    Delta_3 Poisson obs={mean_poi:.5f}  teórico={teorico_poi:.5f}  (L/15)")

    ok = (
        np.isfinite(mean_gue)
        and np.isfinite(mean_poi)
        and mean_gue < mean_poi * (1 - MARGEN)
    )

    if not ok:
        assert False, (
            f"Test 2 FALLÓ: Delta_3_GUE={mean_gue:.5f} no es al menos {100*MARGEN:.0f}% "
            f"menor que Delta_3_Poisson={mean_poi:.5f} para L={L}"
        )

    _report(
        f"GUE < Poisson en Delta_3(L={L})",
        ok,
        f"GUE={mean_gue:.4f}  Poisson={mean_poi:.4f}  ratio={mean_gue/mean_poi:.3f} (debe ser < {1-MARGEN:.2f})",
    )
    return ok


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — Monotonía: Delta_3(L) es monótonamente no-decreciente
# ════════════════════════════════════════════════════════════════════════════
#
# Por definición, Delta_3(L) = min_{A,B} (1/L) ∫₀ᴸ (N(x)−A−Bx)² dx
# es una función de variación cuadrática y, como tal, es no-decreciente en L.
# Cualquier implementación que viole esto tiene un bug en la integración.
#
# Procedimiento:
#   - Para un espectro Poisson fijo y suficientemente largo.
#   - Evaluar Delta_3 en una grilla creciente de L.
#   - Verificar que la secuencia es no-decreciente (se tolera 1% de ruido
#     numérico entre pasos adyacentes).
#
# Este test es independiente del ensemble y captura bugs de implementación
# que ningún test estadístico detectaría fácilmente.
# ════════════════════════════════════════════════════════════════════════════

def test_delta3_monotonia():
    print(f"\n{'─'*70}")
    print("TEST 3 — Monotonía: Delta_3(L) debe ser no-decreciente en L")
    print(f"{'─'*70}")

    # Generar Poisson largo, denso y estable
    rng = np.random.default_rng(seed=7)
    LONGITUD = 5000.0
    espaciados = rng.exponential(1.0, size=int(LONGITUD * 1.3))
    pos = np.cumsum(espaciados)
    pos = pos[pos <= LONGITUD]
    n = len(pos)
    start = n // 3
    end   = 2 * (n // 3)
    u = pos[start:end] - pos[start]

    L_grid = np.linspace(3.0, 60.0, 25)
    d3_vals = np.array([delta3_dyson_mehta(u, L) for L in L_grid])

    # Verificar no-decreciente con tolerancia del 1% (ruido numérico)
    TOLERANCIA_RUIDO = 0.01  # caída máxima admisible entre pasos adyacentes

    violaciones = []
    for i in range(1, len(d3_vals)):
        if not np.isfinite(d3_vals[i]) or not np.isfinite(d3_vals[i - 1]):
            continue
        caida = d3_vals[i - 1] - d3_vals[i]
        caida_rel = caida / max(abs(d3_vals[i - 1]), 1e-10)
        if caida_rel > TOLERANCIA_RUIDO:
            violaciones.append(
                f"L={L_grid[i-1]:.1f}→{L_grid[i]:.1f}: "
                f"Delta_3 bajó de {d3_vals[i-1]:.5f} a {d3_vals[i]:.5f} "
                f"({100*caida_rel:.2f}%)"
            )

    print(f"    Rango L: [{L_grid[0]:.1f}, {L_grid[-1]:.1f}], {len(L_grid)} puntos")
    print(f"    Delta_3 min={np.nanmin(d3_vals):.5f}  max={np.nanmax(d3_vals):.5f}")
    print(f"    Violaciones de monotonía: {len(violaciones)}")
    for v in violaciones:
        print(f"      ✗ {v}")

    ok = len(violaciones) == 0

    if not ok:
        assert False, (
            f"Test 3 FALLÓ: Delta_3 no es monótona. "
            f"{len(violaciones)} violaciones encontradas:\n"
            + "\n".join(violaciones)
        )

    _report(
        "Delta_3(L) monótonamente no-decreciente",
        ok,
        f"Verificado en {len(L_grid)} puntos, L ∈ [{L_grid[0]:.1f}, {L_grid[-1]:.1f}], "
        f"tolerancia ruido {100*TOLERANCIA_RUIDO:.0f}%",
    )
    return ok


# ════════════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("TEST SUITE DE ASSERTIONS — Delta_3 (Dyson–Mehta)")
    print("Verificación matemática real con assert. No solo prints.")
    print("=" * 70)

    fallos = []

    for test_fn in [test_poisson_delta3_formula, test_gue_menor_que_poisson, test_delta3_monotonia]:
        try:
            test_fn()
        except AssertionError as e:
            fallos.append(str(e))
        except Exception as e:
            fallos.append(f"ERROR INESPERADO en {test_fn.__name__}: {e}")

    print(f"\n{'═'*70}")
    pasados = sum(1 for _, ok in _resultados if ok)
    total   = len(_resultados)

    if fallos:
        print(f"{RED}RESULTADO: {pasados}/{total} tests pasaron.{RESET}")
        print(f"\nFallos detallados:")
        for f in fallos:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print(f"{GREEN}RESULTADO: {pasados}/{total} tests pasaron. ✓{RESET}")
        sys.exit(0)
