"""
Validacion rigurosa de delta3_dyson_mehta (auditoria RMT).
- Test Poisson: espectro Poisson densidad 1 -> Delta_3(L) = L/15 (teorico).
- Test GUE: autovalores raw, unfolding por CDF Wigner, tercio central -> pendiente 1/pi^2.
Sin factor empirico; depuracion algebraica si no coincide.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import scipy.linalg as la


def generar_poisson_densidad_1(longitud: float, seed: int = None) -> np.ndarray:
    """
    Proceso de Poisson con densidad 1: posiciones = cumsum(Exp(1)).
    Espectro unfolded por construccion (espaciado medio 1).
    """
    if seed is not None:
        np.random.seed(seed)
    n_esperado = int(longitud * 1.2)
    espaciados = np.random.exponential(1.0, size=n_esperado)
    pos = np.cumsum(espaciados)
    return pos[pos <= longitud]


def generar_gue_raw(N: int, seed: int = None) -> np.ndarray:
    """GUE en escala tipica: H = (A+A')/(2*sqrt(N)), autovalores en [-2,2] (bulk)."""
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    H = (A + A.conj().T) / (2 * np.sqrt(N))
    return np.sort(la.eigvalsh(H))


def unfolding_wigner(evals_raw: np.ndarray) -> np.ndarray:
    """Unfolding por CDF del semicirculo de Wigner. evals_raw en [-2,2]."""
    x = np.asarray(evals_raw, dtype=np.float64)
    n = len(x)
    x_clip = np.clip(x / 2.0, -1.0, 1.0)
    sqrt_term = np.sqrt(np.maximum(4.0 - x * x, 0.0))
    F = 0.5 + (1.0 / (4 * np.pi)) * (x * sqrt_term + 4 * np.arcsin(x_clip))
    return n * F


def tercio_central(u: np.ndarray) -> np.ndarray:
    """Indices [n//3, 2*n//3); re-centrar a 0 para ventanas."""
    n = len(u)
    start = n // 3
    end = 2 * (n // 3)
    if end <= start:
        return np.empty(0)
    return u[start:end] - u[start]


def test_poisson(num_realizaciones: int = 50, longitud: float = 2000.0, L_grid=None):
    """Test Poisson: Delta_3(L) teorico = L/15. Sin factor 0.25."""
    from riemann_spectral.analysis.rigidity import delta3_dyson_mehta

    if L_grid is None:
        L_grid = np.linspace(5.0, 50.0, 18)
    resultados = []
    for r in range(num_realizaciones):
        pos = generar_poisson_densidad_1(longitud, seed=42 + r)
        if len(pos) < L_grid.max() + 50:
            continue
        # Tomar segmento central
        start = len(pos) // 3
        end = 2 * (len(pos) // 3)
        u = pos[start:end] - pos[start]
        if len(u) < L_grid.max() + 10:
            continue
        d3 = [delta3_dyson_mehta(u, L) for L in L_grid]
        resultados.append(d3)
    if not resultados:
        return None, None, L_grid
    mean_d3 = np.nanmean(resultados, axis=0)
    return mean_d3, L_grid, None


def test_gue(N: int = 2000, num_realizaciones: int = 50, L_grid=None):
    """Test GUE: raw evals -> unfolding Wigner -> tercio central -> pendiente 1/pi^2."""
    from riemann_spectral.analysis.rigidity import delta3_dyson_mehta

    if L_grid is None:
        L_grid = np.linspace(5.0, 50.0, 18)
    delta3_por_L = []
    for r in range(num_realizaciones):
        evals = generar_gue_raw(N, seed=42 + r)
        u_full = unfolding_wigner(evals)
        u = tercio_central(u_full)
        if len(u) < L_grid.max() + 10:
            continue
        d3 = [delta3_dyson_mehta(u, L) for L in L_grid]
        delta3_por_L.append(d3)
    if not delta3_por_L:
        return None, None, L_grid
    mean_d3 = np.nanmean(delta3_por_L, axis=0)
    std_d3 = np.nanstd(delta3_por_L, axis=0)
    return mean_d3, std_d3, L_grid


def main():
    from riemann_spectral.analysis.rigidity import delta3_dyson_mehta

    L_grid = np.linspace(5.0, 50.0, 18)
    print("=" * 60)
    print("VALIDACION Delta_3 SIN FACTOR EMPIRICO")
    print("=" * 60)

    # --- Test Poisson (obligatorio): teorico Delta_3 = L/15
    print("\n[1] Test Poisson (densidad 1, 50 realizaciones)")
    mean_poi, _, _ = test_poisson(50, 2000.0, L_grid)
    if mean_poi is not None:
        # Coeficiente observado: mean_d3/L deberia ser 1/15
        ratio_poi = mean_poi / L_grid
        teorico_poi = 1.0 / 15.0
        error_poi = np.abs(ratio_poi - teorico_poi) / (teorico_poi + 1e-12)
        print(f"    Delta_3(L)/L observado (media L): {np.mean(ratio_poi):.6f}")
        print(f"    Teorico 1/15 = {teorico_poi:.6f}")
        print(f"    Ratio observado/teorico: {np.mean(ratio_poi)/teorico_poi:.4f}")
        if np.mean(ratio_poi) > 3.5 * teorico_poi:
            print("    -> Resultado ~4*(L/15): error en integral/ventana (factor 4 global).")
        elif np.abs(np.mean(ratio_poi) - teorico_poi) / teorico_poi < 0.15:
            print("    -> Poisson OK: integral y convencion correctas.")
        else:
            print("    -> Desviacion: revisar algebra de la escalera.")
    else:
        print("    No se pudieron generar realizaciones.")

    # --- Test GUE (unfolding Wigner, tercio central)
    print("\n[2] Test GUE (N=2000, unfolding Wigner, tercio central)")
    mean_gue, std_gue, _ = test_gue(2000, 50, L_grid)
    if mean_gue is not None:
        log_L = np.log(L_grid)
        A = np.column_stack([log_L, np.ones_like(log_L)])
        a_est, b_est = np.linalg.lstsq(A, mean_gue, rcond=None)[0]
        a_teorico = 1.0 / (np.pi ** 2)
        err_rel = abs(a_est - a_teorico) / a_teorico
        print(f"    Pendiente estimada a = {a_est:.6f}")
        print(f"    Teorico 1/pi^2 = {a_teorico:.6f}")
        print(f"    Error relativo = {err_rel*100:.2f}%")
        if err_rel < 0.05:
            print("    -> GUE OK.")
        else:
            print("    -> Revisar unfolding o integral.")
    else:
        print("    No se pudo completar GUE.")

    print("\n" + "=" * 60)
    return {"mean_poisson": mean_poi, "mean_gue": mean_gue, "L_grid": L_grid}


if __name__ == "__main__":
    main()
