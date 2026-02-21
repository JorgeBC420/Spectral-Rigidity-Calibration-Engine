"""
Protocolo de escalamiento y visualizacion: Riemann vs Uniforme vs GUE vs Poisson.
Depende de analysis.spectral y data.generators.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from ..data.generators import generar_uniforme, generar_gue_normalizado, generar_poisson
from ..analysis.spectral import analizar_espectro_completo, analizar_modo_blando


def ejecutar_protocolo_escalamiento(
    N_values: List[int],
    cache_obtener: Optional[Callable[[int], np.ndarray]] = None,
    num_realizaciones_gue: int = 5,
) -> Dict[str, Any]:
    """Ejecuta el protocolo de rigidez para cada N; retorna espectra y modos blandos."""
    resultados = {
        "riemann": [],
        "uniforme": [],
        "gue": [],
        "poisson": [],
    }
    analisis_modo_blando_out = {
        "riemann": [],
        "uniforme": [],
        "gue": [],
        "poisson": [],
    }
    for N in N_values:
        if cache_obtener is not None:
            gamma_r = cache_obtener(N)
            res_r = analizar_espectro_completo(gamma_r, "Riemann")
            resultados["riemann"].append(res_r)
            analisis_modo_blando_out["riemann"].append(analizar_modo_blando(res_r))
        gamma_u = generar_uniforme(N)
        res_u = analizar_espectro_completo(gamma_u, "Uniforme")
        resultados["uniforme"].append(res_u)
        analisis_modo_blando_out["uniforme"].append(analizar_modo_blando(res_u))
        gaps_gue = []
        for r in range(num_realizaciones_gue):
            gamma_g = generar_gue_normalizado(N)
            res_g = analizar_espectro_completo(gamma_g, f"GUE r={r}")
            gaps_gue.append(res_g["gap"])
        resultados["gue"].append({
            "N": N,
            "gap": float(np.mean(gaps_gue)),
            "gap_std": float(np.std(gaps_gue)),
        })
        gamma_p = generar_poisson(N)
        res_p = analizar_espectro_completo(gamma_p, "Poisson")
        resultados["poisson"].append(res_p)
        analisis_modo_blando_out["poisson"].append(analizar_modo_blando(res_p))
    return {
        "espectra": resultados,
        "modos_blandos": analisis_modo_blando_out,
        "N_values": N_values,
    }


def ajustar_exponente_critico(
    N_values: np.ndarray,
    gaps: np.ndarray,
) -> Dict[str, Any]:
    """Ajuste gap ~ C * N^(-alpha) en log-log. Retorna exponente, prefactor, R2."""
    mask = (gaps > 0) & (np.isfinite(gaps))
    N_valid = N_values[mask]
    gaps_valid = gaps[mask]
    if len(N_valid) < 2:
        return {"error": "Datos insuficientes"}
    log_N = np.log(N_valid)
    log_gap = np.log(gaps_valid)
    p = np.polyfit(log_N, log_gap, 1)
    alpha = -p[0]
    C = np.exp(p[1])
    y_pred = p[0] * log_N + p[1]
    SS_res = np.sum((log_gap - y_pred) ** 2)
    SS_tot = np.sum((log_gap - np.mean(log_gap)) ** 2)
    R2 = 1.0 - (SS_res / SS_tot)
    return {"exponente": alpha, "prefactor": C, "R2": R2, "polinomio": p}


def visualizar_protocolo_completo(datos: Dict[str, Any], save_path: str = "rigidez_espectral_protocolo.png") -> None:
    """Graficos: Gap vs N (lineal y log-log), energia vs N, modos blandos, localizacion."""
    resultados = datos["espectra"]
    modos = datos["modos_blandos"]
    N_values = np.array(datos["N_values"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Gap vs N (log-log)
    ax = axes[0, 0]
    for sistema in ["riemann", "uniforme", "gue", "poisson"]:
        if sistema not in resultados or len(resultados[sistema]) == 0:
            continue
        if sistema == "gue":
            gaps = np.array([r["gap"] for r in resultados[sistema]])
        else:
            gaps = np.array([r["gap"] for r in resultados[sistema]])
        ax.loglog(N_values, gaps, "o-", label=sistema.capitalize())
    ax.set_xlabel("N")
    ax.set_ylabel("Gap")
    ax.set_title("Gap vs N (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Energia vs N
    ax = axes[0, 1]
    for sistema in ["riemann", "uniforme", "poisson"]:
        if sistema not in resultados or len(resultados[sistema]) == 0:
            continue
        energias = [r["energia"] for r in resultados[sistema]]
        ax.plot(N_values, energias, "o-", label=sistema.capitalize())
    ax.set_xlabel("N")
    ax.set_ylabel("E")
    ax.set_title("Energia vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Localizacion
    ax = axes[1, 0]
    for sistema in ["riemann", "uniforme", "poisson"]:
        if sistema not in modos or len(modos[sistema]) == 0:
            continue
        locs = [m["localizacion_index"] for m in modos[sistema]]
        ax.plot(N_values, locs, "o-", label=sistema.capitalize())
    ax.axhline(0.3, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("N")
    ax.set_ylabel("Indice localizacion")
    ax.set_title("Modo blando: localizacion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Correlacion sinusoidal
    ax = axes[1, 1]
    for sistema in ["riemann", "uniforme", "poisson"]:
        if sistema not in modos or len(modos[sistema]) == 0:
            continue
        corrs = [m["correlacion_sinusoidal"] for m in modos[sistema]]
        ax.plot(N_values, corrs, "o-", label=sistema.capitalize())
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("N")
    ax.set_ylabel("Corr sin(pi i/N)")
    ax.set_title("Modo blando: periodicidad")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.suptitle("Protocolo Rigidez Espectral")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def ejecutar_analisis_completo(
    cache_obtener: Callable[[int], np.ndarray],
    N_values: Optional[List[int]] = None,
    verbose: bool = True,
    save_path: str = "rigidez_espectral_protocolo.png",
) -> Dict[str, Any]:
    """Ejecuta protocolo y visualiza. N_values por defecto [100, 200, 500, 1000, 2000]."""
    if N_values is None:
        N_values = [100, 200, 500, 1000, 2000]
    datos = ejecutar_protocolo_escalamiento(N_values, cache_obtener=cache_obtener)
    visualizar_protocolo_completo(datos, save_path=save_path)
    return datos
