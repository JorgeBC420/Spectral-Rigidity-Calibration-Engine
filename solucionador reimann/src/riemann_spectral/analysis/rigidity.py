"""
Métricas de rigidez espectral: espaciado mínimo, varianza del número, Delta_3 (Dyson–Mehta).
Funciones núcleo con Numba para rendimiento.
"""

from typing import Tuple

import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True)
def calcular_espaciados(gamma: np.ndarray) -> np.ndarray:
    """Espaciados d_i = gamma_{i+1} - gamma_i."""
    N = len(gamma)
    out = np.empty(N - 1)
    for i in range(N - 1):
        out[i] = gamma[i + 1] - gamma[i]
    return out


@jit(nopython=True, fastmath=True)
def espaciado_minimo(gamma: np.ndarray) -> Tuple[float, int]:
    """Devuelve (d_min, índice del espaciado mínimo)."""
    N = len(gamma)
    if N < 2:
        return np.nan, -1
    d_min = gamma[1] - gamma[0]
    idx = 0
    for i in range(1, N - 1):
        d = gamma[i + 1] - gamma[i]
        if d < d_min:
            d_min = d
            idx = i
    return d_min, idx


@jit(nopython=True, fastmath=True)
def varianza_numero_impl(
    gamma_unfolded: np.ndarray,
    L: float,
) -> float:
    """
    Varianza del número de niveles en un intervalo de longitud L (unfolded).
    Sigma^2(L) = <n^2> - <n>^2 con n = conteo en [x, x+L]; promediado sobre x.
    Para espectro unfolded, <n> = L. Mide rigidez: GUE ~ log(L), Poisson ~ L.
    """
    u = gamma_unfolded
    n = len(u)
    if n < 2 or L <= 0:
        return 0.0
    # Rango de barrido: desde 0 hasta u_max - L
    u_min = u[0]
    u_max = u[-1]
    span = u_max - u_min - L
    if span <= 0:
        return 0.0
    # Muestreo de ventanas (aproximación discreta)
    n_win = min(200, max(20, n // 5))
    xs = np.linspace(u_min, u_max - L, n_win)
    counts = np.empty(n_win)
    for i in range(n_win):
        x = xs[i]
        # Contar cuántos u_j están en [x, x+L]
        c = 0
        for j in range(n):
            if u[j] >= x and u[j] < x + L:
                c += 1
        counts[i] = c
    mean_n = np.mean(counts)
    var_n = np.mean(counts ** 2) - mean_n ** 2
    return float(var_n)


def varianza_numero(gamma_unfolded: np.ndarray, L: float) -> float:
    """Varianza del número (interfaz con chequeos)."""
    if not np.all(np.isfinite(gamma_unfolded)) or len(gamma_unfolded) < 2:
        return np.nan
    return varianza_numero_impl(gamma_unfolded, L)


@jit(nopython=True, fastmath=False)
def _delta3_recta(y: np.ndarray, L: float) -> float:
    """
    Delta_3(L): promedio sobre ventanas [y_i, y_i+L] de
    (1/L) * min_{A,B} int_0^L (N(x) - A - B*x)^2 dx.
    N(x) = escalera: numero de niveles en la ventana con posicion <= x (salto 1 por nivel).
    A y B por minimos cuadrados (ecuaciones normales). fastmath=False para RMT.
    """
    n = len(y)
    if n < 2 or L <= 0:
        return 0.0
    subset = np.empty(n)
    sum_d3 = 0.0
    count_win = 0
    for start in range(n):
        y0 = y[start]
        y1 = y0 + L
        k = 0
        for i in range(n):
            if y[i] >= y0 and y[i] <= y1:
                subset[k] = y[i] - y0
                k += 1
        if k < 2:
            continue
        # Ordenar posiciones en [0, L]
        xs = np.sort(subset[:k])
        # Integrales de la escalera N(x): N=0 en [0,x1), 1 en [x1,x2), ..., k en [xk,L]
        # I1 = int N dx = k*L - sum_j x_j
        # I2 = int x*N dx = (1/2)*(k*L^2 - sum_j x_j^2)
        # I3 = int N^2 dx = k^2*L - sum_j (2j-1)*x_j
        sum_x = 0.0
        sum_x2 = 0.0
        sum_jx = 0.0
        for j in range(k):
            sum_x += xs[j]
            sum_x2 += xs[j] * xs[j]
            sum_jx += (2 * (j + 1) - 1) * xs[j]
        I1 = k * L - sum_x
        I2 = 0.5 * (k * L * L - sum_x2)
        I3 = k * k * L - sum_jx
        # Normal equations: A*L + B*L^2/2 = I1,  A*L^2/2 + B*L^3/3 = I2
        # B = (2*I2 - I1*L) * 6 / L^3  =>  B = 12*I2/L^3 - 6*I1/L^2
        # A = (I1 - B*L^2/2) / L
        L2 = L * L
        L3 = L2 * L
        B = (12.0 * I2 / L3) - (6.0 * I1 / L2)
        A = (I1 - B * L2 * 0.5) / L
        # int (N - A - B*x)^2 = I3 - 2*A*I1 - 2*B*I2 + A^2*L + A*B*L^2 + B^2*L^3/3
        integral = (
            I3
            - 2.0 * A * I1
            - 2.0 * B * I2
            + A * A * L
            + A * B * L2
            + B * B * L3 / 3.0
        )
        if integral < 0.0:
            integral = 0.0
        sum_d3 += integral / L
        count_win += 1
    if count_win == 0:
        return 0.0
    return sum_d3 / count_win


def delta3_dyson_mehta(gamma_unfolded: np.ndarray, L: float) -> float:
    """
    Rigidez de Dyson–Mehta Delta_3(L).
    Mide desviación cuadrática media del conteo de niveles respecto al ajuste lineal.
    GUE: Delta_3(L) ~ (1/pi^2) log(L); Poisson: Delta_3(L) = L/15.
    """
    if not np.all(np.isfinite(gamma_unfolded)) or len(gamma_unfolded) < 2 or L <= 0:
        return np.nan
    return _delta3_recta(gamma_unfolded, L)


@jit(nopython=True, fastmath=True)
def ecuacion_espaciado_minimo_correcta(gamma: np.ndarray, idx: int) -> Tuple[float, float, float]:
    """
    Ecuacion del espaciado: d_i, termino_singular (4/d_i), termino_regular R_i.
    Para d_i = gamma_{i+1} - gamma_i, velocidad d_dot = 4/d_i + R_i.
    Retorna (d_i, termino_singular, termino_regular).
    """
    i = idx
    N = len(gamma)
    d_i = gamma[i + 1] - gamma[i]
    termino_singular = 4.0 / d_i
    R_i = 0.0
    for j in range(N):
        if j != i + 1 and j != i:
            R_i += 2.0 / (gamma[i + 1] - gamma[j])
    for j in range(N):
        if j != i and j != i + 1:
            R_i -= 2.0 / (gamma[i] - gamma[j])
    return d_i, termino_singular, R_i


@jit(nopython=True, fastmath=True)
def descomponer_termino_regular(
    gamma: np.ndarray,
    idx: int,
    radio_cercano: int = 10,
    radio_medio: int = 100,
) -> Tuple[float, float, float, float]:
    """
    Descompone R_i por rango de distancia: cercanos, medios, lejanos.
    Retorna (R_cercano, R_medio, R_lejano, R_total).
    """
    i = idx
    N = len(gamma)
    R_cercano = 0.0
    R_medio = 0.0
    R_lejano = 0.0
    for j in range(N):
        if j == i or j == i + 1:
            continue
        distancia = abs(j - i)
        contribucion_ip1 = 2.0 / (gamma[i + 1] - gamma[j])
        contribucion_i = 2.0 / (gamma[i] - gamma[j])
        contribucion = contribucion_ip1 - contribucion_i
        if distancia < radio_cercano:
            R_cercano += contribucion
        elif distancia < radio_medio:
            R_medio += contribucion
        else:
            R_lejano += contribucion
    R_total = R_cercano + R_medio + R_lejano
    return R_cercano, R_medio, R_lejano, R_total
