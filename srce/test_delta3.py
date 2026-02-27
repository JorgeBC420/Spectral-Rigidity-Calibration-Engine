# -*- coding: utf-8 -*-
"""
tests/test_delta3.py
====================
Suite pytest para Delta_3, EnsembleClassifier y ZScoreEngine.
Incluye tests para los nuevos campos estadísticos de ZScoreEngine v1.2.

Ejecutar:
    cd srce
    pytest tests/ -v
    pytest tests/ -v -m integration
    pytest tests/ -v -m "not slow"
"""

import os
import sys

import numpy as np
import pytest
import scipy.linalg as la

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from riemann_spectral.analysis.rigidity  import delta3_dyson_mehta
from riemann_spectral.analysis.unfolding import unfolding_wigner_gue
from riemann_spectral.engine.ensemble_classifier import (
    EnsembleClassifier,
    PENDIENTE_GUE,
    PENDIENTE_GOE,
    ENSEMBLE_POISSON,
    ENSEMBLE_GUE,
    ENSEMBLE_GOE,
)


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — Poisson: Delta_3(L) ≈ L/15
# ════════════════════════════════════════════════════════════════════════════

class TestPoissonDelta3:

    TOLERANCIA = 0.25

    @pytest.mark.parametrize("L", [5.0, 10.0, 20.0, 30.0])
    def test_formula_L_sobre_15(self, L, poisson_ensemble_promedio):
        d3_mean, L_grid = poisson_ensemble_promedio
        idx = np.where(L_grid == L)[0]
        assert len(idx) == 1
        observado = float(d3_mean[idx[0]])
        teorico   = L / 15.0
        assert np.isfinite(observado), f"Delta_3 NaN para L={L}"
        error_rel = abs(observado - teorico) / teorico
        assert error_rel <= self.TOLERANCIA, (
            f"L={L}: obs={observado:.5f} teórico={teorico:.5f} "
            f"error={100*error_rel:.1f}% > {100*self.TOLERANCIA:.0f}%"
        )

    def test_monotonia(self, poisson_unfolded):
        L_grid = np.linspace(3.0, 50.0, 20)
        d3     = np.array([delta3_dyson_mehta(poisson_unfolded, L) for L in L_grid])
        d3_ok  = d3[np.isfinite(d3)]
        diffs  = np.diff(d3_ok)
        violaciones = int(np.sum(diffs < -0.02 * np.abs(d3_ok[:-1])))
        assert violaciones == 0, f"Delta_3 no es monótona: {violaciones} decrementos"

    def test_todos_finitos(self, poisson_unfolded):
        L_grid    = np.linspace(2.0, 40.0, 15)
        d3        = [delta3_dyson_mehta(poisson_unfolded, L) for L in L_grid]
        nan_count = sum(1 for v in d3 if not np.isfinite(v))
        assert nan_count == 0, f"{nan_count}/{len(d3)} valores NaN/Inf"


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — Separación de ensembles
# ════════════════════════════════════════════════════════════════════════════

class TestSeparacionEnsembles:

    @pytest.mark.parametrize("L", [10.0, 20.0])
    def test_gue_menor_que_poisson(self, L, gue_unfolded, poisson_unfolded):
        d3_gue = delta3_dyson_mehta(gue_unfolded,     L)
        d3_poi = delta3_dyson_mehta(poisson_unfolded, L)
        assert np.isfinite(d3_gue) and np.isfinite(d3_poi)
        assert d3_gue < d3_poi * 0.70, (
            f"L={L}: GUE={d3_gue:.5f} no es 30% menor que Poisson={d3_poi:.5f}"
        )

    @pytest.mark.parametrize("L", [10.0, 20.0])
    def test_goe_entre_gue_y_poisson(self, L, goe_unfolded, gue_unfolded, poisson_unfolded):
        d3_gue = delta3_dyson_mehta(gue_unfolded,     L)
        d3_goe = delta3_dyson_mehta(goe_unfolded,     L)
        d3_poi = delta3_dyson_mehta(poisson_unfolded, L)
        assert np.isfinite(d3_goe)
        assert d3_goe > d3_gue * 0.80, f"GOE={d3_goe:.5f} no supera GUE={d3_gue:.5f}"
        assert d3_goe < d3_poi * 0.90, f"GOE={d3_goe:.5f} no está bajo Poisson={d3_poi:.5f}"


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — EnsembleClassifier: pendientes GOE y GUE
# ════════════════════════════════════════════════════════════════════════════

class TestEnsembleClassifier:

    TOLERANCIA = 0.35

    @pytest.fixture(scope="class")
    def clf(self):
        return EnsembleClassifier(L_min=5.0, L_max=25.0, n_puntos=15)

    def test_pendiente_gue_teorica(self, clf, gue_unfolded):
        res = clf.clasificar(gue_unfolded, label="GUE")
        err = abs(res.pendiente_obs - PENDIENTE_GUE) / PENDIENTE_GUE
        assert err <= self.TOLERANCIA, (
            f"Pendiente GUE obs={res.pendiente_obs:.5f} teórico={PENDIENTE_GUE:.5f} "
            f"error={100*err:.1f}%"
        )

    def test_pendiente_goe_teorica(self, clf, goe_unfolded):
        res = clf.clasificar(goe_unfolded, label="GOE")
        err = abs(res.pendiente_obs - PENDIENTE_GOE) / PENDIENTE_GOE
        assert err <= self.TOLERANCIA, (
            f"Pendiente GOE obs={res.pendiente_obs:.5f} teórico={PENDIENTE_GOE:.5f} "
            f"error={100*err:.1f}%"
        )

    def test_pendiente_goe_es_mitad_gue(self, clf, goe_unfolded, gue_unfolded):
        ratio = (
            clf.clasificar(goe_unfolded).pendiente_obs
            / clf.clasificar(gue_unfolded).pendiente_obs
        )
        assert 0.30 <= ratio <= 0.80, f"Ratio GOE/GUE={ratio:.3f}, esperado ≈ 0.50"

    def test_clasifica_poisson(self, clf, poisson_unfolded):
        res = clf.clasificar(poisson_unfolded, label="Poisson")
        assert res.ensemble == ENSEMBLE_POISSON

    def test_scores_goe_menor_para_goe(self, clf, goe_unfolded):
        res = clf.clasificar(goe_unfolded)
        assert res.scores[ENSEMBLE_GOE] < res.scores[ENSEMBLE_GUE]


# ════════════════════════════════════════════════════════════════════════════
# TEST 4 — Constantes teóricas (smoke)
# ════════════════════════════════════════════════════════════════════════════

class TestConstantesTeoricaas:

    def test_pendiente_gue(self):
        assert abs(PENDIENTE_GUE - 1.0 / np.pi**2) < 1e-10

    def test_pendiente_goe(self):
        assert abs(PENDIENTE_GOE - 1.0 / (2 * np.pi**2)) < 1e-10

    def test_goe_mitad_gue(self):
        assert abs(PENDIENTE_GOE - PENDIENTE_GUE / 2) < 1e-10

    def test_clasificador_parametros(self):
        clf = EnsembleClassifier(L_min=3.0, L_max=20.0, n_puntos=10)
        assert clf.L_min < clf.L_max
        assert clf.n_puntos >= 4


# ════════════════════════════════════════════════════════════════════════════
# TEST 5 — Generators: parámetro rng (v1.1)
# ════════════════════════════════════════════════════════════════════════════

class TestGeneradoresRng:
    """Verifica que generar_gue_normalizado y generar_poisson aceptan rng."""

    def test_gue_con_rng_explicito(self):
        from riemann_spectral.data.generators import generar_gue_normalizado
        rng  = np.random.default_rng(seed=42)
        evals = generar_gue_normalizado(50, rng=rng)
        assert len(evals) == 50
        assert np.all(np.isfinite(evals))
        assert np.all(np.diff(evals) >= 0), "Autovalores no están ordenados"

    def test_gue_reproducible_con_seed(self):
        from riemann_spectral.data.generators import generar_gue_normalizado
        e1 = generar_gue_normalizado(50, seed=99)
        e2 = generar_gue_normalizado(50, seed=99)
        assert np.allclose(e1, e2), "seed no produce resultados reproducibles"

    def test_gue_reproducible_con_rng(self):
        from riemann_spectral.data.generators import generar_gue_normalizado
        e1 = generar_gue_normalizado(50, rng=np.random.default_rng(7))
        e2 = generar_gue_normalizado(50, rng=np.random.default_rng(7))
        assert np.allclose(e1, e2), "rng no produce resultados reproducibles"

    def test_gue_rng_tiene_prioridad_sobre_seed(self):
        from riemann_spectral.data.generators import generar_gue_normalizado
        rng = np.random.default_rng(seed=1)
        e_rng  = generar_gue_normalizado(50, rng=rng, seed=999)
        e_seed = generar_gue_normalizado(50, seed=1)
        # rng=default_rng(1) y seed=1 producen mismo resultado
        assert np.allclose(e_rng, e_seed), "rng no tiene prioridad sobre seed"

    def test_poisson_con_rng(self):
        from riemann_spectral.data.generators import generar_poisson
        rng    = np.random.default_rng(seed=5)
        puntos = generar_poisson(100, rng=rng)
        assert len(puntos) == 100
        assert np.all(np.isfinite(puntos))
        assert np.all(np.diff(puntos) >= 0), "Puntos no están ordenados"

    def test_poisson_reproducible(self):
        from riemann_spectral.data.generators import generar_poisson
        p1 = generar_poisson(100, seed=77)
        p2 = generar_poisson(100, seed=77)
        assert np.allclose(p1, p2)

    def test_goe_disponible(self):
        from riemann_spectral.data.generators import generar_goe_normalizado
        rng   = np.random.default_rng(seed=3)
        evals = generar_goe_normalizado(50, rng=rng)
        assert len(evals) == 50
        assert np.all(np.isfinite(evals))


# ════════════════════════════════════════════════════════════════════════════
# TEST 6 — ZScoreEngine: nuevos campos estadísticos (v1.2)
# ════════════════════════════════════════════════════════════════════════════

class TestZScoreEngineNuevosCampos:
    """
    Verifica que evaluar() devuelve los nuevos campos de v1.2:
        p_gue, p_poisson, ic_gue, ic_poisson, confianza_gue, confianza_poi.
    """

    # Campos que deben estar en TODA respuesta válida
    CAMPOS_REQUERIDOS = {
        "valor", "z_gue", "z_poisson", "anomalia",
        "mean_gue", "std_gue", "mean_poisson", "std_poisson",
        "p_gue", "p_poisson", "ic_gue", "ic_poisson",
        "confianza_gue", "confianza_poi",
    }

    @pytest.fixture(scope="class")
    def engine_y_ceros(self):
        """ZScoreEngine con baseline pequeño (CI rápido) + ceros sintéticos."""
        from riemann_spectral.engine.baseline_factory import BaselineFactory
        from riemann_spectral.engine.zscore_engine    import ZScoreEngine

        factory = BaselineFactory(
            num_realizaciones_gue     = 40,
            num_realizaciones_poisson = 40,
            L_delta3 = 2.0,
            seed     = 42,
        )
        engine = ZScoreEngine(baseline_factory=factory, sigma_umbral=5.0, L_delta3=2.0)

        # Ceros de Riemann sintéticos (aproximación simple)
        rng    = np.random.default_rng(seed=7)
        N      = 80
        # T_n ≈ 2*pi*(n + n/log(n)) — ceros sintéticos con densidad correcta
        n_arr  = np.arange(1, N + 1, dtype=float)
        gamma  = 2 * np.pi * n_arr / np.log(n_arr + 2) * (1 + rng.normal(0, 0.05, N))
        gamma  = np.sort(np.abs(gamma)) + 14.0

        return engine, gamma, N

    def test_todos_los_campos_presentes(self, engine_y_ceros):
        engine, gamma, N = engine_y_ceros
        resultados = engine.evaluar(gamma, N, metricas=["delta3"])
        r = resultados["delta3"]
        faltantes = self.CAMPOS_REQUERIDOS - set(r.keys())
        assert not faltantes, f"Faltan campos: {faltantes}"

    def test_p_valores_en_rango(self, engine_y_ceros):
        engine, gamma, N = engine_y_ceros
        resultados = engine.evaluar(gamma, N, metricas=["delta3", "d_min"])
        for met, r in resultados.items():
            if np.isfinite(r["p_gue"]):
                assert 0.0 <= r["p_gue"] <= 1.0, \
                    f"p_gue={r['p_gue']} fuera de [0,1] para métrica {met}"
            if np.isfinite(r["p_poisson"]):
                assert 0.0 <= r["p_poisson"] <= 1.0, \
                    f"p_poisson={r['p_poisson']} fuera de [0,1] para métrica {met}"

    def test_ic_es_tupla_ordenada(self, engine_y_ceros):
        engine, gamma, N = engine_y_ceros
        resultados = engine.evaluar(gamma, N, metricas=["delta3"])
        r = resultados["delta3"]
        lo_gue, hi_gue = r["ic_gue"]
        lo_poi, hi_poi = r["ic_poisson"]
        if np.isfinite(lo_gue) and np.isfinite(hi_gue):
            assert lo_gue <= hi_gue, f"ic_gue invertido: {lo_gue} > {hi_gue}"
        if np.isfinite(lo_poi) and np.isfinite(hi_poi):
            assert lo_poi <= hi_poi, f"ic_poisson invertido: {lo_poi} > {hi_poi}"

    def test_etiquetas_de_confianza_validas(self, engine_y_ceros):
        engine, gamma, N = engine_y_ceros
        resultados = engine.evaluar(gamma, N)
        valores_validos = {"alto", "medio", "bajo"}
        for met, r in resultados.items():
            assert r["confianza_gue"] in valores_validos, \
                f"confianza_gue='{r['confianza_gue']}' inválido para {met}"
            assert r["confianza_poi"] in valores_validos, \
                f"confianza_poi='{r['confianza_poi']}' inválido para {met}"

    def test_confianza_consistente_con_p_valor(self, engine_y_ceros):
        """La etiqueta debe ser coherente con el p-valor."""
        from riemann_spectral.engine.zscore_engine import _clasificar_confianza
        engine, gamma, N = engine_y_ceros
        resultados = engine.evaluar(gamma, N, metricas=["delta3"])
        r = resultados["delta3"]
        if np.isfinite(r["p_gue"]):
            esperado = _clasificar_confianza(r["p_gue"])
            assert r["confianza_gue"] == esperado, (
                f"confianza_gue={r['confianza_gue']} pero p_gue={r['p_gue']:.4f} "
                f"→ debería ser '{esperado}'"
            )

    def test_valor_gue_dentro_de_ic_propio(self, engine_y_ceros):
        """La media del baseline GUE debe estar dentro de su propio IC 95%."""
        engine, gamma, N = engine_y_ceros
        # Usamos un valor que sabemos que está en el centro del baseline:
        # la media del baseline. Creamos un "gamma falso" que produzca ese valor.
        # Forma más simple: verificar que ic_gue contiene mean_gue.
        resultados = engine.evaluar(gamma, N, metricas=["delta3"])
        r = resultados["delta3"]
        lo, hi = r["ic_gue"]
        if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(r["mean_gue"]):
            assert lo <= r["mean_gue"] <= hi, (
                f"mean_gue={r['mean_gue']:.5f} fuera de IC [{lo:.5f}, {hi:.5f}]"
            )

    def test_p_valor_helper_formula(self):
        """Tests unitarios de _p_valor_dos_colas independientes del engine."""
        from riemann_spectral.engine.zscore_engine import _p_valor_dos_colas

        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Valor en la media → p=1.0 (nadie es más extremo)
        p_media = _p_valor_dos_colas(3.0, baseline)
        assert p_media == 1.0, f"p para media={p_media}, esperado 1.0"

        # Valor muy extremo → p pequeño
        p_extrem = _p_valor_dos_colas(1000.0, baseline)
        assert p_extrem < 0.5, f"p para extremo={p_extrem}, esperado < 0.5"

        # Baseline vacío → p=1.0
        p_vacio = _p_valor_dos_colas(5.0, np.array([]))
        assert p_vacio == 1.0

    def test_clasificar_confianza_umbrales(self):
        """Verifica que los umbrales de clasificación son exactamente 0.01 y 0.05."""
        from riemann_spectral.engine.zscore_engine import _clasificar_confianza
        assert _clasificar_confianza(0.005) == "alto"
        assert _clasificar_confianza(0.01)  == "medio"   # 0.01 no es < 0.01
        assert _clasificar_confianza(0.03)  == "medio"
        assert _clasificar_confianza(0.05)  == "bajo"    # 0.05 no es < 0.05
        assert _clasificar_confianza(0.10)  == "bajo"
        assert _clasificar_confianza(1.00)  == "bajo"


# ════════════════════════════════════════════════════════════════════════════
# TEST 7 — Integración: ZScoreEngine end-to-end
# ════════════════════════════════════════════════════════════════════════════

class TestZScoreEngineIntegracion:
    """
    Test end-to-end del pipeline: ceros sintéticos → ZScoreEngine → coherencia.
    """

    N_CEROS       = 120
    N_REALIZACIONES = 30

    @pytest.fixture(scope="class")
    def zscore_engine(self):
        from riemann_spectral.engine.baseline_factory import BaselineFactory
        from riemann_spectral.engine.zscore_engine    import ZScoreEngine
        factory = BaselineFactory(
            num_realizaciones_gue     = self.N_REALIZACIONES,
            num_realizaciones_poisson = self.N_REALIZACIONES,
            L_delta3 = 2.0, seed = 42,
        )
        return ZScoreEngine(baseline_factory=factory, sigma_umbral=5.0, L_delta3=2.0)

    @pytest.fixture(scope="class")
    def ceros_gue_sinteticos(self):
        rng  = np.random.default_rng(seed=123)
        N    = self.N_CEROS
        u    = np.cumsum(rng.exponential(1.0, N))  # Poisson primero
        T    = _invertir_unfolding(u + 50.0)
        return np.sort(T)

    @pytest.fixture(scope="class")
    def ceros_poisson_sinteticos(self):
        rng  = np.random.default_rng(seed=456)
        N    = self.N_CEROS
        u    = np.cumsum(rng.exponential(1.0, N))
        T    = _invertir_unfolding(u + 50.0)
        return np.sort(T)

    @pytest.mark.integration
    def test_z_scores_finitos(self, zscore_engine, ceros_gue_sinteticos):
        r = zscore_engine.evaluar(ceros_gue_sinteticos, N=self.N_CEROS)
        for met, d in r.items():
            assert np.isfinite(d["valor"]),    f"valor NaN en {met}"
            assert np.isfinite(d["z_gue"]),    f"z_gue NaN en {met}"
            assert np.isfinite(d["z_poisson"]), f"z_poisson NaN en {met}"

    @pytest.mark.integration
    def test_estructura_completa(self, zscore_engine, ceros_gue_sinteticos):
        claves = {
            "valor", "z_gue", "z_poisson", "anomalia",
            "p_gue", "p_poisson", "ic_gue", "ic_poisson",
            "confianza_gue", "confianza_poi",
        }
        r = zscore_engine.evaluar(ceros_gue_sinteticos, N=self.N_CEROS)
        for met, d in r.items():
            assert claves <= set(d.keys()), f"Faltan campos en {met}: {claves - set(d.keys())}"

    @pytest.mark.integration
    def test_gue_no_es_anomalia(self, zscore_engine, ceros_gue_sinteticos):
        r = zscore_engine.evaluar(ceros_gue_sinteticos, N=self.N_CEROS)
        assert not zscore_engine.hay_anomalia(r), \
            "Espectro GUE-like disparó anomalía 5-sigma"


def _invertir_unfolding(u: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """Inversa aproximada de unfolding_riemann via Newton."""
    T = 2 * np.pi * (u + 1.0)
    for _ in range(n_iter):
        log_arg   = np.maximum(T / (2 * np.pi), 1e-10)
        u_actual  = (T / (2 * np.pi)) * (np.log(log_arg) - 1.0)
        u_deriv   = np.log(log_arg) / (2 * np.pi)
        u_deriv   = np.where(np.abs(u_deriv) < 1e-12, 1e-12, u_deriv)
        T         = T - (u_actual - u) / u_deriv
        T         = np.maximum(T, 14.0)
    return T
