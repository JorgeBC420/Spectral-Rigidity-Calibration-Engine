# -*- coding: utf-8 -*-
"""Smoke: carga externa y pipeline RMT."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riemann_spectral.analytics.rmt_pipeline import run_rmt_audit, DISCLAIMER_ES
from riemann_spectral.io.external_loader import (
    detect_numeric_columns,
    load_spectrum,
    load_spectrum_bytes,
    spectrum_from_dataframe,
)


def _gue_like(n: int = 80, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    w = (w + w.conj().T) / 2
    return np.sort(np.linalg.eigvalsh(w.real))


def test_detect_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert detect_numeric_columns(df) == ["a"]


def test_load_spectrum_csv(tmp_path):
    p = tmp_path / "levels.csv"
    pd.DataFrame({"gamma": [1.0, 2.5, 3.1, 4.0]}).to_csv(p, index=False)
    spec = load_spectrum(p, column="gamma")
    assert len(spec.values) == 4
    assert spec.values[0] == 1.0


def test_load_spectrum_bytes():
    csv = b"E\n1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n7.0\n8.0\n9.0\n10.0\n"
    csv += b"11.0\n12.0\n13.0\n14.0\n15.0\n16.0\n17.0\n18.0\n19.0\n20.0\n"
    spec = load_spectrum_bytes(csv, "data.csv")
    assert len(spec.values) >= 20


def test_run_rmt_audit_gue_like():
    levels = _gue_like(100)
    res = run_rmt_audit(levels, unfolding="polynomial", seed=0)
    assert res.n_levels == 100
    assert 0.0 < res.classifier_confidence_pct <= 100.0
    assert res.extra.get("disclaimer") == DISCLAIMER_ES


def test_spectrum_from_dataframe_normalize():
    df = pd.DataFrame({"x": [2.0, 4.0, 6.0] * 10})
    spec = spectrum_from_dataframe(df, column="x", normalize=True)
    assert np.isclose(np.mean(spec.values), 1.0, rtol=0.05)
