# -*- coding: utf-8 -*-
"""Smoke tests: certificación, alturas mpf y bitácora JSONL."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mpmath as mp
import pytest

from riemann_spectral.certification import (
    AcceptanceLevel,
    CertificateBitacora,
    ZeroCertificate,
    FLOAT_SAFE_LOG_T,
    im_float_if_safe,
    t_im_from_offset,
    window_im_mpf,
)


def test_acceptance_level_values():
    assert AcceptanceLevel.EXPLORATORIO.value == "exploratorio"
    assert AcceptanceLevel.CERTIFICADO.value == "certificado"


def test_window_im_mpf_no_float_anchor():
    """A T=10^70 los offsets pequeños no alteran mpf con dps finito; probamos rango amplio."""
    with mp.workdps(80):
        T = mp.power(10, 70)
        t0, t1 = window_im_mpf(T, 0.0, 500.0, 80)
    assert t1 > t0
    f0, safe = im_float_if_safe(t0, 70.0, 80)
    assert safe is False
    assert isinstance(f0, float)


def test_float_safe_at_log3():
    with mp.workdps(40):
        T = mp.power(10, 3)
        t0, _ = window_im_mpf(T, 0.0, 0.5, 40)
    val, safe = im_float_if_safe(t0, 3.0, 40)
    assert safe is True
    assert val >= 1000.0


def test_certificate_jsonl_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "certificates.jsonl"
        bit = CertificateBitacora(path)
        cert = ZeroCertificate(
            log_T=3.0,
            T_anchor_str="1000.0",
            dt_left=0.1,
            dt_right=0.2,
            acceptance_level=AcceptanceLevel.SEMI_RIGUROSO,
            quality_score=0.85,
        )
        bit.append(cert)
        bit.append_run_summary({"n_aceptados": 1})
        rows = CertificateBitacora.read_all(path)
        assert len(rows) == 2
        assert rows[0]["acceptance_level"] == "semi_riguroso"
        assert rows[1]["record_type"] == "run_summary"


def test_t_im_from_offset_matches_mpf():
    with mp.workdps(30):
        T = mp.mpf("1000.5")
        t = t_im_from_offset(T, 0.25, 30)
    assert abs(float(t) - 1000.75) < 1e-9


def test_float_safe_constant():
    assert FLOAT_SAFE_LOG_T == 12.0
