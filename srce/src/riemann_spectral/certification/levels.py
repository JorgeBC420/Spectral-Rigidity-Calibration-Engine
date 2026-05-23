# -*- coding: utf-8 -*-
"""Niveles de aceptación / rigor para resultados SRCE."""

from __future__ import annotations

from enum import Enum


class AcceptanceLevel(str, Enum):
    """
    Etiqueta explícita del rigor de un resultado.

    Evita confundir score operativo (Fase 3) con certificación formal Arb.
    """

    EXPLORATORIO = "exploratorio"
    """Heurística, fase aproximada, o sin validación completa."""

    SEMI_RIGUROSO = "semi_riguroso"
    """mpmath / cotas locales; sin ball arithmetic certificada."""

    CERTIFICADO = "certificado"
    """Arb/python-flint o conteo con es_exacto y radios acotados."""

    DIAGNOSTICO = "diagnostico"
    """Solo conteo Backlund / Gram; sin cero refinado aceptado."""

    def label_es(self) -> str:
        return {
            AcceptanceLevel.EXPLORATORIO: "Exploratorio",
            AcceptanceLevel.SEMI_RIGUROSO: "Semi-riguroso",
            AcceptanceLevel.CERTIFICADO: "Certificado (Arb)",
            AcceptanceLevel.DIAGNOSTICO: "Diagnóstico",
        }[self]
