# -*- coding: utf-8 -*-
"""Certificado reproducible de un candidato/cero en el pipeline zeta."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .levels import AcceptanceLevel


@dataclass
class ZeroCertificate:
    """
    Registro auditable de un intervalo o cero en coordenadas (log_T, dt).

    Los campos T_* y dt_* se guardan como texto para no depender de float(T).
    """

    log_T: float
    T_anchor_str: str
    dt_left: float
    dt_right: float
    dt_refined: Optional[float] = None
    t_im_str: Optional[str] = None

    acceptance_level: AcceptanceLevel = AcceptanceLevel.EXPLORATORIO
    quality_score: Optional[float] = None
    method_backlund: str = "mpmath"
    converged: Optional[bool] = None
    residual: Optional[float] = None
    n_iter: Optional[int] = None
    id_goedel: Optional[str] = None
    goedel_G: Optional[int] = None

    sign_change_phase1: bool = True
    interval_residual_certified: Optional[bool] = None
    backlund_fiable: Optional[bool] = None
    notes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["acceptance_level"] = self.acceptance_level.value
        d["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        return d

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
