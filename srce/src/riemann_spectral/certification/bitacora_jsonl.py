# -*- coding: utf-8 -*-
"""Bitácora append-only certificates.jsonl para auditoría externa."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .zero_certificate import ZeroCertificate


class CertificateBitacora:
    """Escribe certificados en JSONL (una línea por evento)."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, cert: ZeroCertificate) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(cert.to_json_line() + "\n")

    def append_many(self, certs: Iterable[ZeroCertificate]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for c in certs:
                f.write(c.to_json_line() + "\n")

    def append_run_summary(self, summary: Dict[str, Any]) -> None:
        row = {"record_type": "run_summary", **summary}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def read_all(path: Union[str, Path], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        out: List[Dict[str, Any]] = []
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
