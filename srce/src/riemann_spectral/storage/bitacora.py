"""
Bitácora estructurada para persistir hallazgos de experimentos.
Soporta SQLite (consultas, trazabilidad) y export JSON.
"""

import json
import sqlite3
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class Bitacora:
    """
    Persistencia de hallazgos: SQLite + volcado JSON.
    Tabla principal: hallazgos (id, timestamp, tipo, N, métrica, valor, z_score, baseline, extra JSON).
    """

    def __init__(self, db_path: str = "bitacora_riemann.db", json_dir: Optional[str] = None):
        self.db_path = db_path
        self.json_dir = json_dir or os.path.dirname(db_path)
        os.makedirs(self.json_dir, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hallazgos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    tipo TEXT NOT NULL,
                    N INTEGER,
                    metrica TEXT,
                    valor REAL,
                    z_score REAL,
                    baseline TEXT,
                    extra TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hallazgos_tipo ON hallazgos(tipo)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hallazgos_N ON hallazgos(N)
            """)

    def registrar(
        self,
        tipo: str,
        N: Optional[int] = None,
        metrica: Optional[str] = None,
        valor: Optional[float] = None,
        z_score: Optional[float] = None,
        baseline: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Inserta un hallazgo y devuelve el id."""
        ts = datetime.utcnow().isoformat() + "Z"
        extra_str = json.dumps(extra) if extra is not None else None
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO hallazgos (timestamp, tipo, N, metrica, valor, z_score, baseline, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, tipo, N, metrica, valor, z_score, baseline, extra_str),
            )
            return cur.lastrowid or 0

    def listar(
        self,
        tipo: Optional[str] = None,
        N_min: Optional[int] = None,
        N_max: Optional[int] = None,
        limite: int = 500,
    ) -> List[Dict[str, Any]]:
        """Lista hallazgos con filtros opcionales."""
        q = "SELECT id, timestamp, tipo, N, metrica, valor, z_score, baseline, extra FROM hallazgos WHERE 1=1"
        params: List[Any] = []
        if tipo is not None:
            q += " AND tipo = ?"
            params.append(tipo)
        if N_min is not None:
            q += " AND N >= ?"
            params.append(N_min)
        if N_max is not None:
            q += " AND N <= ?"
            params.append(N_max)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limite)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(q, params).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            if d.get("extra"):
                try:
                    d["extra"] = json.loads(d["extra"])
                except Exception:
                    pass
            out.append(d)
        return out

    def exportar_json(self, path: Optional[str] = None) -> str:
        """Exporta todos los hallazgos a un archivo JSON."""
        path = path or os.path.join(self.json_dir, "bitacora_export.json")
        datos = self.listar(limite=10000)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        return path
