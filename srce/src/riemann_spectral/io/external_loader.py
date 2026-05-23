# -*- coding: utf-8 -*-
"""
Carga de espectros / niveles desde archivos externos (CSV, TXT, JSON, XLSX).

Sin lógica Streamlit — reutilizable desde dashboard, CLI y tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class LoadedSpectrum:
    """Resultado de carga con metadatos de limpieza."""

    values: np.ndarray
    column_name: str
    source_path: str
    n_dropped_nan: int = 0
    normalized: bool = False
    numeric_columns: List[str] = field(default_factory=list)


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Columnas con al menos un valor numérico parseable."""
    cols: List[str] = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 3:
            cols.append(str(c))
    return cols


def load_table(path: Union[str, Path]) -> pd.DataFrame:
    """Lee CSV, TSV, JSON (array o records) o XLSX."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf in (".csv", ".tsv", ".txt"):
        sep = "\t" if suf == ".tsv" else ","
        return pd.read_csv(p, sep=sep)
    if suf == ".json":
        return pd.read_json(p)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p)
    raise ValueError(f"Formato no soportado: {suf}")


def spectrum_from_dataframe(
    df: pd.DataFrame,
    column: Optional[str] = None,
    sort_values: bool = True,
    drop_nan: bool = True,
    normalize: bool = False,
    source_label: str = "upload",
) -> LoadedSpectrum:
    """Extrae niveles 1D desde un DataFrame ya cargado (p. ej. Streamlit upload)."""
    numeric_cols = detect_numeric_columns(df)
    if not numeric_cols:
        raise ValueError("No se encontraron columnas numéricas.")

    col = column or numeric_cols[0]
    if col not in df.columns:
        raise ValueError(f"Columna '{col}' no existe. Disponibles: {numeric_cols}")

    s = pd.to_numeric(df[col], errors="coerce")
    n_before = len(s)
    if drop_nan:
        s = s.dropna()
    arr = s.to_numpy(dtype=float)
    if sort_values:
        arr = np.sort(arr)
    if normalize and len(arr) > 0 and np.mean(arr) > 0:
        arr = arr / np.mean(arr)

    return LoadedSpectrum(
        values=arr,
        column_name=col,
        source_path=source_label,
        n_dropped_nan=n_before - len(arr),
        normalized=normalize,
        numeric_columns=numeric_cols,
    )


def load_spectrum_bytes(
    data: bytes,
    filename: str,
    column: Optional[str] = None,
    sort_values: bool = True,
    drop_nan: bool = True,
    normalize: bool = False,
) -> LoadedSpectrum:
    """Carga desde bytes (drag-and-drop en dashboard)."""
    suf = Path(filename).suffix.lower()
    bio = BytesIO(data)
    if suf in (".csv", ".tsv", ".txt"):
        sep = "\t" if suf == ".tsv" else ","
        df = pd.read_csv(bio, sep=sep)
    elif suf == ".json":
        df = pd.read_json(bio)
    elif suf in (".xlsx", ".xls"):
        df = pd.read_excel(bio)
    else:
        raise ValueError(f"Formato no soportado: {suf}")
    return spectrum_from_dataframe(
        df,
        column=column,
        sort_values=sort_values,
        drop_nan=drop_nan,
        normalize=normalize,
        source_label=filename,
    )


def load_spectrum(
    path: Union[str, Path],
    column: Optional[str] = None,
    sort_values: bool = True,
    drop_nan: bool = True,
    normalize: bool = False,
) -> LoadedSpectrum:
    """
    Carga una columna numérica como array 1D de niveles (p. ej. γ_n o eigenvalues).

    Args:
        path: ruta al archivo.
        column: nombre de columna; None = primera columna numérica detectada.
        sort_values: ordenar ascendente (convención espectral).
        drop_nan: eliminar NaN.
        normalize: dividir por media (opcional, solo escala).
    """
    df = load_table(path)
    return spectrum_from_dataframe(
        df,
        column=column,
        sort_values=sort_values,
        drop_nan=drop_nan,
        normalize=normalize,
        source_label=str(path),
    )
