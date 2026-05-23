"""Carga de espectros externos (CSV, JSON, XLSX, …)."""

from .external_loader import (
    LoadedSpectrum,
    detect_numeric_columns,
    load_spectrum,
    load_spectrum_bytes,
    load_table,
    spectrum_from_dataframe,
)

__all__ = [
    "LoadedSpectrum",
    "detect_numeric_columns",
    "load_spectrum",
    "load_spectrum_bytes",
    "load_table",
    "spectrum_from_dataframe",
]
