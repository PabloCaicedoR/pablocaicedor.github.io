"""Lectura y validación de archivos CSV de sEMG."""

from __future__ import annotations

from typing import IO

import numpy as np
import pandas as pd


class ErrorFormato(ValueError):
    """El archivo no cumple el contrato mínimo de columnas."""


def leer_csv(fuente: str | IO[bytes], columna_tiempo: str = "tiempo_s") -> tuple[pd.DataFrame, float, int]:
    """Lee un CSV ancho (una columna de tiempo + un canal por columna).

    Devuelve el DataFrame, la frecuencia de muestreo estimada y el número de
    valores no numéricos que obligaron a descartar filas.
    No modifica el archivo original: la limpieza ocurre sobre la copia en memoria.
    """
    df = pd.read_csv(fuente)
    if columna_tiempo not in df.columns:
        raise ErrorFormato(
            f"Falta la columna '{columna_tiempo}'. Columnas encontradas: {list(df.columns)[:8]}"
        )
    canales = [c for c in df.columns if c != columna_tiempo]
    if not canales:
        raise ErrorFormato("El archivo no contiene canales además del tiempo.")

    df = df[[columna_tiempo, *canales]].apply(pd.to_numeric, errors="coerce")
    n_faltantes = int(df.isna().sum().sum())
    if n_faltantes:
        # Decisión conservadora: no imputar; se informa y se descartan filas.
        df = df.dropna().reset_index(drop=True)

    dt = np.diff(df[columna_tiempo].to_numpy())
    if np.any(dt <= 0):
        raise ErrorFormato("El tiempo no es estrictamente creciente.")
    fs = float(1.0 / np.median(dt))
    df = df.rename(columns={columna_tiempo: "tiempo_s"})
    return df, fs, n_faltantes
