"""Cadena de acondicionamiento digital del sEMG.

Orden: pasabanda → notch (red) → rectificación → envolvente / RMS móvil.
Todos los filtros son de fase cero (``sosfiltfilt`` / ``filtfilt``) para no
desplazar temporalmente los eventos de activación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def _limitar_a_nyquist(f_hz: float, fs: float, margen: float = 0.95) -> float:
    return min(f_hz, margen * fs / 2)


def pasabanda(x: np.ndarray, fs: float, f_baja: float = 20.0, f_alta: float = 450.0, orden: int = 4) -> np.ndarray:
    f_alta = _limitar_a_nyquist(f_alta, fs)
    if not 0 < f_baja < f_alta:
        raise ValueError(f"Banda inválida: {f_baja}–{f_alta} Hz con fs = {fs:.0f} Hz")
    sos = signal.butter(orden, [f_baja, f_alta], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def notch(x: np.ndarray, fs: float, f0: float = 60.0, q: float = 30.0, armonicos: int = 3) -> np.ndarray:
    """Elimina la interferencia de red y sus armónicos por debajo de Nyquist."""
    y = np.asarray(x, dtype=float)
    for k in range(1, armonicos + 1):
        fk = k * f0
        if fk >= fs / 2:
            break
        b, a = signal.iirnotch(fk, q, fs=fs)
        y = signal.filtfilt(b, a, y, axis=0)
    return y


def envolvente(x: np.ndarray, fs: float, fc: float = 8.0, orden: int = 4) -> np.ndarray:
    """Envolvente lineal: rectificación de onda completa + pasabajas."""
    sos = signal.butter(orden, fc, btype="lowpass", fs=fs, output="sos")
    return np.clip(signal.sosfiltfilt(sos, np.abs(x), axis=0), 0, None)


def rms_movil(x: np.ndarray, fs: float, ventana_s: float = 0.2) -> np.ndarray:
    n = max(1, int(round(ventana_s * fs)))
    nucleo = np.ones(n) / n
    return np.sqrt(np.convolve(np.asarray(x, dtype=float) ** 2, nucleo, mode="same"))


def procesar(
    df: pd.DataFrame,
    fs: float,
    f_baja: float = 20.0,
    f_alta: float = 450.0,
    usar_notch: bool = True,
    f_red: float = 60.0,
    fc_env: float = 8.0,
    ventana_rms_s: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """Aplica la cadena completa a todos los canales de ``df``.

    Devuelve un diccionario con las etapas ``cruda``, ``filtrada``,
    ``envolvente`` y ``rms``; cada DataFrame conserva ``tiempo_s``.
    El DataFrame de entrada no se modifica.
    """
    canales = [c for c in df.columns if c != "tiempo_s"]
    t = df["tiempo_s"].to_numpy()
    salida = {etapa: {"tiempo_s": t} for etapa in ("filtrada", "envolvente", "rms")}
    for c in canales:
        x = df[c].to_numpy(dtype=float)
        y = pasabanda(x, fs, f_baja, f_alta)
        if usar_notch:
            y = notch(y, fs, f0=f_red)
        salida["filtrada"][c] = y
        salida["envolvente"][c] = envolvente(y, fs, fc_env)
        salida["rms"][c] = rms_movil(y, fs, ventana_rms_s)
    resultado = {etapa: pd.DataFrame(cols) for etapa, cols in salida.items()}
    resultado["cruda"] = df.copy()
    return resultado
