"""Métricas temporales, espectrales y de coordinación para sEMG."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


# ---------- dominio del tiempo ----------------------------------------------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def mav(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


# ---------- dominio de la frecuencia ----------------------------------------

def espectro(x: np.ndarray, fs: float, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Densidad espectral de potencia por Welch (ventana de Hann, 50 % traslape)."""
    nperseg = int(min(nperseg, len(x)))
    return signal.welch(x, fs=fs, nperseg=nperseg)


def _en_banda(f: np.ndarray, p: np.ndarray, banda: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    m = (f >= banda[0]) & (f <= banda[1])
    return f[m], p[m]


def frecuencia_media(f: np.ndarray, p: np.ndarray, banda: tuple[float, float] = (20, 450)) -> float:
    f, p = _en_banda(f, p, banda)
    return float(np.sum(f * p) / np.sum(p))


def frecuencia_mediana(f: np.ndarray, p: np.ndarray, banda: tuple[float, float] = (20, 450)) -> float:
    f, p = _en_banda(f, p, banda)
    acumulada = np.cumsum(p)
    return float(f[np.searchsorted(acumulada, acumulada[-1] / 2)])


def espectro_por_ventanas(
    x: np.ndarray,
    fs: float,
    ventana_s: float = 1.0,
    paso_s: float = 0.5,
    banda: tuple[float, float] = (20, 450),
    energia_min: float = 0.0,
) -> pd.DataFrame:
    """MNF y MDF en ventanas deslizantes (seguimiento de fatiga).

    Las ventanas cuya energía no supera ``energia_min`` (p. ej. reposo) se
    descartan, porque su espectro está dominado por ruido.
    """
    n, h = int(ventana_s * fs), int(paso_s * fs)
    filas = []
    for i in range(0, len(x) - n + 1, h):
        seg = x[i : i + n]
        if np.mean(seg**2) <= energia_min:
            continue
        f, p = espectro(seg, fs, nperseg=min(512, n))
        filas.append(
            {
                "t_centro_s": (i + n / 2) / fs,
                "mnf_hz": frecuencia_media(f, p, banda),
                "mdf_hz": frecuencia_mediana(f, p, banda),
            }
        )
    return pd.DataFrame(filas)


def pendiente(t: np.ndarray, y: np.ndarray) -> float:
    """Pendiente de la recta de mínimos cuadrados (unidades de y por segundo)."""
    if len(t) < 2:
        return float("nan")
    return float(np.polyfit(t, y, 1)[0])


# ---------- coordinación ----------------------------------------------------

def indice_coactivacion(env_agonista: np.ndarray, env_antagonista: np.ndarray) -> float:
    """Índice de coactivación de Falconer y Winter (1985), en %.

    CI = 2 · Σ min(a, b) / Σ (a + b) · 100. Vale 0 si nunca hay solapamiento
    y 100 si ambas envolventes son idénticas.
    """
    a = np.asarray(env_agonista, dtype=float)
    b = np.asarray(env_antagonista, dtype=float)
    total = np.sum(a + b)
    if total == 0:
        return float("nan")
    return float(200.0 * np.sum(np.minimum(a, b)) / total)


def detectar_activaciones(
    env: np.ndarray,
    fs: float,
    t_base: tuple[float, float] = (0.0, 1.5),
    k: float = 3.0,
    duracion_min_s: float = 0.1,
) -> tuple[pd.DataFrame, float]:
    """Detecta activaciones por umbral μ + k·σ de la línea de base.

    Devuelve la tabla de activaciones y el umbral empleado (en mV).
    """
    i0, i1 = int(t_base[0] * fs), int(t_base[1] * fs)
    base = env[i0:i1]
    umbral = base.mean() + k * base.std()
    activo = env > umbral
    bordes = np.diff(activo.astype(int), prepend=0, append=0)
    inicios, fines = np.flatnonzero(bordes == 1), np.flatnonzero(bordes == -1)
    n_min = int(duracion_min_s * fs)
    filas = [
        {"inicio_s": a / fs, "fin_s": b / fs, "duracion_s": (b - a) / fs}
        for a, b in zip(inicios, fines)
        if b - a >= n_min
    ]
    return pd.DataFrame(filas, columns=["inicio_s", "fin_s", "duracion_s"]), float(umbral)


# ---------- resumen -----------------------------------------------------------

def tabla_resumen(filtrada: pd.DataFrame, fs: float, banda: tuple[float, float] = (20, 450)) -> pd.DataFrame:
    filas = []
    for c in [c for c in filtrada.columns if c != "tiempo_s"]:
        x = filtrada[c].to_numpy()
        f, p = espectro(x, fs)
        filas.append(
            {
                "canal": c,
                "rms_mv": rms(x),
                "mav_mv": mav(x),
                "mnf_hz": frecuencia_media(f, p, banda),
                "mdf_hz": frecuencia_mediana(f, p, banda),
            }
        )
    return pd.DataFrame(filas)
