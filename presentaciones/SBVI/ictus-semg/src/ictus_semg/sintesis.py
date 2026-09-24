"""Generador de sEMG sintético para un protocolo de flexo-extensión de codo.

El modelo NO pretende reproducir la fisiología con fidelidad: produce
señales con las propiedades cualitativas necesarias para ejercitar el
tablero (ruido de banda limitada modulado en amplitud, coactivación,
desplazamiento espectral por fatiga, interferencia de red y deriva).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

CANALES: tuple[str, ...] = (
    "biceps_paretico",
    "triceps_paretico",
    "biceps_no_paretico",
    "triceps_no_paretico",
)


@dataclass(frozen=True)
class ConfigSintesis:
    fs: float = 2000.0          # Hz (Delsys Trigno ≈ 1926 Hz)
    duracion_s: float = 30.0    # s
    reposo_s: float = 2.0       # s de línea de base al inicio
    periodo_s: float = 3.0      # s por ciclo flexión-extensión
    severidad: float = 0.5      # 0 = sin compromiso, 1 = compromiso severo
    fatiga: float = 0.4         # 0 = sin fatiga, 1 = fatiga marcada
    ruido_red_mv: float = 0.02  # amplitud de la interferencia de 60 Hz
    f_red_hz: float = 60.0      # 60 Hz en Colombia
    deriva_mv: float = 0.05     # amplitud de la deriva de línea de base
    semilla: int = 42


def _portadora(n: int, fs: float, peso_alta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ruido gaussiano de banda limitada cuya energía migra hacia bajas frecuencias.

    Mezcla una banda baja (20–120 Hz) y una alta (80–350 Hz). Al reducir el
    peso de la banda alta en el tiempo, la frecuencia mediana desciende, que
    es la firma espectral clásica de la fatiga.
    """
    sos_baja = signal.butter(4, [20, 120], btype="bandpass", fs=fs, output="sos")
    sos_alta = signal.butter(4, [80, 350], btype="bandpass", fs=fs, output="sos")
    baja = signal.sosfiltfilt(sos_baja, rng.standard_normal(n))
    alta = signal.sosfiltfilt(sos_alta, rng.standard_normal(n))
    baja /= baja.std()
    alta /= alta.std()
    mezcla = np.sqrt(1.0 - peso_alta) * baja + np.sqrt(peso_alta) * alta
    return mezcla / mezcla.std()


def _patron_activacion(t: np.ndarray, cfg: ConfigSintesis) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve la activación normalizada (0–1) de flexores y extensores."""
    fase = 2 * np.pi * (t - cfg.reposo_s) / cfg.periodo_s
    flexion = np.clip(np.sin(fase), 0, None) ** 1.5
    extension = np.clip(-np.sin(fase), 0, None) ** 1.5
    en_reposo = t < cfg.reposo_s
    flexion[en_reposo] = 0.0
    extension[en_reposo] = 0.0
    return flexion, extension


def generar_semg(cfg: ConfigSintesis = ConfigSintesis()) -> pd.DataFrame:
    """Genera cuatro canales de sEMG (mV) y la columna ``tiempo_s``."""
    rng = np.random.default_rng(cfg.semilla)
    n = int(round(cfg.duracion_s * cfg.fs))
    t = np.arange(n) / cfg.fs
    flex, ext = _patron_activacion(t, cfg)

    progreso = np.clip((t - cfg.reposo_s) / (cfg.duracion_s - cfg.reposo_s), 0, 1)
    sev = float(np.clip(cfg.severidad, 0, 1))

    # Parámetros por lado: el lado parético recluta menos y coactiva más.
    lados = {
        "paretico": dict(ganancia=1.0 - 0.6 * sev, coact=0.10 + 0.50 * sev, fatiga=cfg.fatiga * (1 + 0.5 * sev)),
        "no_paretico": dict(ganancia=1.0, coact=0.10, fatiga=cfg.fatiga),
    }

    datos: dict[str, np.ndarray] = {"tiempo_s": t}
    for lado, p in lados.items():
        peso_alta = np.clip(0.6 - 0.5 * p["fatiga"] * progreso, 0.05, 0.95)
        # Envolventes: agonista propio + coactivación del antagonista.
        env_biceps = p["ganancia"] * (flex + p["coact"] * ext)
        env_triceps = p["ganancia"] * (ext + p["coact"] * flex)
        for musculo, env in (("biceps", env_biceps), ("triceps", env_triceps)):
            portadora = _portadora(n, cfg.fs, peso_alta, rng)
            piso = 0.01 * rng.standard_normal(n)  # ruido de instrumentación
            datos[f"{musculo}_{lado}"] = 0.5 * env * portadora + piso

    # Artefactos comunes a todos los canales.
    red = cfg.ruido_red_mv * np.sin(2 * np.pi * cfg.f_red_hz * t)
    deriva = cfg.deriva_mv * np.sin(2 * np.pi * 0.3 * t)
    for canal in CANALES:
        datos[canal] = datos[canal] + red + deriva

    return pd.DataFrame(datos)[["tiempo_s", *CANALES]]
