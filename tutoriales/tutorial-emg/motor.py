"""Descriptores exploratorios; no clasificador de enfermedad."""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import cumulative_trapezoid


def preparar(x, fs):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Se requiere un vector finito.')
    if fs != 4000 or len(x) < 2*fs:
        raise ValueError('Se requieren 4000 Hz y al menos 2 s.')
    if np.ptp(x) < 1e-9:
        raise ValueError('Canal plano: no interpretable.')
    sos = signal.butter(4, [20, 1000], fs=fs,
                        btype='bandpass', output='sos')
    return signal.sosfiltfilt(sos, x)


def medir(y, fs):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < int(0.5*fs):
        raise ValueError('Ventana demasiado corta.')
    if not np.isfinite(y).all():
        raise ValueError('Ventana con valores no finitos.')
    z = y - np.mean(y)
    rms = float(np.sqrt(np.mean(z*z)))
    f, p = signal.welch(z, fs=fs, window='hann',
                        nperseg=int(0.25*fs),
                        noverlap=int(0.125*fs),
                        detrend='constant', scaling='density')
    m = (f >= 20) & (f <= 1000)
    fb, pb = f[m], p[m]
    acumulada = cumulative_trapezoid(pb, fb, initial=0)
    if acumulada[-1] <= 1e-12:
        mediana = np.nan
    else:
        mediana = float(np.interp(acumulada[-1]/2, acumulada, fb))
    metricas = dict(rms_uv=rms, pico_pico_uv=float(np.ptp(z)),
                   cresta=float(np.max(np.abs(z))/rms)
                   if rms > 1e-9 else np.nan,
                   fmed_20_1000_hz=mediana)
    return metricas, f, p


def ventanas(y, fs, inicio, fin, W):
    # Índices enteros: intervalos [i, j), sin ambigüedad de redondeo.
    N = round(W*fs)
    paso = N//2
    i0, i1 = round(inicio*fs), round(fin*fs)
    if not (0 <= i0 < i1 <= len(y)) or i1-i0 < N:
        raise ValueError('Intervalo insuficiente o fuera del registro.')
    rows = []
    for i in range(i0, i1-N+1, paso):
        j = i+N
        valores, _, _ = medir(y[i:j], fs)
        rows.append(dict(i=i, j=j, inicio_s=i/fs, fin_s=j/fs,
                         centro_s=(i+j)/(2*fs), **valores))
    return pd.DataFrame(rows)
