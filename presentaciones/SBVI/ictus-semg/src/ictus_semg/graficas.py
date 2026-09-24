"""Figuras Plotly del tablero. Cada función recibe datos y devuelve un ``go.Figure``.

Ninguna función depende de Streamlit: las figuras pueden reutilizarse en un
cuaderno, en un informe Quarto o en otro framework (Dash, Panel).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

COLOR = {
    "cruda": "#9AA5B1",
    "filtrada": "#1F4E79",
    "envolvente": "#C0392B",
    "paretico": "#C0392B",
    "no_paretico": "#1F4E79",
}
PLANTILLA = "simple_white"


def _diezmar(n: int, max_puntos: int) -> slice:
    """Paso de submuestreo solo para visualizar (el análisis usa todas las muestras)."""
    return slice(None, None, max(1, n // max_puntos))


def fig_multicanal(
    etapas: dict[str, pd.DataFrame],
    canales: list[str],
    rango_s: tuple[float, float],
    mostrar_cruda: bool = False,
    max_puntos: int = 6000,
) -> go.Figure:
    t = etapas["filtrada"]["tiempo_s"].to_numpy()
    m = (t >= rango_s[0]) & (t <= rango_s[1])
    s = _diezmar(int(m.sum()), max_puntos)
    fig = make_subplots(
        rows=len(canales), cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=[c.replace("_", " ") for c in canales],
    )
    for i, c in enumerate(canales, start=1):
        if mostrar_cruda:
            fig.add_trace(go.Scattergl(x=t[m][s], y=etapas["cruda"][c].to_numpy()[m][s], name="cruda",
                                       line=dict(color=COLOR["cruda"], width=0.8),
                                       legendgroup="cruda", showlegend=i == 1), row=i, col=1)
        fig.add_trace(go.Scattergl(x=t[m][s], y=etapas["filtrada"][c].to_numpy()[m][s], name="filtrada",
                                   line=dict(color=COLOR["filtrada"], width=0.8),
                                   legendgroup="filtrada", showlegend=i == 1), row=i, col=1)
        fig.add_trace(go.Scattergl(x=t[m][s], y=etapas["envolvente"][c].to_numpy()[m][s], name="envolvente",
                                   line=dict(color=COLOR["envolvente"], width=2),
                                   legendgroup="envolvente", showlegend=i == 1), row=i, col=1)
        fig.update_yaxes(title_text="mV", row=i, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=len(canales), col=1)
    fig.update_layout(template=PLANTILLA, height=190 * len(canales) + 80,
                      margin=dict(l=60, r=20, t=40, b=40),
                      legend=dict(orientation="h", y=1.04, x=0), hovermode="x unified")
    return fig


def fig_psd(filtrada: pd.DataFrame, roles: dict[str, tuple[str, str]], fs: float) -> go.Figure:
    """``roles`` asocia cada canal a (músculo, lado), p. ej. {"ch1": ("biceps", "paretico")}."""
    fig = go.Figure()
    for c, (musculo, lado) in roles.items():
        f, p = signal.welch(filtrada[c].to_numpy(), fs=fs, nperseg=1024)
        estilo = "solid" if musculo == "biceps" else "dot"
        fig.add_trace(go.Scatter(x=f, y=10 * np.log10(p + 1e-20), name=f"{musculo} {lado.replace('_', ' ')}",
                                 line=dict(color=COLOR[lado], dash=estilo)))
    fig.update_layout(template=PLANTILLA, height=420, xaxis_title="Frecuencia (Hz)",
                      yaxis_title="DEP (dB re mV²/Hz)", hovermode="x unified")
    fig.update_xaxes(range=[0, min(500, fs / 2)])
    return fig


def fig_espectrograma(x: np.ndarray, fs: float, titulo: str) -> go.Figure:
    f, t, sxx = signal.spectrogram(x, fs=fs, nperseg=512, noverlap=384)
    m = f <= min(500, fs / 2)
    fig = go.Figure(go.Heatmap(x=t, y=f[m], z=10 * np.log10(sxx[m] + 1e-20),
                               colorscale="Viridis", colorbar=dict(title="dB")))
    fig.update_layout(template=PLANTILLA, height=380, title=titulo,
                      xaxis_title="Tiempo (s)", yaxis_title="Frecuencia (Hz)")
    return fig


def fig_fatiga(series: dict[str, pd.DataFrame], variable: str = "mdf_hz") -> go.Figure:
    """``series`` asocia el lado ("paretico" / "no_paretico") a su tabla de ventanas."""
    fig = go.Figure()
    for lado, df in series.items():
        if len(df) < 2:
            continue
        color = COLOR[lado]
        fig.add_trace(go.Scatter(x=df["t_centro_s"], y=df[variable], mode="markers",
                                 name=lado.replace("_", " "), marker=dict(color=color, size=6, opacity=0.6)))
        a, b = np.polyfit(df["t_centro_s"], df[variable], 1)
        tt = np.array([df["t_centro_s"].min(), df["t_centro_s"].max()])
        fig.add_trace(go.Scatter(x=tt, y=a * tt + b, mode="lines", showlegend=False,
                                 line=dict(color=color, width=2),
                                 hovertemplate=f"pendiente = {a:.2f} Hz/s<extra></extra>"))
    etiqueta = "MDF (Hz)" if variable == "mdf_hz" else "MNF (Hz)"
    fig.update_layout(template=PLANTILLA, height=420, xaxis_title="Tiempo (s)", yaxis_title=etiqueta)
    return fig


def fig_coactivacion(t: np.ndarray, env_ag: np.ndarray, env_ant: np.ndarray, titulo: str,
                     max_puntos: int = 4000) -> go.Figure:
    s = _diezmar(len(t), max_puntos)
    comun = np.minimum(env_ag, env_ant)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[s], y=env_ag[s], name="agonista (bíceps)", line=dict(color="#1F4E79")))
    fig.add_trace(go.Scatter(x=t[s], y=env_ant[s], name="antagonista (tríceps)", line=dict(color="#D68910")))
    fig.add_trace(go.Scatter(x=t[s], y=comun[s], name="área común", fill="tozeroy",
                             line=dict(color="rgba(192,57,43,0.0)"), fillcolor="rgba(192,57,43,0.35)"))
    fig.update_layout(template=PLANTILLA, height=320, title=titulo, xaxis_title="Tiempo (s)",
                      yaxis_title="Envolvente (mV)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


def fig_barras_metrica(resumen: pd.DataFrame, metrica: str, etiqueta: str) -> go.Figure:
    """``resumen`` debe incluir las columnas ``musculo`` y ``lado``."""
    fig = go.Figure()
    for lado in ("paretico", "no_paretico"):
        d = resumen[resumen["lado"] == lado]
        fig.add_trace(go.Bar(x=d["musculo"], y=d[metrica], name=lado.replace("_", " "),
                             marker_color=COLOR[lado]))
    fig.update_layout(template=PLANTILLA, barmode="group", height=340, yaxis_title=etiqueta)
    return fig
