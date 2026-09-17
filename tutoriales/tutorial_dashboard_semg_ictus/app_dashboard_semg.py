from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import wfdb
from scipy import signal
from scipy.ndimage import uniform_filter1d

PN_DIR = "cves/1.0.0/data/24h-electromyography"
SUBJECTS_URL = "https://physionet.org/files/cves/1.0.0/subjects.csv"

# Registros de ejemplo comprobados en el repositorio CVES.
RECORDS = {
    "S0205 · ictus · NIHSS 10 / mRS 3": "s0205-05080904",
    "S0185 · ictus · NIHSS 0 / mRS 0": "s0185-05063014",
    "S0030 · control": "s0030-04051907",
}


def subject_from_record(record_name: str) -> str:
    return record_name.split("-")[0].upper()


@st.cache_data(show_spinner=False)
def load_subjects() -> pd.DataFrame:
    return pd.read_csv(SUBJECTS_URL)


@st.cache_data(show_spinner=False)
def get_header(record_name: str):
    return wfdb.rdheader(record_name, pn_dir=PN_DIR)


@st.cache_data(show_spinner=False)
def load_emg_segment(record_name: str, start_s: int, duration_s: int):
    header = get_header(record_name)
    fs = float(header.fs)
    emg_idx = [i for i, name in enumerate(header.sig_name) if "emg" in name.lower()]
    if not emg_idx:
        raise ValueError("El registro no contiene canales identificados como EMG.")

    sampfrom = int(start_s * fs)
    sampto = min(int((start_s + duration_s) * fs), int(header.sig_len))
    if sampto <= sampfrom:
        raise ValueError("El intervalo solicitado está fuera del registro.")

    rec = wfdb.rdrecord(
        record_name,
        pn_dir=PN_DIR,
        sampfrom=sampfrom,
        sampto=sampto,
        channels=emg_idx,
    )
    return rec.p_signal, float(rec.fs), rec.sig_name


def interpolate_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.all(np.isfinite(x)):
        return x
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if good.sum() < 2:
        return np.full_like(x, np.nan)
    return np.interp(idx, idx[good], x[good])


def bandpass_semg(x: np.ndarray, fs: float, low=20.0, high=450.0, order=4):
    x = interpolate_nan(x)
    if not np.all(np.isfinite(x)):
        return x
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def notch_60(x: np.ndarray, fs: float, q=30.0):
    b, a = signal.iirnotch(60.0, q, fs=fs)
    return signal.filtfilt(b, a, x)


def rms_envelope(x: np.ndarray, fs: float, window_ms=250):
    n = max(3, int(window_ms / 1000.0 * fs))
    return np.sqrt(uniform_filter1d(x**2, size=n, mode="nearest"))


def linear_envelope(x: np.ndarray, fs: float, cutoff=5.0):
    rect = np.abs(x)
    sos = signal.butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, rect)


def robust_activity_threshold(rms: np.ndarray):
    q20 = np.nanquantile(rms, 0.20)
    base = rms[rms <= q20]
    med = np.nanmedian(base)
    mad = 1.4826 * np.nanmedian(np.abs(base - med))
    return med + 3.0 * mad


def run_lengths(binary: np.ndarray, fs: float):
    binary = np.asarray(binary, dtype=bool)
    starts = np.flatnonzero(np.diff(np.r_[False, binary].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[binary, False].astype(int)) == -1) + 1
    durations = (ends - starts) / fs
    return starts, ends, durations


def median_frequency(x: np.ndarray, fs: float, band=(20.0, 450.0)) -> float:
    """Frecuencia mediana espectral dentro de la banda de análisis."""
    x = interpolate_nan(x)
    if not np.all(np.isfinite(x)):
        return np.nan

    nperseg = min(len(x), max(256, int(4 * fs)))
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    fb = f[mask]
    pb = pxx[mask]

    if len(fb) < 2 or np.nansum(pb) <= 0:
        return np.nan

    areas = 0.5 * (pb[:-1] + pb[1:]) * np.diff(fb)
    cumulative = np.r_[0.0, np.cumsum(areas)]
    idx = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    idx = min(idx, len(fb) - 1)
    return float(fb[idx])


def band_power_ratio(x, fs, num_band, den_band):
    x = interpolate_nan(x)
    if not np.all(np.isfinite(x)):
        return np.nan
    nperseg = min(len(x), max(256, int(4 * fs)))
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    num = (f >= num_band[0]) & (f <= num_band[1])
    den = (f >= den_band[0]) & (f <= den_band[1])
    if num.sum() < 2 or den.sum() < 2:
        return np.nan
    p_num = np.trapz(pxx[num], f[num])
    p_den = np.trapz(pxx[den], f[den])
    return np.nan if p_den <= 0 else 100.0 * p_num / p_den


def channel_metrics(raw, filtered, fs):
    rms = rms_envelope(filtered, fs, 250)
    env = linear_envelope(filtered, fs, 5.0)
    thr = robust_activity_threshold(rms)
    active = rms > thr
    starts, ends, durations = run_lengths(active, fs)
    minutes = len(filtered) / fs / 60.0

    return {
        "rms": rms,
        "envelope": env,
        "threshold": thr,
        "active": active,
        "median_rms_uV": float(np.nanmedian(rms)),
        "active_rms_median_uV": (
            float(np.nanmedian(rms[active])) if np.any(active) else np.nan
        ),
        "mdf_hz": median_frequency(filtered, fs, band=(20.0, 450.0)),
        "active_time_pct": float(100.0 * np.nanmean(active)),
        "bursts_per_min": float(len(starts) / minutes) if minutes > 0 else np.nan,
        "median_burst_s": float(np.nanmedian(durations)) if len(durations) else 0.0,
        "line_noise_60_pct": band_power_ratio(raw, fs, (59, 61), (20, 450)),
        "low_freq_pct": band_power_ratio(raw, fs, (0.5, 15), (0.5, 450)),
        "missing_pct": float(100.0 * np.mean(~np.isfinite(raw))),
    }


def main():
    st.set_page_config(page_title="sEMG post-ictus", layout="wide")
    st.title("Dashboard exploratorio de sEMG post-ictus")
    st.caption("Datos abiertos CVES · PhysioNet · uso docente, no diagnóstico")

    with st.sidebar:
        st.header("Selección")
        label = st.selectbox("Registro", list(RECORDS))
        record_name = RECORDS[label]
        header = get_header(record_name)
        total_s = int(header.sig_len / header.fs)
        max_start = max(0, total_s - 30)
        start_s = st.number_input("Inicio [s]", min_value=0, max_value=max_start, value=0, step=30)
        duration_s = st.slider("Duración [s]", min_value=30, max_value=300, value=120, step=30)
        apply_notch = st.checkbox("Aplicar notch 60 Hz", value=False)
        st.divider()
        st.markdown("**Regla:** el notch se activa solo si la inspección espectral o la métrica de ruido lo justifican.")

    subject_id = subject_from_record(record_name)
    subjects = load_subjects()
    subject = subjects.loc[subjects["subject_number"].eq(subject_id)]

    if not subject.empty:
        row = subject.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Participante", subject_id)
        c2.metric("Grupo", str(row.get("group", "n/d")))
        c3.metric("Edad", str(row.get("age", "n/d")))
        c4.metric("NIHSS", str(row.get("NIHSS", "n/d")))
        c5.metric("mRS", str(row.get("MRS", "n/d")))
        st.write(
            "**Lado del ictus:**", row.get("Stroke Side", "n/d"),
            " · **Síntomas:**", row.get("Symptoms", "n/d")
        )

    try:
        raw, fs, names = load_emg_segment(record_name, int(start_s), int(duration_s))
    except Exception as exc:
        st.error(f"No fue posible cargar el segmento: {exc}")
        st.stop()

    processed = {}
    rows = []
    for j, name in enumerate(names):
        r = raw[:, j]
        f = bandpass_semg(r, fs)
        if apply_notch and np.all(np.isfinite(f)):
            f = notch_60(f, fs)
        m = channel_metrics(r, f, fs)
        processed[name] = {"raw": r, "filtered": f, **m}
        rows.append({
            "Canal": name,
            "RMS mediana [µV]": m["median_rms_uV"],
            "RMS activo mediano [µV]": m["active_rms_median_uV"],
            "MDF [Hz]": m["mdf_hz"],
            "Tiempo activo [%]": m["active_time_pct"],
            "Episodios/min": m["bursts_per_min"],
            "Duración mediana [s]": m["median_burst_s"],
            "Ruido 60 Hz [%]": m["line_noise_60_pct"],
            "Baja frecuencia [%]": m["low_freq_pct"],
            "Faltantes [%]": m["missing_pct"],
        })

    metrics_df = pd.DataFrame(rows)
    st.subheader("Indicadores por canal")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    selected = st.selectbox("Canal para inspección", names)
    d = processed[selected]
    t = np.arange(len(d["raw"])) / fs + float(start_s)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=d["raw"], name="Cruda", line=dict(width=0.7)))
    fig.add_trace(go.Scatter(x=t, y=d["filtered"], name="Filtrada", line=dict(width=0.9)))
    fig.update_layout(
        title=f"{selected}: señal cruda y filtrada",
        xaxis_title="Tiempo [s]",
        yaxis_title="Amplitud [µV]",
        height=380,
        legend_orientation="h",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=d["rms"], name="RMS 250 ms"))
    fig2.add_trace(go.Scatter(x=t, y=d["envelope"], name="Envolvente 5 Hz"))
    fig2.add_hline(y=d["threshold"], line_dash="dash", annotation_text="Umbral interno")
    fig2.update_layout(
        title=f"{selected}: magnitud y umbral de actividad",
        xaxis_title="Tiempo [s]",
        yaxis_title="Amplitud [µV]",
        height=380,
        legend_orientation="h",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Uso longitudinal de los nuevos indicadores")
    st.info(
        "Para seguimiento post-ictus, el tutorial propone comparar entre visitas estandarizadas: "
        "(1) el cambio porcentual del RMS activo respecto a la visita basal y "
        "(2) la distancia relativa de MDF frente al músculo contralateral homólogo. "
        "CVES es transversal y no debe interpretarse como una serie de recuperación clínica."
    )

    st.subheader("Interpretación y límites")
    st.info(
        "Los nombres de canal EMG0-EMG3 se conservan porque el encabezado WFDB no identifica por sí solo "
        "el músculo ni la lateralidad. No calcule asimetría bilateral ni co-contracción hasta verificar la "
        "correspondencia anatómica en el protocolo del estudio."
    )
    st.warning(
        "RMS y tiempo activo son indicadores de la señal registrada. No son medidas directas de fuerza, "
        "no constituyen diagnóstico y no deben compararse entre sujetos como si la colocación de electrodos "
        "y la normalización fueran equivalentes."
    )


if __name__ == "__main__":
    main()
