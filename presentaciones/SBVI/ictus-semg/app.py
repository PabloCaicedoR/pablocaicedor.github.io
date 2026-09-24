"""Tablero interactivo de sEMG en ictus — SBVI.

Ejecutar con:  uv run streamlit run app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from ictus_semg import graficas as g
from ictus_semg import metricas as m
from ictus_semg.carga import ErrorFormato, leer_csv
from ictus_semg.procesamiento import procesar
from ictus_semg.sintesis import CANALES, ConfigSintesis, generar_semg

st.set_page_config(page_title="sEMG en ictus", page_icon="🧠", layout="wide")

# (músculo, lado) de cada rol; el canal concreto lo elige el usuario.
ROLES: dict[str, tuple[str, str]] = {
    "Bíceps parético": ("biceps", "paretico"),
    "Tríceps parético": ("triceps", "paretico"),
    "Bíceps no parético": ("biceps", "no_paretico"),
    "Tríceps no parético": ("triceps", "no_paretico"),
}


# ------------------------------------------------------------------ caché ----
@st.cache_data(show_spinner="Generando señal sintética…")
def cargar_sintetica(severidad: float, fatiga: float, ruido_red: float, semilla: int) -> tuple[pd.DataFrame, float]:
    cfg = ConfigSintesis(severidad=severidad, fatiga=fatiga, ruido_red_mv=ruido_red, semilla=semilla)
    return generar_semg(cfg), cfg.fs


@st.cache_data(show_spinner="Leyendo archivo…")
def cargar_csv(contenido: bytes) -> tuple[pd.DataFrame, float, int]:
    return leer_csv(io.BytesIO(contenido))


@st.cache_data(show_spinner="Filtrando…")
def procesar_cache(df: pd.DataFrame, fs: float, f_baja: float, f_alta: float, usar_notch: bool,
                   f_red: float, fc_env: float, ventana_rms_s: float) -> dict[str, pd.DataFrame]:
    return procesar(df, fs, f_baja, f_alta, usar_notch, f_red, fc_env, ventana_rms_s)


# --------------------------------------------------------- barra lateral -----
with st.sidebar:
    st.header("Datos")
    fuente = st.radio("Fuente", ["Señal sintética", "Archivo CSV"], horizontal=True)
    if fuente == "Señal sintética":
        severidad = st.slider("Severidad del compromiso", 0.0, 1.0, 0.5, 0.05,
                              help="0: el lado parético se comporta como el sano. 1: compromiso severo.")
        fatiga = st.slider("Fatiga", 0.0, 1.0, 0.4, 0.05)
        ruido_red = st.slider("Interferencia de red (mV)", 0.0, 0.2, 0.02, 0.01)
        semilla = st.number_input("Semilla", 0, 9999, 42)
        df, fs = cargar_sintetica(severidad, fatiga, ruido_red, int(semilla))
    else:
        archivo = st.file_uploader("CSV con columna tiempo_s y un canal por columna", type="csv")
        if archivo is None:
            st.info("Cargue un archivo para continuar o vuelva a la señal sintética.")
            st.stop()
        try:
            df, fs, descartadas = cargar_csv(archivo.getvalue())
        except ErrorFormato as err:
            st.error(f"Formato no válido: {err}")
            st.stop()
        if descartadas:
            st.warning(f"Se descartaron filas por {descartadas} valores no numéricos (no se imputó nada).")

    canales_disp = [c for c in df.columns if c != "tiempo_s"]
    st.header("Asignación de canales")
    rol_a_canal: dict[str, str] = {}
    for rol, defecto in zip(ROLES, CANALES):
        idx = canales_disp.index(defecto) if defecto in canales_disp else 0
        rol_a_canal[rol] = st.selectbox(rol, canales_disp, index=idx)
    if len(set(rol_a_canal.values())) < len(ROLES):
        st.error("Cada rol debe tener un canal distinto.")
        st.stop()

    st.header("Procesamiento")
    f_baja, f_alta = st.slider("Pasabanda (Hz)", 5, 500, (20, 450))
    usar_notch = st.toggle("Notch de red", value=True)
    f_red = st.radio("Frecuencia de red (Hz)", [60, 50], horizontal=True, disabled=not usar_notch)
    fc_env = st.slider("Corte de la envolvente (Hz)", 2.0, 20.0, 8.0, 0.5)
    ventana_rms = st.slider("Ventana RMS (ms)", 50, 500, 200, 25) / 1000
    t_reposo = st.slider("Reposo inicial para línea de base (s)", 0.5, 5.0, 1.5, 0.5)
    k_umbral = st.slider("Umbral de activación (μ + k·σ)", 1.0, 10.0, 3.0, 0.5)

# ---------------------------------------------------------- procesamiento ----
etapas = procesar_cache(df, fs, float(f_baja), float(f_alta), usar_notch, float(f_red), fc_env, ventana_rms)
t = etapas["filtrada"]["tiempo_s"].to_numpy()
fil, env = etapas["filtrada"], etapas["envolvente"]
bp, tp, bn, tn = (rol_a_canal[r] for r in ROLES)
roles_canal = {rol_a_canal[r]: mus_lado for r, mus_lado in ROLES.items()}  # canal -> (músculo, lado)

ci_p = m.indice_coactivacion(env[bp].to_numpy(), env[tp].to_numpy())
ci_n = m.indice_coactivacion(env[bn].to_numpy(), env[tn].to_numpy())
razon_rms = m.rms(fil[bp].to_numpy()) / m.rms(fil[bn].to_numpy())


def serie_fatiga(canal: str) -> pd.DataFrame:
    x = fil[canal].to_numpy()
    energia_reposo = np.mean(x[: int(t_reposo * fs)] ** 2)
    return m.espectro_por_ventanas(x, fs, banda=(f_baja, f_alta), energia_min=4 * energia_reposo)


fatiga = {"paretico": serie_fatiga(bp), "no_paretico": serie_fatiga(bn)}
pend = {lado: m.pendiente(d["t_centro_s"].to_numpy(), d["mdf_hz"].to_numpy()) if len(d) > 1 else float("nan")
        for lado, d in fatiga.items()}

# ------------------------------------------------------------ indicadores ----
st.title("sEMG de miembro superior en ictus")
st.caption(f"fs = {fs:.0f} Hz · duración = {t[-1]:.1f} s · {len(canales_disp)} canales")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Coactivación lado parético", f"{ci_p:.1f} %",
          delta=f"{ci_p - ci_n:+.1f} pp frente al no parético", delta_color="inverse")
k2.metric("Coactivación lado no parético", f"{ci_n:.1f} %")
k3.metric("Razón RMS bíceps P/NP", f"{razon_rms:.2f}")
k4.metric("Pendiente MDF bíceps P", f"{pend['paretico']:.2f} Hz/s",
          delta=f"{pend['paretico'] - pend['no_paretico']:+.2f} frente al NP", delta_color="inverse")

# --------------------------------------------------------------- pestañas ----
tab_senal, tab_espectro, tab_fatiga, tab_coact, tab_tabla = st.tabs(
    ["Señal", "Espectro", "Fatiga", "Coactivación", "Métricas y exportación"]
)

with tab_senal:
    c1, c2 = st.columns([3, 1])
    rango = c1.slider("Ventana temporal (s)", 0.0, float(t[-1]), (0.0, min(10.0, float(t[-1]))), 0.5)
    ver_cruda = c2.toggle("Superponer señal cruda", value=False)
    st.plotly_chart(g.fig_multicanal(etapas, [bp, tp, bn, tn], rango, mostrar_cruda=ver_cruda), width="stretch")
    activaciones, umbral = m.detectar_activaciones(env[bn].to_numpy(), fs, t_base=(0.0, t_reposo), k=k_umbral)
    st.caption(f"Bíceps no parético: {len(activaciones)} activaciones detectadas (umbral = {umbral:.4f} mV).")

with tab_espectro:
    st.plotly_chart(g.fig_psd(fil, roles_canal, fs), width="stretch")
    canal_sg = st.selectbox("Espectrograma del canal", [bp, tp, bn, tn])
    st.plotly_chart(g.fig_espectrograma(fil[canal_sg].to_numpy(), fs, canal_sg), width="stretch")

with tab_fatiga:
    variable = st.radio("Indicador", ["mdf_hz", "mnf_hz"], horizontal=True,
                        format_func=lambda v: "Frecuencia mediana" if v == "mdf_hz" else "Frecuencia media")
    st.plotly_chart(g.fig_fatiga(fatiga, variable), width="stretch")
    st.caption("Ventanas de 1 s (paso 0,5 s) con energía mayor que 4 veces la del reposo inicial.")

with tab_coact:
    a, b = st.columns(2)
    a.plotly_chart(g.fig_coactivacion(t, env[bp].to_numpy(), env[tp].to_numpy(),
                                      f"Lado parético · CI = {ci_p:.1f} %"), width="stretch")
    b.plotly_chart(g.fig_coactivacion(t, env[bn].to_numpy(), env[tn].to_numpy(),
                                      f"Lado no parético · CI = {ci_n:.1f} %"), width="stretch")

with tab_tabla:
    resumen = m.tabla_resumen(fil[["tiempo_s", bp, tp, bn, tn]], fs, banda=(f_baja, f_alta))
    resumen["musculo"] = resumen["canal"].map(lambda c: roles_canal[c][0])
    resumen["lado"] = resumen["canal"].map(lambda c: roles_canal[c][1])
    c1, c2 = st.columns(2)
    c1.plotly_chart(g.fig_barras_metrica(resumen, "rms_mv", "RMS (mV)"), width="stretch")
    c2.plotly_chart(g.fig_barras_metrica(resumen, "mdf_hz", "MDF (Hz)"), width="stretch")
    st.dataframe(resumen.round(3), hide_index=True)
    st.download_button("Descargar métricas (CSV)", resumen.to_csv(index=False).encode("utf-8"),
                       file_name="metricas_semg.csv", mime="text/csv")
