from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Revisión EEG", layout="wide")
st.title("Revisión retrospectiva de EEG")
st.caption("Prototipo docente | anotaciones y métricas descriptivas")
p = Path("data/derived/metricas.csv")
if not p.exists():
    st.info("Ejecutar primero src/procesamiento.py")
    st.stop()
data = pd.read_csv(p)
channel = st.selectbox("Derivación bipolar", sorted(data.channel.unique()),
                       key="channel")
d = data[data.channel == channel].copy()
lo, hi = float(d.start_s.min()), float(d.end_s.max())
a, b = st.slider("Tiempo desde inicio EDF (s)", lo, hi,
                 (lo, hi), key="interval")
d = d[(d.start_s >= a) & (d.end_s <= b)]
metric = st.selectbox("Indicador", ["rms_uv", "ll_uv", "entropy",
                                      "alpha_rel"], key="metric")
if d.empty:
    st.warning("No hay épocas completas dentro de este intervalo")
else:
    fig = px.line(d, x="start_s", y=metric,
                  hover_data=["state", "overlap_s"])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(d)
    st.download_button("Exportar épocas seleccionadas",
                       d.to_csv(index=False), "metricas_seleccion.csv",
                       "text/csv", key="export")
