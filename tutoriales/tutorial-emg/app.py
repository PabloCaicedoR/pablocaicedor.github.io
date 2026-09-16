import json
import plotly.graph_objects as go
import streamlit as st
from datos import cargar, ETIQUETAS
from motor import preparar, medir, ventanas

st.set_page_config(page_title='EMGDB | Dashboard', layout='wide')
st.title('Dashboard de EMG y radiculopatía L5')
st.caption('Dataset abierto EMGDB 1.0.0 | Análisis retrospectivo docente')
st.info('Tres personas distintas. No son visitas de un mismo paciente.')
registro = st.sidebar.selectbox('Registro', list(ETIQUETAS),
                               format_func=ETIQUETAS.get)
try:
    t, x, meta = cargar(registro)
    fs = meta['fs_hz']
    y = preparar(x, fs)
except Exception as exc:
    st.error(f'No se pudo cargar o procesar: {exc}')
    st.stop()
st.caption(f"{registro} | {meta['modalidad']} | "
           f"{meta['musculo']} | {fs:.0f} Hz | {len(x)/fs:.4f} s")
st.markdown('[Fuente y licencia](https://physionet.org/content/emgdb/1.0.0/)')

# Descarte de 0.25 s en cada extremo del registro ya filtrado.
a, b = st.sidebar.slider('Intervalo de análisis (s)',
    0.25, float(len(x)/fs-0.25), (0.25, 5.25), step=0.25,
    key='intervalo_' + registro)
W = st.sidebar.selectbox('Duración de ventana (s)', [0.5, 1.0, 2.0], index=1)
st.sidebar.caption('Paso = W/2. Banda 20–1000 Hz. Welch 0.25 s.')
nota = st.sidebar.text_area('Anotación de calidad',
                           key='nota_' + registro)
revisado = st.sidebar.checkbox('Inspeccioné el intervalo seleccionado',
                              key=f'revision_{registro}_{a}_{b}')
if b-a < W:
    st.error('Seleccione un intervalo al menos igual a la ventana.')
    st.stop()

fig = go.Figure()
i0, i1 = round(a*fs), round(b*fs)
fig.add_trace(go.Scattergl(x=t[i0:i1], y=x[i0:i1], name='Original'))
fig.add_trace(go.Scattergl(x=t[i0:i1], y=y[i0:i1], name='Filtrada'))
fig.update_layout(title='Control de calidad: todas las muestras del intervalo',
                  xaxis_title='Tiempo (s)', yaxis_title='EMG (µV)')
st.plotly_chart(fig, width='stretch')
st.caption('Use zoom para inspeccionar eventos. La revisión visual '
           'no sustituye anotaciones expertas ni detecta todo artefacto.')
if not revisado:
    st.warning('Revise original y filtrada antes de habilitar indicadores.')
    st.stop()

res = ventanas(y, fs, a, b, W)
seleccion = st.select_slider('Ventana activa', options=list(res.index),
    format_func=lambda k: f"[{res.loc[k,'inicio_s']:.3f}, "
                         f"{res.loc[k,'fin_s']:.3f}) s")
r = res.loc[seleccion]
i, j = int(r.i), int(r.j)
valores, f, p = medir(y[i:j], fs)
for col, clave, titulo in zip(st.columns(4), valores,
    ['RMS (µV)', 'Pico a pico de ventana (µV)',
     'Factor de cresta', 'Frecuencia mediana 20–1000 Hz']):
    col.metric(titulo, f'{valores[clave]:.2f}')
st.caption('Descriptores de la señal; no puntajes de gravedad ni medidas '
           'automáticas de una unidad motora aislada.')

fig = go.Figure(go.Scatter(x=t[i:j], y=y[i:j], name='Ventana activa'))
fig.update_layout(title='Detalle temporal sin reducción de muestras',
                  xaxis_title='Tiempo (s)', yaxis_title='EMG filtrada (µV)')
st.plotly_chart(fig, width='stretch')
fig = go.Figure(go.Scatter(x=f, y=p))
fig.update_layout(title='PSD de la EMG filtrada y centrada',
                  xaxis_title='Frecuencia (Hz)', yaxis_title='PSD (µV²/Hz)',
                  xaxis_range=[0, 1200])
st.plotly_chart(fig, width='stretch')

variable = st.selectbox('Indicador de tendencia', list(valores))
fig = go.Figure(go.Scatter(x=res.centro_s, y=res[variable],
                          mode='lines+markers'))
fig.update_layout(title='Evolución dentro del registro',
                  xaxis_title='Centro de ventana (s)', yaxis_title=variable)
st.plotly_chart(fig, width='stretch')
st.caption('Solapamiento 50 %. Los puntos no son réplicas independientes.')

meta.update(banda_hz=[20,1000], filtro='Butterworth SOS bidireccional',
            orden_diseno=4, borde_s=0.25, ventana_s=W, paso_s=W/2,
            welch_s=0.25, solapamiento_welch_s=0.125,
            intervalo_s=[i0/fs,i1/fs], nota_calidad=nota,
            inspeccion_visual=True, version_pipeline='2.0')
res['registro'] = registro
res['sha256_dat'] = meta['sha256_dat']
st.download_button('Indicadores CSV', res.to_csv(index=False).encode(),
                   registro + '_indicadores.csv', 'text/csv')
st.download_button('Parámetros JSON',
                   json.dumps(meta, ensure_ascii=False, indent=2),
                   registro + '_parametros.json', 'application/json')
