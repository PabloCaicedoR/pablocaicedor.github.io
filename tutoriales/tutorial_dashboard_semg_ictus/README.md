# Tutorial de dashboard sEMG post-ictus

Archivos incluidos:

- `tutorial_dashboard_semg_ictus.qmd`: documento principal Quarto, configurado para PDF.
- `app_dashboard_semg.py`: dashboard Streamlit funcional.
- `referencias.bib`: referencias verificadas con DOI cuando aplica.
- `preamble.tex`: formato LaTeX para el PDF.
- `figuras/pipeline_semg.png`: diagrama conceptual.

## Render del tutorial

Con Quarto y una distribución LaTeX instalados:

```bash
quarto render tutorial_dashboard_semg_ictus.qmd
```

## Ejecutar el dashboard

```bash
uv init
uv add numpy pandas scipy plotly streamlit wfdb pyarrow
uv run streamlit run app_dashboard_semg.py
```

El script lee segmentos parciales desde PhysioNet usando WFDB; no es necesario descargar el dataset CVES completo.

## Indicadores longitudinales añadidos

El tutorial incorpora dos indicadores para seguimiento funcional post-ictus bajo evaluaciones repetidas y estandarizadas:

1. `delta_active_rms_pct`: cambio porcentual del RMS durante periodos activos respecto a la visita basal.
2. `mdf_gap_pct`: distancia relativa de frecuencia mediana entre músculo parético y músculo contralateral homólogo.

Estos indicadores no deben interpretarse como evolución del daño cerebral. Describen cambios neuromusculares compatibles con evolución de la función motora y requieren control del protocolo de adquisición. CVES es transversal, por lo que sirve para enseñar el cálculo, no para demostrar una trayectoria de recuperación longitudinal.
