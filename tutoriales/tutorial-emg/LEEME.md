# Dashboard de EMG con dataset libre

Caso principal: neuropatía por radiculopatía L5. Fuente: EMGDB 1.0.0,
PhysioNet, DOI https://doi.org/10.13026/C24S3D. Licencia ODC-By 1.0.
Los tres registros son personas distintas, no visitas sucesivas.

Con Python 3.11 o 3.12 y uv:

```bash
uv init
uv add -r requirements.txt
uv run python datos.py
uv run python verificar.py
uv run streamlit run app.py
```

Los datos originales DAT/HEA están incluidos; el lector verifica SHA-256.
Para repetir descarga, mueva la carpeta datos a otro lugar y ejecute datos.py.
Un hash incorrecto interrumpe el proceso; no omita esa validación.

```bash
quarto render tutorial-dashboard-emg.qmd --to pdf
```

El QMD no ejecuta Python durante la composición y no requiere archivos externos.
Contiene todos los módulos para reconstruir el proyecto sin el paquete ZIP.
El PDF y el código sustituyen el ejemplo sintético de la versión anterior.
