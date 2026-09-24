# ictus-semg

Tablero interactivo de sEMG de miembro superior en ictus — asignatura SBVI.

```bash
uv sync                      # crea .venv e instala dependencias bloqueadas
uv run pytest                # pruebas de la lógica de señal
uv run streamlit run app.py  # abre el tablero en http://localhost:8501
```

La señal sintética es didáctica; no representa datos de pacientes.
