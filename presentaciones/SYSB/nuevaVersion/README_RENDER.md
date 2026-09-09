# SYSB · Semanas 06–10

El archivo principal es `SYSB_Semanas_06_10.qmd`. Su encabezado YAML produce dos formatos desde la misma fuente:

```bash
quarto render SYSB_Semanas_06_10.qmd --to revealjs
quarto render SYSB_Semanas_06_10.qmd --to beamer
```

## Dependencias

- Quarto
- Una distribución TeX con XeLaTeX para Beamer
- Python con NumPy, SciPy y Matplotlib únicamente si se desean regenerar las figuras

Las figuras ya generadas se encuentran en `assets/`. El script `generate_figures.py` permite reproducirlas con señales sintéticas.

## Salidas esperadas

- `SYSB_Semanas_06_10.html`: presentación RevealJS autocontenida
- `SYSB_Semanas_06_10.pdf`: presentación Beamer 16:9

