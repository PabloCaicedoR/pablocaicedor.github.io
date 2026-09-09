# SYSB · Diapositivas de las semanas 6–10

El proyecto conserva el contenido separado de la presentación visual:

- `semanas-06-10.qmd`: contenido académico único.
- `formato-reveal.yml`: configuración de Reveal.js.
- `formato-beamer.yml`: configuración de Beamer/PDF.
- `estilos/reveal.scss`: tema visual web.
- `estilos/beamer-header.tex`: tema tipográfico y cromático del PDF.
- `referencias.bib`: bibliografía.
- `compilar.sh`: compilación de ambas salidas.

## Compilación

```bash
quarto render semanas-06-10.qmd --to revealjs \
  --metadata-file formato-reveal.yml \
  --output semanas-06-10-reveal.html

TEXINPUTS=.: quarto render semanas-06-10.qmd --to beamer \
  --metadata-file formato-beamer.yml \
  --output semanas-06-10-beamer.pdf
```

El contenido corresponde a las diez sesiones de 90 minutos de las semanas 6–10 del programa de SYSB. Los ejemplos usan Python como herramienta de exploración. La ejecución permanece desactivada para que el render no dependa de un entorno computacional específico.

