# SYSB: diapositivas revisadas

Los tres QMD contienen las notas por diapositiva. Mantenga la carpeta completa para conservar configuración, estilos, bibliografía y figuras.

- `informe-revision.html`: comparación, correcciones y discrepancias de las fuentes.
- `semanas-01-05.qmd`, `semanas-06-10.qmd`, `semanas-11-15.qmd`: fuentes editables.
- Archivos HTML del mismo nombre: presentaciones Revealjs.
- Archivos PDF del mismo nombre: presentaciones Beamer.
- `notas-del-docente.html`: notas con las secciones del libro, en lectura continua.
- `correspondencia-diapositivas.json`: correspondencia completa para reutilización.
- `codigo/ejemplos_verificados.py`: figuras y casos sintéticos reproducibles.

## Uso y compilación

Instale Quarto y una distribución TeX con XeLaTeX, Beamer, `fontawesome5`, fuentes DejaVu y paquetes habituales de Quarto. La compilación se verificó con Quarto 1.7.34 y XeLaTeX de TeX Live 2023. No se ejecutan automáticamente los fragmentos Python durante el render.

```bash
quarto render semanas-01-05.qmd --to revealjs
quarto render semanas-01-05.qmd --to beamer
bash compilar.sh
```

Los HTML se pueden abrir en el navegador; la representación de ecuaciones carga KaTeX desde una CDN y requiere conexión a Internet. Los PDF ya contienen las ecuaciones renderizadas. Para la vista del presentador, sirva la carpeta localmente, abra la presentación y pulse `S`; permita la ventana emergente. La vista de notas puede estar limitada cuando se usa directamente `file://`.

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Abra `http://127.0.0.1:8000/semanas-01-05.html` en el navegador. Las notas también están en `notas-del-docente.html`, que puede abrir sin servidor.

## Ejemplos Python

Con uv:

```bash
uv venv
uv pip install -r codigo/requirements.txt
uv run python codigo/ejemplos_verificados.py
```

El script genera cuatro figuras a partir de datos sintéticos y comprueba RMS, amplitud unilateral, Parseval con ventana, retardo conocido, convolución, Bode y reconstrucción wavelet. Los fragmentos de las diapositivas que usan `x` y `fs` reciben el segmento de la actividad; las etiquetas y sujetos de clasificación deben suministrarse para ese problema.

## Alcance de las referencias

Se usa el libro guía aportado, de 2018 y tercera edición. No se usa como equivalente la numeración de la segunda edición de 2012 que cita el microcurrículo. Las páginas de las notas son posiciones del visor del PDF aportado. Wavelets y clasificación se apoyan explícitamente en los complementos, porque no tienen sección directa en el texto guía.

El paquete no incluye copias de los libros. Los recursos gráficos del logo y los estilos externos originales no estaban adjuntos; los estilos incluidos conservan la proporción 16:9 y la base cromática disponible en los QMD.
