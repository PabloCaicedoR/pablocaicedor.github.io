# Contexto del workspace `pablocaicedor.github.io` (BioForja)

> **Uso de este archivo.** Es el punto de arranque para nuevas tareas en este repositorio.
> Léelo antes de tocar código o contenido. Describe la arquitectura, las convenciones,
> el flujo de build/publicación y los problemas conocidos.
>
> - **Fecha del análisis:** 2026-09-23 · **Commit base:** `43a27e4` (rama `master`)
> - **Nombre con `_` a propósito:** Quarto ignora archivos que empiezan por `_` o `.`.
>   Cualquier otro `.md`/`.qmd`/`.ipynb` del repo **se publica en el sitio público**
>   (solo `README.md` queda excluido). No renombres este archivo sin tenerlo en cuenta.
> - Si algo de aquí contradice el repo, manda el repo: actualiza este archivo.

---

## 1. Qué es

Sitio web estático de docencia del **Ph.D. Pablo Eduardo Caicedo-Rodríguez** (Ingeniería
Biomédica, Escuela Colombiana de Ingeniería Julio Garavito), llamado **BioForja**. Contiene
páginas de asignaturas, presentaciones (Reveal.js/Beamer), laboratorios, tutoriales,
rúbricas, exámenes resueltos y proyectos de investigación. Se construye con **Quarto**
(proyecto `website`). La salida va a `docs/` y se versiona en `master`, que es la fuente
de GitHub Pages (`https://pablocaicedor.github.io/`).

- Remoto: `https://github.com/PabloCaicedoR/pablocaicedor.github.io.git`
- Idioma del contenido: **español** (`lang: es`). Código y comentarios, mayormente en español.
- No hay CI (`.github/` no existe): el sitio se renderiza **en local** y se commitea `docs/`.

---

## 2. Stack y entorno local (verificado 2026-09-23)

| Herramienta | Versión / estado | Notas |
| --- | --- | --- |
| Quarto | 1.8.27 (`/opt/quarto`) | `quarto list tools`: TinyTeX y Chromium **no** instalados |
| TeX | TeX Live 2023 (Debian), XeLaTeX | Para `pdf`/`beamer` |
| R | 4.6.1 | Tiene `reticulate, knitr, rmarkdown, DiagrammeR, kableExtra, tidyverse, cowplot, ggfx` |
| Python | 3.13.11 en `.venv/` (gestionado con **uv** 0.10.9) | `pyproject.toml` + `uv.lock`; `uv sync --dry-run` → al día |
| Paquetes Py clave | numpy 2.4, scipy 1.17, pandas 2.3, matplotlib 3.10, plotly 6.5, scikit-learn 1.8, pywavelets, opencv, **torch 2.14+cu130**, **tensorflow 2.20 [and-cuda]** | `.venv` pesa ~11 GB. **No** incluye `streamlit` ni `wfdb` (los tutoriales de dashboards usan su propio `uv`) |
| Kernels Jupyter | `python3` → `.venv/share/jupyter/kernels/python3` ✅; `bioforge`, `eldermotion`, `laparo`, `spasticassessment` en `~/.local/share/jupyter/kernels` | Estos 4 apuntan a `~/Data_Cantatio/...`, **que ya no existe** |
| `.Rprofile` | `reticulate::use_virtualenv("./.venv", required = TRUE)` | R ejecuta Python con el `.venv` del proyecto |

**Motores de ejecución (importante):**

- Documentos cuyo **primer chunk es `{r}`** usan **knitr**. Python corre vía `reticulate` con `.venv`.
  Es el patrón de casi todas las presentaciones "clásicas" (SYSB/PSIM/APSB/ASIM v1).
- Documentos **solo con chunks `{python}`** usan el motor **Jupyter**. En formato `html`
  heredan `jupyter: eci-class-tf` de `_quarto.yml`, y **ese kernel no existe** (ver §8.1).
  En `revealjs` no se hereda y se usa `python3`, que funciona.
- Varios documentos recientes usan `execute: enabled: false` o `eval: false`: se muestra
  el código, pero no se ejecuta.
- Lenguajes de chunk en el repo: python (~612), r (~59), mermaid (35), dot (31), tikz (8), bash (3).

---

## 3. Configuración del sitio (`_quarto.yml`)

```yaml
project: { type: website, output-dir: docs, resources: ["recursos/imagenes/generales/**"] }
website:
  title: "BioForja"
  favicon / navbar.logo: recursos/imagenes/generales/bioforja_icon.svg
  navbar.left: [index.qmd "Principal", about.qmd "Acerca de mi"]
format.html: theme cosmo, css recursos/estilos/styles_site.css, mermaid theme forest (svg),
             text-align justify, toc false, jupyter: eci-class-tf   # ← kernel inexistente
```

- **No hay lista `render:`**: Quarto renderiza *todos* los `.qmd/.md/.ipynb` del proyecto
  (salvo los que empiezan por `_` o `.`, y `README.md`). Por eso borradores, notas docentes
  y archivos `*_back*.qmd` también se publican (ver §8.5).
- `index.qmd` tiene tres *listings* en cuadrícula: `clases/*.qmd`, `tutoriales/*.qmd`,
  `proyectos/*.qmd`. Muestran los campos `image, date, title, description` y se ordenan por
  `date desc`. **Solo entran los `.qmd` del primer nivel** de esas carpetas.
- `about.qmd` usa la plantilla `about: marquee` con enlaces sociales.
- Hay un bloque `navbar: right:` vacío a nivel raíz (mal ubicado, inofensivo).

---

## 4. Mapa del repositorio

| Ruta | Contenido | ¿Se publica? |
| --- | --- | --- |
| `index.qmd`, `about.qmd` | Portada con *listings* y página personal | Sí |
| `clases/` | Una página por asignatura (`Class_<SIGLA>.qmd`) + `talleres.qmd`. Es el "índice" de cada curso: enlaza presentaciones, labs, recursos y exámenes | Sí (listing) |
| `presentaciones/<SIGLA>/` | Diapositivas `LectNNN_*.qmd` (Reveal.js; algunas Beamer). `SYSB/nuevaVersion*/` = rediseño por semanas. `TALLERES/` = talleres de divulgación | Sí |
| `laboratorios/<SIGLA>/` | Guías de laboratorio (`.qmd`, `.ipynb`). `PythonLanguage/` = labs transversales (Python reproducible, dashboards EMG). `anteriores/` y `Previous/` = versiones antiguas | Sí |
| `tutoriales/` | Tutoriales de herramientas (uv, Quarto, Neovim, kitty, Git/GitHub, terminal, instalación Python/R), IA y educación. Subcarpetas `tutorial-emg/` y `tutorial_dashboard_semg_ictus/` = proyectos Streamlit autocontenidos | Sí (solo el 1.er nivel entra al listing) |
| `proyectos/` | Fichas de proyectos de investigación. `Sabana/` = evaluación inercial del equilibrio (Xsens/Vicon) | Sí (`Sabana/` no entra al listing) |
| `recursos/` | Activos compartidos: `estilos/` (SCSS/CSS/TeX), `imagenes/` (por clase/tutorial/presentación), `documentos/` (apuntes `.qmd` por curso, plantilla LaTeX `LatexTemplate/plantillasCurso.zip`), `examenes/`, `talleres/` (enunciados y soluciones), `Codigo/`, `videos/`, `Legacy/` | Sí (lo referenciado + los `.qmd`) |
| `rubricas/` | Rúbricas (`.qmd` + `.tex`) | Sí, sin enlaces desde el sitio |
| `codigo/` | Notebooks de clase por curso (`cod00N_*.ipynb`), app `PSIM/app/main.py`, animación de descenso de gradiente (`ASIM/video/`) | Sí (los `.ipynb` se renderizan) |
| `data/` | Datasets de ejemplo (EMG, DICOM, NIfTI, CSV, `teaching/` con CT/MRI/US/RX) | Está en `.gitignore`, pero **82 archivos siguen versionados** |
| `accelerolog/` | App Kivy/Android independiente (acelerómetro → CSV). No forma parte del sitio | No (no hay `.qmd`) |
| `Arquitectura_Sitio/` | Documento PDF previo sobre la arquitectura (2026-07-29) | Sí |
| `_extensions/` | `metropolis-theme`, `coatless-quarto/illinois` | — |
| `_freeze/` | Resultados congelados (solo `presentaciones/ASIM` y `site_libs`) | — |
| `docs/` | **Salida generada** (~1.2 GB, 754+ archivos versionados). No editar a mano | Es el sitio |
| `.quarto/` | Caché de Quarto (ignorado) | — |

**Tamaños:** `.git` ≈ 8.7 GB (pack 3.1 GB); `.venv` ≈ 11 GB; `docs` ≈ 1.2 GB; `recursos` ≈ 312 MB.
Hay archivos grandes versionados, por ejemplo: `data/imagen_nii.nii` (62 MB),
`recursos/Legacy/prueba.zip` (35 MB) y HTML autocontenidos de hasta 72 MB en `docs/`.

---

## 5. Asignaturas

| Sigla | Nombre | Página | Estado / notas |
| --- | --- | --- | --- |
| **SYSB** | Sistemas y Señales Biomédicos | `clases/Class_SYSB.qmd` | Activa 2026-II. 10 presentaciones `Lect001–010`. Rediseño por semanas en `nuevaVersion/` (enlazada: semanas 01–05), `nuevaVersion02/` (qmd + `formato-reveal.yml`/`formato-beamer.yml`) y `nuevaVersion03/` (más reciente: semanas 01–15, `notas-del-docente.qmd`, `informe-revision.qmd`; libro guía Semmlow 3.ª ed.). Horario: L y J 10:00 (F204/F206), lab M 10:00 (I1-308) |
| **PSIM** | Procesado de Señales e Imágenes Médicas | `clases/Class_PSIM.qmd` | Activa. Presentaciones de imagen, Fourier y wavelets. Labs 01–03 enlazados |
| **PAIM** | Procesado Avanzado de Imágenes Médicas | `clases/Class_PAIM.qmd` | RX, TAC, RM. Labs `lab00–lab03` (MTF, Beer–Lambert, reconstrucción TAC). Apuntes en `recursos/documentos/PAIM/` |
| **ASIM** | Aprendizaje automático para señales e imágenes médicas | `clases/Class_ASIM.qmd` | Serie `v2Lect001–005` (ML, descenso de gradiente, NN, CNN, RNN/LSTM/GRU). Trabajo más reciente: `laboratorios/ASIM/Taller_Redes_Neuronales_desde_Regresion_Logistica.qmd` (dataset BRFSS2015 en `laboratorios/ASIM/data/`) |
| **SBVI** | Señales Bioeléctricas y Visualización Interactiva | `clases/Class_SBVI.qmd` | Curso nuevo (2026-II). ECG/EMG/EEG + dashboards. Presentación `presentaciones/SBVI/fisiologia_ictus_semg_reveal.qmd`. Labs en `laboratorios/PythonLanguage/`. Tutoriales Streamlit en `tutoriales/tutorial-emg/` y `tutoriales/tutorial_dashboard_semg_ictus/` (**no enlazados** desde la página del curso) |
| **BIST** | Bioestadística | `clases/Class_BIST.qmd` | Esqueleto casi vacío |
| **APSB** | Adq. y Proc. de Señales Biomédicas en Tecnologías de Borde | `clases/Class_APSB.qmd.bak` (desactivada) | Presentaciones y labs aún existen y se publican (Jetson Nano, Linux, EDA) |
| Talleres | Talleres de divulgación | `clases/talleres.qmd` | Tarjetas-imagen que enlazan a `presentaciones/TALLERES/*.qmd` |

Todas las páginas de curso comparten bloques: *Chatbots para la asignatura* (NotebookLM),
*Plantilla para entrega de trabajos* y *Evaluaciones* (típico: parciales 15%+15%, final 20%,
labs 30%, proyecto 20%).

---

## 6. Convenciones de autoría

### 6.1 Nombres y rutas
- Presentaciones: `presentaciones/<SIGLA>/LectNNN_<Tema>.qmd` (ASIM usa `v2LectNNN_*`).
- Labs: `laboratorios/<SIGLA>/labNN_<Tema>.qmd`. Los más nuevos usan nombres largos:
  `Laboratorio_0N_<Tema>_Guia_Estudiante.qmd`.
- Notebooks de clase: `codigo/<SIGLA>/codNNN_<tema>.ipynb` (`_sol` = solución).
- Imágenes: `recursos/imagenes/{clases,Presentaciones,tutoriales,...}/<sigla|tema>/`.
- Las rutas son **relativas** al archivo: desde `presentaciones/X/` o `laboratorios/X/` se usa `../../recursos/...`;
  desde `clases/` o `tutoriales/`, `../recursos/...`.
- Autor habitual en YAML: `"Ph.D. Pablo Eduardo Caicedo R."` o `"Ph.D. Pablo Eduardo Caicedo-Rodríguez"`.

### 6.2 Plantilla de página de curso (`clases/Class_<SIGLA>.qmd`)
```yaml
---
title: "<Nombre de la asignatura>"
description: "Sitio de la asignatura <Nombre> en la Escuela Colombiana de Ingeniería"
lang: es
author: "Ph.D. Pablo Eduardo Caicedo R."
date: last-modified
image: "../recursos/imagenes/generales/<sigla>.png"   # obligatorio para la tarjeta del listing
---
```
Secciones: Introducción → Material del Curso (Presentaciones, Datos, Recursos, Chatbots,
Códigos, Laboratorios, Talleres & Exámenes Anteriores) → Plantilla → Evaluaciones → Horarios.

### 6.3 Plantilla de presentación Reveal.js (patrón dominante)
```yaml
format:
  revealjs:
    code-tools: true
    code-overflow: wrap
    code-line-numbers: true
    code-copy: true
    fig-align: center
    self-contained: true          # 78 archivos; opción obsoleta → preferir embed-resources: true
    theme: [simple, ../../recursos/estilos/metropolis.scss]
    css: ../../recursos/estilos/styles_pres.scss
    logo: ../../recursos/imagenes/generales/Escuela_Rosario_logo.png
    footer: <https://pablocaicedor.github.io/>
    slide-number: true
    preview-links: auto
    transition: fade
    progress: true
    scrollable: true
```
Las presentaciones "clásicas" abren con un chunk `{r}` que carga `DiagrammeR, reticulate,
kableExtra, tidyverse, knitr, cowplot, ggfx`, fija `knitr::opts_chunk$set(echo = FALSE)` y
define un *hook* de tamaño. Luego sigue un chunk `{python}` con `plt.rcParams` y
`text.usetex: True` (requiere LaTeX) y la fuente Fira Code.
Las más nuevas (ASIM v2, SBVI) no usan R: `execute: {echo: false, warning: false, freeze: auto}`.

### 6.4 Plantilla de lab/tutorial HTML (patrón reciente)
`format.html`: `theme: cosmo`, `embed-resources: true`, `toc: true`, `toc-location: left`,
`number-sections`, `code-copy`, `code-tools`, `link-external-newwindow`. Suele añadir
`execute: enabled: false`.

### 6.5 Estilos disponibles (`recursos/estilos/`)
`styles_site.css` (sitio), `metropolis.scss` + `styles_pres.scss` (presentaciones; clases
`.imagen-centrada`, `.ecg-grid`, …), `reveal.scss` + `beamer-header.tex` (SYSB nuevaVersion),
`styles_pres_promise.css`, `styles_unicauca.css`, e iconos de *callouts* en PNG.

---

## 7. Build y publicación

```bash
# Renderizar UN archivo (lo habitual; actualiza docs/ y search.json)
quarto render presentaciones/SYSB/Lect007_DigitalFilters.qmd

# Previsualizar
quarto preview presentaciones/SYSB/Lect007_DigitalFilters.qmd

# Renderizar TODO el sitio: lento y hoy falla por el kernel (§8.1).
# Ejecuta Python/R en todos los documentos. Evitar salvo que sea necesario.
quarto render

# Publicar: commitear las fuentes + docs/ y hacer push a master
git add <fuentes> docs/ && git commit && git push
```

- Casos especiales con su propio README:
  `presentaciones/SYSB/nuevaVersion/README_RENDER.md` (`--to revealjs` / `--to beamer`),
  `presentaciones/SYSB/nuevaVersion02/README.md` (`--metadata-file formato-*.yml`),
  `tutoriales/tutorial-emg/LEEME.md` y `tutoriales/tutorial_dashboard_semg_ictus/README.md`
  (entorno `uv` propio + `streamlit run`), `accelerolog/README.md` (Buildozer/Android).
- Si aparece un `site_libs/` en la **raíz**, es residuo de un render fuera del proyecto.
  No se versiona.
- `.quarto_ipynb` son intermedios del motor Jupyter. Si quedan huérfanos, indican un render fallido.

---

## 8. Problemas conocidos y deuda técnica (priorizados)

### 8.1 🔴 El render de documentos HTML con motor Jupyter falla
`_quarto.yml` fija `jupyter: eci-class-tf` en `format.html`, y ese kernel **no existe**.
Lo verifiqué con un render de prueba el 2026-09-23:
`ERROR: Jupyter kernel 'eci-class-tf' not found. Known kernels: python3, laparo, ...`.
Afecta a 22 `.qmd` que solo tienen chunks Python y salen en HTML: tutoriales (`TutorialPython`,
`ExpansionTaylor`, `FlujoTrabajoSoA`, `tutorial_academico_neovim_desde_cero`), labs
(`PythonLanguage/*`, `PAIM/lab02–03`, `SYSB/Laboratorio_02–03`, `ASIM/Taller_Redes_*`) y apuntes
de `recursos/documentos/*`. Las presentaciones Reveal.js no se ven afectadas.
Opciones: (a) cambiar a `jupyter: python3` (ya apunta al `.venv`); (b) registrar el kernel:
`.venv/bin/python -m ipykernel install --user --name eci-class-tf`.

### 8.2 🔴 `.venv` movido: scripts con *shebang* roto
El entorno se creó en `~/Data_Cantatio/pablocaicedor.github.io/.venv`. `.venv/bin/python` funciona,
pero los ejecutables de consola (`jupyter`, etc.) apuntan a la ruta vieja y fallan.
Solución: `uv sync --reinstall` (o borrar `.venv` y hacer `uv sync`). Los kernels de usuario
`bioforge/eldermotion/laparo/spasticassessment` también apuntan a `~/Data_Cantatio` (proyectos hermanos).

### 8.3 🟠 Seguridad: `.envrc`
Contiene un `GH_TOKEN` (PAT de GitHub) en texto plano. Está en `.gitignore` y **nunca se
commiteó**: busqué en todo el historial con `git log -S`. Además, la sintaxis
`export GH_TOKEN = "..."` (con espacios) es inválida en bash/direnv. Debe ser `export GH_TOKEN="..."`.
**Nunca copiar el valor del token en archivos, commits ni mensajes.**

### 8.4 🟠 Higiene de git

- **(Corregido en local el 2026-09-23, pendiente de commit)**: la regla genérica de Python `dist/`
  en `.gitignore` excluía `docs/site_libs/revealjs/dist/` (núcleo de Reveal.js). Las presentaciones
  **no autocontenidas** (sin `self-contained`/`embed-resources`, por ejemplo `SBVI/fisiologia_ictus_semg_reveal`,
  que usa `chalkboard`, incompatible con `embed-resources`) se veían en GitHub Pages como una página
  HTML sin formato porque `reveal.js`/`reveal.css` daban 404. Se cambió a `/dist/`. Si en el futuro
  una presentación se ve bien en local pero no en Pages, revisa primero con `git check-ignore -v`
  los archivos de `docs/site_libs/` que referencia.
- El patrón `**/*.quarto_ipynb` de `.gitignore` no cubre las variantes `*.quarto_ipynb_1`,
  así que **25 intermedios están versionados**. Conviene usar `**/*.quarto_ipynb*` y `git rm --cached`.
- Hay 25 subproductos LaTeX versionados (`.aux/.log/.fls/.fdb_latexmk/.synctex.gz/.nav/.snm/.toc/.out`),
  por ejemplo en `tutoriales/aux/`, `recursos/documentos/APRENDIZAJE/aux/` y `presentaciones/SYSB/nuevaVersion02/`.
- `/data/` está ignorado, pero 82 archivos siguen versionados.
- Dataset duplicado: `data/teaching/` ≈ `presentaciones/TALLERES/teaching_dataset/` (7.4 MB; el
  segundo tiene además `manifest.xlsx`).
- 9 archivos `*.bak` y varios `*_back*.qmd` sueltos.
- Los mensajes de commit apenas describen los cambios (`update` ×100+, "Refactor code structure…"
  repetido para cambios de contenido).
- `docs/` no tiene `.nojekyll` (hoy no rompe nada porque ninguna carpeta publicada empieza por `_`).

### 8.5 🟠 Contenido publicado que quizá no debería serlo
Al no haber lista `render:`, se publican también: `presentaciones/SYSB/nuevaVersion03/notas-del-docente.html`
e `informe-revision.html`, `presentaciones/PSIM/prueba.html`, `Lect008_Wavelet_back.html`,
`presentaciones/TALLERES/promocionProcesamientoImagenes_back1.html`, `rubricas/*`,
`Arquitectura_Sitio`, `recursos/documentos/Doctorado/*` y los `.md` sueltos (`LEEME`, `ATRIBUCION-DATOS`,
`README_RENDER`, `IndiceExtendido_*`). Para excluir algo: renombrarlo con `_`, o añadir
`project.render: ["**/*.qmd", "!ruta/..."]` en `_quarto.yml`.

### 8.6 🟡 Portada y listings
- `index.qmd` usa las clases `.hero-bioforja` / `.hero-bioforja-img`, pero **no están definidas**
  en ningún CSS. `styles_site.css` define `.hero-cover` / `.hero-cover-img` (y `.home-content`), sin usar.
- El orden `date desc` no es fiable: `ExpansionTaylor` y `TutorialPython` tienen fechas en texto
  ("Febrero 6, 2023"), y `tutInstallPythonR`, `tutorial_academico_neovim_desde_cero`,
  `tutorial_github_popos`, `tutorial_kitty_starship_bioforge` y `tutorial_uv` no tienen `date`.
- No aparecen en ningún listing ni enlace: `tutoriales/tutorial-emg/`, `tutoriales/tutorial_dashboard_semg_ictus/`,
  `tutoriales/PropuestaTallerSonrisa/`, `proyectos/Sabana/InertialBalanceAssessment.qmd`,
  `laboratorios/ASIM/Taller_Redes_Neuronales_desde_Regresion_Logistica.qmd` (aún no enlazado en `Class_ASIM`).

### 8.7 🟡 Errores de contenido detectados
- `description` copiado de otra asignatura: `Class_SBVI` y `Class_BIST` dicen "Sistemas y Señales
  Biomédicoss" (con errata). `Class_ASIM` dice "Procesado de Señales e Imágenes Médicas".
- `Class_SYSB.qmd`: viñeta vacía `* ` antes de "## Laboratorios"; "debioseñales".
- `Class_SBVI.qmd`: "Fisología".
- `presentaciones/PAIM/Lect003_RayosX.qmd`: se coló un texto de asistente IA
  ("A continuación, presento la estructura de las diapositivas en Quarto…").
- README con archivos inexistentes: `accelerolog/` (`buildozer.spec`, `assets/`),
  `codigo/ASIM/video/` (`requirements.txt`), `SYSB/nuevaVersion02/` (`compilar.sh`, `estilos/`).

### 8.8 🟡 Otros
- `self-contained: true` (obsoleto) en 78 documentos genera HTML enormes (hasta 72 MB). La opción
  vigente es `embed-resources: true`, que ya usan 22 archivos.
- `metropolis.scss` importa Fira Code desde `cdn.rawgit.com` (servicio cerrado) y hay fuentes de Google Fonts.
- `pyproject.toml` trae TensorFlow y PyTorch con CUDA. Un entorno nuevo es pesado.
  `README.md` todavía documenta instalación con `mamba`, no con `uv`.
- `compilation.txt` está vacío y versionado.

---

## 9. Recetas para tareas frecuentes

**Agregar una presentación a un curso**
1. Copiar el YAML de una presentación hermana del mismo curso (§6.3). Mantener las rutas `../../recursos/...`.
2. Poner las imágenes en `recursos/imagenes/Presentaciones/<SIGLA>/`.
3. Enlazarla en `clases/Class_<SIGLA>.qmd` → "## Presentaciones".
4. `quarto render presentaciones/<SIGLA>/<archivo>.qmd` y, si cambió, también la página del curso.
5. Commitear la fuente y los archivos nuevos o modificados de `docs/`.

**Agregar un laboratorio**: igual, en `laboratorios/<SIGLA>/`, con la plantilla §6.4,
y enlazarlo en "## Laboratorios". Si es transversal, va en `laboratorios/PythonLanguage/`.

**Agregar un tutorial al listing de la portada**: crear `tutoriales/<nombre>.qmd` en el primer
nivel, con `title`, `description`, `image` y `date` en formato ISO (`YYYY-MM-DD` o `last-modified`).
Luego renderizar el archivo **e** `index.qmd`.

**Agregar una asignatura**: crear `clases/Class_<SIGLA>.qmd` (§6.2) con su imagen en
`recursos/imagenes/generales/<sigla>.png`, y crear `presentaciones/<SIGLA>/` y `laboratorios/<SIGLA>/`.

**Antes de cualquier render con Python**: resolver §8.1 y §8.2, o renderizar solo documentos
con motor knitr o Reveal.js.

---

## 10. Pendiente de verificar
- La configuración de GitHub Pages (rama `master`, carpeta `/docs`) es una **inferencia**: no la
  consulté por API.
- Dónde se hicieron los últimos renders con `eci-class-tf`. Las salidas de `docs/` fechadas el
  2026-09-21 09:27 coinciden con la hora de un checkout (misma marca que `uv.lock`,
  `pyproject.toml`, etc.), así que probablemente vienen de otra máquina o de la ubicación anterior.
