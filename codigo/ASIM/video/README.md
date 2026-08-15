# Descenso de gradiente animado

Este paquete genera un GIF didáctico de **regresión lineal** y **regresión
logística**. En cada fotograma se muestran el modelo actual, sus parámetros y la
reducción de la función de costo.

## Archivos

- `descenso_gradiente_animado.py`: implementación y animación.
- `datos_regresion_lineal.csv`: datos artificiales para la respuesta continua.
- `datos_regresion_logistica.csv`: datos artificiales para las clases 0 y 1.
- `requirements.txt`: dependencias mínimas.
- `descenso_gradiente_regresiones.gif`: resultado listo para las diapositivas.

## Ejecución

Desde esta carpeta, con `uv`:

```bash
uv venv
uv pip install -r requirements.txt
uv run python descenso_gradiente_animado.py
```

Alternativamente, con las herramientas estándar de Python:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python descenso_gradiente_animado.py
```

En Windows PowerShell, la activación es:

```powershell
.venv\Scripts\Activate.ps1
```

La ejecución predeterminada crea un GIF de 1280 × 720 píxeles, con 100
iteraciones y 12 fotogramas por segundo. Estos valores pueden modificarse:

```bash
python descenso_gradiente_animado.py \
  --iteraciones 100 \
  --fps 12 \
  --dpi 100 \
  --salida descenso_gradiente_regresiones.gif
```

## Inserción en Quarto + revealjs

Copie el GIF a la carpeta de recursos del sitio y utilice:

```markdown
![](ruta/descenso_gradiente_regresiones.gif){fig-alt="Descenso de gradiente ajustando una regresión lineal y una regresión logística" width="100%"}
```

Si se desea centrarlo y controlar la altura dentro de una diapositiva:

```markdown
::: {.r-stretch}
![](ruta/descenso_gradiente_regresiones.gif){fig-alt="Convergencia mediante descenso de gradiente"}
:::
```

## Lectura didáctica del GIF

- La línea naranja del panel lineal corresponde a `ŷ = w·x + b`.
- La curva naranja del panel logístico corresponde a `p = σ(w·x + b)`.
- Las líneas verdes discontinuas muestran el ajuste alcanzado en la última
  iteración.
- Los gráficos inferiores presentan la disminución de cada función de costo en
  escala logarítmica.
- La frontera logística se encuentra donde la probabilidad estimada es 0,5.

Los dos entrenamientos usan descenso de gradiente por lotes y registran el
estado inicial como iteración 0. Por tanto, el GIF muestra todas las
actualizaciones desde la iteración 0 hasta la 100.
