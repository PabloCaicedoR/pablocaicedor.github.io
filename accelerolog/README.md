# Accelerolog

Aplicación móvil en Kivy para capturar datos de acelerómetro y sensores adicionales en Android (API 31). Permite visualizar series en tiempo real, calibrar el dispositivo, almacenar mediciones en CSV y compartir resultados mediante el menú del sistema.

## Estructura del proyecto

```
accelerolog/
  main.py                # Punto de entrada de la app Kivy
  ui.py                  # Pantallas (Inicio, Adquisición, Calibración)
  sensors.py             # Gestión de sensores con Plyer, modo demo
  calibration.py         # Rutina de calibración, offsets y persistencia
  storage.py             # Buffer doble y escritura batch a CSV
  export.py              # Compartir CSV con intents de Android
  utils/
    __init__.py
    time.py              # Sellos de tiempo UNIX ms + ISO-8601
  tests/
    __init__.py
    mock_stream.py       # Flujo sintético para pruebas
    sample_output.csv    # Ejemplo de CSV generado
  assets/
    icons/               # Espacio para iconos / recursos estáticos
  buildozer.spec         # Configuración para compilar con Buildozer
  README.md              # Este documento
```

## Requisitos previos

- Python 3.9 o superior (se recomienda 3.10).
- paquete `buildozer` (para empaquetar en Android) y `Cython`.
- SDK/NDK de Android (Buildozer los descarga en la primera compilación).
- Plataforma de desarrollo Linux o WSL2 (Buildozer no soporta Windows nativo).

Dependencias de sistema (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv git zip unzip openjdk-11-jdk \
    libffi-dev libssl-dev autoconf automake libtool pkg-config
pip install --user --upgrade cython buildozer
```

## Preparación del entorno

1. Clona el repositorio o copia la carpeta `accelerolog/` en tu área de trabajo.
2. Crea y activa un entorno virtual:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

3. Instala dependencias Python para ejecución local:

   ```bash
   pip install kivy plyer kivy_garden.graph
   garden install graph  # Necesario para kivy_garden.graph
   ```

4. Si deseas emular sensores en escritorio, habilita el modo demo directamente desde la UI.

## Compilación con Buildozer

La aplicación está configurada para Android API 31.

```bash
cd accelerolog
buildozer android debug
# Primera vez: se descargan SDK/NDK, puede tardar.
```

Instalación y ejecución en un dispositivo conectado por USB:

```bash
buildozer android debug deploy run
```

Observa logs en tiempo real:

```bash
buildozer android logcat
```

### Permisos

- `READ_EXTERNAL_STORAGE` / `WRITE_EXTERNAL_STORAGE`: acceso al almacenamiento compartido para guardar y compartir CSV.
- Permisos en ejecución: la app solicita únicamente los necesarios para acceder a archivos; el acceso a sensores de movimiento no requiere permisos adicionales en Android 12.

## Uso de la aplicación

1. **Pantalla de inicio**: selecciona los sensores disponibles y la tasa de muestreo (10–200 Hz). Si el hardware no soporta la tasa deseada, la app ajusta al valor más cercano y muestra un aviso.
2. **Modo demo**: genera datos sintéticos (senoides + ruido) para pruebas en escritorio. El CSV demo se guarda como `accelerolog_demo.csv`.
3. **Pantalla de adquisición**:
   - Gráficos tiempo-real para cada sensor con triple eje y buffer circular (600 muestras).
   - Lecturas numéricas actualizadas.
   - Botones: *Detener*, *Guardar CSV*, *Compartir*, *Calibrar*.
4. **Calibración**:
   - Guía para apoyar el dispositivo sobre una superficie estable.
   - Captura ~6 segundos de datos, calcula offset y desviación estándar, guarda en `calibration/calibration.json`.
   - Los offsets se aplican automáticamente a lecturas futuras del sensor calibrado.
5. Los CSV se almacenan en `records/accelerolog_<timestamp>.csv` (dentro de `user_data_dir` en Android). El archivo incluye:
   - Fila inicial con metadatos (dispositivo, versión Android, app, tasa objetivo).
   - Cabecera `timestamp_unix_ms,timestamp_iso,sensor,ax,ay,az,extra1,extra2`.
   - Tasa de escritura en lotes (buffer doble) para minimizar bloqueos.

## Verificación y pruebas

- Para validar el flujo sin hardware real:

  ```bash
  python -m tests.mock_stream
  ```

  (Puedes importar `MockSensorStream` en scripts personalizados para alimentar la UI).

- CSV de ejemplo: `tests/sample_output.csv`. Verificación rápida:

  ```bash
  python - <<'PY'
  import csv, pathlib
  path = pathlib.Path("tests/sample_output.csv")
  with path.open() as fh:
      reader = csv.reader(fh)
      for i, row in enumerate(reader):
          print(row)
          if i > 3:
              break
  PY
  ```

- Logs de ejecución: `logs/app.log` (rotación manual; borra o rota este archivo si crece demasiado).

## Resolución de problemas comunes

- **Falta `kivy_garden.graph`**: ejecuta `garden install graph` antes del build. Si Buildozer falla, borra `.buildozer/` y recompila.
- **Tiempo real lento**: reduce la tasa de muestreo a 25/50 Hz o desactiva sensores redundantes.
- **Compartir CSV falla en Android 12**: verifica que la app tenga permisos de almacenamiento y que el archivo exista en `records/`. Reintenta tras presionar *Guardar CSV*.
- **Sensores ausentes**: la UI desactiva los que Plyer no expone en el dispositivo. Usa Modo demo para pruebas.
- **Errores de compilación NDK**: asegúrate de ejecutar `buildozer android clean` antes de reconstruir tras cambios significativos.

## Desarrollo y extensiones

- Ajusta estilos visuales en `ui.py` (clases `SensorTile`, `HomeScreen`).
- Añade sensores nuevos ampliando `SENSOR_SPECS` en `sensors.py`, especificando nombre y atributo Plyer.
- Modifica `storage.py` para usar otras estrategias de persistencia (SQLite, Parquet) manteniendo el buffer circular.
- Integra pruebas automatizadas conectando `MockSensorStream` con el pipeline de almacenamiento para validar CSV en CI.

## Pruebas en dispositivo real (Android 12/API 31)

1. Activa *Opciones de desarrollador* y *Depuración USB* en el dispositivo.
2. Conecta por USB y ejecuta `adb devices` para confirmar conexión.
3. Compila e instala con `buildozer android debug deploy run`.
4. Verifica:
   - Las gráficas responden al movimiento del dispositivo.
   - El archivo CSV se genera en `Android/data/<package>/files/records/`.
   - La acción *Compartir* abre el diálogo del sistema y permite enviar el CSV.
   - La calibración reduce ruido al mantener el teléfono quieto.

## Licencia

No se distribuye archivo de licencia específico; añade el que corresponda antes de publicar.

