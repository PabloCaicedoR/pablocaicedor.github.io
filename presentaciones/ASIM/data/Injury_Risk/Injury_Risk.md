# Guía metodológica de cálculo de variables (Biomecánica) — _Injury_Risk_

## 1) Diccionario de datos (recordatorio)

- **Knee_Flex_deg** (deg): Ángulo de flexión de rodilla (variable objetivo).
- **EMG_Quad_RMS_mV** (mV): RMS del EMG del cuádriceps (rectificado y promediado).
- **EMG_Ham_RMS_mV** (mV): RMS del EMG de isquiotibiales.
- **GRF_Vert_Norm_BW** (BW): Componente vertical de la fuerza de reacción del suelo, normalizada por peso corporal.
- **Omega_Shank_deg_s** (deg/s): Velocidad angular del segmento pierna (shank) medida/estimada con IMU.
- **Hip_Flex_deg** (deg): Ángulo de flexión de cadera.

## 2) Cómo se _calcula_ cada variable en un pipeline real

> Nota: El CSV ofrece **valores ya procesados**. A continuación se describe **cómo se obtendrían** a partir de señales crudas estándar.

### 2.1 EMG (EMG_Quad_RMS_mV, EMG_Ham_RMS_mV)

**Entradas crudas:** EMG diferencial (µV) por canal (cuádriceps / isquiotibiales).
**Salida:** **RMS** en mV de ventana deslizante.

1. **Filtrado pasa-banda** (artefactos de movimiento y línea):
   - Típico: 20–450 Hz (Butterworth 4º orden, bidireccional).
2. **Notch** (si hay ruido de red): 50/60 Hz (y armónicos si procede).
3. **Rectificación**: valor absoluto, $x_r(t) = |x(t)|$.
4. **RMS por ventana** de $N$ muestras (p.ej., 100 ms):
   $$x*{\mathrm{RMS}}[k] \,=\, \sqrt{\frac{1}{N} \sum*{i=0}^{N-1} x_r^2[k-i]} $$
5. **Conversión de unidades** a mV (si procede) y **normalización opcional** (p.ej., a MVC).

**Observaciones didácticas:**

- La RMS reduce sensibilidad a fase y captura “energía” muscular.
- Elegir ventana (50–250 ms) equilibra suavizado vs. latencia.
- Documentar **MVC**, electrodos, piel, tasa de muestreo y filtros.

### 2.2 GRF vertical normalizada (GRF_Vert_Norm_BW)

**Entradas crudas:** fuerza vertical de plataforma (N), masa corporal $m$.
**Salida:** **GRF vertical normalizada por peso (BW)**.

1. **Fuerza vertical** cruda $F_z(t)$ (ya calibrada).
2. **Filtrado** pasa-bajo (p.ej., 20–50 Hz, Butterworth 4º orden, bidireccional) para reducir ruido.
3. **Normalización a BW**:
   $$\mathrm{GRF_Vert_Norm_BW}(t) \,=\, \frac{F_z(t)}{m\,g}$$
   donde $g = 9.81\,\mathrm{m/s^2}$.

**Observaciones:**

- Permite comparar individuos con distinta masa.
- Revisar **offset** (fuerza en quietud ≈ 1 BW de pie).
- En gestos de impacto usar cortes de fase (contacto/no contacto) por umbral.

### 2.3 Velocidad angular del shank (Omega_Shank_deg_s)

**Entradas crudas:** giroscopio de IMU en la pierna (rad/s), orientación del sensor.
**Salida:** **velocidad angular** en deg/s respecto al eje relevante (flexo-extensión).

1. **Selección de eje** (p.ej., eje mediolateral para flexión sagital).
2. **Corrección de deriva/bias** del giroscopio (estimado en reposo).
3. **Filtrado** pasa-bajo (5–20 Hz) si se requiere suavizado.
4. **Conversión** a deg/s: $\omega*{deg/s} = \omega*{rad/s} \times 180/\pi$.

**Observaciones:**

- Para **ángulo** se integraría $\omega$ con fusión sensorial (Madgwick/Kalman) para limitar deriva.
- Cuidar alineación sensor-segmento (matriz de rotación/calibración funcional).

### 2.4 Ángulos articulares (Hip_Flex_deg, y por analogía Knee_Flex_deg)

**Entradas crudas:** cinemática 3D (sistema óptico o IMU fusionadas).
**Salida:** **ángulo de flexión** en grados.

1. **Definir marcos anatómicos** (ISB) para pelvis, muslo y pierna.
2. **Calcular orientación segmentaria** (cuaterniones/matrices).
3. **Ángulo relativo** en plano sagital:
   - Cadera: muslo vs. pelvis.
   - Rodilla: pierna vs. muslo.
4. **Extracción del componente sagital** (flexo-extensión) y conversión a **grados**.

**Observaciones:**

- Con **IMUs**, usar métodos de fusión (Madgwick/Kalman).
- Con **mocap**, aplicar **filtros** (6–12 Hz) a marcadores y IK.
- Reportar **convención de signos** y rango fisiológico.

---

## 3) Función generativa (dataset sintético)

La variable **Knee_Flex_deg** del CSV fue simulada con la siguiente relación no lineal ($a$ = ángulo de rodilla), recortada a $[-5, 140]\,deg$, con ruido mixto gaussiano + t de Student:

$$
\begin{aligned}
a \,=&\; 15

- 25\,\sigma\!\big(1.6\,(\mathrm{GRF}-0.6)\big)
- 0.06\,\operatorname{sign}(\omega)\,|\omega|^{1.2}
- 0.55\,\mathrm{Hip} + 9\,\sin(\mathrm{Hip}^\circ) \\
  &- 10\,\frac{\mathrm{EMG}_{ham}}{\mathrm{EMG}_{quad}+0.05}
- 6\,\mathrm{EMG}\_{quad}\,\mathrm{GRF}

* 40\,(\mathrm{EMG}\_{quad}-0.12)^2

- 8\,\mathrm{GRF}\,\sin(\mathrm{Hip}^\circ)
- \varepsilon
  \end{aligned}
$$

donde $\sigma(x)=1/(1+e^{-x})$ y $\omega=\Omega\_\mathrm{Shank}$. **Propósito didáctico:** mostrar cómo _interacciones_ (p.ej., EMG×GRF) y _no linealidades_ afectan una salida biomecánica.

## 4) Preprocesamiento, sincronización y control de calidad (QC)

- **Sincronización**: alinear plataformas de fuerza, IMUs y mocap (pulsos TTL o _cross-correlation_).
- **Resampleo**: llevar todas las series a una misma tasa (interpolación/anti-aliasing).
- **Filtros**: documentar orden, frecuencias de corte y método (cero-fase).
- **Anotación de fases**: contacto/no-contacto a partir de GRF (umbral p.ej., 20 N) o cinemática.
- **Outliers**: inspección visual + z-score/IQR.
- **Unidades**: verificar SI y consistencia (deg, deg/s, mV, BW).
- **Normalizaciones**: a **BW** (fuerzas) y a **MVC** (EMG) si se comparan sujetos.

## 5) Cómo replicar (mini-ejemplos en Python)

> Suponga señales crudas `emg_quad`, `emg_ham` (V), `fz` (N), `gyro` (rad/s), y orientaciones segmentarias ya estimadas.

```python
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass(sig, fs, f1=20, f2=450, order=4):
    b, a = butter(order, [f1/(fs/2), f2/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def rms_window(x, fs, win_ms=100):
    N = int(win_ms*fs/1000)
    N = max(N, 1)
    # Convolución eficiente de x^2 con ventana uniforme
    power = np.convolve(x**2, np.ones(N)/N, mode='same')
    return np.sqrt(np.maximum(power, 0))

# EMG → RMS (mV)
fs_emg = 1000
emg_bp = bandpass(emg_quad, fs_emg, 20, 450)
emg_rect = np.abs(emg_bp)
emg_quad_rms_mV = rms_window(emg_rect, fs_emg, 100) * 1e3

# GRF → normalizada por BW
m = 70.0  # kg
g = 9.81
fs_f = 1000
fz_filt = filtfilt(*butter(4, 50/(fs_f/2), btype='low'), fz)
grf_vert_norm_bw = fz_filt / (m*g)

# Gyro → deg/s
omega_deg_s = gyro * 180/np.pi
```

## 6) Estadísticos descriptivos del CSV

| Variable          | count |    mean |     std |      min |      25% |      50% |     75% |     max |
| :---------------- | ----: | ------: | ------: | -------: | -------: | -------: | ------: | ------: |
| Knee_Flex_deg     |  1000 | 36.6432 |  36.988 |       -5 |  -1.6713 |  30.6706 | 64.7409 | 129.671 |
| EMG_Quad_RMS_mV   |  1000 |  0.1328 |  0.0672 |   0.0211 |   0.0743 |   0.1343 |  0.1912 |  0.2499 |
| EMG_Ham_RMS_mV    |  1000 |  0.1063 |  0.0555 |   0.0106 |   0.0558 |   0.1086 |  0.1545 |  0.1999 |
| GRF_Vert_Norm_BW  |  1000 |  0.6531 |  0.3779 |        0 |   0.3398 |   0.6508 |  0.9868 |  1.2972 |
| Omega_Shank_deg_s |  1000 |    -7.7 | 229.191 | -399.477 | -206.479 | -12.5766 | 190.033 | 399.646 |
| Hip_Flex_deg      |  1000 | 14.7053 | 14.3405 |  -9.9985 |   2.2487 |  14.7299 | 26.9997 | 39.8875 |

## 7) Supuestos y limitaciones

- Este dataset es **sintético** y no reemplaza mediciones clínicas.
- Las fórmulas de cálculo son **estándares** en biomecánica, pero los **parámetros** (ventanas/filtros) deben ajustarse a cada laboratorio y tarea.
- La interpretación de EMG depende de colocación de electrodos, crosstalk y normalización (MVC).
