# Índice extendido — Curso de Aprendizaje Automático para Señales e Imágenes Médicas (24 h)

**Rol**: Diseñador instruccional (AA en señales e imágenes médicas)
**Audiencia**: Pregrado (profundización, nivel avanzado)
**Duración total**: **24 horas** (8 sesiones × 3 h)
**Stack principal**: Python 3.12, scikit-learn, PyTorch, MONAI, MNE (según caso)
**Política de validación**: _subject-wise k-fold_ en todas las prácticas
**Evaluación sumativa**: 7 mini-labs (70%) + quiz final (30%)
**Énfasis clínico**: Musculoesquelético (EMG/RX/TC)

---

## Tabla de sesiones (24 h = 8 sesiones × 3 h)

| #   | Título                                                            | Bloque            | Horas | Entregable                                      |
| --- | ----------------------------------------------------------------- | ----------------- | ----: | ----------------------------------------------- |
| 1   | Fundamentos de AA en salud: tareas, métricas y pipeline           | Fundamentos       |     3 | Mini-lab 1: pipeline clásico (sklearn)          |
| 2   | Validación rigurosa y reproducibilidad (subject-wise)             | Fundamentos       |     3 | Mini-lab 2: split subject-wise + baseline       |
| 3   | Señales biomédicas I: preprocesado y _features_ (ECG/EMG)         | Señales           |     3 | Mini-lab 3: filtro → _features_ → clasificación |
| 4   | Señales biomédicas II: ML clásico vs CNN 1D                       | Señales           |     3 | Mini-lab 4: baseline vs CNN1D (comparativa)     |
| 5   | Señales biomédicas III: LSTM/GRU/Transformers 1D + explicabilidad | Señales           |     3 | Mini-lab 5: secuenciales + saliency/IG          |
| 6   | Imágenes médicas I: DICOM, preprocesado y _transfer learning_     | Imágenes          |     3 | Mini-lab 6: _fine-tuning_ (MONAI)               |
| 7   | Imágenes médicas II: U-Net/ResNet, métricas y Grad-CAM            | Imágenes          |     3 | Mini-lab 7: segmentación + Grad-CAM             |
| 8   | Integración, buenas prácticas y _quiz_ final                      | Imágenes (cierre) |     3 | _Quiz_ final + repositorio reproducible         |

**Balance de horas por bloque (aprox.)**: Fundamentos 6 h (≈30%), Señales 9 h (≈35%), Imágenes 9 h (≈35%).
**Evaluación (modelo 5B)**: 7 mini-labs × 10% c/u = **70%**; _Quiz_ final = **30%**.

---

## 1. Fundamentos de AA en salud: tareas, métricas y pipeline (3 h)

- **Resultados de aprendizaje**:
  RA1. Diferenciar tareas (clasificación, regresión, segmentación).
  RA2. Seleccionar métricas apropiadas por tarea (Acc, AUC, F1, MAE; Dice/IoU para segmentación).
  RA3. Describir un pipeline reproducible de AA en salud.
- **Conceptos clave**: problema/tarea, _train/val/test_, _baseline_, _feature engineering_, normalización, _data leakage_.
- **Actividad práctica (PhysioNet, sklearn)**:
  Clasificar latidos ECG (e.g., **MIT-BIH** o **PTB-XL** de PhysioNet). Preproceso básico, _split_ estratificado por paciente (sin fuga), baseline con _LogReg_/SVM.
- **Stack**: Python 3.12, `scikit-learn`, `numpy`, `pandas`, `matplotlib`.
- **Validación**: **subject-wise k-fold** (k=5).
- **Evaluación**: Mini-lab 1 (10%).
- **Referencias sugeridas (verificar DOI/ISBN antes de publicar)**:
  - Bishop, _Pattern Recognition and Machine Learning_, 2006, **ISBN 978-0387310732**.
  - Pedregosa et al., _Scikit-learn: Machine Learning in Python_, JMLR 12:2825–2830, 2011.
- **Riesgos comunes**: fuga por segmentos del mismo paciente; métricas inadecuadas por clase desbalanceada.

---

## 2. Validación rigurosa y reproducibilidad (3 h)

- **Resultados**:
  RA1. Implementar _subject-wise k-fold_ y reportes por clase.
  RA2. Configurar entorno reproducible y _seeds_.
  RA3. Documentar experimentos (versionado y _data cards_).
- **Conceptos**: _subject-wise_ vs aleatorio, _nested CV_, semillas, hojas de datos, _readme_ de experimentos.
- **Actividad (PhysioNet, sklearn)**:
  Repetir baseline sesión 1 con particionamiento **subject-wise k-fold** y reporte de varianza entre _folds_.
- **Stack**: `scikit-learn`, `mlflow` (opcional), `pyproject.toml` o `conda` env, control de versiones.
- **Validación**: **subject-wise k-fold** con estratificación si aplica.
- **Evaluación**: Mini-lab 2 (10%).
- **Referencias**:
  - Géron, _Hands-On Machine Learning_, 2ª/3ª ed., O’Reilly, **ISBN 978-1492032649**.
- **Riesgos**: comparar métricas de _folds_ no homólogos; no fijar semillas; no congelar versiones.

---

## 3. Señales biomédicas I: preprocesado y _features_ (3 h)

- **Resultados**:
  RA1. Aplicar filtrado, _resampling_ y normalización en ECG/EMG.
  RA2. Extraer _features_ en tiempo/frecuencia (RMS, picos R, bandas).
  RA3. Entrenar un clasificador clásico y evaluar robustez.
- **Conceptos**: filtros pasa-banda, remoción de línea base, _windowing_, PSD, _z-score_.
- **Actividad (PhysioNet)**:
  ECG **MIT-BIH**: detección de picos R + _features_ → clasificación de latidos con `RandomForest` vs `LogReg`.
- **Stack**: `scikit-learn`, `scipy`, `wfdb` (lectura PhysioNet), `matplotlib`.
- **Validación**: **subject-wise k-fold**; métricas: F1 macro, sensibilidad por clase.
- **Evaluación**: Mini-lab 3 (10%).
- **Referencias**:
  - Goldberger et al., _PhysioNet_ (portal oficial).
- **Riesgos**: _overfitting_ por _feature selection_ en todo el set; filtrado que distorsiona morfología.

---

## 4. Señales biomédicas II: ML clásico vs CNN 1D (3 h)

- **Resultados**:
  RA1. Implementar una CNN 1D simple y compararla con ML clásico.
  RA2. Seleccionar métricas y _early stopping_.
  RA3. Analizar errores por sujeto.
- **Conceptos**: convolución 1D, _receptive field_, _padding/stride_, _learning rate_.
- **Actividad (PhysioNet)**:
  ECG **PTB-XL** (multi-derivación) o MIT-BIH re-muestreado: baseline `SVM` vs `CNN1D` en PyTorch.
- **Stack**: `PyTorch`, `scikit-learn`.
- **Validación**: **subject-wise k-fold**; _confusion matrix_ por sujeto.
- **Evaluación**: Mini-lab 4 (10%).
- **Referencias**:
  - Goodfellow et al., _Deep Learning_, MIT Press, **ISBN 978-0262035613**.
- **Riesgos**: comparar modelos con _splits_ distintos; no balancear clases.

---

## 5. Señales biomédicas III: LSTM/GRU/Transformers 1D + explicabilidad (3 h)

- **Resultados**:
  RA1. Implementar LSTM/GRU/Transformers 1D para detección de eventos.
  RA2. Aplicar explicabilidad 1D (saliency, Integrated Gradients).
  RA3. Reportar variabilidad entre _folds_.
- **Conceptos**: dependencias temporales, enmascaramiento, longitudes variables, atención.
- **Actividad (PhysioNet)**:
  PTB-XL (o EEG PhysioNet para eventos): modelo secuencial + **saliency/IG** (Captum).
- **Stack**: `PyTorch`, `captum`, `mne` (si EEG).
- **Validación**: **subject-wise k-fold**; métricas por evento.
- **Evaluación**: Mini-lab 5 (10%).
- **Referencias**:
  - Hochreiter & Schmidhuber, LSTM (1997).
  - Vaswani et al., _Attention Is All You Need_ (2017).
- **Riesgos**: secuencias truncadas sesgando etiquetas; explicaciones no estables entre _folds_.

---

## 6. Imágenes médicas I: DICOM, preprocesado y _transfer learning_ (3 h)

- **Resultados**:
  RA1. Cargar DICOM/series y normalizar intensidades.
  RA2. Ejecutar _transfer learning_ con MONAI.
  RA3. Documentar _data transforms_ y _augmentations_.
- **Conceptos**: spacing, _windowing_, normalización, recortes, _augment_.
- **Actividad (TCIA)**:
  **LIDC-IDRI** (TCIA): clasificación simple de nódulos (benigno/sospechoso como etiqueta didáctica) con _fine-tuning_ de ResNet.
- **Stack**: `MONAI`, `PyTorch`, `pydicom`.
- **Validación**: **subject-wise k-fold** por paciente.
- **Evaluación**: Mini-lab 6 (10%).
- **Referencias**:
  - MONAI (docs oficiales).
  - Armato et al., LIDC-IDRI (descripción del _dataset_).
- **Riesgos**: fuga por cortes múltiples del mismo paciente en _train/test_; _augment_ que altera anatomía.

---

## 7. Imágenes médicas II: U-Net/ResNet, métricas y Grad-CAM (3 h)

- **Resultados**:
  RA1. Entrenar U-Net para segmentación 2D.
  RA2. Evaluar con Dice/IoU y curvas de volumen.
  RA3. Aplicar **Grad-CAM** para inspección de atención.
- **Conceptos**: U-Net (encoder-decoder), pérdida (Dice/BCE), _tiling_, posprocesado.
- **Actividad (TCIA)**:
  **LIDC-IDRI**: segmentación de nódulos con U-Net (MONAI) + **Grad-CAM** sobre cortes con ResNet para inspección.
- **Stack**: `MONAI`, `PyTorch`, `captum`.
- **Validación**: **subject-wise k-fold**; reporte Dice por paciente.
- **Evaluación**: Mini-lab 7 (10%).
- **Referencias**:
  - Ronneberger et al., U-Net, MICCAI 2015 (DOI disponible).
  - He et al., ResNet, CVPR 2016.
- **Riesgos**: entrenar/validar en cortes del mismo estudio; segmentación con máscaras ruidosas.

---

## 8. Integración, buenas prácticas y _quiz_ final (3 h)

- **Resultados**:
  RA1. Integrar pipeline completo (datos → modelo → validación → reporte).
  RA2. Elaborar _model card_ y _data card_ resumidas.
  RA3. Defender decisiones de diseño con evidencia.
- **Conceptos**: estructura de repositorio, _model/data cards_, _checklist_ de reproducibilidad, trazabilidad de experimentos.
- **Actividad**:
  Integrar un caso señal (ECG) o imagen (TCIA) a elección: _notebook_ final con `README` y _report_ de métricas.
- **Stack**: `PyTorch`, `MONAI`, `scikit-learn`, `matplotlib`.
- **Validación**: **subject-wise k-fold**; reporte agregando media±DE.
- **Evaluación**: _Quiz_ final (30%).
- **Referencias**:
  - Mitchell et al., _Model Cards for Model Reporting_ (FAccT).
- **Riesgos**: falta de _seeds_, rutas relativas no reproducibles, dependencias sin fijar versión.

---

## Requisitos de reproducibilidad

- `Python==3.12.*`, _seed_ fija (e.g., 42), `requirements.txt` o `environment.yml`, carpetas `data/`, `notebooks/`, `models/`, `reports/`.
- _Subject-wise k-fold_ obligatorio; no mezclar cortes/derivaciones del mismo paciente entre _splits_.
- Reportar **media±DE** por _fold_ y por paciente cuando aplique.

---

## Apéndice: Referencias globales y datasets (portal oficial)

- **Textos**:
  - Bishop (2006), _PRML_, **ISBN 978-0387310732**.
  - Goodfellow, Bengio, Courville (2016), _Deep Learning_, **ISBN 978-0262035613**.
- **Software**:
  - Pedregosa et al. (2011), _Scikit-learn_, JMLR 12.
  - Paszke et al. (2019), _PyTorch_ (NeurIPS).
  - MONAI (documentación oficial).
- **Señales (PhysioNet)**: **MIT-BIH**, **PTB-XL**, EEG Motor/Imagery. _(Usar portal oficial; añadir DOI/URL confirmados antes de publicar)_
- **Imágenes (TCIA)**: **LIDC-IDRI**. _(Añadir DOI/URL confirmados antes de publicar)_
