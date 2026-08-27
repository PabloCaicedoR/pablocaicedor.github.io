# Dataset docente de imágenes médicas

Este directorio fue creado automáticamente por `build_teaching_dataset.sh`.

## Propósito

Crear una colección pequeña y visualmente útil para una clase de divulgación
sobre cómo se forman, representan y procesan las imágenes médicas.

**No es un dataset para diagnóstico clínico.**

## Criterios de selección

### Radiografía

Fuente Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

La ficha de Kaggle describe 5.863 radiografías pediátricas AP, organizadas en
NORMAL/PNEUMONIA, y señala que las imágenes fueron sometidas a control de
calidad y revisión médica. Para enseñanza se seleccionan dos ejemplos de cada
grupo disponible en los nombres del dataset:

- normal;
- neumonía con token `bacteria` en el nombre;
- neumonía con token `virus` en el nombre.

El subtipo se conserva como **metadato del archivo**, no como inferencia
diagnóstica del script.

### CT

Fuente Kaggle:
https://www.kaggle.com/datasets/abbymorgan/cranial-ct

Kaggle describe este conjunto como una serie DICOM craneal preparada
específicamente para experimentar con cortes secuenciales y planos axial,
sagital y coronal.

Se producen:

- tres cortes axiales informativos;
- un plano coronal central;
- un plano sagital central;
- el mismo corte axial con dos visualizaciones.

Ventanas usadas para demostración:

- cerebro: W=80 HU, L=40 HU;
- hueso: W=2800 HU, L=600 HU.

El cambio de ventana modifica la visualización, no los valores HU originales.

### MRI

Fuente Kaggle:
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

El dataset contiene imágenes MRI de 110 pacientes con glioma de bajo grado,
junto con máscaras manuales de anormalidad FLAIR. Los TIFF tienen tres canales
(pre-contraste, FLAIR, post-contraste); para la clase se exporta FLAIR.

Se selecciona, por paciente, el corte con mayor máscara y luego se toman
ejemplos cercanos a P25/P50/P75 del área relativa de la máscara. Esto muestra
variabilidad sin depender de una selección aleatoria.

También se incluye un corte con máscara vacía. Esto **no significa paciente
sano**: significa solamente que en ese corte no hay anormalidad segmentada.

### Ultrasonido BUSI

Fuente Kaggle:
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

BUSI contiene 780 imágenes de ultrasonido mamario de 600 mujeres, organizadas
como normal, benigno y maligno, con ground truth de segmentación.

Para enseñanza se seleccionan:

- 2 normales;
- 2 benignas;
- 2 malignas.

En benigno/maligno se prefieren lesiones con máscara visible, tamaño no extremo,
buena centralidad y una única máscara para simplificar la primera explicación.

## Estructura

```text
teaching_dataset/
├── xray/
├── ct/
│   ├── axial/
│   ├── mpr/
│   └── windowing/
├── mri/
│   ├── images/
│   ├── masks/
│   └── overlays/
├── ultrasound/
├── contact_sheets/
├── manifest.csv
└── README.md
```

## Trazabilidad

`manifest.csv` registra:

- modalidad;
- rol didáctico;
- clase/etiqueta;
- archivo original;
- archivo derivado;
- criterio de selección;
- URL de Kaggle;
- nota de licencia.

Los datos fuente no se modifican.

## Restricción de uso

Material para enseñanza y demostración de conceptos de imágenes médicas.
No utilizar las imágenes ni los criterios de este script para diagnóstico,
triaje o decisiones clínicas.
