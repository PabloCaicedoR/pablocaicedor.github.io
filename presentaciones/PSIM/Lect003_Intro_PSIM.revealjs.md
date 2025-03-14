---
title: "Procesamiento de Señales e Imagenes"
description: "PSIM -- 101849"
subtitle: "Ingeniería Biomédica"
lang: es
author: "Ph.D. Pablo Eduardo Caicedo Rodríguez"
date: "2024-08-12"
format:
  revealjs: 
    code-tools: true
    code-overflow: wrap
    code-line-numbers: true
    code-copy: true
    fig-align: center
    self-contained: true
    theme: 
      - simple
      - ../../recursos/estilos/metropolis.scss
    slide-number: true
    preview-links: auto
    logo: ../../recursos/imagenes/generales/Escuela_Rosario_logo.png
    css: ../../recursos/estilos/styles_pres.scss
    footer: <https://pablocaicedor.github.io/>
    transition: fade
    progress: true
    scrollable: true
resources:
  - demo.pdf
---


::: {.cell}

:::

::: {.cell}

:::



# Procesado de Señales e Imágenes Médicas - PSIM

## Introduction

![](../../recursos/imagenes/Presentaciones/PSIM/signalProcessingWorkflow.svg){height="160%"}

* _Data acquisition_ is to capture the signal and encode in a form suitable for computer processing.
* _Signal conditioning_ is to remove noise and artifacts from the signal.
* _Feature extraction_ is to extract relevant information from the signal.
* _Hypothesis testing_ is to test the hypothesis based on the extracted features.

## Signal condtioning

::: {.callout-important title="Base Information"}

* [A 12-lead electrocardiogram for arrhythmia study - Article](https://www.nature.com/articles/s41597-020-0386-x)

* [A 12-lead electrocardiogram for arrhythmia study - Data](https://physionet.org/content/ecg-arrhythmia/1.0.0/)

:::


## Signal conditioning



::: {.cell}

```{.python .cell-code}
data  = sio.loadmat(path_ecg+"/JS00001.mat")
```
:::

::: {.cell}

```{.python .cell-code}
print(type(data))
```

::: {.cell-output .cell-output-stdout}

```
<class 'dict'>
```


:::

```{.python .cell-code}
print(data.keys())
```

::: {.cell-output .cell-output-stdout}

```
dict_keys(['val'])
```


:::

```{.python .cell-code}
print(type(data['val']))
```

::: {.cell-output .cell-output-stdout}

```
<class 'numpy.ndarray'>
```


:::

```{.python .cell-code}
print(data['val'].shape)
```

::: {.cell-output .cell-output-stdout}

```
(12, 5000)
```


:::
:::



## Signal conditioning



::: {.cell}

:::