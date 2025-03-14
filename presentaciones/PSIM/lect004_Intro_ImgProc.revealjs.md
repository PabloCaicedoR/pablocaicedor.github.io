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



# Procesamiento de imágenes

## What is Digital Image Processing?

::: {.callout-note title="Definition" collapsible="false"}

- Two-dimensional function, f(x, y)
- Where x and y are spatial coordinates.
- The amplitude of f at any pair of coordinates (x, y) is called the intensity.

:::

::: {.callout-warning title="The digital image" collapsible="false"}

If the coordinates and the intensity are discrete quantities the image turns into a digital image.

:::

## What is Digital Image Processing?

::: {.callout-tip title="Definition" collapsible="false"}

A digital image is composed by a finite number of elements called PIXEL.

:::

::::{.columns}

:::{.column width="45%"}

![https://www.researchgate.net/figure/Digital-image-representation-by-pixels-vii_fig2_311806469](../../recursos/imagenes/Presentaciones/PSIM/pixels.png)

:::

:::{.column width="45%"}

::: {.callout-tip title="Depth" collapsible="false"}

:::{.small_font}

A digital image is composed by a finite number of elements called PIXEL. Bpp( Bits per pixel)

- 1bpp. B/W image, monochrome.
- 2bpp. CGA Image.
- 4bpp. Minimun for VGA standard.
- 8bpp. Super-VGA image.
- 24bpp. Truecolor image.
- 48bpp. Professional-level images.
  

:::

:::



:::

::::

## What is Digital Image Processing?

::::{.columns}

:::{.column width="45%"}

![https://www.researchgate.net/figure/Digital-image-representation-by-pixels-vii_fig2_311806469](../../recursos/imagenes/Presentaciones/PSIM/pixels.png)

:::

:::{.column width="45%"}

::: {.callout-tip title="Color Space" collapsible="false"}

How can i represent the color

- RGB.
- CMYK.
- HSV.
- Among others.

:::


:::

::::

## What is Digital Image Processing?

::::{.columns}

:::{.column width="45%"}



::: {.cell}
::: {.cell-output-display}
![](lect004_Intro_ImgProc_files/figure-revealjs/unnamed-chunk-3-1.png){width=960}
:::
:::

::: {.cell}

```{.python .cell-code}
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(image_path+"image01.tif")
fig001 = plt.figure()
plt.imshow(img)
```
:::




:::

:::{.column width="45%"}




::: {.cell}
::: {.cell-output-display}
![](lect004_Intro_ImgProc_files/figure-revealjs/unnamed-chunk-5-3.png){width=960}
:::
:::

::: {.cell}

```{.python .cell-code}
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(image_path+"lena.tif")
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig002 = plt.figure()
plt.imshow(RGB_img)
```
:::




:::

::::


## Images and vision

- The paradigm surrounding the conceptualization of light and perception has undergone significant evolution. 
- Initially, the prevailing understanding within humanity posited that visual stimuli emanated from the eye itself. 
- However, contemporary knowledge has elucidated that light originates from external sources, undergoes reflection from objects within the environment, and is subsequently captured by the eye.

## Images and vision

::: {.callout-note title="Important" collapsible="false"}


We also understand that light is a type of electromagnetic radiation, and its wavelength falls within a range from 400 nanometers to 700 nanometers.

:::


![Taken from Corke 2023](../../recursos/imagenes/Presentaciones/PSIM/ligh_spectrum.png){fig-align="center"}

## Images and vision

::: {.callout-note title="Important" collapsible="false"}

- The most common way light is made is by something getting really hot. This makes energy that comes out as light.

- Some important term are: 
  - _Absortion:_ It is the fraction of light which a body absorbs depending on the wavelength.
  - _Reflectance:_ It is the fraction of the incoming light which a body reflects. It's a number between 0 to 1 and also depends on wavelength.
  - _Luminance:_ It is the fraction of the incoming light which a surface reflects. It's a function of absortion and reflectance, and because of that luminance depends on wavelength.

:::

## Images and vision

::: {.callout-tip title="The eye" collapsible="false"}

- Our eye has two types of cells. _Cones_ and _Rods_.
- Cones are the most sensitive cells but above all these are color sensitive.
- Rods responds only two intensity and they used on night, mostly.
- Humans, like most primates, are trichomats. This means that humans have three types of cones (Long, Medium and shorts).
  - 65% of longs (Sense red)
  - 33% of mediums (Sense green)
  - 2% of shortsv(Sense blue)

:::

## Images and vision

::: {.callout-tip title="The artificial eye" collapsible="false"}

![Taken from Corke 2023](../../recursos/imagenes/Presentaciones/PSIM/CameraSensor.png){fig-align="center"}

:::

The currents from each sensor are function of the luminance and the spectral response filter.

## Images and vision

![Taken from https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf](../../recursos/imagenes/Presentaciones/PSIM/pinhole.png){fig-align="center"  width=1000}

## Images and vision

![Taken from https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf](../../recursos/imagenes/Presentaciones/PSIM/pinhole2.png){fig-align="center"  width=1000}

## Images and vision

![Taken from https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf](../../recursos/imagenes/Presentaciones/PSIM/pinhole3.png){fig-align="center"  width=500}

## Images and vision

![Taken from https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf](../../recursos/imagenes/Presentaciones/PSIM/CameraModel.png){fig-align="center"  width=1000}


## Images and vision

![Taken from https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf](../../recursos/imagenes/Presentaciones/PSIM/CameraModel2.png){fig-align="center"  width=1000}

## Sampling and quantization

::: {.callout-tip title="Definition" collapsible="false"}

__Sampling:__ Digitalization of the spatial coordinates.

:::

::: {.callout-tip title="Definition" collapsible="false"}

__Quantiazation:__ Digitalization of the light intensity (amplitude).

:::


## Sampling and quantization



![Tomado de Gonzalez, Rafael C., y Richard E. Woods. 2018. Digital Image Processing. New York, NY: Pearson.](../../recursos/imagenes/Presentaciones/PSIM/sampling_quantization_01.png){fig-align="center"  width=600}




## Sampling and quantization

:::{layout-nrow=2}



::: {.cell}
::: {.cell-output-display}
![](lect004_Intro_ImgProc_files/figure-revealjs/loading image 1-5.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![](lect004_Intro_ImgProc_files/figure-revealjs/loading image 2-7.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![](lect004_Intro_ImgProc_files/figure-revealjs/loading image 3-9.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![](lect004_Intro_ImgProc_files/figure-revealjs/loading image 4-11.png){width=960}
:::
:::



:::

## Sampling and quantization

:::{layout-ncol=2}



::: {.cell}
::: {.cell-output-display}
![](lect004_Intro_ImgProc_files/figure-revealjs/Loading kirby-13.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![](lect004_Intro_ImgProc_files/figure-revealjs/Gray kirby-15.png){width=960}
:::
:::



:::

:::{layout-ncol=4}



::: {.cell}
::: {.cell-output-display}
![1bit](lect004_Intro_ImgProc_files/figure-revealjs/kirby1bit-17.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![2bit](lect004_Intro_ImgProc_files/figure-revealjs/kirby2bit-19.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![3bit](lect004_Intro_ImgProc_files/figure-revealjs/kirby3bit-21.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![4bit](lect004_Intro_ImgProc_files/figure-revealjs/kirby4bit-23.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![5bit](lect004_Intro_ImgProc_files/figure-revealjs/kirby5bit-25.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![6bit](lect004_Intro_ImgProc_files/figure-revealjs/kirby6bit-27.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![7bit](lect004_Intro_ImgProc_files/figure-revealjs/kirby7bit-29.png){width=960}
:::
:::

::: {.cell}
::: {.cell-output-display}
![8bit](lect004_Intro_ImgProc_files/figure-revealjs/kirby8bit-31.png){width=960}
:::
:::



:::

## Sampling and quantization

![Tomado de Gonzalez, Rafael C., y Richard E. Woods. 2018. Digital Image Processing. New York, NY: Pearson.](../../recursos/imagenes/Presentaciones/PSIM/sampling_quantization_02.png){fig-align="center"  width=600}

## Linear indexing

![Tomado de Gonzalez, Rafael C., y Richard E. Woods. 2018. Digital Image Processing. New York, NY: Pearson.](../../recursos/imagenes/Presentaciones/PSIM/linear_indexing.png){fig-align="center"  width=600}

::: {.columns}

:::{.column width="50%"}

::: {.callout-tip title="From normal to linear" collapsible="false"}

$$\alpha = My+x$$

:::



:::

:::{.column width="50%"}

::: {.callout-tip title="From linear to normal" collapsible="false"}

$$x = \alpha \bmod M$$

$$y = \frac{\alpha - x}{M}$$

:::



:::

:::

## Spatial resolution

![Tomado de Gonzalez, Rafael C., y Richard E. Woods. 2018. Digital Image Processing. New York, NY: Pearson.](../../recursos/imagenes/Presentaciones/PSIM/dpi01.png){fig-align="center"  width=600}

## Intensity resolution

![Tomado de Gonzalez, Rafael C., y Richard E. Woods. 2018. Digital Image Processing. New York, NY: Pearson.](../../recursos/imagenes/Presentaciones/PSIM/intensity_resolution.png){fig-align="center"  width=600}

## Intensity resolution

![Tomado de Gonzalez, Rafael C., y Richard E. Woods. 2018. Digital Image Processing. New York, NY: Pearson.](../../recursos/imagenes/Presentaciones/PSIM/intensity_resolution_02.png){fig-align="center"  width=600}

## "A simple problem"
![Tomado de https://medium.com/@abhishekjainindore24/semantic-vs-instance-vs-panoptic-segmentation-b1f5023da39f](../../recursos/imagenes/Presentaciones/PSIM/Personas01.png)

## "A simple problem"
![Tomado de https://medium.com/@abhishekjainindore24/semantic-vs-instance-vs-panoptic-segmentation-b1f5023da39f](../../recursos/imagenes/Presentaciones/PSIM/Personas02.png)

## Relationships between pixels

::: {.callout-tip title="Neighborhood" collapsible="false"}

:::{#fig-neighborhoods layout-ncol=3}


 ![N4](../../recursos/imagenes/Presentaciones/PSIM/n4.png){fig-align="center" width=250} 

 ![ND](../../recursos/imagenes/Presentaciones/PSIM/nd.png){fig-align="center" width=250} 

 ![N8](../../recursos/imagenes/Presentaciones/PSIM/n8.png){fig-align="center" width=250} 


:::

:::


## Relationships between pixels -- Neighborhood

::: {.callout-tip title="Neighborhood" collapsible="false"}

:::{#fig-neighborhoods layout-ncol=3}


 ![N4-$N_4\left(p\right)$](../../recursos/imagenes/Presentaciones/PSIM/n4.png){fig-align="center" width=250} 

 ![ND-$N_D\left(p\right)$](../../recursos/imagenes/Presentaciones/PSIM/nd.png){fig-align="center" width=250} 

 ![N8-$N_8\left(p\right)$](../../recursos/imagenes/Presentaciones/PSIM/n8.png){fig-align="center" width=250} 

Neighborhoods

:::

:::

## Relationships between pixels -- Adjacency

::: {.callout-tip title="Rules for adjecency" collapsible="false"}

- 4-Adjecncy: Two pixels p and q with values from V are 4-adjacent if q is in the set $N_4\left(p\right)$

- 8-adjacency. Two pixels p and q with values from V are 8-adjacent if q is in the set $N_8\left(p\right)$

- m-adjacency (also called mixed adjacency). Two pixels p and q with values from V are m-adjacent if:
    - q is in $N_4\left(p\right)$.
    - q is in $N_D\left(p\right)$ and the set $N_4\left(p\right) \cap N_4\left(q\right)$ has no pixels whose values are from V.

:::


## Relationships between pixels

 ![Adjacency](../../recursos/imagenes/Presentaciones/PSIM/Adjacency.png){fig-align="center" width=500} 

## Relationships between pixels

:::{layout="[[1,1], [1]]"}

 ![A4](../../recursos/imagenes/Presentaciones/PSIM/a4.png){fig-align="center" width=250 .lightbox} 

 ![A8](../../recursos/imagenes/Presentaciones/PSIM/a8.png){fig-align="center" width=250 .lightbox} 

 ![A-m](../../recursos/imagenes/Presentaciones/PSIM/am.png){fig-align="center" width=250 .lightbox}

:::

## Relationships between pixels -- Path

::: {.callout-tip title="Digital path" collapsible="false"}

It is a sequence of adjacent pixels. 

$$\left(x_0, y_0\right), \left(x_1, y_1\right), \left(x_2, y_2\right), \dots \left(x_n, y_n\right)$$

If $\left(x_0, y_0\right)=\left(x_n, y_n\right)$ the path is known as closed path

Let S represent a subset of pixels in an image. Two pixels p and q are said to be connected in S if there exists a path between them consisting entirely of pixels in S.

:::

## Relationships between pixels -- Path, Connected Subset

![](../../recursos/imagenes/Presentaciones/PSIM/connectedSet.png){fig-align="center" width=500 .lightbox}

## Relationships between pixels -- Regions

![](../../recursos/imagenes/Presentaciones/PSIM/two_regions.png){fig-align="center" width=500 .lightbox}


## Relationships between pixels -- Boundary

![](../../recursos/imagenes/Presentaciones/PSIM/boundary.png){fig-align="center" width=500 .lightbox}

## Relationships between pixels -- Distance

::: {.callout-tip title="Distance" collapsible="false"}

::: columns

:::{.column width="45%"}

![](../../recursos/imagenes/Presentaciones/PSIM/city_block.png){fig-align="center" width=500 .lightbox}

:::

:::{.column width="45%"}

- __City block distance__: $D_4\left(p,q\right) = \lvert x-u\rvert + \lvert y-v \rvert$
- Chessboard distance: $D_8\left(p,q\right) = max \left(\lvert x-u\rvert , \lvert y-v \rvert \right)$
- Euclidean distance: $D_e\left(p,q\right) = \sqrt{\left(x-u\right)^2 + \left(y-v\right)^2}$

:::

:::



:::

## Relationships between pixels

::: {.callout-tip title="Distance" collapsible="false"}

::: columns

:::{.column width="45%"}

![](../../recursos/imagenes/Presentaciones/PSIM/chessboard_distance.png){fig-align="center" width=500 .lightbox}

:::

:::{.column width="45%"}

- City block distance: $D_4\left(p,q\right) = \lvert x-u\rvert + \lvert y-v \rvert$
- __Chessboard distance__: $D_8\left(p,q\right) = max \left(\lvert x-u\rvert , \lvert y-v \rvert \right)$
- Euclidean distance: $D_e\left(p,q\right) = \sqrt{\left(x-u\right)^2 + \left(y-v\right)^2}$

:::

:::



:::

