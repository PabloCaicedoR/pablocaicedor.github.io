---
title: "Procesado de Señales e Imágenes Médicas"
description: "ASIM_M -- 104399"
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


## Importance of Frequency-Response Filters
- **Frequency-response filters** are critical for enhancing specific features or reducing noise in images.
- Widely used in MRI, CT, and ultrasound imaging.

## What is a Frequency-Response Filter?
- A frequency-response filter modifies the frequency components of a signal.
- Applied in image processing to control **which frequencies** (details) pass or are suppressed.

## Types of Frequency-Response Filters
- **Low-pass filters**: Allow low frequencies, suppress high frequencies (smoothes image).
- **High-pass filters**: Allow high frequencies, suppress low frequencies (sharpens image).
- **Band-pass filters**: Allow frequencies in a certain range.

## Spatial vs. Frequency Domain
- **Spatial domain**: Operations on pixel values directly.
- **Frequency domain**: Operations on the image's frequency components.

## Fourier Transform
- Converts an image from the spatial domain to the frequency domain.
- **Formula**: $$F\left(u,v\right) = \sum\sum f\left(x,y\right) e^{-j2\pi(\frac{ux}{M} + \frac{vy}{N})}$$


## Low-Pass Filters
- Removes high-frequency components (e.g., noise, sharp edges).
- **Example**: Gaussian filter, Butterworth filter.

## High-Pass Filters
- Enhances edges and high-frequency details.
- **Example**: Laplacian filter.

## Band-Pass Filters
- Allows frequencies within a specific range.
- Useful for isolating specific image features.


## Noise Reduction in MRI
- **Low-pass filters** reduce noise and artifacts in MRI scans.
- Smoothes the image without losing crucial details.

## Edge Enhancement in Ultrasound Images
- **High-pass filters** help in detecting tissue boundaries by enhancing edges.
- Improves clarity of anatomical structures.

## Feature Extraction in CT Scans
- Filters can help in extracting features like **tumors** or **vessels**.
- Band-pass filters isolate structures of interest at specific frequency ranges. 

## Case Study: Applying a Low-Pass Filter in MRI. Step-by-step Process
1. Load MRI image.
2. Apply **Fourier Transform** to move the image into the frequency domain.
3. Design and apply a **low-pass filter**.
4. Perform **Inverse Fourier Transform** to return to the spatial domain.
5. Visualize the result.


## Summary
- Frequency-response filters play a crucial role in biomedical image processing.
- Help enhance key features and suppress unwanted noise.

## Future Trends
- **Deep learning** integration with traditional filters.
- Development of adaptive filters for real-time processing.
