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

## What is Thresholding?

- Definition: A technique to separate objects from the background in images.
- Concept: Converting a grayscale image into a binary image.
- Importance: Simplification for further analysis (e.g., edge detection, segmentation).

---

## Global Thresholding

- Explanation of **Global Thresholding**.
- Example of a fixed threshold \( T \):
  - If \( I(x, y) > T \), the pixel becomes white (1), otherwise black (0).
- Limitations: Sensitivity to uneven lighting.

---

## Adaptive Thresholding

- Definition of **Adaptive Thresholding**.
- Instead of a global threshold, the threshold is calculated for different regions of the image.
- Advantages: Effective in images with varying illumination.
- Algorithm: Example of an adaptive method based on the local mean of neighboring pixels.

---

## Otsu's Algorithm

- Explanation of **Otsu's Algorithm**.
  - Automatic global thresholding that minimizes the within-class variance.
- Steps of the algorithm:
  1. Compute image histograms.
  2. Evaluate the between-class variance function for every possible threshold.
  3. Select the threshold that minimizes the within-class variance.
- Advantages: Automatic and effective in bimodal images.

---

## Justification for Using Thresholding

- When thresholding is useful:
  - Images with a clear contrast between the object and the background.
  - Situations requiring quick segmentation.
- Example applications:
  - Text detection, object recognition, medical images (e.g., X-rays).
- Limitations: Less effective in noisy or low-quality images.

---

## Comparison of Thresholding Algorithms

| Method           | Precision | Processing Speed | Ease of Implementation | Typical Applications  |
|------------------|-----------|------------------|------------------------|-----------------------|
| Global Threshold  | Medium    | Fast             | Simple                 | High-contrast images  |
| Adaptive Threshold| High      | Moderate         | Moderate               | Unevenly lit images   |
| Otsu's Algorithm  | High      | Moderate         | Moderate               | Bimodal distributions |

---

## Practical Examples

- Show examples of original images and the result after applying:
  - Global Thresholding.
  - Adaptive Thresholding.
  - Otsu's Algorithm.
- Visualizations highlighting the differences.

---

## Practical Examples

:::{.panel-tabset}

## Results



::: {.cell layout-ncol="3"}
::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-1.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-2.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-3.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-4.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-5.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-6.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-7.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-8.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-9.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-10.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-11.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/Image_Thresholding_Creation-12.png){width=960}
:::
:::



## Code



::: {.cell}

```{.python .cell-code}
plt.imshow(image_circle, cmap="gray")
plt.axis("off")
plt.show()

plt.imshow(image_gradient, cmap="gray")
plt.axis("off")
plt.show()

plt.imshow(noisy_circle, cmap="gray")
plt.axis("off")
plt.show()

_, global_thresh1 = cv2.threshold(image_circle, 127, 255, cv2.THRESH_BINARY)
plt.imshow(global_thresh1, cmap="gray")
plt.axis("off")
plt.show()

_, global_thresh2 = cv2.threshold(image_gradient, 127, 255, cv2.THRESH_BINARY)
plt.imshow(global_thresh2, cmap="gray")
plt.axis("off")
plt.show()

_, global_thresh3 = cv2.threshold(noisy_circle, 127, 255, cv2.THRESH_BINARY)
plt.imshow(global_thresh3, cmap="gray")
plt.axis("off")
plt.show()

adaptive_thresh1 = cv2.adaptiveThreshold(image_circle, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
plt.imshow(adaptive_thresh1, cmap="gray")
plt.axis("off")
plt.show()

adaptive_thresh2 = cv2.adaptiveThreshold(image_gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
plt.imshow(adaptive_thresh2, cmap="gray")
plt.axis("off")
plt.show()

adaptive_thresh3 = cv2.adaptiveThreshold(noisy_circle, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
plt.imshow(adaptive_thresh3, cmap="gray")
plt.axis("off")
plt.show()

_, otsu_thresh1 = cv2.threshold(image_circle, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(otsu_thresh1, cmap="gray")
plt.axis("off");
plt.show()

_, otsu_thresh2 = cv2.threshold(image_gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(otsu_thresh2, cmap="gray")
plt.axis("off")
plt.show()

_, otsu_thresh3 = cv2.threshold(noisy_circle, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(otsu_thresh3, cmap="gray")
plt.axis("off")
plt.show()
```
:::



:::


---

## Conclusion and Questions

- Summary of key points:
  - Thresholding as a simple yet powerful technique.
  - Importance of choosing the right algorithm depending on the context.
  - Otsu's algorithm as an effective solution for bimodal images.

##  Introduction to Morphological Operations

- Definition: Morphological operations apply a structuring element to an image to alter its structure.
- Focus: Primarily used for binary images.
- Key applications: Noise removal, object extraction, shape analysis.

---

## Structuring Element

- A small matrix used to probe and interact with a given image.
- Common shapes: Rectangular, circular, elliptical.
- Example: A 3x3 square structuring element.

$$\begin{bmatrix}
  1 & 1 & 1 \\
  1 & 1 & 1 \\
  1 & 1 & 1 \\
\end{bmatrix}$$

## Common Morphological Operations

- **Erosion**:
  - Removes pixels on object boundaries.
  - Shrinks the size of objects in the image.
  - Used to eliminate small noise or detach connected objects.

- **Dilation**:
  - Adds pixels to object boundaries.
  - Enlarges the object in an image.
  - Helps fill small holes and gaps within objects.

- **Opening**:
  - Erosion followed by dilation.
  - Used to remove small objects (noise) while maintaining the shape of larger objects.

- **Closing**:
  - Dilation followed by erosion.
  - Fills small holes and gaps in an object’s boundaries.

- **Top-Hat Transformation**:
  - The difference between the original image and its **opening**.
  - Used to highlight **bright regions** on a dark background.
  - Detecting small, bright objects or details in an unevenly illuminated image.
  
- **Black-Hat Transformation**:
  - The difference between the **closing** of an image and the original image.
  - Used to highlight **dark regions** on a bright background.
  - Emphasizing dark objects or shadows in an image.

## Example of Morphological Operations



::: {.cell}
::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/unnamed-chunk-5-25.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/unnamed-chunk-5-26.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/unnamed-chunk-5-27.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/unnamed-chunk-5-28.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/unnamed-chunk-5-29.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/unnamed-chunk-5-30.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/unnamed-chunk-5-31.png){width=960}
:::

::: {.cell-output-display}
![](Lect006_Imag_Proc_002_files/figure-revealjs/unnamed-chunk-5-32.png){width=960}
:::
:::



## Introduction to Frequency Response

- **What is Frequency Response?** 
  - The frequency response of an image shows how spatial details in the image are distributed across different frequencies.
  - In image processing, this is typically analyzed using the **Fourier Transform**.
- **Why Frequency Analysis?**
  - Useful for identifying patterns, noise, and image structures not easily observed in the spatial domain.

---

## The Fourier Transform

- **Fourier Transform (FT)**:
  - Converts an image from the **spatial domain** (pixels) to the **frequency domain** (sinusoids).
  - Each point in the frequency domain represents a specific frequency in the image.
- **Mathematical Basis**:
  - \( F(u,v) = \sum_x \sum_y f(x,y) e^{-j 2 \pi (ux/M + vy/N)} \)
  - Where \( F(u,v) \) is the frequency representation of the image.

---

## Low and High Frequencies

- **Low Frequencies**: 
  - Represent **slow variations** or large structures in the image (e.g., background or smooth gradients).
- **High Frequencies**:
  - Represent **rapid variations** or fine details (e.g., edges, noise).
- **Key Insight**: Most of the important structural information in an image is captured in the low-frequency range.

---

## Frequency Domain Representation

- The Fourier Transform of an image produces a **frequency spectrum**.
- **DC Component** (center of the spectrum): Represents the average intensity of the image.
- **Higher frequencies**: Spread out from the center and capture finer details.
  
- **Logarithmic scale**: Often used to visualize the frequency spectrum due to the wide range of values.

---

## The 2D Discrete Fourier Transform (DFT)

- The **2D DFT** is used to convert a 2D image into its frequency components:
  - **Input**: A 2D grayscale image.
  - **Output**: A complex matrix representing amplitude and phase for each frequency.
  
- **Inverse DFT**: Converts the frequency representation back to the spatial domain.

---

## Low-Pass and High-Pass Filtering

- **Low-Pass Filter (LPF)**:
  - Allows **low frequencies** to pass, attenuates high frequencies.
  - Used to **blur** images, removing high-frequency details like noise and edges.
  
- **High-Pass Filter (HPF)**:
  - Allows **high frequencies** to pass, attenuates low frequencies.
  - Used to **sharpen** images by enhancing edges and fine details.

---

## Band-Pass Filtering

- **Band-Pass Filter**:
  - Allows frequencies within a certain range (band) to pass.
  - Useful for **selectively enhancing specific frequency components** while filtering others.
- Applications: Used in image enhancement and texture analysis.

---

## Frequency Response Visualization

- **Magnitude Spectrum**: 
  - Represents the amplitude of each frequency component.
  - Typically visualized using the logarithmic scale to manage the large range of values.

- **Phase Spectrum**: 
  - Represents the phase of each frequency component.
  - Less important for human perception but crucial for reconstructing the image.

---

## Applications of Frequency Domain Processing

- **Noise Removal**: Low-pass filters can smooth out high-frequency noise.
- **Edge Detection**: High-pass filters enhance edges and sharp transitions.
- **Image Compression**: Frequency domain analysis helps identify redundant information.
- **Pattern Recognition**: Useful for detecting repetitive patterns like textures.




---