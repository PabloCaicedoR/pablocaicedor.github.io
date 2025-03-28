---
title: "Sistemas y Señales Biomédicos"
description: "SYSB"
subtitle: "Ingeniería Biomédica"
lang: es
author: "Ph.D. Pablo Eduardo Caicedo Rodríguez"
date: "2025-03-14"
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



# Sistemas y Señales Biomedicos - SYSB

# Digital Filters

## Introduction

- The **Z-transform** is a fundamental tool in **digital signal processing (DSP)**, widely used in **biomedical signal processing**.
- It allows analysis of **discrete-time biomedical signals** such as:
  - **ECG (Electrocardiogram)**: Heart activity
  - **EEG (Electroencephalogram)**: Brain waves
  - **EMG (Electromyogram)**: Muscle activity
- It helps design **digital filters**, analyze **system stability**, and perform **signal reconstruction**.

---

## Definition of the Z-Transform

The **Z-transform** of a discrete-time signal $x[n]$ is defined as:

$$X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}$$

where:
- $X(z)$ is the **Z-domain representation** of $x[n]$.
- $z$ is a **complex variable**: $z = r e^{j\omega}$, where $r$ is the magnitude and $\omega$ is the frequency.
- The Z-transform provides a way to study **frequency characteristics** of biomedical signals.

---

## The Inverse Z-Transform

- The **Inverse Z-Transform** allows us to retrieve the original discrete-time signal from $X(z)$.
- It is given by:

 $$x[n] = \frac{1}{2\pi j} \oint_{C} X(z) z^{n-1} dz$$

- Common methods to compute the inverse Z-transform:
  1. **Partial Fraction Expansion**: Used when dealing with rational functions.
  2. **Power Series Expansion**: Expanding $X(z)$ as a series to identify coefficients.
  3. **Residue Method**: Using contour integration for more complex cases.

### **Example: Step Response in Biomedical Systems**
- A simple **low-pass filter** used for smoothing an ECG signal has:
  
 $$H(z) = \frac{1}{1 - 0.9z^{-1}}$$
  
  Expanding in a power series:
  
 $$H(z) = 1 + 0.9z^{-1} + 0.81z^{-2} + 0.729z^{-3} + ...$$
  
  The inverse Z-transform reveals an **exponentially decaying impulse response**, modeling the smoothing effect of the filter.

---

## Relationship with Sampling Method

- **Biomedical signals originate in continuous time** and must be sampled for digital processing.
- **Sampling frequency ($f_s$)** determines the accuracy of the digital representation:
  
 $$T_s = \frac{1}{f_s}$$
  
- The Z-transform relates to sampling through the **Discrete-Time Fourier Transform (DTFT)**:
  
 $$X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n] e^{-j\omega n}$$
  
  where $X(e^{j\omega})$ is obtained by evaluating the Z-transform along the unit circle: $z = e^{j\omega}$.

### **Example: ECG Sampling Rate**
- Standard ECG systems sample at **250 Hz or 500 Hz**.
- The Nyquist frequency is **125 Hz or 250 Hz**, respectively.
- Using the **Z-transform**, we analyze how filters modify the frequency content of ECG signals.

---

## Region of Convergence (ROC)

- The **Region of Convergence (ROC)** determines where the Z-transform sum **converges**.
- The ROC provides insights into:
  - **System stability**
  - **Causality**
  - **Invertibility**

### **Types of ROC:**
1. **Right-sided signals (causal systems):**
   - The ROC is **outside the outermost pole**.
   - The system is **stable if the ROC includes the unit circle** ($|z| = 1$).
   
2. **Left-sided signals (anti-causal systems):**
   - The ROC is **inside the innermost pole**.
   
3. **Two-sided signals:**
   - The ROC lies **between poles**.

### **Example: Stability of a Biomedical Filter**
- A **high-pass ECG filter** with transfer function:
  
 $$H(z) = \frac{1 - z^{-1}}{1 - 0.95 z^{-1}}$$
  
  - The **pole at 0.95** means the system is stable since $|0.95| < 1$.
  - If the pole were **outside the unit circle**, the system would be unstable.

---

## Relationship with Convolution

- In biomedical DSP, **filtering operations rely on convolution**.
- **Convolution in time domain**:
  
 $$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-k]$$

- **Multiplication in Z-domain**:
  
 $$Y(z) = X(z) H(z)$$

- This simplifies filter design, allowing us to analyze biomedical signals efficiently.

### **Example: EEG Band-Pass Filtering**
- EEG signals contain different **frequency bands**:
  - **Delta (0.5–4 Hz)**: Deep sleep
  - **Theta (4–8 Hz)**: Relaxation
  - **Alpha (8–12 Hz)**: Resting state
  - **Beta (12–30 Hz)**: Active thinking
- A **band-pass filter** for extracting **alpha waves (8–12 Hz)** is designed as:
  
 $$H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$$
  
  The Z-transform allows us to **analyze and optimize this filter**.

---

## Conclusion

- The **Z-transform** is crucial for **analyzing and processing biomedical signals**.
- It enables:
  - **Stability analysis** (Region of Convergence)
  - **Filtering and feature extraction** (EEG, ECG, EMG signals)
  - **Efficient signal convolution**
- The **inverse Z-transform** reconstructs signals for further analysis.
- Understanding the Z-transform helps in **filter design, denoising, and feature extraction** in biomedical applications.

---

## References

- Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing*.
- Ingle, V. K., & Proakis, J. G. (2011). *Digital Signal Processing using MATLAB*.
- Rangayyan, R. M. (2015). *Biomedical Signal Analysis*.
