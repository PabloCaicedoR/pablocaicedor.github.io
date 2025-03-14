---
title: "Adquisición y Procesamiento de Señales Biomédicas en Tecnologías de Borde"
description: "APSB"
subtitle: "Ingeniería Biomédica"
lang: es
author: "Ph.D. Pablo Eduardo Caicedo Rodríguez"
date: "2025-01-20"
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



# Adquisición y Procesamiento de Señales Biomédicas en Tecnologías de Borde - APSB

## Methodology for designing an edge ai device

1. Problem Definition & Use Case Analysis
2. Data Collection & Preprocessing
3. Model Selection & Optimization
4. Hardware Selection
5. Deployment & Model Inference
6. Testing, Validation, and Continuous Improvement
7. Final Deployment & Scaling

## Problem Definition & Use Case Analysis

- Identify the specific AI task (e.g., real-time ECG analysis, fall detection, predictive maintenance in IoT).
- Determine operational constraints, including:
    - Power consumption (battery-operated vs. wired).
    - Latency requirements (real-time processing vs. periodic updates).
    - Communication needs (Wi-Fi, Bluetooth, LoRa, standalone processing).

## Data Collection & Preprocessing

- Sensor Selection: Choose sensors relevant to the application (e.g., accelerometers for motion tracking, biosensors for health monitoring).
- Edge-Compatible Data Acquisition: Optimize data formats to reduce memory and computational load.
- Preprocessing on Edge:
    - Signal filtering (e.g., noise reduction in biomedical signals).
    - Feature extraction (e.g., time-series features for motion classification).

## Model Selection & Optimization

- Model Selection:
    - Lightweight CNNs (for image processing).
    - Recurrent Neural Networks (RNNs) / LSTMs (for time-series data like ECG).
    - TinyML models optimized for microcontrollers (e.g., TensorFlow Lite, PyTorch Mobile).

- Model Optimization for Edge Deployment:
    - Quantization: Convert floating-point models to int8 or int16 to reduce size and computation load.
    - Pruning: Remove unnecessary neurons or layers while preserving accuracy.
    - Distillation: Train a smaller model using knowledge from a larger one.

## Hardware Selection

- Processing Unit:
    - Microcontrollers (MCUs) (e.g., ARM Cortex-M, ESP32) → Low-power, simple AI tasks.
    - Edge AI Accelerators (e.g., Google Edge TPU, NVIDIA Jetson Nano) → More complex AI processing.
    - FPGAs (Field-Programmable Gate Arrays) → Custom AI workloads for high-speed processing.

- Memory & Storage:
    - RAM Optimization: Choose embedded SRAM or external DRAM depending on model size.
    - Flash Storage: Store inference models efficiently.

- Connectivity:
    - Offline processing for low-latency applications.
    - Edge-to-cloud integration for periodic updates.

## Deployment & Model Inference
- Convert trained AI models into optimized edge-compatible formats (e.g., TensorFlow Lite, ONNX).
- Implement real-time inference using hardware-accelerated libraries (e.g., TensorRT, OpenVINO).
- Optimize firmware for energy efficiency using duty-cycling techniques (process only when necessary).

## Testing, Validation, and Continuous Improvement

- Edge Benchmarking:
  - Measure inference speed and power consumption.
  - Validate model accuracy on real-world edge-generated data.

- Security & Reliability:
    - Implement secure boot & firmware updates to prevent cyber threats.
    - Ensure robust error handling for sensor malfunctions.

- Feedback & Model Updating:
    - If connected to a cloud system, update models periodically using federated learning.
    - Optimize AI pipelines with incremental learning on-device where feasible.

## Final Deployment & Scaling

- Deploy at scale, ensuring the Edge AI model adapts to different environments.
- Implement remote monitoring & diagnostics for predictive maintenance.
- Enable over-the-air (OTA) updates to improve AI models post-deployment.

## Abstract

The hardware-software co-design approach is the most widely used methodology for Edge AI device development. It ensures:

- Real-time performance with optimized AI models.
- Energy-efficient processing for battery-operated or low-power devices.
- Scalability and security in edge environments.

This methodology is industry-standard and used by leading companies in healthcare, automotive, and industrial IoT, ensuring robust and reliable Edge AI solutions.

## Example of application

::: {.callout-important title="Use case"}
A wearable ECG monitoring device designed for continuous heart health tracking and arrhythmia detection. This Edge AI-based solution analyzes ECG signals in real-time on a low-power microcontroller, providing instant alerts for cardiac irregularities without relying on cloud computing.
:::

## Step 1: Problem Definition & Use Case Analysis

::: {.callout-note title="Objective"}
Detect abnormal heart rhythms (arrhythmias) in real-time using a wearable ECG device.
:::

### Operational Constraints:
- Must be energy-efficient (battery-operated, low power consumption).
- Needs real-time inference for immediate alerts.
- Should operate offline, but sync with mobile apps for periodic review.

### Key Challenges:
- Processing ECG data on a low-power Edge device.
- Minimizing false positives/negatives in arrhythmia detection.
- Ensuring high reliability and accuracy.

## Step 2: Data Collection & Preprocessing

### Sensor Selection:
- ECG sensor (e.g., AD8232) captures raw heart signals.
- Accelerometer (optional) for motion artifacts reduction.

### Edge-Compatible Data Acquisition:
- Sample rate: 250 Hz (sufficient for arrhythmia detection).
- Use on-device filtering (low-pass filters) to remove noise.

### Preprocessing on Edge:
- Apply Butterworth filters for noise reduction.
- R-peak detection using Pan-Tompkins algorithm for heart rate calculation.
- Extract features like RR intervals, QRS width, and HR variability.


## Step 3: Model Selection & Optimization


### AI Model:
- Use 1D CNN + LSTM hybrid model (efficient for ECG signal processing).
- Train the model using MIT-BIH Arrhythmia Database.

### Model Optimization for Edge AI:
- Quantization: Convert model to int8 precision using TensorFlow Lite.
- Pruning: Remove redundant neurons to reduce computation load.
- Knowledge Distillation: Train a smaller model from a high-performing one.

## Step 4: Hardware Selection

### Microcontroller (MCU):
- Nordic nRF52840 (low-power ARM Cortex-M4 + BLE connectivity).
- Alternative: ESP32 (for low-cost AI inference).

### Memory & Storage:
- RAM: 512KB (optimized for Edge AI processing).
- Flash storage: 4MB (stores ECG data logs for later analysis).

### Connectivity:
- Bluetooth Low Energy (BLE) for periodic sync with mobile apps.
- Can function offline with real-time alerts.


## Step 5: Deployment & Model Inference

- Convert trained TensorFlow model → TensorFlow Lite for Edge AI inference.
- Deploy on the Nordic nRF52840 MCU using TensorFlow Lite for Microcontrollers.
- Use hardware-accelerated inference for efficient processing.
- Implement event-driven processing (AI runs only on abnormal detections to save power).

## Step 6: Testing, Validation, and Continuous Improvement

### Edge Benchmarking:
- Real-time inference latency: <10 ms per ECG segment.
- Power consumption: 5mW (optimized for long battery life).

### Security & Reliability:
- Secure Boot & Firmware Updates to prevent hacking.
- Adaptive AI Models: Learns individual patient heart patterns to reduce false alarms.

### Feedback & Model Updating:
- Sync detected arrhythmia events with a cloud server for validation.
- Use federated learning to improve AI models without sharing raw patient data.


## Step 7: Final Deployment & Scaling
- Mass production of the device for hospitals, clinics, and home use.
- Integration with mobile apps for patient-doctor communication.
- Regulatory Approval: Submit for FDA/CE certification for medical device compliance.
- Over-the-Air (OTA) Updates: Allow model updates based on new ECG patterns.