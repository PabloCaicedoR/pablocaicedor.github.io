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

# Machine Learning Introduction[^1]


::: {.class="x_small_font"}
[^1]: Diapositivas basadas en el texto: J. D. Kelleher, B. Mac Namee, y A. D’Arcy, Fundamentals of machine learning for predictive data analytics: algorithms, worked examples, and case studies, 2nd ed. Cambridge: The MIT press, 2020.
:::

##  Introduction

:::: {.columns}

::: {.column width="45%"}

::: {.callout-important title="Definitions"}

**Machine Learning:**  is defined as an automated process that extracts patterns from data.

:::

:::

::: {.column width="45%"}

![](../../recursos/imagenes/Presentaciones/ASIM/machine_learning_process.png)

:::
::::


## Introduction 

### How machine learning works?    
Machine learning algorithms work by searching through a set of possible prediction models for the model that best captures the relationship between the descriptive features and target feature in a dataset.



::: {.cell}
::: {.cell-output .cell-output-stdout}

```
   Pregnancies  Glucose  BloodPressure  ...  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72  ...                     0.627   50        1
1            1       85             66  ...                     0.351   31        0
2            8      183             64  ...                     0.672   32        1
3            1       89             66  ...                     0.167   21        0
4            0      137             40  ...                     2.288   33        1

[5 rows x 9 columns]
```


:::
:::



## Introduction

::: {.callout-caution title="What can be wrong???"}
* When we are dealing with large datasets, it is likely that there is noise.
* When we are dealing with large datasets, it is likely that there is missing data.
* When we are dealing with large datasets, it is likely that there is data leakage.
:::

::: {.callout-important title="Ill-posed problem"}
Ill-posed problem, that is, a problem for which a unique solution cannot be determined using only the information that is available
:::

## Introduction

![](../../recursos/imagenes/Presentaciones/ASIM/ds_workflow.png)

## Data Understanding Workflow.

::: {.callout-important title="Exploratory data analysis"}
1. Data Loading.
2. Basic Statistics: Displays summary statistics.
3. Missing Values Check: Identifies missing values.
4. Feature Distributions: Visualizes distributions using histograms or countplots.
5. Relationship between variables.
:::

## Data Understanding Workflow



::: {.cell}

```{.python .cell-code}
# Identify variable types
discrete_vars = ["Pregnancies"]  # Discrete numerical variable
categorical_vars = ["Outcome"]  # Class label
continuous_vars = [
    col
    for col in data.select_dtypes(include=[np.number]).columns
    if col not in discrete_vars + ["Outcome"]
]

# Basic dataset information
print("Dataset Information:\n", data.info())
print("\nSummary Statistics:\n", data.describe())
print("\nMissing Values:\n", data.isnull().sum())

# Ensure numeric data and handle NaN or infinite values
numeric_data = data.select_dtypes(include=[np.number]).dropna()
numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan).dropna()

# Dynamically determine the number of rows for subplots
num_cont_vars = len(continuous_vars)
rows = (num_cont_vars // 3) + (num_cont_vars % 3 > 0)  # Ensures proper grid layout

# Plot distributions for continuous variables
plt.figure(figsize=(12, 4 * rows))
for i, column in enumerate(continuous_vars, 1):
    plt.subplot(rows, 3, i)
    sns.histplot(numeric_data[column], kde=True, bins=20, color="skyblue")
    plt.title(f"Distribution of {column}")
plt.tight_layout()
plt.show()

# Plot distribution for discrete variable (Pregnancies) using a countplot
plt.figure(figsize=(8, 4))
sns.countplot(x="Pregnancies", data=numeric_data, palette="viridis")
plt.title("Count of Pregnancies")
plt.show()

# Plot class distribution for Outcome
plt.figure(figsize=(6, 4))
sns.countplot(x="Outcome", data=data, palette="coolwarm")
plt.title("Class Distribution of Outcome")
plt.xlabel("Diabetes Diagnosis (0: No, 1: Yes)")
plt.ylabel("Count")
plt.show()

# Correlation heatmap to check relationships
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
```
:::