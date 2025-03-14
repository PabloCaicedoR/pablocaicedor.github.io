---
title: '[Predict the onset of diabetes based on diagnostic measures](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)'
---





## Context

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

[Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.](https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC2245318&blobtype=pdf)

## Variables

- **Pregnancies**: Number of times pregnant

- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test

- **BloodPressur**e: Diastolic blood pressure (mm Hg)

- **SkinThickness**: Triceps skin fold thickness (mm)

- **Insulin**: 2-Hour serum insulin (mu U/ml)

- **BMI**: Body mass index (weight in kg/(height in m)^2)

- **DiabetesPedigreeFunction**: Diabetes pedigree function

- **Age**: Age (years)

- **Outcome**: Class variable (0 or 1) 268 of 768 are 1, the others are 0

::: {#cell-5 .cell execution_count=1}
``` {.python .cell-code}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import statsmodels.api as sm
```
:::


## Load data

::: {#cell-7 .cell execution_count=2}
``` {.python .cell-code}
data = pd.read_csv("../../data/diabetes.csv")
```
:::


## Check any missing values

## Explore the data relationship

## Normalize and standarize the data

## Create neural network data

## Train model

## Eval Model

