---
title: Health Care Cost Predictor
---




The data for this example is located in Kaggle in the following [URL](https://www.kaggle.com/datasets/mirichoi0218/insurance), but the modified file for this class is located [here](../../data/insurance_2.csv)

## Context of the data

The content is adapted from kaggle. 

The datasets utilized in 'Machine Learning with R' by Brett Lantz are a valuable resource for learners, providing a foundation for hands-on experience with machine learning concepts. Although Packt Publishing does not make these datasets readily available online, they can be accessed through public domain sources, requiring only minor preprocessing and formatting to match the book's specifications. This presents an opportunity for readers to engage deeply with the material, reproducing and building upon the book's examples to reinforce their understanding of machine learning principles.

## Variables

**age**: age of primary beneficiary

**sex**: insurance contractor gender, female, male

**bmi**: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight $\left(kg / m^2\right)$ using the ratio of height to weight, ideally 18.5 to 24.9

**children**: Number of children covered by health insurance / Number of dependents

**smoker**: Smoking

**salary**: Salary of the insurance contractor

**region**: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.

**charges**: Individual medical costs billed by health insurance


## Configuration of solution

::: {#cell-3 .cell execution_count=1}
``` {.python .cell-code}
data_path = "../../data/"
```
:::


## Library Load

::: {#cell-5 .cell execution_count=2}
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


## Data Load

::: {#cell-7 .cell execution_count=3}
``` {.python .cell-code}
data = pd.read_csv(data_path+"insurance_2.csv")
```
:::


::: {#cell-8 .cell execution_count=4}
``` {.python .cell-code}
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device = 0  # "Select the index of the GPU you wish to use"
torch.cuda.set_device(device)
print(f"GPU selection: {torch.cuda.get_device_name(device)}")
```

::: {.cell-output .cell-output-stdout}
```
Number of GPUs available: 1
GPU 0: NVIDIA GeForce MX110
GPU selection: NVIDIA GeForce MX110
```
:::
:::


## Understanding the data

1. Loading and summarizing data
2. Visualizing distributions
3. Exploring relationships between variables
4. Analyzing categorical variables

### 1. Loading and summarizing data

::: {#cell-11 .cell execution_count=5}
``` {.python .cell-code}
data.info()
```

::: {.cell-output .cell-output-stdout}
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 8 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       1338 non-null   int64  
 1   sex       1338 non-null   object 
 2   bmi       1338 non-null   float64
 3   children  1338 non-null   int64  
 4   smoker    1338 non-null   object 
 5   salary    1338 non-null   float64
 6   region    1338 non-null   object 
 7   charges   1338 non-null   float64
dtypes: float64(3), int64(2), object(3)
memory usage: 83.8+ KB
```
:::
:::


::: {#cell-12 .cell execution_count=6}
``` {.python .cell-code}
data.describe()
```

::: {.cell-output .cell-output-display execution_count=6}

```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>salary</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
      <td>1338.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.207025</td>
      <td>30.663397</td>
      <td>1.094918</td>
      <td>159064.411451</td>
      <td>13270.422265</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.049960</td>
      <td>6.098187</td>
      <td>1.205493</td>
      <td>41741.994963</td>
      <td>12110.011237</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>15.960000</td>
      <td>0.000000</td>
      <td>104622.922023</td>
      <td>1121.873900</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>26.296250</td>
      <td>0.000000</td>
      <td>130087.161933</td>
      <td>4740.287150</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>30.400000</td>
      <td>1.000000</td>
      <td>146740.897257</td>
      <td>9382.033000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>51.000000</td>
      <td>34.693750</td>
      <td>2.000000</td>
      <td>171897.191284</td>
      <td>16639.912515</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>53.130000</td>
      <td>5.000000</td>
      <td>338460.517246</td>
      <td>63770.428010</td>
    </tr>
  </tbody>
</table>
</div>
```

:::
:::


::: {#cell-13 .cell execution_count=7}
``` {.python .cell-code}
data.select_dtypes("object")
```

::: {.cell-output .cell-output-display execution_count=7}

```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>smoker</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>yes</td>
      <td>southwest</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>no</td>
      <td>southeast</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>no</td>
      <td>southeast</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>no</td>
      <td>northwest</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>no</td>
      <td>northwest</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>male</td>
      <td>no</td>
      <td>northwest</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>female</td>
      <td>no</td>
      <td>northeast</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>female</td>
      <td>no</td>
      <td>southeast</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>female</td>
      <td>no</td>
      <td>southwest</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>female</td>
      <td>yes</td>
      <td>northwest</td>
    </tr>
  </tbody>
</table>
<p>1338 rows × 3 columns</p>
</div>
```

:::
:::


::: {#cell-14 .cell execution_count=8}
``` {.python .cell-code}
data["sex"] = data["sex"].astype("category")
data["smoker"] = data["smoker"].astype("category")
data["region"] = data["region"].astype("category")
```
:::


::: {#cell-15 .cell execution_count=9}
``` {.python .cell-code}
data.select_dtypes("number")
```

::: {.cell-output .cell-output-display execution_count=9}

```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>children</th>
      <th>salary</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>27.900</td>
      <td>0</td>
      <td>159272.812482</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>33.770</td>
      <td>1</td>
      <td>117088.625944</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>33.000</td>
      <td>3</td>
      <td>129043.852213</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>22.705</td>
      <td>0</td>
      <td>194635.486180</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>28.880</td>
      <td>0</td>
      <td>113585.904592</td>
      <td>3866.85520</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>50</td>
      <td>30.970</td>
      <td>3</td>
      <td>145933.927725</td>
      <td>10600.54830</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>18</td>
      <td>31.920</td>
      <td>0</td>
      <td>117665.917758</td>
      <td>2205.98080</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>18</td>
      <td>36.850</td>
      <td>0</td>
      <td>133402.353115</td>
      <td>1629.83350</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>21</td>
      <td>25.800</td>
      <td>0</td>
      <td>133975.682996</td>
      <td>2007.94500</td>
    </tr>
    <tr>
      <th>1337</th>
      <td>61</td>
      <td>29.070</td>
      <td>0</td>
      <td>216658.755628</td>
      <td>29141.36030</td>
    </tr>
  </tbody>
</table>
<p>1338 rows × 5 columns</p>
</div>
```

:::
:::


::: {#cell-16 .cell execution_count=10}
``` {.python .cell-code}
data.info()
```

::: {.cell-output .cell-output-stdout}
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1338 entries, 0 to 1337
Data columns (total 8 columns):
 #   Column    Non-Null Count  Dtype   
---  ------    --------------  -----   
 0   age       1338 non-null   int64   
 1   sex       1338 non-null   category
 2   bmi       1338 non-null   float64 
 3   children  1338 non-null   int64   
 4   smoker    1338 non-null   category
 5   salary    1338 non-null   float64 
 6   region    1338 non-null   category
 7   charges   1338 non-null   float64 
dtypes: category(3), float64(3), int64(2)
memory usage: 56.8 KB
```
:::
:::


### 2. Visualizing distributions

::: {#cell-18 .cell execution_count=11}
``` {.python .cell-code}
sns.histplot(data["bmi"], stat="probability")
```

::: {.cell-output .cell-output-display}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-12-output-1.png){}
:::
:::


### 3. Exploring relationships between variables

::: {#cell-20 .cell execution_count=12}
``` {.python .cell-code}
sns.scatterplot(data=data, x="bmi", y="charges", hue="smoker")
```

::: {.cell-output .cell-output-display}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-13-output-1.png){}
:::
:::



### 4. Analyzing categorical variables

::: {#cell-22 .cell execution_count=13}
``` {.python .cell-code}
sns.countplot(data=data, x="smoker", stat="probability")
```

::: {.cell-output .cell-output-display}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-14-output-1.png){}
:::
:::


::: {#cell-23 .cell execution_count=14}
``` {.python .cell-code}
sns.boxplot(data=data, y="charges", x="smoker")
```

::: {.cell-output .cell-output-display}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-15-output-1.png){}
:::
:::


::: {#cell-24 .cell execution_count=15}
``` {.python .cell-code}
sns.pointplot(data=data, x="sex", y="charges", hue="smoker")
```

::: {.cell-output .cell-output-display}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-16-output-1.png){}
:::
:::


::: {#cell-25 .cell execution_count=16}
``` {.python .cell-code}
g001 = sns.FacetGrid(data=data, col="smoker", row="sex")
g001.map(plt.scatter, "bmi", "charges")
```

::: {.cell-output .cell-output-display}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-17-output-1.png){}
:::
:::


::: {#cell-26 .cell execution_count=17}
``` {.python .cell-code}
sns.regplot(data=data, x="salary", y="charges",
            scatter_kws={"color": "blue"},  # Color de los puntos
            line_kws={"color": "red"})
```

::: {.cell-output .cell-output-display}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-18-output-1.png){}
:::
:::


## 5. Checking availability of GPU

::: {#cell-28 .cell execution_count=30}
``` {.python .cell-code}
device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device1}")
device1
```

::: {.cell-output .cell-output-stdout}
```
Using device: cuda:0
```
:::

::: {.cell-output .cell-output-display execution_count=30}
```
device(type='cuda', index=0)
```
:::
:::


## 6. Splitting data

::: {#cell-30 .cell execution_count=19}
``` {.python .cell-code}
entrada = data["salary"].to_numpy().reshape(-1, 1)
salida = data["charges"].to_numpy().reshape(-1, 1)
```
:::


::: {#cell-31 .cell execution_count=20}
``` {.python .cell-code}
standarScaler_features = StandardScaler().fit(entrada)
standarScaler_output = StandardScaler().fit(salida)
```
:::


::: {#cell-32 .cell execution_count=21}
``` {.python .cell-code}
salary_train, salary_test, charges_train, charges_test = train_test_split(
    standarScaler_features.transform(entrada),
    standarScaler_output.transform(salida),
    train_size=0.7,
    shuffle=True,
)
```
:::


## 7. Converting Data To Tensor

::: {#cell-34 .cell execution_count=22}
``` {.python .cell-code}
t_salary_train = torch.tensor(salary_train, dtype=torch.float32, device=device1)
t_salary_test = torch.tensor(salary_test, dtype=torch.float32, device=device1)
t_charges_train = torch.tensor(charges_train, dtype=torch.float32, device=device1)
t_charges_test = torch.tensor(charges_test, dtype=torch.float32, device=device1)
```
:::


## 8. Model Implementation

::: {#cell-36 .cell execution_count=23}
``` {.python .cell-code}
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```
:::


::: {#cell-37 .cell execution_count=24}
``` {.python .cell-code}
model = LinearRegression().to(device1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
:::


## 10. Train Model

::: {#cell-39 .cell execution_count=25}
``` {.python .cell-code}
num_epochs = 1000
for epoch in range(num_epochs):

     # Fordward Pass and loss

     charges_predicted = model(t_salary_train)
     loss = criterion(charges_predicted, t_charges_train)

     # Backward pass
     loss.backward()

     #wweights update
     optimizer.step()
     optimizer.zero_grad()

     # Progress tracking

     if (epoch+1)%10 ==0:
          print(f"Epoch: {epoch+1}, loss={loss.item():.4f}")

```

::: {.cell-output .cell-output-stdout}
```
Epoch: 10, loss=1.2763
Epoch: 20, loss=0.8460
Epoch: 30, loss=0.5655
Epoch: 40, loss=0.3828
Epoch: 50, loss=0.2637
Epoch: 60, loss=0.1861
Epoch: 70, loss=0.1355
Epoch: 80, loss=0.1026
Epoch: 90, loss=0.0811
Epoch: 100, loss=0.0671
Epoch: 110, loss=0.0579
Epoch: 120, loss=0.0520
Epoch: 130, loss=0.0481
Epoch: 140, loss=0.0456
Epoch: 150, loss=0.0439
Epoch: 160, loss=0.0428
Epoch: 170, loss=0.0421
Epoch: 180, loss=0.0417
Epoch: 190, loss=0.0414
Epoch: 200, loss=0.0412
Epoch: 210, loss=0.0411
Epoch: 220, loss=0.0410
Epoch: 230, loss=0.0409
Epoch: 240, loss=0.0409
Epoch: 250, loss=0.0409
Epoch: 260, loss=0.0408
Epoch: 270, loss=0.0408
Epoch: 280, loss=0.0408
Epoch: 290, loss=0.0408
Epoch: 300, loss=0.0408
Epoch: 310, loss=0.0408
Epoch: 320, loss=0.0408
Epoch: 330, loss=0.0408
Epoch: 340, loss=0.0408
Epoch: 350, loss=0.0408
Epoch: 360, loss=0.0408
Epoch: 370, loss=0.0408
Epoch: 380, loss=0.0408
Epoch: 390, loss=0.0408
Epoch: 400, loss=0.0408
Epoch: 410, loss=0.0408
Epoch: 420, loss=0.0408
Epoch: 430, loss=0.0408
Epoch: 440, loss=0.0408
Epoch: 450, loss=0.0408
Epoch: 460, loss=0.0408
Epoch: 470, loss=0.0408
Epoch: 480, loss=0.0408
Epoch: 490, loss=0.0408
Epoch: 500, loss=0.0408
Epoch: 510, loss=0.0408
Epoch: 520, loss=0.0408
Epoch: 530, loss=0.0408
Epoch: 540, loss=0.0408
Epoch: 550, loss=0.0408
Epoch: 560, loss=0.0408
Epoch: 570, loss=0.0408
Epoch: 580, loss=0.0408
Epoch: 590, loss=0.0408
Epoch: 600, loss=0.0408
Epoch: 610, loss=0.0408
Epoch: 620, loss=0.0408
Epoch: 630, loss=0.0408
Epoch: 640, loss=0.0408
Epoch: 650, loss=0.0408
Epoch: 660, loss=0.0408
Epoch: 670, loss=0.0408
Epoch: 680, loss=0.0408
Epoch: 690, loss=0.0408
Epoch: 700, loss=0.0408
Epoch: 710, loss=0.0408
Epoch: 720, loss=0.0408
Epoch: 730, loss=0.0408
Epoch: 740, loss=0.0408
Epoch: 750, loss=0.0408
Epoch: 760, loss=0.0408
Epoch: 770, loss=0.0408
Epoch: 780, loss=0.0408
Epoch: 790, loss=0.0408
Epoch: 800, loss=0.0408
Epoch: 810, loss=0.0408
Epoch: 820, loss=0.0408
Epoch: 830, loss=0.0408
Epoch: 840, loss=0.0408
Epoch: 850, loss=0.0408
Epoch: 860, loss=0.0408
Epoch: 870, loss=0.0408
Epoch: 880, loss=0.0408
Epoch: 890, loss=0.0408
Epoch: 900, loss=0.0408
Epoch: 910, loss=0.0408
Epoch: 920, loss=0.0408
Epoch: 930, loss=0.0408
Epoch: 940, loss=0.0408
Epoch: 950, loss=0.0408
Epoch: 960, loss=0.0408
Epoch: 970, loss=0.0408
Epoch: 980, loss=0.0408
Epoch: 990, loss=0.0408
Epoch: 1000, loss=0.0408
```
:::
:::


::: {#cell-40 .cell execution_count=26}
``` {.python .cell-code}
with torch.no_grad():
    prediction = model(t_salary_test)
    mse = mean_squared_error(t_charges_test.cpu().numpy(), prediction.cpu().numpy())
    r2 = r2_score(t_charges_test.cpu().numpy(), prediction.cpu().numpy())

    plt.plot(
        standarScaler_features.inverse_transform(salary_test),
        standarScaler_output.inverse_transform(charges_test),
        "ro",
    )
    plt.plot(
        standarScaler_features.inverse_transform(salary_test),
        standarScaler_output.inverse_transform(prediction.cpu().numpy()),
        "b",
    )
```

::: {.cell-output .cell-output-display}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-27-output-1.png){}
:::
:::


## Model Performance

::: {#cell-42 .cell execution_count=27}
``` {.python .cell-code}
with torch.no_grad():
    prediction = model(t_salary_test)
    mse = mean_squared_error(t_charges_test.cpu().numpy(), prediction.cpu().numpy())
    r2 = r2_score(t_charges_test.cpu().numpy(), prediction.cpu().numpy())

    plt.plot(
        standarScaler_features.inverse_transform(salary_test),
        standarScaler_output.inverse_transform(charges_test),
        "ro",
    )
    plt.plot(
        standarScaler_features.inverse_transform(salary_test),
        standarScaler_output.inverse_transform(prediction.cpu().numpy()),
        "b",
    )
```

::: {.cell-output .cell-output-display}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-28-output-1.png){}
:::
:::


::: {#cell-43 .cell execution_count=28}
``` {.python .cell-code}
s_predicha = standarScaler_output.inverse_transform(prediction.cpu().numpy())
s_real = standarScaler_output.inverse_transform(charges_test)

residuos = s_real- s_predicha

sm.graphics.tsa.plot_acf(residuos, lags=100)

```

::: {.cell-output .cell-output-display execution_count=28}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-29-output-1.png){}
:::

::: {.cell-output .cell-output-display}
![](cod003_sol_LinearRegression_InsuranceCosts_files/figure-html/cell-29-output-2.png){}
:::
:::


