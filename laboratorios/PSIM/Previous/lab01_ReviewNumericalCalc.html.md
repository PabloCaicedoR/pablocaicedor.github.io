---
title: Numerical Calculus Review
---





## Data Creation

::: {#cell-3 .cell execution_count=2}
``` {.python .cell-code}
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 4*x**2 + 4

t = np.linspace(-10, 10, 1000)
f_t = f(t)

plt.plot(t,f_t)
```

::: {.cell-output .cell-output-display}
![](lab01_ReviewNumericalCalc_files/figure-html/cell-2-output-1.png){}
:::
:::


## Numerical Diferentation

Determine the numerical derivative of the function f, represented as $\left(\frac{df}{dt}\right)$, via finite difference approximation. Subsequently, contrast this result with the numerical evaluation of the symbolic derivative, derived through analytical differentiation, to assess the precision of the numerical approach


## Numerical Integration

Determine the numerical integration of the function f, represented as $\left(\int_{-10}^{10}{f(t)dt}\right)$, via trapezoidal rule. Subsequently, contrast this result with the numerical evaluation of the symbolic integral, derived through analytical differentiation, to assess the precision of the numerical approach


## Numerical solutions of ordinary differential equations (ODEs)

Solve the ODE $\frac{dy}{dx} = 2x - 3y$ with initial condition y(0) = 1 using Euler's Method. With a step 0f 0.1. Suppose that: $x \in \left[0, 10\right]$ and $y \in \left[1, 10\right]$


## Numerical optimization

Minimize the function f(x) = x^4 - 4x^2 + 4 using Gradient Descent.


