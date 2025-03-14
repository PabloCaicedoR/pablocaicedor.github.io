---
title: "Computación de seno y coseno usando expansión de Taylor"
subtitle: "Un ejemplo de clase"
description: "Un ejemplo de clase del cálculo de una serie de Taylor sin uso de librerías especiales de Python -- En construcción --"
lang: es
author: "Ph.D. Pablo Eduardo Caicedo R."
date: "Febrero 6, 2023"
date-modified: "Mayo 25, 2023"
image: "../recursos/imagenes/tutoriales/taylor_expansion/taylor_expansion.jpg"
---





Las ecuaciones de las expansiones de Taylor (centradas en cero) fueron extraídas de la recopilación que hizo [Wikipedia](https://es.wikipedia.org/wiki/Serie_de_Taylor)

$$cos\left(x\right) = \sum_{n=0}^{\infty}{\frac{x^{2n}}{2n!}\left(-1\right)^{n}}$$
$$sin\left(x\right) = \sum_{n=0}^{\infty}{\frac{\left(-1\right)^{n}}{\left(2n+1\right)!}x^{2n+1}}$$

::: {#198a0f80 .cell execution_count=1}
``` {.python .cell-code}
def factorial(x):
    output = 1
    for k in range(1,x+1):
        output = output*k
    return output
```
:::


::: {#351cf2d0 .cell execution_count=2}
``` {.python .cell-code}
def sin_taylor_expansion(x,n):
    pi = 3.141592653589793238462643383279502884197169399375105820974944
    x = pi*x/180
    output = 0
    for k in range(0, n):
        term = (((-1)**k)/factorial(2*k + 1))*(x**(2*k+1))
        output = output+term
    return output
```
:::


::: {#2326dbbd .cell execution_count=3}
``` {.python .cell-code}
v_est = sin_taylor_expansion(30,5)

print(v_est)

print("Error Relativo:", abs(0.5-v_est)/0.5)
```

::: {.cell-output .cell-output-stdout}
```
0.5000000000202799
Error Relativo: 4.0559777758630844e-11
```
:::
:::


