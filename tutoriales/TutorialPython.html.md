---
title: "Tutorial de Python"
subtitle: "Tomado de Ciencia de Datos para Ciencias Naturale - Keilor Rojas"
description: "Breve Tutorial de Python"
lang: es
author: "Ph.D. Pablo Eduardo Caicedo R."
date: "Febrero 6, 2023"
date-modified: "Mayo 25, 2023"
image: "../recursos/imagenes/tutoriales/python_programming/python_programming.jpg"
---





# Google Colab

Tomado del libro [Ciencia de Datos para Ciencias Naturales](https://bookdown.org/keilor_rojas/CienciaDatos/)

Si no tiene experiencia con el lenguaje Markdown utilice esta [guía](https://www.markdownguide.org/basic-syntax/) para enriquecer sus celdas de texto.


## Características

- Plataforma de Google Research.
- Permite a cualquier persona escribir y ejecutar código Python o R a través del navegador.
- Se base se basa en la interfase de Jupyter Notebook.
- Los documentos son denominados notebooks de Colab.
- Los entornos son interactivos.
- Permite la utilizar Python y R.
- Manejo de celdas de código


## Ventajas

- No requiere configuración del programa.
- La mayoría de librerías y programas ya están preinstalados.
- Acceso gratuito a GPU, es decir, se ejecuta en los servidores de Google.
- Facilidad para compartir documentos.

## Desventajas de Colab

- No se ejecuta sin conexión.
- Conjuntos de datos que se importan al entorno sin ser montado desde Google Drive se perderán cuando la máquina virtual se apague.
- Experiencia más sencilla que otras opciones.
- Permite utilizar más lenguajes: Posgres, Julia.

## Tipos de celdas

- **Código:** Para abrir una celda de código simplemente haga click en la barra + Código. Para ejecutar el código puede presionar el símbolo de play a la izquierda de la celda o las teclas Cmd/Ctrl+Enter.
- **Texto:** Para abrir una celda de texto simplemente haga click en la barra + Texto. Las celdas de texto utilizan la sintaxis de Markdown. Para ver el texto fuente de Markdown, haga doble click en una celda de texto.


# Python


Python es un lenguaje de programación de alto nivel, interpretado, multiparadigma y general-proposito. Su filosofia de diseno enfatiza la legibilidad del codigo. Es uno de los lenguajes de programacion mas populares en uso hoy en dia, y se utiliza para una amplia gama de aplicaciones, incluyendo desarrollo web, ciencia de datos, machine learning e inteligencia artificial.

# Ventajas
- Python es un lenguaje de programación multiparadigma, lo que significa que puede ser utilizado para diferentes tipos de programación, como la programación orientada a objetos, la programación imperativa y la programación funcional.
- Python es un lenguaje de programación interpretado, lo que significa que no necesita ser compilado antes de ser ejecutado. Esto hace que Python sea muy rápido de desarrollar y depurar.
- Python es un lenguaje de programación muy portable, lo que significa que puede ser ejecutado en diferentes plataformas, como Windows, Mac OS X y Linux. También puede ser ejecutado en la nube.
- Python tiene una gran comunidad de usuarios y desarrolladores, lo que significa que hay muchos recursos disponibles para aprender y usar Python.

# Desventajas
- Python puede ser un poco más lento que los lenguajes compilados, como C o C++.
- Python tiene una sintaxis un poco más compleja que algunos otros lenguajes de programación, como Java o JavaScript.
- Python puede no ser el mejor lenguaje de programación para ciertos tipos de aplicaciones, como las aplicaciones de juegos o las aplicaciones de alto rendimiento.

# Tipos básicos de variables en Python

|         **Tipo**         | **Nombre** |             **Uso común**            |     **Ejemplo**     |
|:------------------------:|:----------:|:------------------------------------:|:-------------------:|
|     Entero o integer     |     int    |    Representar <br>números enteros   |     1,2,-3,4,...    |
|         Flotante         |    float   |   Representar<br>números decimales   |   1.2,-3.1,4.5,...  |
| Cadenas de<br>caracteres |     str    | Representar<br>palabras y caracteres | "hola","PYTHON",... |
|         Booleano         |    bool    |    Representar <br>datos binarios    | True, False         |

La función *type()* permite determinar el tipo de variable.  
La función *print()* muestra la variable.

::: {#32176c06 .cell execution_count=1}
``` {.python .cell-code}
x = 20
print("El tipo de variable de X es:", type(x))

y = -5.1
print("El tipo de variable de Y es:", type(y))

w = "HOLA"
print("El tipo de variable de W es:", type(w))

v = True
print("El tipo de variable de V es:", type(v))
```

::: {.cell-output .cell-output-stdout}
```
El tipo de variable de X es: <class 'int'>
El tipo de variable de Y es: <class 'float'>
El tipo de variable de W es: <class 'str'>
El tipo de variable de V es: <class 'bool'>
```
:::
:::


# Operaciones con variables básicas

## Strings

::: {#22b0cedb .cell execution_count=2}
``` {.python .cell-code}
cadena_caracteres = " Diplomado en Analítica para la Banca "

#Tamaño de la cadena de caracteres
print(len(cadena_caracteres))

#Corte de variable
print(cadena_caracteres[0:10])
print(cadena_caracteres[20:30])

#Convertir la variable a mayúsculas
print(cadena_caracteres.upper())

#Convertir la variable a minúscula
print(cadena_caracteres.lower())

#Contar cuantas veces aparece una cadena de caracteres
print(cadena_caracteres.count("ca"))

#Reemplazar en una cadena, una letra con otra
print(cadena_caracteres.replace("a", "0"))

#Partir la cadena de caracteres cada vez que se encuentre un caracter
print(cadena_caracteres.split(" "))

#Concatenar dos cadenas de caracteres
cadena01 = "Pablo Eduardo"
cadena02 = "Caicedo Rodríguez"
print(cadena01+" "+cadena02)
```

::: {.cell-output .cell-output-stdout}
```
38
 Diplomado
ica para l
 DIPLOMADO EN ANALÍTICA PARA LA BANCA 
 diplomado en analítica para la banca 
2
 Diplom0do en An0lític0 p0r0 l0 B0nc0 
['', 'Diplomado', 'en', 'Analítica', 'para', 'la', 'Banca', '']
Pablo Eduardo Caicedo Rodríguez
```
:::
:::


# Conversión de Datos

::: {#c52baca4 .cell execution_count=3}
``` {.python .cell-code}
edad = input("Cuál es tu edad?")
print("Tipo de la variable edad: %s", type(edad))
edad = int(edad)
print("Tipo de la variable edad: %s", type(edad))
```
:::


# Operadores

Son símbolos que indican al compilador o intérprete que es necesario ejecutar alguna manipulación de los datos. Existen operadores aritméticos (p.e. suma, resta, multiplicación), de comparación (p.e. menor que, mayor o igual que), lógicos (and, or, not) y de pertenencia (in, not in).

::: {#02424491 .cell execution_count=4}
``` {.python .cell-code}
x = 5+6
print(x)
print('Hola'*3)
5 <= -3
c = 20
print(((c < 90) or (c > 60)))
nombre = "Pablo Eduardo"
print("a" in nombre)
```

::: {.cell-output .cell-output-stdout}
```
11
HolaHolaHola
True
True
```
:::
:::


# Estructura de datos

Las estructuras de datos son formas de organización de los datos que permiten leerlos, manipularlos y establecer relaciones entre ellos. Entre las formas más comunes tenemos **listas**, **diccionarios** y **tuplas**.

Las **listas** se tratan de colecciones de valores encerrados entre paréntesis cuadrados []. Son estructuras muy utilizadas en Python porque tienen mucha versatilidad. Al igual que los strings, tienen posiciones asignadas donde se puede verificar o incluso modificar su contenido con una gran cantidad de funciones disponibles. Las listas pueden tener distintos tipos de datos y pueden ser cambiadas en cualquier momento de la ejecución del programa.

Los **diccionarios** constituyen otra forma de organización de los datos donde existe una clave y un valor asociado. Para definirlos se usa el símbolo {} y para diferenciar entre clave y valor se usa el símbolo :. La mayoría de funciones utilizadas para modificar listas, también pueden ser utilizadas con diccionarios.

Las **tuplas** son otra forma de organizar los datos. Sin embargo, a diferencia de las listas y los diccionarios son inmutables, es decir, no se pueden modificar. Se definen entre paréntesis (). Su procesamiento es más rápido.

## Listas

::: {#5fa9e0fc .cell execution_count=5}
``` {.python .cell-code}
lista = [3, 2, 1, 0.5, "hora del cafe", "torta chilena", "pinto", "jugo"]
print(lista)
lista.append("empanadita")
print(lista)
"pinto" in lista
```

::: {.cell-output .cell-output-stdout}
```
[3, 2, 1, 0.5, 'hora del cafe', 'torta chilena', 'pinto', 'jugo']
[3, 2, 1, 0.5, 'hora del cafe', 'torta chilena', 'pinto', 'jugo', 'empanadita']
```
:::

::: {.cell-output .cell-output-display execution_count=4}
```
True
```
:::
:::


## Diccionarios

::: {#74c68dc7 .cell execution_count=6}
``` {.python .cell-code}
tel = {'Maria': 4098, 'Jorge': 4139}
print(tel)
print(tel["Maria"])
print(tel.keys())
print(tel.values)
'Maria' in tel
```

::: {.cell-output .cell-output-stdout}
```
{'Maria': 4098, 'Jorge': 4139}
4098
dict_keys(['Maria', 'Jorge'])
<built-in method values of dict object at 0x7fc13bff2a40>
```
:::

::: {.cell-output .cell-output-display execution_count=5}
```
True
```
:::
:::


## Tuplas

::: {#a1d7de7b .cell execution_count=7}
``` {.python .cell-code}
frutas = ('naranja', 'mango', 'sandia', 'banano', 'kiwi')
print(type(frutas))
frutas[1]
```

::: {.cell-output .cell-output-stdout}
```
<class 'tuple'>
```
:::

::: {.cell-output .cell-output-display execution_count=6}
```
'mango'
```
:::
:::


# Condicionales

Son bloques de código que dependiendo de una condición se ejecutan o no. Los test condicionales usan la palabra clave if que tienen como resultado un valor booleano de true o false.

Por supuesto, se puede complementar con operadores lógicos (“and” y “or”) para evaluar múltiples condiciones.

Un aspecto importante de la estructura de condicionales es la identación. Este término se refiere a un tipo de notación que delimita la estructura del programa estableciendo bloques de código. En sencillo, es la inclusión de algunos espacios (sangría) en la segunda parte de cada condicional.

::: {#23d2049f .cell execution_count=8}
``` {.python .cell-code}
if si la siguiente condición es verdad:
    realizar la siguiente acción
#2
if si la siguiente condición es verdad:
    realizar esta acción
else:
  sino realizar una la siguiente alternativa
#3
if si la siguiente condición es verdad:
  realizar esta acción

elif si esta otra condición es verdad:
  realizar esta acción alternativa
else:
  sino realizar esta otra acción alternativa
```
:::


::: {#680605b5 .cell execution_count=9}
``` {.python .cell-code}
c=8
if c>6:
    c-=3 # c = c-3
elif -2 < c < 4:
    c**=2 # c = c**2
else:
    c+=2 # c = c+2
c
```

::: {.cell-output .cell-output-display execution_count=7}
```
5
```
:::
:::


# Ciclos

Los ciclos son bloques de código que se ejecutan iterativamente. En Python se usan las funciones **while** y **for** para llevar a cabo los ciclos, los cuales se ejecutan hasta que la condición especificada se cumpla. Las funciones while y for normalmente van acompañadas de un iterador conocida como contador y que se designa con la letra *i**, aunque en realidad puede ser cualquier otra letra.

::: {#423f9279 .cell execution_count=10}
``` {.python .cell-code}
i = 1

while i<5:
    print(i)
    i+=1
```

::: {.cell-output .cell-output-stdout}
```
1
2
3
4
```
:::
:::


Dentro de while existen los argumentos **break** y **continue** que permiten detener el ciclo aún cuando la condición se cumple o detener la iteración actual y continuar con la siguiente.

::: {#bdf8aa98 .cell execution_count=11}
``` {.python .cell-code}
i = 5
while i<10:
    print(i)
    if i==9:
        break
    i+=1
```

::: {.cell-output .cell-output-stdout}
```
5
6
7
8
9
```
:::
:::


::: {#00f09683 .cell execution_count=12}
``` {.python .cell-code}
i = 5
while i<8:
    i+=1
    if i==7:
        continue
    print(i)
```

::: {.cell-output .cell-output-stdout}
```
6
8
```
:::
:::


Los ciclos (loops) que utilizan **for** son probablemente los más utilizados en Python y sirven para iterar sobre una secuencia (p.e. un string, estructura de datos, etc). La sintaxis se leería de la siguiente manera:

::: {#2220be87 .cell execution_count=13}
``` {.python .cell-code}
S="Buenos dias"

for i in S:
    print(i)
```

::: {.cell-output .cell-output-stdout}
```
B
u
e
n
o
s
 
d
i
a
s
```
:::
:::


También se puede iterar sobre números. Para esto se puede utilizar la función range(). Recordar que en Python siempre se empieza en cero.

::: {#605b952b .cell execution_count=14}
``` {.python .cell-code}
for num in range(10):
    print(num, num**2)
```

::: {.cell-output .cell-output-stdout}
```
0 0
1 1
2 4
3 9
4 16
5 25
6 36
7 49
8 64
9 81
```
:::
:::


# Funciones

Las funciones son las estructuras esenciales de código en los lenguajes de programación. Constituyen un grupo de instrucciones para resolver un problema muy concreto.

En Python, la definición de funciones se realiza mediante la instrucción **def** más un nombre descriptivo de la función, seguido de paréntesis y finalizando con dos puntos (:). El algoritmo que la compone, irá identado. Un parámetro es un valor que la función espera recibir a fin de ejecutar acciones específicas. Una función puede tener uno o más parámetros. La función es ejecutada hasta que sea invocada, es decir, llamada por su nombre, sea con **print()** o con **return()**.

::: {#368e496d .cell execution_count=15}
``` {.python .cell-code}
def Matarile(nombreviejo,nombrenuevo):
    print('Usted ya no se llama %s, ha elegido llamarse %s.' %(nombreviejo,nombrenuevo))

nombreviejo = input('Escriba su nombre: ')
nombrenuevo = input('Escriba cómo quiere llamarse: ')

Matarile(nombreviejo, nombrenuevo)
```
:::


# Librerías y Módulos

Una de las principales características de Python es que dispone de diferentes tipos de librerías o bibliotecas. En síntesis, una librería responde al conjunto de funcionalidades que permiten al usuario llevar a cabo nuevas tareas que antes no se podían realizar. Cada una de las librerías disponen de diferentes módulos que son los que le otorgan funciones específicas. Python posee una gran cantidad de librerías útiles en diferentes campos como la visualización, cienca de datos, cálculos numéricos, bioinformática o inteligencia artificial.

Algunas utilizadas en ciencia de datos:
- Numpy
- Pandas
- Matplotlib
- Seaborn

::: {#1c7d54ca .cell execution_count=16}
``` {.python .cell-code}
import numpy
import matplotlib.pyplot # se importa módulo pyplot dentro de matplotlib
import numpy.random as npr
```
:::


# Manipulación de datos

## Numpy

**NumPy** (Numerical Python), es una biblioteca de Python que da soporte para crear vectores y matrices grandes multidimensionales, junto con una gran colección de funciones matemáticas de alto nivel. La funcionalidad principal de **NumPy** es su estructura de datos ndarray (arreglos), para una matriz de n dimensiones, sobre las cuales se pueden realizar operaciones matemátias de manera eficiente.

Crearemos una lista usando código nativo de Python y lo convertiremos en una matriz unidimensional con la función np.array()

::: {#9101f53f .cell execution_count=17}
``` {.python .cell-code}
import numpy as np

list1 = [6,8,10,12]
array1 = np.array(list1)
print(array1)
```

::: {.cell-output .cell-output-stdout}
```
[ 6  8 10 12]
```
:::
:::


Los **ndarrays** son estructuras de datos genéricas para almacenar datos homogéneos. Son equivalentes a las matrices y los vectores en álgebra, por lo que también se les puede aplicar operaciones matemáticas. Notar que las operaciones matemáticas se pueden realizar en todos los valores en un ndarray a la vez.

::: {#3641a388 .cell execution_count=18}
``` {.python .cell-code}
print(array1 - 2)
print(array1 * array1, "\n\n")
```

::: {.cell-output .cell-output-stdout}
```
[ 4  6  8 10]
[ 36  64 100 144] 


```
:::
:::


Los arreglos se encierran entre **[]**, pero al imprimirlos no están separados por comas. Hay diferentes formas de crear arreglos con propiedades específicas, lo que les provee bastante flexibilidad.

::: {#4e111374 .cell execution_count=19}
``` {.python .cell-code}
# Crea una matriz con datos específicos
print(np.array([[1,2],[3,4]]),'\n')
# Crea una matriz con unos: tres filas y cuatro columnas
print(np.ones((3,4)),'\n')
# Crea una matriz con ceros: tres filas y cuatro columnas
print(np.zeros((3,4)),'\n')
# Crea una matriz con un dato específico: tres filas y cuatro columnas
print(np.full((3,4), 7.3),'\n')
# Crea un arreglo con datos seguidos: empieza en 10 termina en 30(sin incluir) con incrementos de 5.
print(np.arange(10,30,5),'\n')
# # Crea un arreglo con inicio y fin y una cantidad de datos: arreglo de 6 datos entre 0 y 5/3 .
print(np.linspace(0,5/3,6),'\n')
# Crea una matriz con datos aleatorios entre 0 y 1: dos filas y tres columnas
print(np.random.rand(2,3),'\n')
```

::: {.cell-output .cell-output-stdout}
```
[[1 2]
 [3 4]] 

[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]] 

[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]] 

[[7.3 7.3 7.3 7.3]
 [7.3 7.3 7.3 7.3]
 [7.3 7.3 7.3 7.3]] 

[10 15 20 25] 

[0.         0.33333333 0.66666667 1.         1.33333333 1.66666667] 

[[0.9629078  0.00511719 0.4519879 ]
 [0.86245333 0.45132958 0.63662918]] 

```
:::
:::


::: {#ccdde540 .cell execution_count=20}
``` {.python .cell-code}
arr1 = np.array([np.arange(0,5), np.arange(0,5)*5])
#Arreglo
print(arr1, "\n")
# Forma
print(arr1.shape, "\n")
# Tamaño
print(arr1.size, "\n")
# Número de Dimensiones
print(arr1.ndim, "\n")
# Transpuesta
print(arr1.T, "\n")
```

::: {.cell-output .cell-output-stdout}
```
[[ 0  1  2  3  4]
 [ 0  5 10 15 20]] 

(2, 5) 

10 

2 

[[ 0  0]
 [ 1  5]
 [ 2 10]
 [ 3 15]
 [ 4 20]] 

```
:::
:::


::: {#3f482f91 .cell execution_count=21}
``` {.python .cell-code}
arr = np.array([1,2,3,4,5,6,7])
# Porcionar
print(arr[1:3])# de 1 al 3 en índice
print(arr[4:])# de la posición 4 en adelante
print(arr[::2])# de uno por medio
```

::: {.cell-output .cell-output-stdout}
```
[2 3]
[5 6 7]
[1 3 5 7]
```
:::
:::


