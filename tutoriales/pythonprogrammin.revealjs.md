---
title: "Python programming"
description: "A small tutorial in python in slides"
subtitle: "Ingeniería Biomédica"
lang: es
author: "Ph.D. Pablo Eduardo Caicedo Rodríguez"
date: "2024-08-12"
image: "../recursos/imagenes/tutoriales/python_programming/novice_python_programmer.jpeg"
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
      - ../recursos/estilos/styles_pres.scss
    slide-number: true
    preview-links: auto
    logo: ../recursos/imagenes/generales/Escuela_Rosario_logo.png
    css: ../recursos/estilos/styles_pres.scss
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




# Python Programming

## Python

* Python is a high-level, interpreted, multi-paradigm, and general-purpose programming language. 
* Its design philosophy emphasizes code readability. 
* It is one of the most popular programming languages in use today, and is used for a wide range of applications, including web development, data science, machine learning, and artificial intelligence.


## Advantages
* Python is a multi-paradigm programming language, meaning it can be used for different types of programming, such as object-oriented programming, imperative programming, and functional programming.
* Python is an interpreted programming language, meaning it does not need to be compiled before being executed. This makes Python very fast to develop and debug.
* Python is a highly portable programming language, meaning it can be run on different platforms, such as Windows, Mac OS X, and Linux. It can also be run in the cloud.
Python has a large community of users and developers, meaning there are many resources available to learn and use Python.

## Disadvantages
* Python can be a bit slower than compiled languages, such as C or C++.
* Python has a slightly more complex syntax than some other programming languages, such as Java or JavaScript.
* Python may not be the best programming language for certain types of applications, such as game applications or high-performance applications.

## Basic Syntax

* Indentation is used to denote block-level structure
* Variables are assigned using the `=` operator
* Print output using the `print()` function
* Comments: `#` for single-line, `'''` or `"""` for multi-line

## Data Types

* Integers: `int` (e.g., `1`, `2`, `3`)
* Floats: `float` (e.g., `3.14`, `-0.5`)
* Strings: `str` (e.g., `'hello'`, `"hello"`)
* Boolean: `bool` (e.g., `True`, `False`)
* List: `list` (e.g., `[1, 2, 3]`, `['a', 'b', 'c']`)
* Dictionary: `dict` (e.g., `{'name': 'John', 'age': 30}`)

## Control Structures

* Conditional statements:
	+ `if` statements: `if` condition `:` code
	+ `elif` statements: `elif` condition `:` code
	+ `else` statements: `else` `:` code
* Loops:
	+ `for` loops: `for` variable `in` iterable `:` code
	+ `while` loops: `while` condition `:` code

## Functions

* Reusable blocks of code
* Take arguments and return values
* Can be used to:
	+ Organize code
	+ Reduce repetition
	+ Encapsulate complex logic
* Function definition: `def` function_name `(arguments)` `:` code

##  Modules

* Pre-written code libraries
* Imported using the `import` statement
* Examples:
	+ `math`: mathematical functions (e.g., `sin()`, `cos()`)
	+ `random`: random number generation (e.g., `randint()`, `uniform()`)
	+ `time`: time-related functions (e.g., `time()`, `sleep()`)


## Exception Handling

* Try-except blocks:
	+ `try` block: code that might raise an exception
	+ `except` block: code to handle the exception
* Catching specific exceptions:
	+ `except` ValueError `:` code
	+ `except` TypeError `:` code
* Raising exceptions:
	+ `raise` ValueError(`message`)

## Object-Oriented Programming

* Classes:
	+ Define custom data types
	+ Encapsulate data and behavior
* Objects:
	+ Instances of classes
	+ Have attributes (data) and methods (behavior)
* Inheritance:
	+ Create new classes based on existing ones
	+ Inherit attributes and methods

## Advanced Topics

* Decorators:
	+ Modify function behavior
	+ Use `@` symbol to apply
* Generators:
	+ Special type of iterable
	+ Use `yield` statement to define
* Lambda functions:
	+ Small, anonymous functions
	+ Use `lambda` keyword to define

## Decorators




::: {.cell}

```{.python .cell-code}
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")

    return wrapper


@my_decorator
def say_hello():
    print("Hello!")


say_hello()
```

::: {.cell-output .cell-output-stdout}

```
Something is happening before the function is called.
Hello!
Something is happening after the function is called.
```


:::
:::




## Generators




::: {.cell}

```{.python .cell-code}
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

gen = infinite_sequence()
print(next(gen))
```

::: {.cell-output .cell-output-stdout}

```
0
```


:::

```{.python .cell-code}
print(next(gen))
```

::: {.cell-output .cell-output-stdout}

```
1
```


:::

```{.python .cell-code}
print(next(gen))
```

::: {.cell-output .cell-output-stdout}

```
2
```


:::
:::




## Lambda Function




::: {.cell}

```{.python .cell-code}
rectangle_area_calculation = lambda base, height: base * height
print(rectangle_area_calculation(4, 6))  # Salida: 24
```

::: {.cell-output .cell-output-stdout}

```
24
```


:::
:::




## Conclusion

* Python is a powerful and versatile language
* Continuously learning and practicing will help you master it
* Explore advanced topics and libraries to become proficient