---
title: Algoritmos básicos de procesamiento de imágenes
---





Usando la imagen [fresas.png](../../recursos/imagenes/Presentaciones/PSIM/imagen_fresas.png) la cual fue tomada de la página [Geeks for geeks](https://www.geeksforgeeks.org/matlab-intensity-transformation-operations-on-images/) resuelva las siguientes cuestiones.

1. Aplique las siguientes transformaciones y describa el efecto de cada transformación:
   1. Transformación n-potencial con $1<n<2$
   2. Transformación n-potencial con $0.5<n<1$
   3. Transformación LOG (Logaritmo Natural)
   4. Transformación exponencial
   5. Describa en un diagrama de bloques el algoritmo necesario para realizar este tipo de transformaciones.
   6. Investigue y desarrolle el algoritmo de la transformación $\Gamma$. La información básica la puede encontrar [aqui](https://pablocaicedor.github.io/presentaciones/PSIM/Lect005_Imag_Proc_001.html#/basic-mathematic---pixel-intensity-2)

Recuerde: Si una imagen queda completamente blanca o completamente negra, es probable que sea por mal manejo de rangos de intensidad

2. Sea los siguientes kernels de convolución:
   1. $\begin{bmatrix}
1 & 1 & 1 \\
1 & -8 & 1 \\
1 & 1 & 1 
\end{bmatrix}$

   2. $\begin{bmatrix}
-1 & -1 & -1 \\
-1 & 8 & -1 \\
-1 & -1 & -1 
\end{bmatrix}$  

Explique las siguientes cuestiones:

   i) Investigue las formas de realizar la convolución con opencv.
   
   ii) Aplique cada uno de los kernels de convolución y compare los resultados.
   
   iii) Explique cuales son las respectivas resoluciónes de pixel de las imagenes resultantes así como su máximo y su mínimo.

3. Utilizando la imagen [radiografía](../../recursos/imagenes/Presentaciones/PSIM/female-chest-x-ray.jpg), aplique cada tipo de matriz afine presente en las [diapositivas de clase](https://pablocaicedor.github.io/presentaciones/PSIM/Lect005_Imag_Proc_001.html#/neighborhood-operations-5) 

4. Determinar proyecto final para PSIM.
   1. Determinar el problema a resolver.
   2. Determinar el objetivo a alcanzar
   3. Determinar el dataset
   
Recuerde, en su proyecto final,  **NO** podrá hacer uso de algoritmos de machine learning.


