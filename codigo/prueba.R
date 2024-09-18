setwd(".")
# Instalar y cargar la librería ggplot2
library(ggplot2)
library("readxl")

# Crear un dataframe con los datos
data <- read_excel("./codigo/datos.xlsx", 1)

ggplot(data) +
    geom_tile(aes(y = factor(Cantidad), x = factor(Año))) +
    coord_radial() +
    facet_wrap(~Tipo)
