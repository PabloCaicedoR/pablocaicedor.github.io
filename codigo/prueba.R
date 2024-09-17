setwd(".")
# Instalar y cargar la librería ggplot2
library(ggplot2)

# Crear un dataframe con los datos
data <- read.csv("datossilvia.csv")

# Crear el gráfico circular de barras
ggplot(data, aes(x = factor(year), y = count, fill = category)) +
    geom_bar(stat = "identity", position = "dodge2") +
    coord_polar(start = 0) +
    theme_minimal() +
    labs(title = "Distribución de Artículos por Categoría", x = "Año", y = "Cantidad de Artículos") +
    scale_fill_manual(values = c(
        "Acceso equidad" = "#1f77b4",
        "Pedagogía curriculum" = "#ff7f0e",
        "Naturaleza de la ciencia" = "#2ca02c",
        "Identidad" = "#d62728"
    ))

ggplot(data, aes(x = factor(year), y = category, fill = factor(count))) +
    geom_tile()
