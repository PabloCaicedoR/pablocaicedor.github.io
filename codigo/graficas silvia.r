library(ggplot2)


p <- ggplot(datos, aes(Año, Cantidad)) +
  geom_col(aes(fill = Tipo), position = position_dodge(width = 1)) +
  geom_vline(xintercept = 1:13 - 0.5, color = "gray90") +
  geom_hline(yintercept = 0:3 * 5, color = "gray90") +
  scale_fill_manual(values = c("deeppink", "chartreuse3","goldenrod1","darkorchid1")) +
  ggtitle("vvvv") +
  theme_bw() +
  theme(panel.border = element_blank(),
        axis.text.x = element_text(size = 14),
        axis.title.x = element_blank(),
        panel.grid.major = element_blank()) 

p + coord_polar()


ggplot(datos, aes(Año, Cantidad)) +
  geom_col(aes(fill = Tipo)) +  coord_polar(theta = "x")









ggplot(datos, aes(x=Año, y=Cantidad, fill = Tipo)) + geom_bar(stat = "identity") +  coord_polar()

