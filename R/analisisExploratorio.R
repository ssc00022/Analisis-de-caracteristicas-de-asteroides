# ----------------------- ANALISIS EXPLORATORIO --------------------------------

# Carga de datos
asteroids <- read.csv("asteroids_initial.csv")

# Estructura interna
str(asteroids)

# Resumen del contenido
summary(asteroids)

# Descripción del contenido
describe(asteroids)

# Diagrama de frecuencia de atributos con más valores perdidos
diagrama_frecuencia(asteroids, c("BV", "UB", "IR", "G", "GM"),
                                     c("extent", "spec_B", "spec_T"))

# Diagramas de cajas de la mayoría de atributos numéricos
diagramas_cajas(asteroids, c("data_arc", "n_obs_used", "epoch_mjd", "GM", "IR",
                             "G", "BV", "UB"))

# Gráficas de correlación de la mayoría de atributos numéricos
grafica_correlacion(asteroids, c("data_arc", "n_obs_used", "epoch_mjd", "GM", "IR",
                                 "G", "BV", "UB"))

# Gráficas de sectores de neo y pha
prev <- par(mfrow = c(1, 2))
grafica_sectores(asteroids, "neo")
grafica_sectores(asteroids, "pha")
par(prev)

# Gráfica de barras horizontales de class y condition_code
grafica_barras(asteroids, "class")
grafica_barras(asteroids, "condition_code")

