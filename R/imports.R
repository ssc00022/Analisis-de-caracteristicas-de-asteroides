# -------------------------- IMPORTACIONES -------------------------------------

# Instalar el paquete Hmisc si es preciso y cargar
install.packages("Hmisc")
library(Hmisc)

# Instalar el paquete ellipse y cargar
install.packages("ellipse")
library(ellipse)

# Instalar el paquete multiUS y cargar
install.packages("multiUS")
library(multiUS)

# Instalar el paquete dplyr y cargar
install.packages("dplyr")
library(dplyr)

# Instalar el paquete smotefamily y cargar
install.packages("smotefamily")
library(smotefamily)

# ---------------------------- FUNCIONES ---------------------------------------

# Función para elaborar diagramas de frecuencia
diagrama_frecuencia <- function(datos, atributos_numericos, atributos_nominales) {
  # Extraer datos numéricos y nominales del data frame
  datos_numericos <- datos[, atributos_numericos, drop = FALSE]
  datos_nominales <- datos[, atributos_nominales, drop = FALSE]
  
  # Calcular la cantidad de valores no nulos para atributos numéricos
  valores_numericos <- colSums(!is.na(datos_numericos))
  
  # Calcular la cantidad de valores no nulos para atributos nominales
  valores_nominales <- colSums(datos_nominales != "")
  
  # Combinar los valores en un solo vector
  valores_visualizar <- c(valores_numericos, valores_nominales)
  
  # Crear la gráfica de barras
  contador_valores_perdidos <- barplot(valores_visualizar, 
                                       main =  "Diagrama de frecuencia",
                                       names.arg = c(atributos_numericos, atributos_nominales),
                                       col = "skyblue")
  
  # Añadir etiquetas de frecuencia en las barras
  text(x = contador_valores_perdidos, y = valores_visualizar,
       labels = valores_visualizar,
       pos = 3, cex = 1.2, col = "black")
}

# Función para elaborar diagramas de cajas
diagramas_cajas <- function(datos, atributos_excluidos) {
  # Seleccionar solo las columnas numéricas del dataframe 'datos'
  numerical_columns <- sapply(datos, is.numeric)
  asteroids_numerical <- datos[, numerical_columns]
  
  # Inicializar un contador para la disposición de gráficos
  n <- 0
  
  # Excluir las columnas especificadas
  asteroids_numerical <- asteroids_numerical[, !(names(asteroids_numerical) %in% atributos_excluidos)]
  
  # Iterar sobre las columnas numéricas restantes
  for (col in colnames(asteroids_numerical)) {
    # Configurar la disposición de gráficos en filas de 3
    if (n %% 6 == 0) {
      prev <- par(mfrow = c(2, 3))
    }
    
    # Incrementar el contador
    n <- n + 1
    
    # Obtener los datos de la columna actual
    col_datos <- asteroids_numerical[[col]]
    
    # Contar el número de outliers en la columna actual
    outliers <- identificar_outliers(col_datos)
    n_outliers <- sum(outliers)
    
    # Contar el número de valores nulos en la columna actual
    n_na <- sum(is.na(col_datos))
    
    # Obtener el número total de filas en los datos numéricos
    n_filas <- nrow(asteroids_numerical)
    
    # Crear el gráfico de caja para la columna actual
    boxplot(asteroids_numerical[[col]],
            main = paste("Diagrama de cajas de", col, "\n", 
                         n_outliers, "outliers (", round(n_outliers * 100 / n_filas, 2), "%)\n", 
                         n_na, "valores nulos (", round(n_na * 100 / n_filas, 2), "%)")
    )
  }
}

# Función para elaborar gráficas de correlación
grafica_correlacion <- function(datos, atributos_excluidos) {
  # Declarar gráfica múltiple
  prev <- par(mfrow = c(1, 2))
  
  # Filtrar solo las columnas numéricas
  numerical_columns <- sapply(datos, is.numeric)
  asteroids_numerical <- datos[, numerical_columns]
  
  # Quitar columnas específicas
  asteroids_numerical <- asteroids_numerical[, !(names(asteroids_numerical) %in% atributos_excluidos)]
  
  # Crear conjunto con valores NA eliminados
  asteroids_numerical_na <- na.omit(asteroids_numerical)
  
  # Calcular las matrices de correlación
  cor_matrix <- cor(asteroids_numerical)
  cor_matrix_na <- cor(asteroids_numerical_na)
  
  # Calcular porcentaje del conjunto con valores NA eliminados
  porcentaje <- round(nrow(asteroids_numerical_na) / nrow(asteroids_numerical) * 100, 1)
  
  # Crear las gráficas de correlación
  titulo <- paste("Conjunto de datos completo")
  plotcorr(cor_matrix, col = heat.colors(10), main = titulo)
  titulo <- paste(nrow(asteroids_numerical_na), " instancias (", porcentaje, "%)")
  plotcorr(cor_matrix_na, col = heat.colors(10), main = titulo)
  par(prev)
}

# Función para elaborar gráficas de sectores
grafica_sectores <- function(datos, col) {
  # Filtrar datos eliminando valores NA y vacíos
  datos_filtrados <- subset(datos, !is.na(datos[[col]]) & datos[[col]] != "")
  
  # Contar las ocurrencias del atributo col
  counts <- table(datos_filtrados[[col]])
  
  # Calcular los porcentajes
  percentages <- round(counts / sum(counts) * 100, 1)
  
  # Crear etiquetas con los porcentajes
  labels <- paste(percentages, "%", sep=" ")
  
  # Establecer esquema de colores
  colores <- topo.colors(length(labels))
  
  # Crear el gráfico de sectores con las etiquetas
  pie(counts, 
      labels = labels, 
      main = paste("Gráfica de sectores de", col, "\n", nrow(datos_filtrados), "instancias"), 
      col = colores
  )
  
  # Incluir leyenda de la gráfica
  legend("bottom",
         inset = -0.2,
         title = "Categorías", 
         names(counts), 
         cex = 1,
         fill = colores,
         ncol = 2,
         xpd = TRUE
  )
}

# Función para elaborar gráficas de barras
grafica_barras <- function(datos, atributo) {
  # Contar la frecuencia de cada categoría
  frecuencia <- table(datos[[atributo]])
  
  # Crear una paleta de colores para las barras
  colores <- topo.colors(length(frecuencia))
  
  # Crear la gráfica de barras horizontales
  par(mar = c(5, 8, 4, 2))
  barplot(frecuencia, 
          main = paste("Categorías de", atributo),
          names.arg = names(frecuencia),
          horiz = TRUE,
          col = colores,
          las = 1,
          cex.names = 1
  )
}

# Función para eliminar atributos del dataset
eliminar_atributos <- function(datos, atributos) {
  # Excluir las columnas especificadas
  datos_final <- datos[, !(names(datos) %in% atributos)]
  
  # Devolver dataset con atributos eliminados
  return(datos_final)
}

# Función para reemplazar valores anómalos en NA
reemplazar_outliers_na <- function(col) {
  # Identificar valores anómalos
  outliers <- boxplot.stats(col)$out
  
  # Reemplazar valores anómalos por NA
  col[col %in% outliers] <- NA
  return(col)
}

# Función para elaborar gráficos de dispersión
grafico_dispersion <- function(datos, col_x, col_y) {
  # Filtrar datos eliminando valores NA en las dos columnas especificadas
  datos_filtrados <- subset(datos, !is.na(datos[[col_x]]) & !is.na(datos[[col_y]]))
  
  # Calcular la correlación de Pearson entre las dos columnas
  correlacion <- cor(datos_filtrados[[col_x]], datos_filtrados[[col_y]])
  
  # Crear el título con la correlación incluida
  titulo <- paste("Gráfico de dispersión entre", col_x, "y", col_y, "\n", 
                  nrow(datos_filtrados), " instancias con r = ", round(correlacion, 4))
  
  # Crear el gráfico de dispersión
  plot(datos_filtrados[[col_x]],
       datos_filtrados[[col_y]],
       main = titulo,
       xlab = col_x,
       ylab = col_y)
}

# Función para normalizar valores numéricos en formato min-max
min_max_norm <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}

# Función para balancear los datos
muestreo <- function(datos) {
  # Identificar las clases mayoritaria y segunda mayoritaria
  conteo_clases <- datos %>% count(class)
  orden_clases <- conteo_clases %>% arrange(desc(n))
  primera_clase <- orden_clases$class[1]
  segunda_clase <- orden_clases$class[2]
  
  # Submuestreo de la clase mayoritaria al doble de la segunda clase mayoritaria
  # sin reemplazamiento
  muestras_primera_clase <- datos %>% filter(class == primera_clase)
  muestras_segunda_clase <- datos %>% filter(class == segunda_clase)
  n_instancias <- nrow(muestras_segunda_clase) * 2
  submuestreo <- muestras_primera_clase %>% 
    sample_n(size = n_instancias, replace = FALSE)
  
  # Combinar los datos submuestreados con los datos restantes
  datos_balanceados <- bind_rows(submuestreo,
                                 datos %>% filter(class != primera_clase))
  
  # Determinar coeficiente de sobremuestreo para que la clase minoritaria alcance
  # alrededor de 500 instancias y el resto aumenten proporcionalmente
  ultima_clase <- orden_clases$class[nrow(orden_clases)]
  muestras_ultima_clase <- datos %>% filter(class == ultima_clase)
  coeficiente <- round(500 / nrow(muestras_ultima_clase), 1)
  
  # Aplicar sobremuestreo tantas veces como clases existan sin contar las dos primeras
  for (i in 1:(nrow(conteo_clases) - 2)) {
    # Separar las características (X) y la variable objetivo (y)
    X <- datos_balanceados[, -which(names(datos_balanceados) == "class")]
    y <- datos_balanceados$class
    
    # Aplicar SMOTE del paquete smotefamily según el coeficiente calculado antes
    smote_output <- SMOTE(X = X, target = y, K = 5, dup_size = coeficiente)
    
    # Combinar los datos sintéticos generados con los datos originales
    datos_balanceados <- smote_output$data
    names(smote_data)[ncol(datos_balanceados)] <- "class"  # Renombrar la columna de class
  }
  
  # Devolver datos muestreados
  return(datos_balanceados)
}

