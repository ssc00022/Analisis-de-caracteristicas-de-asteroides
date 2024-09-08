# ----------------------- PREPROCESAMIENTO PREDICCION --------------------------

# Reducción de atributos por carencia de utilidad y pocas instancias
asteroids_pred <- eliminar_atributos(asteroids_final, c("BV", "UB", "spec_T", "spec_B", "neo", "pha"))

# Gráfica de correlación de atributos numéricos restantes
grafica_correlacion(asteroids_pred, "")

# Gráfica de correlación sin atributos con mayor número de valores vacíos
grafica_correlacion(asteroids_pred, c("diameter", "albedo"))

# Gráfico de dispersión entre H y diameter
grafico_dispersion(asteroids_pred, "H", "diameter")

# Reducción de atributos por correlaciones
asteroids_pred <- eliminar_atributos(asteroids_pred, c("q", "ad", "per_y", "n", "moid"))

# Normalización min-max
for (col in colnames(asteroids_pred)) {
  if (is.numeric(asteroids_pred[[col]])) {
    asteroids_pred[[col]] <- min_max_norm(asteroids_pred[[col]])
  }
}

# Selección de instancias con valores no vacíos de diameter y albedo
asteroids_pred <- subset(asteroids_pred, !is.na(diameter) & !is.na(albedo))

# Diagramas de cajas de atributos restantes
diagramas_cajas(asteroids_pred, "")

# Imputación con KNN de atributos con mayor número de valores vacíos
datos_numericos <- asteroids_pred[, sapply(asteroids_pred, is.numeric)]
datos_imputados <- as.data.frame(seqKNNimp(datos_numericos))
asteroids_pred <- cbind(datos_imputados, class = asteroids_pred$class)

# Eliminar atributo class y guardar archivo para regresión
asteroids_regr <- eliminar_atributos(asteroids_pred, "class")
write.csv(asteroids_regr, "asteroids_regr.csv", row.names = FALSE)

# Tabla de frecuencias del atributo class
table(asteroids_pred$class)

# Eliminar clases con escasos ejemplos
asteroids_clas <- subset(asteroids_pred, class != 'AST' & class != 'CEN')

# Gráfica de barras horizontales de class
grafica_barras(asteroids_pred, "class")

# Submuestreo de la clase mayoritaria al doble de la segunda mayoritaria y
# sobremuestreo estratificado del resto de clases con la clase minoritaria con
# un mínimo de 500 instancias
asteroids_clas <- muestreo(asteroids_clas)
table(asteroids_clas$class)
grafica_barras(asteroids_clas, "class")

# Guardar archivo para clasificación
write.csv(asteroids_clas, "asteroids_clas.csv", row.names = FALSE)
