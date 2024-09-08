# ----------------------- PREPROCESAMIENTO CLUSTERING --------------------------

# Reducción de atributos por carencia de utilidad y pocas instancias
asteroids_clus <- eliminar_atributos(asteroids_final, c("spec_T", "neo", "pha", "class"))

# Normalización min-max
for (col in colnames(asteroids_clus)) {
  if (is.numeric(asteroids_clus[[col]])) {
    asteroids_clus[[col]] <- min_max_norm(asteroids_clus[[col]])
  }
}

# Selección de instancias con valores no vacíos de spec_B
asteroids_clus <- subset(asteroids_clus, spec_B != "")

# Diagrama de frecuencias
diagrama_frecuencia(asteroids_clus, c("H", "diameter", "BV", "UB"), c("spec_B"))

# Eliminación de atributo H
asteroids_clus <- eliminar_atributos(asteroids_clus, c("H"))
                                     
# Imputación con KNN de atributos con valores nulos
datos_numericos <- asteroids_clus[, sapply(asteroids_clus, is.numeric)]
datos_imputados <- as.data.frame(seqKNNimp(datos_numericos))
asteroids_clus <- cbind(datos_imputados, spec_B = asteroids_clus$spec_B)

# Frecuencia de categorías de spec_B
describe(asteroids_clus$spec_B)
table(asteroids_clus$spec_B)

# Eliminar instancias con valores ruidosos o poco precisos
asteroids_clus <- subset(asteroids_clus, !grepl("U", spec_B) &  !grepl(":", spec_B) & !grepl("\\(IV\\)", spec_B))

# Crear el vector de mapeo
mapeo <- c(
  'Cb' = 'C', 'Cg' = 'C', 'Cgh' = 'C', 'Ch' = 'C', 'B' = 'C',
  'Ld' = 'L',
  'Sa' = 'S', 'Sk' = 'S', 'Sl' = 'S', 'Sq' = 'S', 'Sr' = 'S',
  'Xc' = 'X', 'Xe' = 'X', 'Xk' = 'X'
)

# Aplicar el mapeo para transformar las subclases a superclases
asteroids_clus <- asteroids_clus %>%
  mutate(spec_B = recode(spec_B, !!!mapeo))
describe(asteroids_clus$spec_B)
table(asteroids_clus$spec_B)

# Guardar archivo para clustering
write.csv(asteroids_clus, "asteroids_clus.csv", row.names = FALSE)
