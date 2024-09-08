# ------------------- PREPROCESAMIENTO REGLAS ASOCIACION -----------------------

# Reducción de atributos por carencia de utilidad y pocas instancias
asteroids_ra <- eliminar_atributos(asteroids_final, c("BV", "UB", "spec_T", "spec_B", "neo", "pha", "class"))

# Selección de instancias sin valores nulos
asteroids_ra <- na.omit(asteroids_ra)

# Discretización con la regla de Sturges
for (col in colnames(asteroids_ra)) {
  n <- length(asteroids_ra[[col]])
  k <- ceiling(log2(n) + 1)
  if (is.numeric(asteroids_ra[[col]])) {
    asteroids_ra[[col]] <- cut(asteroids_ra[[col]], breaks = k)
  }
}

# Guardar archivo para reglas de asociación
write.csv(asteroids_ra, "asteroids_ra.csv", row.names = FALSE)
