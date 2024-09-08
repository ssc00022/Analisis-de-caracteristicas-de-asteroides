# ------------------------- PREPROCESAMIENTO -----------------------------------

# Reducción de atributos carentes de utilidad
asteroids_final <- eliminar_atributos(asteroids, c("full_name", "data_arc",
                                                   "n_obs_used", "epoch_mjd"))

# Reducción de atributos con muy pocos ejemplos
asteroids_final <- eliminar_atributos(asteroids_final, c("IR", "G", "GM", "extent"))
describe(asteroids_final)
summary(asteroids_final)

# Reducción del atributo rot_per
asteroids_final <- eliminar_atributos(asteroids_final, c("rot_per"))
describe(asteroids_final)
summary(asteroids_final)

# Selección de instancias con mejores mediciones
asteroids_final <- subset(asteroids_final, condition_code == 0)
asteroids_final <- eliminar_atributos(asteroids_final, c("condition_code"))

# Conversión a valores nulos de valores anómalos
for (col in colnames(asteroids_final)) {
  if (is.numeric(asteroids_final[[col]])) {
    asteroids_final[[col]] <- reemplazar_outliers_na(asteroids_final[[col]])
  }
}
