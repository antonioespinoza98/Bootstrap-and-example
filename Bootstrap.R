# Bootstrap ---------------------------------------------------------------
# Autor: Marco Espinoza
# Comisión Económica para América Latina y el Caribe
# Julio - 2024

# 1. El sample consiste del error con el replace TRUE para todo el censo para 
# el error. 

# 2. De ahi con la prediccion y el vector de efectos aleatorios, 
# estimamos F(X1) + Z1 + E1 para el censo. 

# 3. Media para Dominio para PB e {1,... N} 

# 4. Luego se hace una esperanza para todas las poblaciones Bootstrap

# 5. Se hace lo mismo para la varianza


# Del censo vamos a necesitar:
# 1.1 F(X) que es nuestra funcion estimada
# 1.2 El dominio


# Preparación -------------------------------------------------------------
library(srvyr)
library(survey)
library(data.table)
library(ggplot2)
library(dplyr)
library(tidyr)
library(XboostingMM)

data <- readRDS("data/encuesta_df_agg.rds") |>
  mutate_if(is.character, as.factor)

censo <- readRDS("data/cens0.rds") |>
  select(dam) |>
  mutate(dam = recode(dam,
                      "01" = "1",
                      "02" = "2",
                      "03" = "3",
                      "04" = "4",
                      "05" = "5",
                      "06" = "6"
                      ))
data <- as.data.frame(data)
# Leemos la predicción
f <- readRDS("output/prediction.rds")
length(f)

# pegamos la predicción al censo
censo$f <- f

# 2. Efectos aleatorios
fit <- readRDS("output/fit.rds")
randomEffects <- fit$raneffs

# 3. Errores
errorTerms <- fit$errorTerms

# Calculamos el population B1

dominio <- sort(unique(censo$dam))
len_dom <- length(censo$dam)

# Tamaño de la base
ext <- dim(censo)[1]


# bias corrected term -----------------------------------------------------

# De acuerdo con Krennmair, Schmid (2022). Como estimador de la varianza
# residual del modelo, \sigma^2_e esta positivamente sesgado, debido a que
# incluye incertidumbre de la estimacion de la funcion \hat{f}()
# Es por esta razon, que los autores proponen una correccion de este sesgo.

# 1. Utlizar las predicciones del modelo final despues de converger

last_pred <- fit$boosting_ensemble$fhat

# 2. Generar B Bootstrap samples
# 2.1 y*(b) = f(x) + e*(b) donde e*(b) son los valores remuestreados
# con reemplazo de los residuales marginales centrados 
y <- data$ingreso

residual_cent <- y - last_pred

# Ahora deberiamos sacar una muestra con reemplazo de los residuales
# centrados para B bootstrap 

error_samp <- matrix(0, nrow = 681, ncol = 100)

for(i in 1:100){
  error_samp[, i] <- sample(residual_cent, size = 681, replace = TRUE)
}

# Hacemos una estimacion de y_b

y_B <- matrix(0, nrow = 681, ncol = 100)

for(i in 1:100) {
  y_B[, i] <- last_pred + error_samp[, i]
}

colnames(y_B) <- paste0("y_b", 1:100)

sqrterror <- function(preds, dtrain){
  labels <- xgboost::getinfo(dtrain, "label")
  grad <- preds - labels
  hess <- rep(1, length(grad))
  return(list(grad = grad, hess = hess))
}

formula <- y_b ~ area + sexo + edad + F182013_stable_lights  + X2016_crops.coverfraction + X2016_urban.coverfraction + X2016_gHM + accessibility + accessibility_walking_only + area1 + sexo2 + edad2 + edad3 + edad4 + edad5 + anoest2 + anoest3 + anoest4 + discapacidad1 + etnia1 + tiene_alcantarillado + tiene_electricidad + tiene_acueducto + tiene_gas + eliminar_basura + tiene_internet + piso_tierra + material_paredes + material_techo + rezago_escolar + alfabeta + hacinamiento + tasa_desocupacion

f_boot <- matrix(0, nrow = 681, ncol = 100)

for(i in 1:100) {
  # Primero debemos pegarle la variable a la base de datos,
  # tomando en cuenta que la base debe reiniciarse despues de cada
  # iteracion para evitar un traslape.
  temp_data <- data
  temp_data$y_b <- y_B[, i]
  
  
  # Ahora debemos: estimar la funcion para cada b bootstrap, con y_b(b)
  #  como variable dependiente.
  
  modelo <- xboosting(
    formula = formula,
    data = temp_data,
    loss = sqrterror,
    n.trees = 1,
    shrinkage = 0.6,
    interaction.depth = 9,
    minsplit = 1,
    lambda = 1,
    alpha = 4,
    subsample = 0.5
  )
  
  # extraemos la prediccion 
  
  f_boot[,i] <- modelo$fhat
  
}

# Estimamos el termino de correccion k(f)
k <- 1/100 * sum( (last_pred - f_boot)^2   )

saveRDS(k, "fact_correction.rds")


# bias-corrected estimator for the residual variance ----------------------

k <- readRDS("fact_correction.rds")

sigma_corrected <- fit$errorVar - k


# Bootstrap de la incertidumbre -------------------------------------------

# A lo planteado por Krennmair, Schmid (2022)

# 1. Se calcula el vector de residuales marginales como 
# \hat{e}_i = y_i - \hat{f}(X_i)

f <- readRDS("prediction_on_training.rds")

resid_marg <- y - f

data$resid_marg <- resid_marg


# 2. residuales nivel-2

r_bar_by_dam <- data |>
  group_by(dam) |>
  summarise(
   r_bar = 1/681 * sum(resid_marg) 
  )

r_bar <- r_bar_by_dam$r_bar

# 3. Residuales marginales nivel-1

r_hat <- resid_marg - 1*r_bar

# Centrar y escalar por los factores de corrección



# Ciclo -------------------------------------------------------------------
count <- 1
limit <- 100
ext <- dim(censo)[1]
PBS <- matrix(0, nrow = ext, ncol = limit)
result <-  vector(mode = "numeric", length = ext)

while (count <= limit) {
  # sample del error
  errorPB <- sample(errorTerms, ext, replace = TRUE)
  
  for (dom in dominio) {
    # Primero verificamos que en el ciclo identificamos el dominio en el censo
    # Del cual deberia devolver las 1437686 observaciones con los dominios
    # identificados.
    
    # Hay que obtener el vector F filtrado para cada dominio
    indices <- which(censo$dam == dom)
    
    random_effect <- randomEffects[as.numeric(dom)]
    
    result[indices] <- censo$f[indices] + random_effect + errorPB[indices]
  }
  
  PBS[, count] <- result
  count <- count + 1
  
}

colnames(PBS) <- paste0("PB", 1:limit)

PBS <- as.data.frame(PBS)

PBS$dam <- censo$dam


# Cálculo de medias para cada dominio -------------------------------------
PBS_long <- PBS |>
  pivot_longer(cols = starts_with("PB"),
               names_to = "PB",
               values_to = "value")

mean_df <- PBS_long |>
  group_by(dam, PB) |>
  summarise(media = mean(value, na.rm = TRUE)) |>
  pivot_wider(names_from = PB, values_from = media)

# Cálculo de medias para todas las PB -------------------------------------

medias <- numeric(6)
varianzas <- numeric(6)

for (i in 1:6) {
  medias[i] <- mean(as.numeric(mean_df[i, -1]), na.rm = TRUE)
  varianzas[i] <- var(as.numeric(mean_df[i, -1]), na.rm = TRUE)
}

resultado <- matrix(c(medias, varianzas), nrow = 6, ncol = 2,
                    dimnames = list(
                      "Dam" = c("01", "02",
                                "03", "04",
                                "05", "06"),
                      "Estimación" = c("value","Varianza")
                    ))

resultado <- as_tibble(resultado)
resultado$dam <- c("01","02","03","04","05","06")

data <- readRDS("val_set.rds") |>
  mutate_if(is.character, as.factor) |>
  select(dam, fep, dam)

diseno <- data  %>% 
  as_survey_design(weights = fep)

media_est_srvyr <- svyby(
  ~ingreso,
  by = ~dam,
  design = diseno,
  svymean
)

media_est_srvyr$dam <- c("01","02","03","04","05","06")


mse <- numeric(6)

for (i in 1:6) {
  dam_value <- media_est_srvyr$dam[i]
  observed_mean <- media_est_srvyr$ingreso[i]
  
  bootstrap_means <- as.numeric(mean_df[[which(mean_df$dam == dam_value)]])
  mse[i] <- mean((bootstrap_means - observed_mean)^2, na.rm = TRUE)
}

mse_result <- tibble(
  dam = media_est_srvyr$dam,
  MSE = mse
)


# Validación --------------------------------------------------------------

confint_int <- resultado |>
  mutate(
    SE = sqrt(Varianza), # Standard Error
    min_inter = value - 1.96 * SE, # Lower bound of the 95% CI
    max_inter = value + 1.96 * SE  # Upper bound of the 95% CI
  )

# Calcular los intervalos de confianza
IC_df <- mean_df |>
  pivot_longer(cols = starts_with("PB"),
               names_to = "PB",
               values_to = "value") |>
  group_by(dam) |>
  summarise(
    lower = quantile(value, probs = 0.025, na.rm = TRUE),
    upper = quantile(value, probs = 0.975, na.rm = TRUE)
  )


IC_df$dam <- c("01","02","03","04","05","06")
# Mostrar los intervalos de confianza junto con las medias y varianzas
final <- resultado |>
  left_join(IC_df, by = "dam")


confint_int <- data.table(confint_int, keep.rownames = TRUE)
colnames(confint_int) <- c("dam","min_inter", "max_inter")
confint_int$dam <- c("01","02","03","04","05","06")
confint_int$value <- media_est_srvyr$value

mapping <- rbind(media_est_srvyr[,-3], resultado[,-2])
mapping$dam <- rep(c("01","02","03","04","05","06"), 2)
estim <- c(rep("Directo", 6), rep("XGBoost MM", 6))

mapping <- cbind(mapping, estim)
options(scipen = 999)
mapping |>
  ggplot(aes(x = dam, y = value)) + geom_point(aes(color = estim), size = 2, position = "jitter") +
  scale_color_manual(values = c("red", "green")) +
  ylim(150000,450000) +
  labs(x = "Región de planificación", y = "Ingreso", col = "Modelo") +
  geom_errorbar(data = confint_int, aes(x = dam, ymin = min_inter, ymax = max_inter)) +
  theme_minimal()

final |>
  ggplot(aes(x = dam, y = value)) + geom_point(col = "green") +
  labs(x = "Región de planificación económica", y = "Ingreso") +
  # ylim(150000,450000) +
  geom_errorbar(data = final, aes(x = dam, ymin = AdjustedLower,
                                  ymax = AdjustedUpper)) +
  theme_minimal()



