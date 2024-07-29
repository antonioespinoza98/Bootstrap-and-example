################################################################################
# Illustrative Data Example                                                 
################################################################################

#==============================================================================
# 0. Preparation
#==============================================================================

#- load packages and import data:
library(lme4)
library(stan4bart)
library(dplyr)
library(XboostingMM)

set.seed(1998)

data <- readRDS("encuesta_df_agg.rds") |>
  mutate_if(is.character, as.factor)

censo <- readRDS("cens0.rds") |>
  mutate(
  region = recode(dam,
                  "01" = "Central",
                  "02" = "Chorotega",
                  "03" = "Pacífico Central",
                  "04" = "Brunca",
                  "05" = "Huetar Caribe",
                  "06" = "Huetar Norte")) %>%
  mutate(region = as.factor(region))


data <- as.data.frame(data)
censo <- as.data.frame(censo)

dfsTrain <- data
dfsTest1 <- censo

#- set parameters 
conv_memboost = 0.001; maxIter_memboost = 1000; verbose_memboost = F;minIter_memboost = 0

#- specify model
formula <- ingreso ~ area + sexo + edad + F182013_stable_lights  + X2016_crops.coverfraction + X2016_urban.coverfraction + X2016_gHM + accessibility + accessibility_walking_only + area1 + sexo2 + edad2 + edad3 + edad4 + edad5 + anoest2 + anoest3 + anoest4 + discapacidad1 + etnia1 + tiene_alcantarillado + tiene_electricidad + tiene_acueducto + tiene_gas + eliminar_basura + tiene_internet + piso_tierra + material_paredes + material_techo + rezago_escolar + alfabeta + hacinamiento + tasa_desocupacion
random <- ~ 1|dam

modelLMM <- ingreso ~ (1 | dam) + (1|area) + (1|sexo) + (1|edad) + (1|discapacidad1) + (1|etnia1) + F182013_stable_lights  + X2016_crops.coverfraction + X2016_urban.coverfraction + X2016_gHM + accessibility + accessibility_walking_only + area1 + sexo2 + edad2 + edad3 + edad4 + edad5 + anoest2 + anoest3 + anoest4 + tiene_alcantarillado + tiene_electricidad + tiene_acueducto + tiene_gas + eliminar_basura + tiene_internet + piso_tierra + material_paredes + material_techo + rezago_escolar + alfabeta + hacinamiento + tasa_desocupacion

#- estimate the ICC in training data
fitICCLMM <- lme4::lmer(ingreso ~ (1 | dam), data = dfsTrain, REML = FALSE)
summary(fitICCLMM)
dfV <- lme4::VarCorr(fitICCLMM)
ICC <- dfV$dam[1] / ( dfV$dam[1] + sigma(fitICCLMM)^2 )
ICC 
#==============================================================================
# 1. Model estimation
#==============================================================================  

#-----------------------------------
#- some preparations
#-----------------------------------
result <- NULL
PredNames <- attr( stats::terms( formula ), "term.labels" )
OutcomeName <-  all.vars(formula)[1]
XTrain <- as.data.frame(dfsTrain[, PredNames, drop = FALSE]) 
XTest1 <- as.data.frame(dfsTest1[, PredNames, drop = FALSE])
YTrain <- as.matrix(dfsTrain[, OutcomeName, drop = FALSE]) 


#-------------------------------------------------------------------------------
#- MEM Boosting 
#-------------------------------------------------------------------------------
s_BoostMERT_L2 <- Sys.time()
# ================================================================
# CUSTOM OBJECTIVE FUNCTION
# ================================================================

sqrterror <- function(preds, dtrain){
  labels <- xgboost::getinfo(dtrain, "label")
  grad <- preds - labels
  hess <- rep(1, length(grad))
  return(list(grad = grad, hess = hess))
}

fitBoostMERT_L2 <- boost_mem(
  formula,
  data = dfsTrain,
  random = random,
  shrinkage = 0.6,
  interaction.depth = 9,
  n.trees = 100,
  loss = sqrterror,
  minsplit = 1,
  subsample = 0.5,
  lambda = 1,
  alpha = 4,
  verbose_memboost = verbose_memboost,
  minIter_memboost = minIter_memboost,
  maxIter_memboost = maxIter_memboost
)

e_BoostMERT_L2 <- Sys.time()
time_BoostMERT_L2 <- difftime(e_BoostMERT_L2, s_BoostMERT_L2, units = "secs")[[1]]

#- Calculate predicted values
# tmpdfsTest1 <- merge(dfsTest1, data.frame(dam = unique(dfsTrain$dam), raneffs = fitBoostMERT_L2$raneffs))

fhat_Test1 <- XboostingMM:::predict.xgb(fitBoostMERT_L2$boosting_ensemble,
                          newdata = dfsTest1,
                          n.trees = 100)

saveRDS(fhat_Test1, file = "prediction.rds")
saveRDS(fitBoostMERT_L2, file = "fit.rds")



# perform prediction on training set --------------------------------------

pred_training <- XboostingMM:::predict.xgb(fitBoostMERT_L2$boosting_ensemble,
                          newdata = dfsTrain,
                          n.trees = 100)


saveRDS(pred_training, "prediction_on_training.rds")

Yhat_Test1 <-  readRDS("C:/Users/marqu/OneDrive/Documents/osfstorage-archive - Copy/Functions for (MEM) gradient tree boosting/prediction.rds")


val = validation(test = dfsTest1, prediction = prediction, validation_set = val_set,
           weights = "fep", label = "ingreso", region = "dam", model = 
           )
#-------------------------------------------------------------------------------
#- Bayesian Additive Regression Tree with random intercept
#-------------------------------------------------------------------------------
s_BART <- Sys.time()
fitBART <- stan4bart::stan4bart(
  formula = ingreso ~ (1|dam) + bart(F182013_stable_lights + X2016_crops.coverfraction + X2016_urban.coverfraction + X2016_gHM + accessibility + accessibility_walking_only + area1 + sexo2 + edad2 + edad3 + edad4 + edad5 + anoest2 + anoest3 + anoest4 + discapacidad1 + etnia1 + tiene_alcantarillado + tiene_electricidad + tiene_acueducto + tiene_gas + eliminar_basura + tiene_internet + piso_tierra + material_paredes + material_techo + rezago_escolar + alfabeta + hacinamiento + tasa_desocupacion),
  verbose = -1, # suppress ALL output
  data = dfsTrain,
  bart_args = list(keepTrees = T) ) # needed for prediction of (new) test data

e_BART <- Sys.time()
time_BART <- difftime(e_BART, s_BART, units = "secs")[[1]]

#- Calculate predicted values
Yhat_BART_Test1 <- rowMeans( predict(fitBART, dfsTest1, type = "ev") )
Yhat_BART_Test2 <- rowMeans( predict(fitBART, dfsTest2, type = "ev") )


#-------------------------------------------------------------------------------
#- MERF
#-------------------------------------------------------------------------------

library(SAEforest)

ingreso <- dfsTrain$ingreso
X_covar <- dfsTrain[,PredNames]

model1 <- SAEforest_model(Y = ingreso,
                          X = X_covar,
                          dName = "dam",
                          smp_data = dfsTrain,
                          pop_data = dfsTest1)

summary(model1)
#-------------------------------------------------------------------------------
#- Standard Random Forest 
#-------------------------------------------------------------------------------
s_RF <- Sys.time()
fitRF <- randomForest::randomForest(formula = model, data = dfsTrain)
e_RF <- Sys.time()
time_RF <- difftime(e_RF, s_RF, units = "secs")[[1]]

#- Calculate predicted values
Yhat_RF_Test1 <- predict(fitRF, dfsTest1)


#-------------------------------------------------------------------------------
#- Linear Mixed Model
#-------------------------------------------------------------------------------
s_LMM <- Sys.time()
fitLMM <- lme4::lmer(modelLMM, weights = n, data = dfsTrain)
e_LMM <- Sys.time()
time_LMM <- difftime(e_LMM, s_LMM, units = "secs")[[1]]

Yhat_LMM_Test1 <- predict( fitLMM, dfsTest1 )


#-------------------------------------------------------------------------------
#- Calculate PMSE and R2 for both test sets
#-------------------------------------------------------------------------------
Yhat_list_Test1 <- list( BoostMERT_L2 = Yhat_MEMBoost_Test1[, grepl("BoostMERT_L2", colnames(Yhat_MEMBoost_Test1))],
                         LMM = Yhat_LMM_Test1 )

nEstimators <- length(Yhat_list_Test1)
EstimatorNames <- names(Yhat_list_Test1)
resList <- sapply(EstimatorNames, function(x) NULL) 

for (x in 1:nEstimators){
  PMSE_Test1 <- mean( (YTest1 - Yhat_list_Test1[[x]])^2 )
  R2_Test1 <- cor(YTest1, Yhat_list_Test1[[x]])^2
  resList[[x]] <- rbind( resList[[x]],
                         c(  Estimator = x,
                             PMSE_Test1 = PMSE_Test1,
                             R2_Test1 = R2_Test1
                             ) )
}
resPMSE_R2 <- do.call("rbind", resList) 
resPMSE_R2 <- data.frame(resPMSE_R2)
resPMSE_R2$Estimator <- EstimatorNames 
cbind(resPMSE_R2$Estimator, round(resPMSE_R2[, 2:3], 2))

# computation times of the different approaches
resCompTimes <- data.frame( Estimator = EstimatorNames,
                            ComputationTime = c(time_BoostMERT_L2, time_BoostREEM_L2,
                              time_LMM ) )
cbind(resCompTimes$Estimator, round(resCompTimes$ComputationTime, 2) )



#==============================================================================
#- 2. Get estimated variance of random intercepts and estimated error variance
#==============================================================================
Dhat <- matrix(
  c(
    fitBoostMERT_L2$var_random_effects,
    fitted(fitBART,  type = "Sigma")$dam,
    lme4::VarCorr(fitLMM)$dam[1]
  ),
  nrow = 3,
  dimnames = list(Model = c("BoostMERT_L2","BART", "LMM"), "Variance of random intercepts")
  
)

# Varibility betweeen planification regions (between-group variance)
Dhat


ErrorVar <- matrix(
  c(fitBoostMERT_L2$errorVar,
    47468214004,
    fitted(fitBART,  type = "sigma")^2,
    sigma(fitICCLMM) ^ 2),
  nrow = 4,
  dimnames = list(Model = c("BoostMERT_L2","MERF","BART", "LMM"), "Variance of residual errors")
)
# Variance of the residual errors (not explained by either fixed effects or random effects)
ErrorVar




