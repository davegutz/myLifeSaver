# Taylor Community Data Analysis
# 31-Mar-2026

# SETUP ####
# load packages
library(lubridate)
library(readxl)
library(ggplot2)
library(tidyr)
library(effects)
library(dplyr)
library(zoo) # rollmean
 
 
# Import Data
D <- read.csv("lhs_taylor_results.csv")
plot( added_lc_worth_norm ~ yrs_sum_al, data=D)
lm_sum_al <- lm(yrs_sum_al ~ added_lc_worth_norm, data=D)
summary(lm_sum_al)

plot(added_lc_worth_norm ~ yrs_il_single, data=D)
lm_sum_il_single <- lm(yrs_il_single ~ added_lc_worth_norm, data=D)
summary(lm_sum_il_single)

plot(added_lc_worth_norm ~ yrs_il_double, data=D)
lm_sum_il_double <- lm(yrs_il_double ~ added_lc_worth_norm, data=D)
summary(lm_sum_il_double)


glm_anD <- glm(added_lc_worth_norm ~ yrs_sum_al + yrs_il_single + yrs_il_double, family=gaussian, data=D)
summary(glm_anD)
plot(allEffects(glm_anD))

#install.packages("leaps")
library(leaps)
#install.packages("regsubsets")
#library(regsubsets)
model_added <- leaps::regsubsets(added_lc_worth_norm ~ yrs_sum_al + yrs_il_single +
                             yrs_il_double + total_living_yrs + elapsed_time_yrs +
                             earning_potential + earning_potential_cc	+ 
                             man_independent_yrs + woman_independent_yrs + 
                             man_assisted_yrs + woman_assisted_yrs + apy_roi + 
                             apy_cpi + roi_one_dollar_at_end + cpi_one_dollar_at_end + 
                             man_goes_to_al + woman_goes_to_al + constant_monthly_roi + 
                             constant_monthly_cpi,
                             data=D, nbest=1)
model_cc <- leaps::regsubsets(final_worth_cc_norm ~ yrs_sum_al + yrs_il_single +
                             yrs_il_double + total_living_yrs + elapsed_time_yrs +
                             earning_potential + earning_potential_cc	+ 
                             man_independent_yrs + woman_independent_yrs + 
                             man_assisted_yrs + woman_assisted_yrs + apy_roi + 
                             apy_cpi + roi_one_dollar_at_end + cpi_one_dollar_at_end + 
                             man_goes_to_al + woman_goes_to_al + constant_monthly_roi + 
                             constant_monthly_cpi,
                             data=D, nbest=1)
model_lc <- leaps::regsubsets(final_worth_lc_norm ~ yrs_sum_al + yrs_il_single +
                              yrs_il_double + total_living_yrs + elapsed_time_yrs +
                                earning_potential + earning_potential_cc	+ 
                                man_independent_yrs + woman_independent_yrs + 
                                man_assisted_yrs + woman_assisted_yrs + apy_roi + 
                                apy_cpi + roi_one_dollar_at_end + cpi_one_dollar_at_end + 
                                man_goes_to_al + woman_goes_to_al + constant_monthly_roi + 
                                constant_monthly_cpi,
                              data=D, nbest=1)

# potential more models
#final_worth_cc_norm	final_worth_lc_norm
#final_worth_norm_cc	final_worth_norm_lc	

summary_models_added <- summary(model_added)
summary(model_added)
summary_models_added$cp
summary_models_added$bic
summary_models_added$adjr2
# Find the index of the best model (e.g., minimum BIC)
which.min(summary_models_added$bic)
plot(model_added, scale = "bic") # Options: "bic", "cp", "adjr2", "r2"
coef(model_added, id = 6) # Get coefficients for the best model (e.g., model with 9 predictors)


summary_models_cc <- summary(model_cc)
summary(model_cc)
summary_models_cc$cp
summary_models_cc$bic
summary_models_cc$adjr2
# Find the index of the best model (e.g., minimum BIC)
which.min(summary_models_cc$bic)
plot(model_cc, scale = "bic") # Options: "bic", "cp", "adjr2", "r2"
coef(model_cc, id = 6) # Get coefficients for the best model (e.g., model with 9 predictors)


summary_models_lc <- summary(model_lc)
summary(model_lc)
summary_models_lc$cp
summary_models_lc$bic
summary_models_lc$adjr2
# Find the index of the best model (e.g., minimum BIC)
which.min(summary_models_lc$bic)
plot(model_lc, scale = "bic") # Options: "bic", "cp", "adjr2", "r2"
coef(model_lc, id = 6) # Get coefficients for the best model (e.g., model with 9 predictors)



######### Ranch
lm_faRa <- lm(Total.Value ~ Fin.area, data=Ra)
summary(lm_faRa)
plot(allEffects(lm_faRa))

plot(Total.Value ~ Fin.area, data=Ra)
abline(lm_faRa)


glm_Ra <- glm(Total.Value ~ Fin.area + Lot.size + Beds + Baths, family=gaussian, data=Ra)
summary(glm_Ra)
plot(allEffects(glm_Ra))

new_fa=data.frame(Fin.area=c(0, 5000))
new_Ra=predict(lm_faRa, new_fa, type="response")

######### Cape
lm_faCp <- lm(Total.Value ~ Fin.area, data=Cp)
summary(lm_faCp)
plot(allEffects(lm_faCp))

glm_Cp <- glm(Total.Value ~ Fin.area + Lot.size + Beds + Baths, family=gaussian, data=Cp)
summary(glm_Cp)
plot(allEffects(glm_Cp))

######### Colonial
lm_faCo <- lm(Total.Value ~ Fin.area, data=Co)
summary(lm_faCo)
plot(allEffects(lm_faCo))

glm_Co <- glm(Total.Value ~ Fin.area + Lot.size + Beds + Baths, family=gaussian, data=Co)
summary(glm_Co)
plot(allEffects(glm_Co))

######### Split Level
lm_faSl <- lm(Total.Value ~ Fin.area, data=Sl)
summary(lm_faSl)
plot(allEffects(lm_faSl))

glm_Sl <- glm(Total.Value ~ Fin.area + Lot.size + Beds + Baths, family=gaussian, data=Sl)
summary(glm_Sl)
plot(allEffects(glm_Sl))

#############Summarize

# Create an empty plot
plot(Ra$Fin.area, Ra$Total.Value, type="n", xlim=c(500, 5500), ylim=c(200000, 1200000), 
     xlab="Finished Area, sqft", ylab="Total Assesed Value, $", main="Wenham Neighborhood 2000")
lines(Ra$Fin.area, Ra$Total.Value, type="p", pch=1, col="red")
abline(lm_faRa, col="red")
lines(Cp$Fin.area, Cp$Total.Value, type="p", pch=2, col="blue")
abline(lm_faCp, col="blue")
lines(Co$Fin.area, Co$Total.Value, type="p", pch=3, col="green")
abline(lm_faCo, col="green")
lines(Sl$Fin.area, Sl$Total.Value, type="p", pch="S", col="black")
abline(lm_faSl, col="black")
legend("topleft", legend=c("Ranch", "Cape", "Colonial", "Split Level"), col=c("red", "blue", "green", "black"), lty=1)

