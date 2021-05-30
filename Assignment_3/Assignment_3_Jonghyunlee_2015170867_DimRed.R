install.packages("glmnet")
install.packages("GA")
install.packages("ggplot2")

library(glmnet)
library(GA)
library(psych)
library(moments)
library(corrplot)
library(ggplot2)


#performance Evaluation Function
Perf_Table <- matrix(0, nrow=8, ncol=6)
rownames(Perf_Table) <- c("All", "Forward","Backward","Stepwise","GA","Ridge","Lasso","Elastic Net")
colnames(Perf_Table) <- c("TPR","Precision","TNR","Accurarcy","BCR","F1-Measure")

#calling dataset,preprocessing,MLR
admission <- read.csv("Admission_Predict.csv") 
nStud <- nrow(admission)
nVar <- ncol(admission)
id_idx <- c(1)

admission_data <- cbind(admission[,-c(id_idx)])
describe(admission_data)

admission_mlr_data <- cbind(admission[,-c(id_idx)])
set.seed(12345) 
admission_trn_idx <- sample(1:nStud, round(0.7*nStud))
admission_trn <- admission_mlr_data[admission_trn_idx,]
admission_tst <- admission_mlr_data[-admission_trn_idx,]
admission_tst
full_model_admission <- lm(Chance.of.Admit~ ., data=admission_trn)
summary(full_model_admission)

full_model_admission_coeff <- as.matrix(full_model_admission$coefficients,8,1)
full_model_admission_coeff

#MAE, MAPE, RMSA
perf_eval_reg <- function(tgt_y, pre_y){
  rmse <- sqrt(mean((tgt_y-pre_y)^2))
  mae <- mean(abs(tgt_y- pre_y))
  mape <- 100*mean(abs((tgt_y-pre_y)/tgt_y))
  return(c(rmse,mae,mape))
}
perf_mat <- matrix(0, nrow=5, ncol=3)
rownames(perf_mat) <- c("All Variables","Forward Selection","Backward Elimination","Stepwise Selection","Genetic Algorithm")
colnames(perf_mat) <-c("RMSE","MAE","MAPE")
perf_mat

#all variables
full_model_admission_prob <- predict(full_model_admission,newdata=admission_tst)
perf_mat[1,] <- perf_eval_reg(admission_tst$Chance.of.Admit, full_model_admission_prob)
perf_mat

#Forward Selection 
tmp_admission_x <- paste(colnames(admission_trn)[-1], collapse="+")
tmp_admission_xy <- paste("Chance.of.Admit~", tmp_admission_x, collapse="")
as.formula(tmp_admission_xy)

start_time_FS <- proc.time()
forward_admission_model <- step(lm(Chance.of.Admit ~ 1, data = admission_trn),
                      scope = list(upper = as.formula(tmp_admission_xy), lower = Chance.of.Admit ~ 1),
                      direction = "forward", trace = 1)
end_time_FS <- proc.time()
end_time_FS - start_time_FS

summary(forward_admission_model)
forward_model_coeff <- as.matrix(forward_admission_model$coefficients,8,1)
forward_model_coeff

#prediction
forward_admission_model_prob <- predict(forward_admission_model, type="response",newdata=admission_tst)
perf_mat[2,] <- perf_eval_reg(admission_tst$Chance.of.Admit,forward_admission_model_prob)
perf_mat

#Backward Elimination
backward_admission_model <- step(full_model_admission, scope=list(upper=as.formula(tmp_admission_xy),lower=Chance.of.Admit~1), direction="backward",trace=1)

start_time_BE <- proc.time()
backward_admission_model <- step(full_model_admission, scope = list(upper = as.formula(tmp_admission_xy),
                                                 lower = Chance.of.Admit ~ 1), direction = "backward",trace = 1)
end_time_BE <- proc.time()
end_time_BE - start_time_BE

summary(backward_admission_model)
backward_admission_model_coeff <- as.matrix(backward_admission_model$coefficients,8,1)
backward_admission_model_coeff

#prediction 
backward_admission_model_prob <- predict(backward_admission_model, type="response",newdata=admission_tst)
perf_mat[3,] <- perf_eval_reg(admission_tst$Chance.of.Admit, backward_admission_model_prob)
perf_mat


#Stepwise Selection
start_time_SS <- proc.time()
stepwise_admission_model <- step(lm(Chance.of.Admit ~ 1, data = admission_trn),
                       scope = list(upper = as.formula(tmp_admission_xy),
                                    lower = Chance.of.Admit ~ 1), direction = "both", trace = 1)
end_time_SS <- proc.time()
end_time_SS - start_time_SS

summary(stepwise_admission_model)
stepwise_admission_model_coeff <- as.matrix(stepwise_admission_model$coefficients,8,1)
stepwise_admission_model_coeff

#prediction
stepwise_admission_model_prob <- predict(stepwise_admission_model, type="response",newdata=admission_tst)
perf_mat[4,] <- perf_eval_reg(admission_tst$Chance.of.Admit, stepwise_admission_model_prob)
perf_mat

#Adjusted R squared
summary(full_model_admission)$adj.r.squared
summary(forward_admission_model)$adj.r.squared
summary(backward_admission_model)$adj.r.squared
summary(stepwise_admission_model)$adj.r.squared


#Genetic Algorithm
fit_AR <- function(string){
  sel_var_idx <-  which(string==1)
  sel_x <- x[,sel_var_idx]
  xy <- data.frame(sel_x, y)
  GA_mlr <- lm(y ~ ., data = xy)
  return(summary(GA_mlr)$adj.r.square)
}

#variable selection via Genetic Algorithm
x <- as.matrix(admission_trn[,-1])
y <- admission_trn[,1]

start_time_GA_mlr <- proc.time()
GA_AR <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x),
            names = colnames(x), popSize = 50, pcrossover = 0.5, pmutation = 0.01,
            maxiter = 100, elitism = 2, seed = 123)
end_time_GA_mlr <- proc.time()
end_time_GA_mlr - start_time_GA_mlr

best_mlr_idx <- which(GA_AR@solution == 1)
best_mlr_idx
summary(GA_AR)

input_ga_idx <- c(2,3,6,7,8)
target_ga_idx <- 1

admission_ga_data <- cbind(admission_trn[c(input_ga_idx,target_ga_idx)])
mlr_admission_ga <- lm(Chance.of.Admit ~ ., data = admission_ga_data)
mlr_admission_ga

#prediction
mlr_admission_ga_haty <- predict(mlr_admission_ga, type="response",newdata = admission_tst)
perf_mat[5,] <- perf_eval_reg(admission_tst$Chance.of.Admit, mlr_admission_ga_haty)
perf_mat

#Changing Hyperparameters

#Changing Population Size
#Population Size 10
start_time_GA_mlr <- proc.time()
GA_AR <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 10, pcrossover = 0.5, pmutation = 0.01,
            maxiter = 100, elitism = 2, seed = 123)
end_time_GA_mlr <- proc.time()
end_time_GA_mlr - start_time_GA_mlr
summary(GA_AR)

#Population Size 500
start_time<- proc.time()
GA_popsize1 <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 500, pcrossover = 0.5, pmutation = 0.01,
            maxiter = 100, elitism = 2, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_popsize1)

#Population Size 2000
start_time<- proc.time()
GA_popsize2 <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 2000, pcrossover = 0.5, pmutation = 0.01,
                  maxiter = 100, elitism = 2, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_popsize2)

#Changing Crossover Rate
#Crossover Rate 0.1
start_time_GA_mlr <- proc.time()
GA_AR <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.1, pmutation = 0.01,
            maxiter = 100, elitism = 2, seed = 123)
end_time_GA_mlr <- proc.time()
end_time_GA_mlr - start_time_GA_mlr
summary(GA_AR)

#Crossover Rate 0.3
start_time <- proc.time()
GA_CR1 <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.3, pmutation = 0.01,
            maxiter = 100, elitism = 2, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_CR1)

#Crossover Rate 1
start_time <- proc.time()
GA_CR2 <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 1, pmutation = 0.01,
            maxiter = 100, elitism = 2, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_CR2)


#Changing Mutation Rate
#Mutation Rate 0.001
start_time_GA_mlr <- proc.time()
GA_AR <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.5, pmutation = 0.001,
            maxiter = 100, elitism = 2, seed = 123)
end_time_GA_mlr <- proc.time()
end_time_GA_mlr - start_time_GA_mlr
summary(GA_AR)

#Mutation Rate 0.1
start_time <- proc.time()
GA_MR1 <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.5, pmutation = 0.1,
             maxiter = 100, elitism = 2, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_MR1)

#Mutation Rate 1
start_time <- proc.time()
GA_MR2 <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.5, pmutation = 1,
             maxiter = 100, elitism = 2, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_MR2)

#Changing Max iteration
#Max iteration 5
start_time <- proc.time()
GA_AR <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.5, pmutation = 0.01,
            maxiter = 5, elitism = 2, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_AR)

#Max iteration 10
start_time <- proc.time()
GA_iter1 <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.5, pmutation = 0.01,
            maxiter = 10, elitism = 2, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_iter1)

#Max iteration 500
start_time <- proc.time()
GA_iter2 <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.5, pmutation = 0.01,
            maxiter = 500, elitism = 2, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_iter2)

#Changing elitism
#Elitism 1
start_time <- proc.time()
GA_AR <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.5, pmutation = 0.01,
            maxiter = 100, elitism = 1, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_AR)

#Elitism 20
start_time <- proc.time()
GA_elit1 <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.5, pmutation = 0.01,
            maxiter = 100, elitism = 20, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_elit1)

#Elitism 50
start_time <- proc.time()
GA_elit2 <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.5, pmutation = 0.01,
            maxiter = 100, elitism = 50, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_elit2)

#optimal speed
start_time <- proc.time()
GA_optspeed <- ga(type = 'binary', fitness = fit_AR, nBits = ncol(x), names = colnames(x), popSize = 10, pcrossover = 0.3, pmutation = 0.001,
               maxiter = 10, elitism = 1, seed = 123)
end_time <- proc.time()
end_time - start_time
summary(GA_optspeed)

#===============================================================================================================================================================

#performance Evaluation Function for Logistic Regression
perf_eval <- function(cm){
  #True Positive Rate: TPR
  TPR <- cm[2,2]/sum(cm[2,])
  #Precision
  PRE <- cm[2,2]/sum(cm[,2])
  #True Negative Rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  #Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  #Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  #F1- Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  return(c(TPR, PRE, TNR, ACC, BCR, F1))
}

#preparing function for calculating AUROC
auroc <- function(model, data){
  lr_response <- predict(model, type = "response", newdata = data)
  roc_data_val <- cbind(data, lr_response)
  roc_data_val <- roc_data_val[c(order(-roc_data_val$lr_response)),]
  rownames(roc_data_val) <- NULL
  real_1 <- length(which(roc_data_val$predicthd_target == 1))
  real_0 <- nrow(roc_data_val) - real_1
  real_1
  real_0
  roc_tpr <- rep(0,nrow(roc_data_val))
  roc_fpr <- rep(0,nrow(roc_data_val))
  roc_data_val <- cbind(roc_data_val, roc_tpr, roc_fpr)
  for(i in 1:nrow(roc_data_val)){
    roc_data_val$roc_tpr[i] <- length(which(roc_data_val$predicthd_target[1:i]==1))/real_1
    roc_data_val$roc_fpr[i] <- length(which(roc_data_val$predicthd_target[1:i]==0))/real_0
  }
  roc_data_val
  roc_df <- data.frame(roc_data_val$roc_tpr, roc_data_val$roc_fpr)
  roc_df
  plot_roc<-ggplot(roc_df, aes(x=roc_df[,2], y=roc_df[,1])) +geom_line(col='red') +geom_point(col = 'red', size = 0)
  plot_roc
  auroc <- 0
  for(i in 1:(nrow(roc_df)-1)){
    if(roc_df[i,2] == roc_df[i+1,2]){
      next
    }
    else{
      ti <- (roc_df[i+1,2]-roc_df[i,2])*((roc_df[i+1,1]+roc_df[i,1])/2)
    }
    auroc <- auroc + ti
  }
  print(plot_roc)
  return(auroc)
}


#preparing performance matrix for Logistic Regressions
perf_mat_lr <- matrix(0, 5, 10)
colnames(perf_mat_lr) <- c("AUROC(trn)", "AUROC(val)", "TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1", 
                           "number of reduced variables", "time spent")
rownames(perf_mat_lr) <- c("Original", "Forward", "Backward", "Stepwise", "GA")


#calling dataset,preprocessing, LR
predicthd <- read.csv("framingham.csv")
predicthd2 <- na.exclude(predicthd)
input_idx <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
target_idx <- 16
predicthd_input <- predicthd2[,input_idx]
predicthd_input <- scale(predicthd_input, center = TRUE, scale = TRUE)
predicthd_target <- as.factor(predicthd2[,target_idx])
predicthd_data <- data.frame(predicthd_input, predicthd_target)

describe(predicthd_data)

set.seed(12345)
trn_idx <- sample(1:nrow(predicthd_data), round(0.7*nrow(predicthd_data)))
predicthd_trn <- predicthd_data[trn_idx,]
predicthd_val <- predicthd_data[-trn_idx,]

#All variables

full_lr <- glm(predicthd_target ~ ., family=binomial, data = predicthd_trn)
summary(full_lr)

full_lr_coef <- as.matrix(full_lr$coefficients, 16, 1)
full_lr_coef
perf_mat_lr[1,1]<-auroc(full_lr, predicthd_trn)
perf_mat_lr[1,2]<-auroc(full_lr, predicthd_val)

full_lr_prob <- predict(full_lr, type = "response", newdata = predicthd_val)
full_lr_prey <- rep(0, nrow(predicthd_val))
full_lr_prey[which(full_lr_prob >= 0.2)] <- 1
full_lr_cm <- matrix(0, nrow = 2, ncol = 2)
full_lr_cm[1,1] <- length(which(predicthd_val$predicthd_target == 0 & full_lr_prey == 0))
full_lr_cm[1,2] <- length(which(predicthd_val$predicthd_target == 0 & full_lr_prey == 1))
full_lr_cm[2,1] <- length(which(predicthd_val$predicthd_target == 1 & full_lr_prey == 0))
full_lr_cm[2,2] <- length(which(predicthd_val$predicthd_target == 1 & full_lr_prey == 1))
full_lr_cm

perf_mat_lr[1,3:8] <- perf_eval(full_lr_cm)
perf_mat_lr[1,9:10] <- c(NA, NA)
perf_mat_lr

#forward selection
tmp_x <- paste(colnames(predicthd_trn)[-16], collapse = " + ")
tmp_xy <- paste("predicthd_target ~ ", tmp_x, collapse = "")
as.formula(tmp_xy)

start_time_forward_lr <- proc.time()
forward_lr <- step(glm(predicthd_target ~1,family=binomial, data = predicthd_trn),scope = list(upper = as.formula(tmp_xy), lower = predicthd_target ~ 1),
                   direction = 'forward', trace = 1)
end_time_forward_lr <- proc.time()
time_spent_forward_lr <- end_time_forward_lr - start_time_forward_lr
time_spent_forward_lr
perf_mat_lr[2,10] <- time_spent_forward_lr[3]

summary(forward_lr)
forward_lr_coef <- as.matrix(forward_lr$coefficients, 8, 1)
forward_lr_coef
perf_mat_lr[2,9] <- 16 - length(forward_lr_coef)

perf_mat_lr[2,1] <- auroc(forward_lr, predicthd_trn)
perf_mat_lr[2,2] <- auroc(forward_lr, predicthd_val)

forward_lr_prob <- predict(forward_lr, type = "response", newdata = predicthd_val)
forward_lr_prey <- rep(0, nrow(predicthd_val))
forward_lr_prey[which(forward_lr_prob >= 0.2)] <- 1
forward_lr_cm <- matrix(0, nrow = 2, ncol = 2)
forward_lr_cm[1,1] <- length(which(predicthd_val$predicthd_target == 0 & forward_lr_prey == 0))
forward_lr_cm[1,2] <- length(which(predicthd_val$predicthd_target == 0 & forward_lr_prey == 1))
forward_lr_cm[2,1] <- length(which(predicthd_val$predicthd_target == 1 & forward_lr_prey == 0))
forward_lr_cm[2,2] <- length(which(predicthd_val$predicthd_target == 1 & forward_lr_prey == 1))
forward_lr_cm
perf_mat_lr[2,3:8] <- perf_eval(forward_lr_cm)
perf_mat_lr

#backward elimination
tmp_x <- paste(colnames(predicthd_trn)[-16], collapse = " + ")
tmp_xy <- paste("predicthd_target ~ ", tmp_x, collapse = "")
as.formula(tmp_xy)

start_time_backward_lr <- proc.time()
backward_lr <- step(full_lr,scope = list(upper = as.formula(tmp_xy), lower = predicthd_target ~ 1),
                    direction = 'backward', trace = 1)
end_time_backward_lr <- proc.time()
time_spent_backward_lr <- end_time_backward_lr - start_time_backward_lr
time_spent_backward_lr
perf_mat_lr[3,10] <- time_spent_backward_lr[3]

summary(backward_lr)
backward_lr_coef <- as.matrix(backward_lr$coefficients, 8, 1)
backward_lr_coef
perf_mat_lr[3,9] <- 16 - length(backward_lr_coef)

perf_mat_lr[3,1] <- auroc(backward_lr, predicthd_trn)
perf_mat_lr[3,2] <- auroc(backward_lr, predicthd_val)

backward_lr_prob <- predict(backward_lr, type = "response", newdata = predicthd_val)
backward_lr_prey <- rep(0, nrow(predicthd_val))
backward_lr_prey[which(backward_lr_prob >= 0.2)] <- 1
backward_lr_cm <- matrix(0, nrow = 2, ncol = 2)
backward_lr_cm[1,1] <- length(which(predicthd_val$predicthd_target == 0 & backward_lr_prey == 0))
backward_lr_cm[1,2] <- length(which(predicthd_val$predicthd_target == 0 & backward_lr_prey == 1))
backward_lr_cm[2,1] <- length(which(predicthd_val$predicthd_target == 1 & backward_lr_prey == 0))
backward_lr_cm[2,2] <- length(which(predicthd_val$predicthd_target == 1 & backward_lr_prey == 1))
backward_lr_cm
perf_mat_lr[3,3:8] <- perf_eval(backward_lr_cm)
perf_mat_lr

#stepwise selection
tmp_x <- paste(colnames(predicthd_trn)[-16], collapse = " + ")
tmp_xy <- paste("predicthd_target ~ ", tmp_x, collapse = "")
as.formula(tmp_xy)

start_time_stepwise_lr <- proc.time()
stepwise_lr <- step(glm(predicthd_target ~ 1,family=binomial, data = predicthd_trn),scope = list(upper = as.formula(tmp_xy), lower = predicthd_target ~ 1),
                    direction = 'both', trace = 1)
end_time_stepwise_lr <- proc.time()
time_spent_stepwise_lr <- end_time_stepwise_lr - start_time_stepwise_lr
time_spent_stepwise_lr
perf_mat_lr[4,10] <- time_spent_stepwise_lr[3]

summary(stepwise_lr)
stepwise_lr_coef <- as.matrix(stepwise_lr$coefficients, 8, 1)
stepwise_lr_coef
perf_mat_lr[4,9] <- 16 - length(stepwise_lr_coef)

perf_mat_lr[4,1] <- auroc(stepwise_lr, predicthd_trn)
perf_mat_lr[4,2] <- auroc(stepwise_lr, predicthd_val)

stepwise_lr_prob <- predict(stepwise_lr, type = "response", newdata = predicthd_val)
stepwise_lr_prey <- rep(0, nrow(predicthd_val))
stepwise_lr_prey[which(stepwise_lr_prob >= 0.2)] <- 1
stepwise_lr_cm <- matrix(0, nrow = 2, ncol = 2)
stepwise_lr_cm[1,1] <- length(which(predicthd_val$predicthd_target == 0 & stepwise_lr_prey == 0))
stepwise_lr_cm[1,2] <- length(which(predicthd_val$predicthd_target == 0 & stepwise_lr_prey == 1))
stepwise_lr_cm[2,1] <- length(which(predicthd_val$predicthd_target == 1 & stepwise_lr_prey == 0))
stepwise_lr_cm[2,2] <- length(which(predicthd_val$predicthd_target == 1 & stepwise_lr_prey == 1))
stepwise_lr_cm
perf_mat_lr[4,3:8] <- perf_eval(stepwise_lr_cm)
perf_mat_lr

#Genetic algorithm
auroc_2 <- function(model, data){
  lr_response <- predict(model, type = "response", newdata = data)
  roc_data_val <- cbind(data, lr_response)
  roc_data_val <- roc_data_val[c(order(-roc_data_val$lr_response)),]
  rownames(roc_data_val) <- NULL
  real_1 <- length(which(roc_data_val$y == 1))
  real_0 <- nrow(roc_data_val) - real_1
  real_1
  real_0
  roc_tpr <- rep(0,nrow(roc_data_val))
  roc_fpr <- rep(0,nrow(roc_data_val))
  roc_data_val <- cbind(roc_data_val, roc_tpr, roc_fpr)
  for(i in 1:nrow(roc_data_val)){
    roc_data_val$roc_tpr[i] <- length(which(roc_data_val$y[1:i]==1))/real_1
    roc_data_val$roc_fpr[i] <- length(which(roc_data_val$y[1:i]==0))/real_0
  }
  roc_data_val
  roc_df <- data.frame(roc_data_val$roc_tpr, roc_data_val$roc_fpr)
  roc_df
  auroc <- 0
  for(i in 1:(nrow(roc_df)-1)){
    if(roc_df[i,2] == roc_df[i+1,2]){
      next
    }
    else{
      ti <- (roc_df[i+1,2]-roc_df[i,2])*((roc_df[i+1,1]+roc_df[i,1])/2)
    }
    auroc <- auroc + ti
  }
  return(auroc)
}

#fitness function: AUROC for the training dataset
fit_AUROC <- function(string){
  sel_var_idx <- which(string == 1)
  sel_x <- x[,sel_var_idx]
  xy <- data.frame(sel_x, y)
  GA_lr <- glm(y ~ ., family = binomial, xy)
  return(auroc_2(GA_lr, xy))
}

x = as.matrix(predicthd_val[,-16])
y = predicthd_val[,16]

# Variable selection by Genetic Algorithm
start_time_GA_lr <- proc.time()
GA_AUROC <- ga(type = "binary", fitness = fit_AUROC, nBits = ncol(x), names = colnames(x), popSize = 50, pcrossover = 0.5, 
               pmutation = 0.01, maxiter = 5, elitism = 2, seed = 123)
end_time_GA_lr <- proc.time()

time_spent_GA_lr <- end_time_GA_lr-start_time_GA_lr
time_spent_GA_lr

perf_mat_lr[5,10] <- time_spent_GA_lr[3]

best_vars_mat <- GA_AUROC@solution
c_nvar <- rep(0, nrow(best_vars_mat))
c_nvar
for(i in 1:nrow(best_vars_mat)){
  c_nvar[i] <- length(which(best_vars_mat[i,]==1))
}
c_nvar
best_var_idx_lr <- which(best_vars_mat[which.min(c_nvar),]==1)
best_var_idx_lr
GA_lr_trn <- predicthd_trn[,c(best_var_idx_lr, 16)]
GA_lr_val <- predicthd_val[,c(best_var_idx_lr, 16)]

GA_lr <- glm(predicthd_target ~ ., family=binomial, GA_lr_trn)
summary(GA_lr)
GA_lr_coeff <- as.matrix(GA_lr$coefficients, 16, 1)
GA_lr_coeff
perf_mat_lr[5,9] <- 16-length(GA_lr_coeff)

perf_mat_lr[5,1] <- auroc(GA_lr, predicthd_trn)
perf_mat_lr[5,2] <- auroc(GA_lr, predicthd_val)

GA_lr_prob <- predict(GA_lr, type = "response", newdata = predicthd_val)
GA_lr_prey <- rep(0, nrow(predicthd_val))
GA_lr_prey[which(GA_lr_prob >= 0.2)] <- 1
GA_lr_cm <- matrix(0, nrow = 2, ncol = 2)
GA_lr_cm[1,1] <- length(which(predicthd_val$predicthd_target == 0 & GA_lr_prey == 0))
GA_lr_cm[1,2] <- length(which(predicthd_val$predicthd_target == 0 & GA_lr_prey == 1))
GA_lr_cm[2,1] <- length(which(predicthd_val$predicthd_target == 1 & GA_lr_prey == 0))
GA_lr_cm[2,2] <- length(which(predicthd_val$predicthd_target == 1 & GA_lr_prey == 1))
GA_lr_cm
perf_mat_lr[5,3:8] <- perf_eval(GA_lr_cm)
perf_mat_lr