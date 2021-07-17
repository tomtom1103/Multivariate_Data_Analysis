install.packages("nnet")
install.packages("tidyverse")
install.packages("psych")
install.packages("glmnet")
install.packages("GA")
install.packages("tree")
install.packages("party")
install.packages("ROCR")
install.packages("ipred")
install.packages("mlbench")
install.packages("caret")
install.packages("doParallel")
install.packages("randomForest")
install.packages("ada")
install.packages("adabag")
install.packages("gbm")
library(gbm)
library(ada)
library(adabag)
library(randomForest)
library(tidyverse)
library(dplyr)
library(nnet)
library(ggplot2)
library(psych)
library(glmnet)
library(GA)
library(tree)
library(party)
library(ROCR)
library(ipred)
library(rpart)
library(mlbench)
library(caret)
library(parallel)
library(doParallel)

Damage <- read.csv("Earthquake_Damage.csv")
str(Damage)

class_1 <- sample_n(Damage[which(Damage$damage_grade == 1),],2520)
class_2 <- sample_n(Damage[which(Damage$damage_grade == 2),],14825)
class_3 <- sample_n(Damage[which(Damage$damage_grade == 3),],8725)

Damage <- rbind(class_1,class_2,class_3)

#Performance Evaluation
perf_eval_multi <- function(cm){
  #Simple Accuracy
  ACC <- sum(diag(cm))/sum(cm)
  #Balanced Correction Rate
  BCR <- 1
  for(i in 1:dim(cm)[1]){
    BCR = BCR * (cm[i,i]/sum(cm[i,]))
  }
  BCR = BCR^(1/dim(cm)[1])
  return(c(ACC,BCR))
}

#Performance Table
perf_table <- matrix(0,nrow = 8, ncol = 2)
colnames(perf_table) <- c("ACC","BCR")
rownames(perf_table) <- c("MLR","CART","ANN","Bagging CART",
                          "Random Forests","Bagging ANN","AdaBoost","GBM")
perf_table

#Data preprocessing
#Convert character variable to binary
cha_idx <- c(9:15,27)
character <- Damage[,cha_idx]
a <- names(character)

frame <- 0
for(i in 1:length(character)){
  tmp <- class.ind(character[,i])
  for(j in 1:ncol(tmp)){
    colnames(tmp)[j] <-  paste0(a[i],"_",colnames(tmp)[j])
  }
  assign(paste0("dummy_",a[i]),tmp)
  frame <- cbind(frame,tmp)
}

cha_input <- frame[,c(2:39)]

#Convert non numeric variables to binary
num_idx <- c(2:8,28)
num_input <- scale(Damage[,num_idx], center = TRUE, scale = TRUE)

bin_idx <- c(16:26,29:39)
bin_input <- lapply(Damage[,bin_idx],factor)

input <- data.frame(cha_input,num_input,bin_input)
target <- as.factor(Damage[,c(40)])
Final_data <- data.frame(input, Class = target)

#Split the Data
set.seed(12345)
trn <- Final_data[sample(nrow(Final_data),15044),]
val <- Final_data[sample(nrow(Final_data),5013),]
tst <- Final_data[sample(nrow(Final_data),6016),]


#Question 1
#Multinomial Logistic Regression
trn_data <- rbind(trn,val)

start.time1 <- proc.time()
ml_logit <- multinom(Class ~ ., data = trn_data)
end.time1 <- proc.time()

time1 <- end.time1 - start.time1
time1

summary(ml_logit)
t(summary(ml_logit)$coefficients)

ml_logit_prey <- predict(ml_logit, newdata = tst)
mlr_cfm <- table(tst$Class, ml_logit_prey)
mlr_cfm

perf_table[1,] <- perf_eval_multi(mlr_cfm)
perf_table

#Classification and Regression Tree
input_idx <- c(1:68)
target_idx <- 69

set.seed(12345)
CART_trn <- data.frame(trn[,input_idx], GradeYN = trn[,target_idx])
CART_val <- data.frame(val[,input_idx], GradeYN = val[,target_idx])
CART_tst <- data.frame(tst[,input_idx], GradeYN = tst[,target_idx])

gs_CART <- list(min_criterion = c(0.9,0.95,0.99),min_split = c(10,30,50,100),max_depth = c(0,20,5)) %>%
  cross_df()

CART_result = matrix(0,nrow(gs_CART),5)
colnames(CART_result) <- c("min_criteron","min_split","max_depth","ACC","BCR")

iter_cnt = 1

start.time2 <- proc.time()
for(i in 1:nrow(gs_CART)){
  cat("CART Min Criterion:",gs_CART$min_criterion[i],",Minsplit:",
      gs_CART$min_split[i],",Max depth:",gs_CART$max_depth[i],"\n")
  
  tmp_control = ctree_control(mincriterion = gs_CART$min_criterion[i],
                              minsplit = gs_CART$min_split[i],maxdepth = gs_CART$max_depth[i])
  tmp_tree <- ctree(GradeYN ~., data = CART_trn, controls = tmp_control)
  tmp_tree_val_prediction <- predict(tmp_tree, newdata = CART_val)
  
  tmp_tree_val_cm <- table(CART_val$GradeYN, tmp_tree_val_prediction)
  tmp_tree_val_cm
  
  CART_result[iter_cnt,1] = gs_CART$min_criterion[i]
  CART_result[iter_cnt,2] = gs_CART$min_split[i]
  CART_result[iter_cnt,3] = gs_CART$max_depth[i]
  
  CART_result[iter_cnt,4:5] = perf_eval_multi(tmp_tree_val_cm)
  iter_cnt = iter_cnt + 1
}
end.time2 <- proc.time()
time2 <- end.time2 - start.time2
time2

CART_result <- CART_result[order(CART_result[,5],decreasing = T),]
CART_result

best_criterion <- CART_result[1,1]
best_split <- CART_result[1,2]
best_depth <- CART_result[1,3]

tree_control = ctree_control(mincriterion = best_criterion,
                             minsplit = best_split, maxdepth = best_depth)
CART_data <- rbind(CART_trn,CART_val)

CART_final <- ctree(GradeYN ~., data = CART_data, controls = tree_control)

CART_prediction <- predict(CART_final, newdata = CART_tst)
CART_cm <- table(CART_tst$GradeYN, CART_prediction)
CART_cm

perf_table[2,] <- perf_eval_multi(CART_cm)
perf_table

plot(CART_final, type = "simple")

#Artificial Neural Network
gs_ANN <- list(nH = seq(from = 20, to = 40, by = 5),
               maxit = seq(from = 100, to =300 , by = 50),
               rang = c(0.3, 0.5, 0.9)) %>%
  cross_df()

input <- trn[,input_idx] #for training ANN
target <- class.ind(trn[,target_idx]) 
val_input <- val[,input_idx]
val_target <- class.ind(val[,target_idx]) 

data <- rbind(trn,val) #for testing ANN
data_input <- data[,input_idx]
data_target <- class.ind(data[,target_idx])
tst_input <- tst[,input_idx]
tst_target <- class.ind(tst[,target_idx])

ANN_result <- matrix(0,nrow(gs_ANN),5)

set.seed(12345)

start.time3 <- proc.time()
for(i in 1:nrow(gs_ANN)){
  cat("Training ANN: the number of hidden nodes:",gs_ANN$nH[i],",maxit:",
      gs_ANN$maxit[i],",rang:",gs_ANN$rang[i],"\n")
  evaluation <- c()
  
  # Training the model
  tmp_nnet <- nnet(input,target,size = gs_ANN$nH[i],maxit = gs_ANN$maxit[i],
                   rang = gs_ANN$rang[i],silent = TRUE,MaxNWts = 10000)
  
  #Evaluate the model
  real <- max.col(val_target)
  pred <- max.col(predict(tmp_nnet,val_input))
  evaluation <- rbind(evaluation,cbind(real,pred))
  
  #Confusion Matrix
  ann_cfm <- matrix(0,nrow = 3, ncol = 3)
  ann_cfm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  ann_cfm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  ann_cfm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  ann_cfm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  ann_cfm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  ann_cfm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  ann_cfm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  ann_cfm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  ann_cfm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  ANN_result[i,1] <- gs_ANN$nH[i]
  ANN_result[i,2] <- gs_ANN$maxit[i]
  ANN_result[i,3] <- gs_ANN$rang[i]
  ANN_result[i,4:5] <- t(perf_eval_multi(ann_cfm))
}
end.time3 <- proc.time()

time3 <- end.time3 - start.time3
time3

#Check best combination of hyperparameters of ANN
best_ANN_result <- ANN_result[order(ANN_result[,5],decreasing = TRUE),]
colnames(best_ANN_result) <- c("nH","Maxit","rang","ACC","BCR")
best_ANN_result

#Train final Model
best_nH1 <- best_ANN_result[1,1]
best_maxit1 <- best_ANN_result[1,2]
best_rang1 <- best_ANN_result[1,3]

ANN_final <- nnet(data_input,data_target,size = best_nH1, 
                  maxit = best_maxit1, rang = best_rang1, MaxNWts = 10000)

ANN_pred <- predict(ANN_final, tst_input)
ANN_cfm1 <- table(max.col(tst_target), max.col(ANN_pred))
ANN_cfm1

perf_table[3,] <- perf_eval_multi(ANN_cfm1)
perf_table

#single model elapsed time
time_table[1,] <- time1[3]
time_table[2,] <- time2[3]
time_table[3,] <- time3[3]
time_table


#Question 2
#Ensemble Model 1: Bagging with CART

set.seed(12345)

#Train bagged model
nbagg <- seq(from = 30, to = 300, by = 30)
Bagging_Result <- matrix(0,length(nbagg),3)

iter_cnt = 1

start.time4 <- proc.time()
for(i in 1:length(nbagg)){
  cat("Bagging Training: the number of bootstrap:",nbagg[i],"\n")
  evaluation <- c()
  
  tmp_bagging <- cforest(GradeYN ~ ., data = CART_trn, controls = 
                           cforest_control(mincriterion = best_criterion, minsplit = best_split, 
                                           maxdepth = best_depth,mtry = 0, ntree = nbagg[i]))
  
  real <- CART_val$GradeYN
  pred <- predict(tmp_bagging, newdata = CART_val)
  evaluation <- rbind(evaluation,cbind(real,pred))
  
  cfm <- matrix(0,nrow = 3, ncol = 3)
  cfm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  cfm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  cfm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  cfm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  cfm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  cfm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  cfm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  cfm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  cfm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  Bagging_Result[iter_cnt,1] = nbagg[i]
  Bagging_Result[iter_cnt,2:3] = t(perf_eval_multi(cfm))
  iter_cnt = iter_cnt + 1
}
end.time4 <- proc.time()

time4 <- end.time4 - start.time4
time4

Bagging_Result_order <- Bagging_Result[order(Bagging_Result[,3],decreasing = T),]
colnames(Bagging_Result_order) <- c("No.of Bootstrap","ACC","BCR")
Bagging_Result_order

best_bootstrap <- Bagging_Result[1,1]

Bagging_Final <- cforest(GradeYN ~ ., data = CART_data, controls = 
                           cforest_control(mincriterion = best_criterion, minsplit = best_split, 
                                           maxdepth = best_depth, mtry = 0, ntree = best_bootstrap))
Bagging_pred <- predict(Bagging_Final, newdata = CART_tst)

Bagging_cfm <- table(CART_tst$GradeYN, Bagging_pred)
Bagging_cfm

perf_table[4,] <- perf_eval_multi(Bagging_cfm)
perf_table

#Question 3
#Ensemble Model 2: Random Forest

ntree <- seq(from = 30, to = 300, by = 30)
RF_Result <- matrix(0,length(ntree),3)
colnames(RF_Result) <- c("No.of Tree","ACC","BCR")

iter_cnt = 1

start.time5 <- proc.time()
for(i in 1:length(ntree)){
  cat("RandomForest Training:",ntree[i],"\n")
  tmp_RF <- randomForest(GradeYN ~., data = CART_trn, ntree = ntree[i], 
                         mincriterion = best_criterion, min_split = best_split, 
                         maxdepth = max_depth, importance = TRUE, do.trace = TRUE)
  
  RF.pred <- predict(tmp_RF, newdata = CART_val, type = "class")
  RF.cfm <- table(CART_val$GradeYN, RF.pred)
  print(tmp_RF)
  RF_Result[iter_cnt,1] = ntree[i]
  RF_Result[iter_cnt,2:3] = t(perf_eval_multi(RF.cfm))
  iter_cnt = iter_cnt + 1
}
end.time5 <- proc.time()
time5 <- end.time5 - start.time5
time5

RF_Result <- RF_Result[order(RF_Result[,3],decreasing = T),]
RF_Result
best_bootstrap2 <- RF_Result[1,1]

RF_Final <- randomForest(GradeYN ~ ., data = CART_data, ntree = best_bootstrap2, 
                         importance = TRUE, do.trace = TRUE)

print(RF_Final)
plot(RF_Final)

Var.imp <- importance(RF_Final)
summary(Var.imp)
barplot(Var.imp[order(Var.imp[,4],decreasing = TRUE),4])

RF_pred <- predict(RF_Final, newdata = CART_tst, type = "class")
RF_cfm <- table(CART_tst$GradeYN, RF_pred)
RF_cfm

perf_table[5,] <- perf_eval_multi(RF_cfm)
perf_table

#Compare CART with Bagging and RandomForest
Bagging_data <- Bagging_Result[order(Bagging_Result[,1],decreasing = F),]
No.Bootstrap <- Bagging_data[,1]
CART_Bagging_BCR <- Bagging_data[,3]
RF_BCR <- (RF_Result[order(RF_Result[,1],decreasing = F),])[,3]
BCR_summary <- data.frame(No.Bootstrap,CART_Bagging_BCR,RF_BCR)

GRAPH <- ggplot(data = BCR_summary)+
  geom_line(aes(x=No.Bootstrap,y=CART_Bagging_BCR))+geom_line(aes(x=No.Bootstrap,y=RF_BCR))
GRAPH + ylab("Value of BCR")

#Question 4: Single Model ANN with iterations
best_nH <- best_nH1
best_maxit <- best_maxit1
best_rang <- best_rang1
val_perf <- c()

iter_cnt = 1

start.time6 <- proc.time()
for(i in 1:30){
  evaluation <- c()
  
  tmp_nnet <- nnet(data_input, data_target, size = 
                     best_nH, maxit = best_maxit, rang = best_rang, MaxNWts = 10000)
  
  real <- max.col(tst_target)
  pred <- max.col(predict(tmp_nnet,tst_input))
  evaluation <- rbind(evaluation, cbind(real,pred))
  
  Final_cm <- matrix(0,nrow = 3, ncol = 3)
  Final_cm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  Final_cm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  Final_cm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  Final_cm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  Final_cm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  Final_cm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  Final_cm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  Final_cm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  Final_cm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  val_perf <- rbind(val_perf,t(perf_eval_multi(Final_cm)))
}
end.time6 <- proc.time()
time6 <- end.time6 - start.time6
time6

#Mean and S.D of ACC, BCR
colnames(val_perf) <- c("ACC","BCR")

mean_ACC <- mean(val_perf[,1])
sd_ACC <- sd(val_perf[,1])
ANN_ACC <- cbind(mean_ACC,sd_ACC)
ANN_ACC

mean_BCR <- mean(val_perf[,2])
sd_BCR <- sd(val_perf[,2])
ANN_BCR <- cbind(mean_BCR,sd_BCR)
ANN_BCR
ANN_iter_summary <- data.frame(ANN_ACC,ANN_BCR)
ANN_iter_summary

#without zero values of BCR
zero_idx <- c(3,5,10,12,14,18,24,26,27,30)
val_perf_nozero <- val_perf[-zero_idx,]
val_perf_nozero

mean_BCR_nozero <- mean(val_perf_nozero[,2])
sd_BCR_nozero <- sd(val_perf_nozero[,2])
ANN_BCR_nozero <- cbind(mean_BCR_nozero,sd_BCR_nozero)
ANN_BCR_nozero


#Question 5
#Ensemble Model 3: Bagging with Neural Network

cl <- makeCluster(4)
registerDoParallel(cl)

nrepeats = seq(from = 30, to = 300, by = 30)
ann.bagging.result <- c()
summary.table <- matrix(0,10,5)
colnames(summary.table) <- c("Bootstrap","mean_ACC","sd_ACC","mean_BCR","sd_BCR")

set.seed(12345)

for(i in 1:length(nrepeats)){
  cat("Training Bagging ANN: The Number of Bootstrap:",nrepeats[i],"\n")
  
  for(j in 1:10){
    cat("The Number of Repeats:",j,"\n")
    evaluation <- c()
    
    tmp_ann.bagging.model <- avNNet(data[,input_idx],data[,target_idx], 
                                    size = best_nH1, maxit = best_maxit1, rang = best_rang1, 
                                    repeats = nrepeats[i], bag = TRUE, trace = TRUE, MaxNWts = 10000)
    
    real <- max.col(tst_target)
    pred <- max.col(predict(tmp_ann.bagging.model,tst_input))
    evaluation <- rbind(evaluation, cbind(real,pred))
    
    Final_cm <- matrix(0,nrow = 3,ncol = 3)
    Final_cm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
    Final_cm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
    Final_cm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
    Final_cm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
    Final_cm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
    Final_cm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
    Final_cm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
    Final_cm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
    Final_cm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
    
    ann.bagging.result <- rbind(ann.bagging.result,t(perf_eval_multi(Final_cm)))
  }
  summary.table[i,1] <- nrepeats[i]
  summary.table[i,2] <- mean(ann.bagging.result[,1])
  summary.table[i,3] <- sd(ann.bagging.result[,1])
  summary.table[i,4] <- mean(ann.bagging.result[,2])
  summary.table[i,5] <- sd(ann.bagging.result[,2])
}
ann.bagging.result
summary.table

#Train with best number of bootstrap
best_repeats <- 270

ann.bagging.best <- c()

evaluation <- c()
annb.model <- avNNet(data[,input_idx], data[,target_idx], 
                     size = best_nH1 , maxit = best_maxit1, rang = best_rang1, 
                     repeats = best_repeats, bag = TRUE, trace = TRUE, MaxNWts = 10000)

real <- max.col(tst_target)
pred <- max.col(predict(annb.model,tst_input))
evaluation <- rbind(evaluation, cbind(real,pred))

annb_cm <- matrix(0,nrow = 3, ncol = 3)
annb_cm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
annb_cm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
annb_cm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
annb_cm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
annb_cm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
annb_cm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
annb_cm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
annb_cm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
annb_cm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))

ann.bagging.best <- rbind(ann.bagging.best,t(perf_eval_multi(annb_cm)))
ann.bagging.result

perf_table[6,] <- perf_eval_multi(annb_cm)
perf_table

#Question 6
#Ensemble Model 4: Adaboost

gs <- list(iter = c(100,200,300),
           bag.frac = c(0.1,0.3,0.5)) %>%
  cross_df()

ada_perf <- data.frame()

start.time8 <- proc.time()
iter_cnt = 1
for(i in 1:nrow(gs)){
  cat("Training adaboost: the number of population:",gs$iter[i],",ratio:",gs$bag.frac[i],"\n")
  evaluation <- c()
  
  tmp_iter <- gs[i,1]
  tmp_frac <- gs[i,2]
  
  tmp_adaboost <- boosting(GradeYN ~., data = CART_trn, boos = TRUE, 
                           mfinal = gs$iter[i], bag.frac = gs$bag.frac[i], 
                           control = rpart.control(mincriterion = best_criterion, minsplit = best_split))
  
  real <- CART_val$GradeYN
  pred <- predict(tmp_adaboost, CART_val[,input_idx])
  evaluation <- rbind(evaluation, cbind(real,pred$class))
  
  cfm <- matrix(0,nrow = 3, ncol = 3)
  cfm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  cfm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  cfm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  cfm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  cfm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  cfm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  cfm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  cfm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  cfm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  tmp_gs <- data.frame(tmp_iter, tmp_frac)
  tmp_ada <- t(perf_eval_multi(cfm))
  
  ada_perf <- rbind(ada_perf,cbind(tmp_gs,tmp_ada))
}
end.time8 <- proc.time()
time8 <- end.time8 - start.time8
time8

colnames(ada_perf) <- c("iteration","bag.frac","ACC","BCR")
ada_perf <- ada_perf[order(ada_perf[,4], decreasing = TRUE),]
ada_perf

best_iter <- ada_perf[1,1]
best_ration <- ada_perf[1,2]

adaboost.model <- boosting(GradeYN ~., data = CART_data, 
                           boos = TRUE, iter = best_iter, bag.frac = best_ration)
ada_print <- print(adaboost.model)

adaboost.pred <- predict(adaboost.model,CART_tst[,input_idx])
adaboost.cfm <- table(CART_tst$GradeYN, adaboost.pred$class)
adaboost.cfm

perf_table[7,] <- perf_eval_multi(adaboost.cfm)
perf_table

#Question 7
#Ensemble Model 5: GBM
GBM.trn <- data.frame(trn[,input_idx],GradeYN = trn[,target_idx])
GBM.val <- data.frame(val[,input_idx],GradeYN = val[,target_idx])
GBM.tst <- data.frame(tst[,input_idx],GradeYN = tst[,target_idx])

gbmGrid <-  expand.grid(n.trees = c(400,500,800,1000),
                        shrinkage = c(0.02, 0.05, 0.1,0.3))

gbm_perf <- matrix(0,nrow(gbmGrid),4)

start.time9 <- proc.time()
iter_cnt = 1
for(i in 1:nrow(gbmGrid)){
  cat("Training GBM: the number of population:",gbmGrid$n.trees[i],",shrinkage:",gbmGrid$shrinkage[i],"\n")
  evaluation <- c()
  
  tmp_gbm <- gbm.fit(GBM.trn[,input_idx],GBM.trn[,target_idx], 
                     distribution = "multinomial",verbose = TRUE, 
                     n.trees =gbmGrid$n.trees[i], shrinkage = gbmGrid$shrinkage[i])
  
  real <- GBM.val$GradeYN
  pred <- as.data.frame(predict(tmp_gbm, GBM.val[,input_idx], type = "response", n.trees = gbmGrid$n.trees[i]))
  pred <- max.col(pred)
  evaluation <- rbind(evaluation,cbind(real,pred))
  
  #Confusion Matrix
  gbm_cfm <- matrix(0,nrow = 3, ncol = 3)
  gbm_cfm[1,1] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 1))
  gbm_cfm[1,2] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 2))
  gbm_cfm[1,3] <- length(which(evaluation[,1] == 1 & evaluation[,2] == 3))
  gbm_cfm[2,1] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 1))
  gbm_cfm[2,2] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 2))
  gbm_cfm[2,3] <- length(which(evaluation[,1] == 2 & evaluation[,2] == 3))
  gbm_cfm[3,1] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 1))
  gbm_cfm[3,2] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 2))
  gbm_cfm[3,3] <- length(which(evaluation[,1] == 3 & evaluation[,2] == 3))
  
  gbm_perf[iter_cnt,1] <- gbmGrid$n.trees[i]
  gbm_perf[iter_cnt,2] <- gbmGrid$shrinkage[i]
  gbm_perf[iter_cnt,3:4] <- t(perf_eval_multi(gbm_cfm))
  iter_cnt = iter_cnt +1
}
end.time9 <- proc.time()
time9 <- end.time9 - start.time9
time9

gbm_perf <- gbm_perf[order(gbm_perf[,4],decreasing = TRUE),]
colnames(gbm_perf) <- c("n.trees","shrinkage","ACC","BCR")
gbm_perf

best_tree <- gbm_perf[1,1]
best_shrinkage <- gbm_perf[1,2]

gbm.model <- gbm.fit(data[,input_idx],data[,target_idx], distribution = "multinomial",
                     verbose = TRUE, n.trees = best_tree, shrinkage = best_shrinkage)

gbm.pred <- as.data.frame(predict(gbm.model, GBM.tst[,input_idx], type = "response", n.trees = best_tree))
gbm.cfm <- table(max.col(gbm.pred), GBM.tst$GradeYN)
gbm.cfm

perf_table[8,] <- perf_eval_multi(gbm.cfm)
perf_table

#Question 8: Performance Evaluation of All Constructed models
perf_table_test <- perf_table
perf_table_test

ann_bagging_othercomp <- c(0.657214, 0.4521863)
perf_table_test[6,] <- ann_bagging_othercomp
perf_table_test

perf_table <- perf_table_test
perf_table

perf_table_final <- perf_table[order(perf_table[,2],decreasing = TRUE),]
perf_table_final

#training time comparison
time_table <- matrix(0, nrow = 9, ncol = 1)
colnames(time_table) <- c("Elapsed Time")
rownames(time_table) <- c("MLR","CART","ANN","Bagging CART",
                          "Random Forests","ANN with iterations","Bagging ANN","AdaBoost","GBM")
time_table

time_table[1,] <- time1[3]
time_table[2,] <- time2[3]
time_table[3,] <- time3[3]
time_table[4,] <- time4[3]
time_table[5,] <- time5[3]
time_table[6,] <- time6[3]
time_table[7,] <- time7[3]
time_table[8,] <- time8[3]
time_table[9,] <- time9[3]
time_table

time_table_final <- time_table[order(time_table[,1],decreasing = TRUE),]
time_table_final

#Extra Question - Regularization by upsampling
Damage_up <- upSample(subset(Final_data, select = -Class),Final_data$Class)
table(Damage_up$Class)

set.seed(12345)
trn_up <- Damage_up[sample(nrow(Damage_up),15044),]
val_up <- Damage_up[sample(nrow(Damage_up),5013),]
tst_up <- Damage_up[sample(nrow(Damage_up),6016),]

input_idx <- c(1:68)
target_idx <- 69

CART_trn_up <- data.frame(trn_up[,input_idx], GradeYN = trn_up[,target_idx])
CART_val_up <- data.frame(val_up[,input_idx], GradeYN = val_up[,target_idx])
CART_tst_up <- data.frame(tst_up[,input_idx], GradeYN = tst_up[,target_idx])

RF.ensem <- randomForest(GradeYN ~., data = CART_trn_up, ntree = 300, 
                         mincriterion = best_criterion, min_split = best_split, 
                         maxdepth = max_depth, importance = TRUE, do.trace = TRUE)

RF.ensem.pred <- predict(RF.ensem, newdata = CART_tst_up, type = "class")
RF.ensem.cfm <- table(CART_tst_up$GradeYN,RF.ensem.pred)
RF.ensem.cfm
perf_eval_multi(RF.ensem.cfm)

