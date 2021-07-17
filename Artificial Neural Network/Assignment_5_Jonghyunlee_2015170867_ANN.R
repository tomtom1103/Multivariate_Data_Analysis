install.packages("nnet")
install.packages("tidyverse")
install.packages("psych")
install.packages("glmnet")
install.packages("GA")
install.packages("tree")
install.packages("party")
install.packages("ROCR")
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

damage <- read.csv("train_values.csv")

##Question 1##
#Check type of variables
str(damage)

#Plot Bar plot for not numeric variable
id_idx <- 1
cha_idx <- c(9,10,11,12,13,14,15,27)
character <- damage[,cha_idx]

a <- names(character)

for(i in 1:length(character)){
  tmp_var <- character[,i]
  plot <- ggplot(damage, aes(x=tmp_var))+geom_bar()
  print(plot)
  b <- paste0(a[i],".png")
  ggsave(b)
}

##Question 2##
#Convert non-numeric variables to binary

frame <- 0

for(i in 1:length(character)){
  tmp <- class.ind(character[,i])
  for(j in 1:ncol(tmp)){
    colnames(tmp)[j]<-paste0(a[i],"_",colnames(tmp)[j]) 
  }
  assign(paste0("dummy_",a[i]),tmp)
  frame <- cbind(frame,tmp)
}

zero_idx <- 1
dummy_data <- frame[,-c(zero_idx)]

#Data Preprocessing
#Performance Evaluation
perf_eval_multi <- function(cm){
  #Simple Accuracy
  ACC = sum(diag(cm))/sum(cm)
  #Balanced Correction Rate
  BCR = 1
  for(i in 1:dim(cm)[1]){
    BCR = BCR * (cm[i,i]/sum(cm[i,]))
  }
  BCR = BCR^(1/dim(cm)[1])
  return(c(ACC,BCR))
}

#Combine numeric variables and dummy variables
Final_data <- cbind(damage[,-c(id_idx,cha_idx)],dummy_data)
#Target Variable
tar_idx <- 31
target <- Final_data[,tar_idx]
EQ_target <- as.factor(target)
#numeric input
num_idx <- c(1:7,19)
num_input <- scale(Final_data[,num_idx], center = TRUE, scale = TRUE)
#binary input
bin_idx <- c(8:18,20:30)
bin_input <- lapply(Final_data[,bin_idx], factor)
#dummy input
dummy_input <- Final_data[,-c(num_idx,bin_idx,tar_idx)]
#Final Input Variable
EQ_input <- data.frame(num_input,bin_input,dummy_input)
EQ_data_normalized <- data.frame(EQ_input, Class = EQ_target)

#Split the data
set.seed(12345)
trn <- EQ_data_normalized[sample(nrow(EQ_data_normalized),150000),]
val <- EQ_data_normalized[sample(nrow(EQ_data_normalized),50000),]
tst <- EQ_data_normalized[sample(nrow(EQ_data_normalized),60602),]

input_idx <- c(1:68)
target_idx <- 69

##Question3##
#Train ANN(Artificial Neural Network)
ANN_trn_input <- trn[,input_idx]
ANN_trn_target <- class.ind(trn[,target_idx])

ANN_val_input <- val[,input_idx]
ANN_val_target <- class.ind(val[,target_idx])

#Find the best number of hidden nodes in terms of BCR
#Candidate hidden nodes
gs <- list(nH = seq(from = 5, to = 35, by = 5),
           maxit = seq(from = 100, to =300 , by = 100)) %>%
  cross_df()

val_perf <- matrix(0,nrow(gs),4)

for(i in 1:nrow(gs)){
  cat("Training ANN: the number of hidden nodes:",gs$nH[i],",maxit:",gs$maxit[i],"\n")
  evaluation <- c()
  
  # Training the model
  trn_input <- ANN_trn_input
  trn_target <- ANN_trn_target
  tmp_nnet <- nnet(trn_input,trn_target,size = gs$nH[i],maxit = gs$maxit[i],silent = TRUE,MaxNWts = 10000)
  
  #Evaluate the model
  val_input <- ANN_val_input
  val_target <- ANN_val_target
  
  real <- max.col(val_target)
  pred <- max.col(predict(tmp_nnet,val_input))
  evaluation <- rbind(evaluation,cbind(real,pred))
  #Confusion Matrix
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
  
  val_perf[i,1] <- gs$nH[i]
  val_perf[i,2] <- gs$maxit[i]
  val_perf[i,3:4] <- t(perf_eval_multi(cfm))
}

#Check best and worst combination of ANN
best_val_perf <- val_perf[order(val_perf[,4],decreasing = TRUE),]
colnames(best_val_perf) <- c("nH","Maxit","ACC","BCR")
best_val_perf

worst_val_perf <- val_perf[order(val_perf[,4],decreasing = FALSE),]
colnames(worst_val_perf) <- c("nH","Maxit","ACC","BCR")
worst_val_perf

##Question 4##
best_nH <- best_val_perf[1,1]
best_maxit <- best_val_perf[1,2]

rang = c(0.3,0.5,0.9)
val_perf_rang = matrix(0,length(rang),3)

for(i in 1:length(rang)){
  evaluation <- c()
  
  trn_input <- ANN_trn_input
  trn_target <- ANN_trn_target
  tmp_nnet <- nnet(trn_input,trn_target,size = best_nH, maxit = best_maxit, rang = rang[i], MaxNWts = 10000)
  
  val_input <- ANN_val_input
  val_target <- ANN_val_target
  evaluation <- rbind(evaluation, cbind(max.col(val_target),
                                        max.col(predict(tmp_nnet,val_input))))
  
  cfm <- table(evaluation[,1],evaluation[,2])
  val_perf_rang[i,1] <- rang[i]
  val_perf_rang[i,2:3] <- t(perf_eval_multi(cfm))
}

best_val_perf_rang <- val_perf_rang[order(val_perf_rang[,3],decreasing = TRUE),]
colnames(best_val_perf_rang) <- c("rang","ACC","BCR")
best_val_perf_rang

best_rang <- best_val_perf_rang[1,1]

##Question 5##
#Combine training and validation dataset
nnet_trn <- rbind(trn,val)
nnet_input <- nnet_trn[,input_idx]
nnet_target <- class.ind(nnet_trn[,target_idx])

#Test the ANN
tst_input <- tst[,input_idx]
tst_target <- class.ind(tst[,target_idx])

val_perf_final <- matrix(0,10,6)
colnames(val_perf_final) <- c("iteration","nH","maxit","rang","ACC","BCR")
val_perf_final

for (i in c(1:10)){
  cat("Training ANN with optimal parameters : iteration ",i,'\n')
  eval_val <- c()
  
  tmp_nnet <- nnet(nnet_input, nnet_target, size = best_nH, decay = 5e-4, maxit = best_maxit, MaxNWts = 10000, rang = best_rang)
  
  eval_val <- rbind(eval_val, cbind(max.col(tst_target),
                                    max.col(predict(tmp_nnet, tst_input))))
  
  cfm <- matrix(0, 3, 3)
  
  cfm[1,1] <- length(which(eval_val[,1] == 1 & eval_val[,2] == 1))
  cfm[1,2] <- length(which(eval_val[,1] == 1 & eval_val[,2] == 2))
  cfm[1,3] <- length(which(eval_val[,1] == 1 & eval_val[,2] == 3))
  cfm[2,1] <- length(which(eval_val[,1] == 2 & eval_val[,2] == 1))
  cfm[2,2] <- length(which(eval_val[,1] == 2 & eval_val[,2] == 2))
  cfm[2,3] <- length(which(eval_val[,1] == 2 & eval_val[,2] == 3))
  cfm[3,1] <- length(which(eval_val[,1] == 3 & eval_val[,2] == 1))
  cfm[3,2] <- length(which(eval_val[,1] == 3 & eval_val[,2] == 2))
  cfm[3,3] <- length(which(eval_val[,1] == 3 & eval_val[,2] == 3))
  
  val_perf_final[i,1] <- i
  val_perf_final[i,2] <- best_nH
  val_perf_final[i,3] <- best_maxit
  val_perf_final[i,4] <- best_rang
  
  perf_eval_multi(cfm)
  val_perf_final[i,5:6] <- perf_eval_multi(cfm)
}

val_perf_final
val_perf_final <- val_perf_final[order(val_perf_final[,6], decreasing = T),]
val_perf_final


check_volitality_BCR <- val_perf_final[,c("BCR")]
describe(check_volitality_BCR)
check_volitality_ACC <- val_perf_final[,c("ACC")]
describe(check_volitality_ACC)

##Q6##
fit_BCR <- function(string){
  sel_var_idx <- which(string == 1)
  sel_x <- nnet_input[,sel_var_idx]
  eval_val <- c()
  GA_ann <- nnet(sel_x, nnet_target, size = best_nH, decay = 5e-4, 
                 maxit = best_maxit, MaxNWts = 10000, rang = best_rang)
  eval_val <- rbind(eval_val, cbind(max.col(tst_target),
                                    max.col(predict(GA_ann, tst_input))))
  
  cfm <- matrix(0, 3, 3)
  
  cfm[1,1] <- length(which(eval_val[,1] == 1 & eval_val[,2] == 1))
  cfm[1,2] <- length(which(eval_val[,1] == 1 & eval_val[,2] == 2))
  cfm[1,3] <- length(which(eval_val[,1] == 1 & eval_val[,2] == 3))
  cfm[2,1] <- length(which(eval_val[,1] == 2 & eval_val[,2] == 1))
  cfm[2,2] <- length(which(eval_val[,1] == 2 & eval_val[,2] == 2))
  cfm[2,3] <- length(which(eval_val[,1] == 2 & eval_val[,2] == 3))
  cfm[3,1] <- length(which(eval_val[,1] == 3 & eval_val[,2] == 1))
  cfm[3,2] <- length(which(eval_val[,1] == 3 & eval_val[,2] == 2))
  cfm[3,3] <- length(which(eval_val[,1] == 3 & eval_val[,2] == 3))
  GA_perf <- perf_eval_multi(cfm)
  return(GA_perf[2])
}

x <- nnet_input
y <- nnet_target

#GA1
start_time <- proc.time()
GA_BCR_1 <- ga(type = 'binary', fitness = fit_BCR, nBits = ncol(x), names = colnames(x), 
               popSize = 10, pcrossover = 0.5, pmutation = 0.01,
               maxiter = 10, elitism = 1, seed = 123)
end_time <- proc.time()

GA_time_1 <- end_time - start_time
best_idx_1 <- which(GA_BCR_1@solution[1,] == 1)
best_idx_1

#GA2
start_time <- proc.time()
GA_BCR_2 <- ga(type = 'binary', fitness = fit_BCR, nBits = ncol(x), names = colnames(x), 
               popSize = 10, pcrossover = 0.5, pmutation = 0.01,
               maxiter = 10, elitism = 1, seed = 1234)
end_time <- proc.time()

GA_time_2 <- end_time - start_time
best_idx_2 <- which(GA_BCR_2@solution[1,] == 1)
best_idx_2

#GA3
start_time <- proc.time()
GA_BCR_3 <- ga(type = 'binary', fitness = fit_BCR, nBits = ncol(x), names = colnames(x), 
               popSize = 10, pcrossover = 0.5, pmutation = 0.01,
               maxiter = 10, elitism = 1, seed = 12345)
end_time <- proc.time()

GA_time_3 <- end_time - start_time
best_idx_3 <- which(GA_BCR_3@solution[1,] == 1)
best_idx_3

#GA all variables
GA_result <- rbind(GA_BCR_1@solution, GA_BCR_2@solution, GA_BCR_3@solution)
all_GA <- rep(0, ncol(GA_result))
GA_result<-rbind(GA_result, all_GA)
rownames(GA_result) <- c("GA_1","GA_2","GA_3", 'summary')

for(i in 1:ncol(GA_result)){
  GA_result[4,i] <- sum(GA_result[1:3,i])
}

GA_result


##Q7##
unused_idx <- c()
for (i in 1:ncol(GA_result)){
  if (GA_result[4,i] == 0){
    unused_idx <- append(unused_idx, i, after = length(unused_idx))
  }
}

unused_idx

eval_val <- c()

reduced_ann <- nnet(nnet_input[,-unused_idx],nnet_target, size = best_nH, 
                    decay = 5e-4, maxit = best_maxit, MaxNWts = 10000, rang = best_rang)
eval_val <- rbind(eval_val, cbind(max.col(tst_target), 
                                  max.col(predict(reduced_ann, tst_input[,-unused_idx]))))

r_cfm <- matrix(0, 3, 3)

r_cfm[1,1] <- length(which(eval_val[,1] == 1 & eval_val[,2] == 1))
r_cfm[1,2] <- length(which(eval_val[,1] == 1 & eval_val[,2] == 2))
r_cfm[1,3] <- length(which(eval_val[,1] == 1 & eval_val[,2] == 3))
r_cfm[2,1] <- length(which(eval_val[,1] == 2 & eval_val[,2] == 1))
r_cfm[2,2] <- length(which(eval_val[,1] == 2 & eval_val[,2] == 2))
r_cfm[2,3] <- length(which(eval_val[,1] == 2 & eval_val[,2] == 3))
r_cfm[3,1] <- length(which(eval_val[,1] == 3 & eval_val[,2] == 1))
r_cfm[3,2] <- length(which(eval_val[,1] == 3 & eval_val[,2] == 2))
r_cfm[3,3] <- length(which(eval_val[,1] == 3 & eval_val[,2] == 3))
r_cfm
reduced_performance <- matrix(0,1,2)
colnames(reduced_performance) <- c("ACC","BCR")
reduced_performance[1,] <- perf_eval_multi(r_cfm)
reduced_performance

##Q8##
tree_trn <- trn
tree_val <- val
tree_tst <- tst

min_criterion = c(0.3,0.4,0.6)
min_split = c(2000,3000,4000)
max_depth = c(5,6,7)
eq_pre_search_result = matrix(0,length(min_criterion)*length(min_split)*length(max_depth),6)
colnames(eq_pre_search_result) <- c("min_criterion", "min_split", "max_depth", "ACC", "BCR", "N_leaves")

iter_cnt = 1

for (i in 1:length(min_criterion)){
  for (j in 1:length(min_split)){
    for (k in 1:length(max_depth)){
      
      cat("CART Min criterion:", min_criterion[i], ", Min split:", min_split[j], ", Max depth:", max_depth[k], "\n")
      tmp_control = ctree_control(mincriterion = min_criterion[i], minsplit = min_split[j], maxdepth = max_depth[k])
      tmp_tree <- ctree(Class ~ ., data = tree_trn, controls = tmp_control)
      tmp_tree_val_prediction <- predict(tmp_tree, newdata = tree_val)
      
      # Confusion matrix for the validation dataset
      tmp_tree_val_cm <- matrix(0,3,3)
      
      tmp_tree_val_cm[1,1] <- length(which(tree_val[,ncol(tree_val)] == 1 & tmp_tree_val_prediction == 1))
      tmp_tree_val_cm[1,2] <- length(which(tree_val[,ncol(tree_val)] == 1 & tmp_tree_val_prediction == 2))
      tmp_tree_val_cm[1,3] <- length(which(tree_val[,ncol(tree_val)] == 1 & tmp_tree_val_prediction == 3))
      tmp_tree_val_cm[2,1] <- length(which(tree_val[,ncol(tree_val)] == 2 & tmp_tree_val_prediction == 1))
      tmp_tree_val_cm[2,2] <- length(which(tree_val[,ncol(tree_val)] == 2 & tmp_tree_val_prediction == 2))
      tmp_tree_val_cm[2,3] <- length(which(tree_val[,ncol(tree_val)] == 2 & tmp_tree_val_prediction == 3))
      tmp_tree_val_cm[3,1] <- length(which(tree_val[,ncol(tree_val)] == 3 & tmp_tree_val_prediction == 1))
      tmp_tree_val_cm[3,2] <- length(which(tree_val[,ncol(tree_val)] == 3 & tmp_tree_val_prediction == 2))
      tmp_tree_val_cm[3,3] <- length(which(tree_val[,ncol(tree_val)] == 3 & tmp_tree_val_prediction == 3))
      
      # parameters
      eq_pre_search_result[iter_cnt,1] = min_criterion[i]
      eq_pre_search_result[iter_cnt,2] = min_split[j]
      eq_pre_search_result[iter_cnt,3] = max_depth[k]
      # Performances from the confusion matrix
      eq_pre_search_result[iter_cnt,4:5] = perf_eval_multi(tmp_tree_val_cm)
      # Number of leaf nodes
      eq_pre_search_result[iter_cnt,6] = length(nodes(tmp_tree, unique(where(tmp_tree))))
      iter_cnt = iter_cnt + 1
    }
  }
}

eq_pre_search_result <- eq_pre_search_result[order(eq_pre_search_result[,5], decreasing = T),]
eq_pre_search_result

#Best hyperparameters
eq_best_criterion <- eq_pre_search_result[1,1]
eq_best_split <- eq_pre_search_result[1,2]
eq_best_depth <- eq_pre_search_result[1,3]

#Classification Tree test data testing
eq_tree_contol <- ctree_control(mincriterion = eq_best_criterion, 
                                minsplit = eq_best_split, maxdepth = eq_best_depth)

tree_newtrn <- rbind(tree_trn, tree_val)

eq_tree_opt <- ctree(Class ~., data = tree_newtrn, controls = eq_tree_contol)
opt_tree_tst_prediction <- predict(eq_tree_opt, newdata = tree_tst)
opt_tree_tst_cm <- matrix(0,3,3)

opt_tree_tst_cm[1,1] <- length(which(tree_tst[,ncol(tree_tst)] == 1 & opt_tree_tst_prediction == 1))
opt_tree_tst_cm[1,2] <- length(which(tree_tst[,ncol(tree_tst)] == 1 & opt_tree_tst_prediction == 2))
opt_tree_tst_cm[1,3] <- length(which(tree_tst[,ncol(tree_tst)] == 1 & opt_tree_tst_prediction == 3))
opt_tree_tst_cm[2,1] <- length(which(tree_tst[,ncol(tree_tst)] == 2 & opt_tree_tst_prediction == 1))
opt_tree_tst_cm[2,2] <- length(which(tree_tst[,ncol(tree_tst)] == 2 & opt_tree_tst_prediction == 2))
opt_tree_tst_cm[2,3] <- length(which(tree_tst[,ncol(tree_tst)] == 2 & opt_tree_tst_prediction == 3))
opt_tree_tst_cm[3,1] <- length(which(tree_tst[,ncol(tree_tst)] == 3 & opt_tree_tst_prediction == 1))
opt_tree_tst_cm[3,2] <- length(which(tree_tst[,ncol(tree_tst)] == 3 & opt_tree_tst_prediction == 2))
opt_tree_tst_cm[3,3] <- length(which(tree_tst[,ncol(tree_tst)] == 3 & opt_tree_tst_prediction == 3))

opt_tree_tst_cm
opt_tree_performance <- matrix(0,1,2)
colnames(opt_tree_performance) <- c('ACC','BCR')
opt_tree_performance[1,] <- perf_eval_multi(opt_tree_tst_cm)
opt_tree_performance

#plotting optimal tree
plot(eq_tree_opt, type = 'simple')

##Q9##
ml_newtrn <- tree_newtrn
ml_newtrn[,ncol(ml_newtrn)] <- relevel(ml_newtrn[,ncol(ml_newtrn)], ref = '1')
ml_tst <- tree_tst
ml_tst[,ncol(ml_tst)] <- relevel(ml_tst[,ncol(ml_tst)], ref = '1')

ml_logit <- multinom(Class ~ ., data = ml_newtrn)
ml_logit_prey <- predict(ml_logit, newdata = ml_tst)
ml_logit_cm <- table(ml_tst$Class, ml_logit_prey)

ml_logit_performance <- matrix(0,1,2)
colnames(ml_logit_performance) <- c("ACC", "BCR")
ml_logit_performance[1,] <- perf_eval_multi(ml_logit_cm)
ml_logit_performance

#All algorithms
total_perf <- matrix(0,2,4)
colnames(total_perf) <- c('ANN', 'ANN(Selected by GA)', 'Decision Tree', 'Multinomial LogReg')
rownames(total_perf) <- c('Accuracy', 'BCR')
total_perf[,1] <- val_perf_final[1,5:6]
total_perf[,2] <- reduced_performance
total_perf[,3] <- opt_tree_performance
total_perf[,4] <- ml_logit_performance
total_perf

