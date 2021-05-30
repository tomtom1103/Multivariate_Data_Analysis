install.packages("dplyr")
library("psych")
library("ggplot2")
library("corrplot")
library("dplyr")


#Performance Evaluation Function
perf_eval <- function(cm){
  
  # True positive rate: TPR (Recall)
  TPR <- cm[2,2]/sum(cm[2,])
  # Precision
  PRE <- cm[2,2]/sum(cm[,2])
  # True negative rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  # Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  # Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  # F1-Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(TPR, PRE, TNR, ACC, BCR, F1))
}

# Initializing the performance matrix
perf_mat <- matrix(0, 1, 6)
colnames(perf_mat) <- c("TPR (Recall)", "Precision", "TNR", "ACC", "BCR", "F1")
rownames(perf_mat) <- "Logstic Regression"

#Main dataset
predicthd <- read.csv("framingham.csv")
input_idx <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
target_idx <- 16
predicthd_input <- predicthd[,input_idx]
predicthd_input <- scale(predicthd_input, center = TRUE, scale = TRUE)
predicthd_target <- as.factor(predicthd[,target_idx])
predicthd_data <- data.frame(predicthd_input, predicthd_target)

describe(predicthd_data)


#Q3 (without nominal variables)
predicthd2 <- read.csv("framingham.csv")
input_idx2 <- c(2,3,5,10,11,12,13,14,15)
target_idx2 <- 16
predicthd_input2 <- predicthd2[,input_idx2]
predicthd_target2 <- as.factor(predicthd2[,target_idx2])
predicthd_data2 <- data.frame(predicthd_input2, predicthd_target2)

describe(predicthd_data2)

#normalizing data
predicthd_input2 <- predicthd2[,input_idx2]
predicthd_input2 <- scale(predicthd_input2, center = TRUE, scale = TRUE)
predicthd_target2 <- predicthd2[,target_idx2]
predicthd_data2_norm <- data.frame(predicthd_input2, predicthd_target2)

describe(predicthd_data2_norm)

#boxplot normalized
boxplot(predicthd_data2_norm)
boxplot(predicthd_data2_norm$age,xlab = "Age")
boxplot(predicthd_data2_norm$education,xlab = "Education")
boxplot(predicthd_data2_norm$cigsPerDay,xlab = "Cigarettes per day")
boxplot(predicthd_data2_norm$totChol,xlab = "Total cholesterol")
boxplot(predicthd_data2_norm$sysBP,xlab = "sysBP")
boxplot(predicthd_data2_norm$diaBP,xlab = "diaBP")
boxplot(predicthd_data2_norm$BMI,xlab = "BMI")
boxplot(predicthd_data2_norm$heartRate,xlab = "Heartrate")
boxplot(predicthd_data2_norm$glucose,xlab = "Glucose")

#Q4

#Removing outliers, and checking
#totChol
Q1 = quantile(predicthd_data2$totChol, probs = c(0.25),na.rm = TRUE)
Q3 = quantile(predicthd_data2$totChol, probs = c(0.75),na.rm = TRUE)
IQR = Q3 - Q1

LC = Q1 - 1.5 * IQR
UC = Q3 + 1.5 * IQR

predicthd_data_outliersremoved = subset(predicthd_data2, totChol >= LC & totChol <= UC)

par(mfrow = c(2,1))
boxplot(predicthd_data2$totChol, xlab = "totChol, Before removing outliers", horizontal = TRUE)
boxplot(predicthd_data_outliersremoved$totChol, xlab = "totChol, After removing outliers", horizontal = TRUE)


#sysBP
Q1 = quantile(predicthd_data2$sysBP, probs = c(0.25),na.rm = TRUE)
Q3 = quantile(predicthd_data2$sysBP, probs = c(0.75),na.rm = TRUE)
IQR = Q3 - Q1

LC = Q1 - 1.5 * IQR
UC = Q3 + 1.5 * IQR

predicthd_data_outliersremoved = subset(predicthd_data2, sysBP >= LC & sysBP <= UC)

par(mfrow = c(2,1))
boxplot(predicthd_data2$sysBP, xlab = "sysBP, Before removing outliers", horizontal = TRUE)
boxplot(predicthd_data_outliersremoved$sysBP, xlab = "sysBP, After removing outliers", horizontal = TRUE)

#diaBP
Q1 = quantile(predicthd_data2$diaBP, probs = c(0.25),na.rm = TRUE)
Q3 = quantile(predicthd_data2$diaBP, probs = c(0.75),na.rm = TRUE)
IQR = Q3 - Q1

LC = Q1 - 1.5 * IQR
UC = Q3 + 1.5 * IQR

predicthd_data_outliersremoved = subset(predicthd_data2, diaBP >= LC & diaBP <= UC)

par(mfrow = c(2,1))
boxplot(predicthd_data2$diaBP, xlab = "diaBP, Before removing outliers", horizontal = TRUE)
boxplot(predicthd_data_outliersremoved$diaBP, xlab = "diaBP, After removing outliers", horizontal = TRUE)

#BMI
Q1 = quantile(predicthd_data2$BMI, probs = c(0.25),na.rm = TRUE)
Q3 = quantile(predicthd_data2$BMI, probs = c(0.75),na.rm = TRUE)
IQR = Q3 - Q1

LC = Q1 - 1.5 * IQR
UC = Q3 + 1.5 * IQR

predicthd_data_outliersremoved = subset(predicthd_data2, BMI >= LC & BMI <= UC)

par(mfrow = c(2,1))
boxplot(predicthd_data2$BMI, xlab = "BMI, Before removing outliers", horizontal = TRUE)
boxplot(predicthd_data_outliersremoved$BMI, xlab = "BMI, After removing outliers", horizontal = TRUE)

#glucose
Q1 = quantile(predicthd_data2$glucose, probs = c(0.25),na.rm = TRUE)
Q3 = quantile(predicthd_data2$glucose, probs = c(0.75),na.rm = TRUE)
IQR = Q3 - Q1

LC = Q1 - 1.5 * IQR
UC = Q3 + 1.5 * IQR

predicthd_data_outliersremoved = subset(predicthd_data2, glucose >= LC & glucose <= UC)

par(mfrow = c(2,1))
boxplot(predicthd_data2$glucose, xlab = "glucose, Before removing outliers", horizontal = TRUE)
boxplot(predicthd_data_outliersremoved$glucose, xlab = "glucose, After removing outliers", horizontal = TRUE)

#final check
describe(predicthd_data2)
describe(predicthd_data_outliersremoved)

#Q5-1
new_idx <- c(10)
predicthd_data_outliersremoved_new <- predicthd_data_outliersremoved[-new_idx]
pairs(~age+education+cigsPerDay+totChol+sysBP+diaBP+BMI+heartRate+glucose,data=predicthd_data_outliersremoved_new)
ChdCor <- cor(predicthd_data_outliersremoved_new)
ChdCor
corrplot(ChdCor,method = "number")

#Q5-2
new_idx2 <- c(5,6)
predicthd_data_outliersremoved_new2 <-predicthd_data_outliersremoved_new[-new_idx2]
describe(predicthd_data_outliersremoved_new2)

#Q6-1,2
set.seed(12345)
trn_idx <- sample(1:nrow(predicthd_data), round(0.7*nrow(predicthd_data)))
predicthd_trn <- predicthd_data[trn_idx,]
predicthd_tst <- predicthd_data[-trn_idx,]

full_lr <- glm(predicthd_target ~ ., family=binomial, data = predicthd_trn)
summary(full_lr)

#Q6-3
lr_response <- predict(full_lr, type = "response", newdata = predicthd_tst)
lr_target <- predicthd_tst$predicthd_target
lr_predicted <- rep(0, length(lr_target))
lr_predicted[which(lr_response >= 0.2)] <- 1
cm_full <- table(lr_target, lr_predicted)
cm_full

#Performance Evaluation Function
perf_eval <- function(cm){
  
  # True positive rate: TPR (Recall)
  TPR <- cm[2,2]/sum(cm[2,])
  # Precision
  PRE <- cm[2,2]/sum(cm[,2])
  # True negative rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  # Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  # Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  # F1-Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(ACC, BCR, F1))
}

# Initializing the performance matrix
perf_mat <- matrix(0, 1, 3)
colnames(perf_mat) <- c("ACC", "BCR", "F1")
rownames(perf_mat) <- "Logstic Regression"

perf_mat[1,] <- perf_eval(cm_full)
perf_mat

#Q6-4
Clas_perf <- data.frame(lr_response, predicthd_tst$predicthd_target)
Clas_perf_mat<- arrange(Clas_perf, desc(lr_response),predicthd_tst$predicthd_target)
colnames(Clas_perf_mat) <- c("P(Positive)", "TenYearCHD")
Clas_perf_mat

#Define and update TPR and FPR for every iteration
TPR1 <- length(which(Clas_perf_mat$`TenYearCHD`==1))
FPR1 <- length(which(Clas_perf_mat$`TenYearCHD`==0))

TPR_FPR <- cbind(0,0)
colnames(TPR_FPR) <- c("TPR","FPR")

TPR2 = 0 
FPR2 = 0

for(i in 1:nrow(Clas_perf_mat)){
  if(Clas_perf_mat[i,2]==1){
    TPR2 <- TPR2 + 1
  }else{
    FPR2 <- FPR2 + 1
  }
  TPR_tmp <- TPR2/TPR1
  FPR_tmp <- FPR2/FPR1
  TPR_FPR_tmp <- data.frame(TPR_tmp,FPR_tmp)
  colnames(TPR_FPR_tmp) <- c("TPR","FPR")
  TPR_FPR <- rbind(TPR_FPR,TPR_FPR_tmp)
}
TPR_FPR

#ROC table
z <- c("0","0")
Clas_perf_mat2 <- rbind(z,Clas_perf_mat)
Clas_perf_mat2
ROC_table <- data.frame(Clas_perf_mat2,TPR_FPR)
colnames(ROC_table) <- c("P(Positive)","TenYearCHD","True_Positive","False_Positive")
ROC_table

#ROC Curve
ggplot(data = ROC_table, aes(x=`False_Positive`,y=`True_Positive`))+geom_line()+geom_abline(color = "blue")

#AUROC
TPR_FPR %>%
  arrange(FPR) %>%
  mutate(area_rectangle = (lead(FPR)-FPR)*pmin(TPR,lead(TPR)),
         area_triangle = 0.5 * (lead(FPR)-FPR)*abs(TPR-lead(TPR))) %>%
  summarise(area = sum(area_rectangle + area_triangle, na.rm = TRUE))

#Q7-1
forqseven_idx <- c(5,6)
predicthd_data_outliersremoved_qseven <- predicthd_data_outliersremoved[-forqseven_idx]

set.seed(12345)
trn_idx2 <- sample(1:nrow(predicthd_data_outliersremoved_qseven), round(0.7*nrow(predicthd_data_outliersremoved_qseven)))
predicthd_trn2 <- predicthd_data_outliersremoved_qseven[trn_idx2,]
predicthd_tst2 <- predicthd_data_outliersremoved_qseven[-trn_idx2,]

full_lr2 <- glm(predicthd_target2 ~ ., family=binomial, data = predicthd_trn2)
summary(full_lr2)

#Q7-2
lr_response2 <- predict(full_lr2, type = "response", newdata = predicthd_tst2)
lr_target2 <- predicthd_tst2$predicthd_target2
lr_predicted2 <- rep(0, length(lr_target2))
lr_predicted2[which(lr_response2 >= 0.2)] <- 1
cm_full2 <- table(lr_target2, lr_predicted2)
cm_full2

#Performance Evaluation Function
perf_eval2 <- function(cm){
  
  # True positive rate: TPR (Recall)
  TPR <- cm[2,2]/sum(cm[2,])
  # Precision
  PRE <- cm[2,2]/sum(cm[,2])
  # True negative rate: TNR
  TNR <- cm[1,1]/sum(cm[1,])
  # Simple Accuracy
  ACC <- (cm[1,1]+cm[2,2])/sum(cm)
  # Balanced Correction Rate
  BCR <- sqrt(TPR*TNR)
  # F1-Measure
  F1 <- 2*TPR*PRE/(TPR+PRE)
  
  return(c(ACC, BCR, F1))
}

# Initializing the performance matrix
perf_mat2 <- matrix(0, 1, 3)
colnames(perf_mat2) <- c("ACC", "BCR", "F1")
rownames(perf_mat2) <- "Logstic Regression"

perf_mat2[1,] <- perf_eval2(cm_full2)
perf_mat2

#Q7-3
Clas_perf2 <- data.frame(lr_response2, predicthd_tst2$predicthd_target2)
Clas_perf_mat2<- arrange(Clas_perf2, desc(lr_response2),predicthd_tst2$predicthd_target2)
colnames(Clas_perf_mat2) <- c("P(Positive)", "TenYearCHD")
Clas_perf_mat2

#Define and update TPR and FPR for every iteration
TPR1 <- length(which(Clas_perf_mat2$`TenYearCHD`==1))
FPR1 <- length(which(Clas_perf_mat2$`TenYearCHD`==0))

TPR_FPR <- cbind(0,0)
colnames(TPR_FPR) <- c("TPR","FPR")

TPR2 = 0 
FPR2 = 0

for(i in 1:nrow(Clas_perf_mat2)){
  if(Clas_perf_mat2[i,2]==1){
    TPR2 <- TPR2 + 1
  }else{
    FPR2 <- FPR2 + 1
  }
  TPR_tmp <- TPR2/TPR1
  FPR_tmp <- FPR2/FPR1
  TPR_FPR_tmp <- data.frame(TPR_tmp,FPR_tmp)
  colnames(TPR_FPR_tmp) <- c("TPR","FPR")
  TPR_FPR <- rbind(TPR_FPR,TPR_FPR_tmp)
}
TPR_FPR

#ROC table
z <- c("0","0")
Clas_perf_mat3 <- rbind(z,Clas_perf_mat2)
Clas_perf_mat3
ROC_table <- data.frame(Clas_perf_mat3,TPR_FPR)
colnames(ROC_table) <- c("P(Positive)","TenYearCHD","True_Positive","False_Positive")
ROC_table

#ROC Curve
ggplot(data = ROC_table, aes(x=`False_Positive`,y=`True_Positive`))+geom_line()+geom_abline(color = "blue")

#AUROC
TPR_FPR %>%
  arrange(FPR) %>%
  mutate(area_rectangle = (lead(FPR)-FPR)*pmin(TPR,lead(TPR)),
         area_triangle = 0.5 * (lead(FPR)-FPR)*abs(TPR-lead(TPR))) %>%
  summarise(area = sum(area_rectangle + area_triangle, na.rm = TRUE))


