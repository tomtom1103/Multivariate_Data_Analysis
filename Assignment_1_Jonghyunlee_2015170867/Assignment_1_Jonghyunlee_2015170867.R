install.packages("moments")
library(moments)
library(psych)
library(corrplot)

perf_eval_reg <- function(tgt_y, pre_y){
  rmse <- sqrt(mean ((tgt_y - pre_y)^2))
  mae <- mean(abs(tgt_y - pre_y))
  mape <- 100*mean(abs((tgt_y - pre_y)/tgt_y))
  
  return(c(rmse,mae,mape))
}

perf_mat <- matrix(0, nrow = 1, ncol = 3)
rownames(perf_mat) <- c("Admission Predict")
colnames(perf_mat) <- c("RMSE", "MAE", "MAPE")
perf_mat

#Q3
admission <- read.csv("Admission_Predict.csv") 
nStud <- nrow(admission)
nVar <- ncol(admission)
id_idx <- c(1)

admission_data <- cbind(admission[,-c(id_idx)])
describe(admission_data)

scale_admission <- scale(admission_data)
boxplot(scale_admission)
boxplot(admission_data$GRE.Score,xlab = "GRE")
boxplot(admission_data$TOEFL.Score,xlab = "TOEFL")
boxplot(admission_data$University.Rating,xlab = "rating")
boxplot(admission_data$SOP,xlab = "SOP")
boxplot(admission_data$LOR,xlab = "LOR")
boxplot(admission_data$CGPA,xlab = "CGPA")
boxplot(admission_data$Research,xlab = "Research")

#Q5
pairs(~Chance.of.Admit+GRE.Score+TOEFL.Score+University.Rating+SOP+LOR+CGPA+Research,data=admission_data)

AdmCor <- cor(admission_data)
AdmCor
corrplot(AdmCor,method = "number")

#Q6
admission_mlr_data <- cbind(admission[,-c(id_idx)])
set.seed(12345) 
admission_trn_idx <- sample(1:nStud, round(0.7*nStud))
admission_trn_data <- admission_mlr_data[admission_trn_idx,]
admission_val_data <- admission_mlr_data[-admission_trn_idx,]

mlr_admission <- lm(Chance.of.Admit ~ ., data = admission_trn_data)
mlr_admission
summary(mlr_admission)
plot(mlr_admission)

#Q8
mlr_admission_haty <- predict(mlr_admission, newdata = admission_val_data)

perf_mat[1,] <- perf_eval_reg(admission_val_data$Chance.of.Admit, mlr_admission_haty)
perf_mat

#Q10
id_idx2 <- c(2,3,4,5)

admission_data2 <- cbind(admission_data[,-c(id_idx2)])

nStud2 <- nrow(admission_data2)
nVar2 <- ncol(admission_data2)
set.seed(12345)
admission_trn_idx2 <- sample(1:nStud2, round(0.7*nStud2))
admission_trn_data2 <- admission_data2[admission_trn_idx2,]
admission_val_data2 <- admission_data2[-admission_trn_idx2,]
mlr_admission2 <- lm(Chance.of.Admit ~ ., data = admission_trn_data2)

mlr_admission2
summary(mlr_admission2)
plot(mlr_admission2)

perf_mat2 <- matrix(0, nrow=1, ncol=3)

rownames(perf_mat2) <- c("3 Variables")
colnames(perf_mat2) <- c("RMSE", "MAE", "MAPE")
mlr_admission_haty2 <- predict(mlr_admission2, newdata = admission_val_data2)

perf_mat2[1,] <- perf_eval_reg(admission_val_data2$Chance.of.Admit, mlr_admission_haty2)
perf_mat2

#Extra Question
install.packages("car")
library(car)
vif(mlr_admission)
anova(mlr_admission)

