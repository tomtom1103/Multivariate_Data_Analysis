install.packages("wordcloud")
install.packages("psych")
library(arules)
library(arulesViz)
library(wordcloud)
library(psych)

#Step1: Data Preprocessing
mooc_dataset <- read.csv("big_student_clear_third_version.csv")
Institute <- mooc_dataset[,c(2)]
Course <- mooc_dataset[,c(3)]
Region <- gsub(" ","",mooc_dataset[,c(10)])
Degree <- gsub(" ","",mooc_dataset[,c(11)])

Transaction_ID <- mooc_dataset[,c(6)]

RawTransactions <- paste(Institute, Course, Region, Degree, sep = '_')
MOOC_transactions <- paste(Transaction_ID, RawTransactions, sep = ' ')
write.csv(MOOC_transactions, file = "MOOC_User_Course.csv", 
          row.names = FALSE, quote = FALSE)

#Step2: Data Reading
#Question 2-1
MOOC_single <- read.transactions("MOOC_User_Course.csv", format = "single", 
                                 header = TRUE, cols = c(1,2), 
                                 rm.duplicates = TRUE, skip = 1)
summary(MOOC_single)
str(MOOC_single)

#Question 2-2 Wordcloud
itemName <- itemLabels(MOOC_single)
itemCount <- itemFrequency(MOOC_single)*nrow(MOOC_single)
wordcloud(words = itemName, freq = itemCount, min.freq = 700, 
          scale = c(1.2,0.2), col = brewer.pal(8,"Dark2"), random.order = FALSE)

#Question 2-3 Bar chart
itemFrequencyPlot(MOOC_single, support = 0.01, cex.names = 0.8)
itemFrequencyPlot(MOOC_single, topN = 5, type = "absolute", cex.names = 0.8)

#Step3: Generate Rules and Interpret Results
#Question 3-1
support_rule <- c(0.0005, 0.0015, 0.002, 0.0025)
confidence_rule <- c(0.005, 0.01, 0.1)

matrix_rules <- matrix(0,4,3)
rownames(matrix_rules) <- paste0("Support = ",support_rule)
colnames(matrix_rules) <- paste0("Confidence = ",confidence_rule)
matrix_rules

start.time <- proc.time()
for(i in 1:4){
  for(j in 1:3){
    tmp_a <- support_rule[i]
    tmp_b <- confidence_rule[j]
    cat("Support:",support_rule[i],",Confidence:",confidence_rule[j],"\n")
    
    rules_tmp <- apriori(MOOC_single, parameter = list(support = tmp_a, confidence = tmp_b))
    rules_tmp <- data.frame(length(rules_tmp), tmp_a, tmp_b)
    
    tmp_cnt <- rules_tmp[,1]
    matrix_rules[i,j] <- tmp_cnt
  }
}
end.time <- proc.time()
time <- end.time - start.time
time

matrix_rules

#Question 3-2: fixed hyperparameters
fixed_h <- apriori(MOOC_single, parameter = list(support = 0.001, confidence = 0.05))
inspect(fixed_h)
inspect(sort(fixed_h, by = "support"))
inspect(sort(fixed_h, by = "confidence"))
inspect(sort(fixed_h, by = "lift"))
fixed_h
str(fixed_h)

#support x confidence x lift
fixed_h_df <- DATAFRAME(fixed_h)
fixed_h_df$Perf_New <- fixed_h_df$support * fixed_h_df$confidence * fixed_h_df$lift
fixed_h_df <- fixed_h_df[order(fixed_h_df[,8],decreasing = T),]
fixed_h_df

plot(fixed_h, method = "graph")
rules <- subset(fixed_h, lhs %pin% c("HarvardX_CS50x_UnitedStates_Master's"))
inspect(rules)
rules1 <- subset(fixed_h, lhs %pin% c("MITx_6.00x_UnitedStates_Master's"))
inspect(rules1)

rules2 <- subset(fixed_h, lhs %pin% c("MITx_6.00x_UnitedStates_Secondary"))
inspect(rules2)
rules3 <- subset(fixed_h, lhs %pin% c("MITx_6.002x_UnitedStates_Secondary"))
inspect(rules3)

rules4 <- subset(fixed_h, lhs %pin% c("MITx_6.002x_India_Bachelor's"))
inspect(rules4)
rules5 <- subset(fixed_h, lhs %pin% c("MITx_6.00x_India_Bachelor's"))
inspect(rules5)


#Extra Question
plot(fixed_h, method = "grouped")
plot(fixed_h, method = "matrix", engine = "3d", measure = "lift")
plot(fixed_h, measure = c("support", "lift"), shading = "confidence", jitter = 0)
plot(fixed_h, method = "two-key plot")
plot(fixed_h, method = "paracoord")
