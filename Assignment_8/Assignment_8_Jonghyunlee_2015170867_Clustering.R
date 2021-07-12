install.packages("clValid")
install.packages("plotrix")
install.packages("factoextra")
install.packages("dbscan")
install.packages("ISLR")
install.packages("fpc")
install.packages("pheatmap")
install.packages("NbClust")
install.packages("ggplot2")
library(NbClust)
library(ggfortify)
library(tidyverse)
library(pheatmap)
library(fpc)
library(ISLR)
library(factoextra)
library(dbscan)
library(clValid)
library(plotrix)

#Q1: Data Preparation
College <- read.csv("College.csv")
str(College)

College_class <- College[,1]
College_F <- College[,-1]
College_F <- College_F[,-1]

College_F_scaled <- as.data.frame(scale(College_F, center = TRUE, scale = TRUE))

#Q2: K-Means Clustering
start_time <- proc.time()
College_clValid <- clValid(College_F_scaled, 2:15, clMethods = "kmeans", 
                           validation = c("internal","stability"),maxitems = 10000)
end_time <- proc.time()
time1 <- end_time - start_time
time1

summary(College_clValid)

#Q3: 10 times iteration Centroid and Size
for(i in 1:10){
  cat("Number of Iterations:",i,"\n")
  College_tmp <- kmeans(College_F_scaled, 3)
  
  print(College_tmp$centers)
  print(College_tmp$size)
}

#Q4: Radar Chart
College_kmc <- kmeans(College_F_scaled,3)
kmc_cluster <- College_kmc$cluster
cluster_kmc <- data.frame(College_F_scaled, clusterID = as.factor(College_kmc$cluster))
kmc_summary <- data.frame()

for(i in 1:(ncol(cluster_kmc)-1)){
  kmc_summary = rbind(kmc_summary, tapply(cluster_kmc[,i],
                                          cluster_kmc$clusterID,mean))
}
colnames(kmc_summary) <- paste("cluster", c(1:3))
rownames(kmc_summary) <- colnames(College_F)
kmc_summary

par(mfrow = c(1,3))
for(i in 1:3){
  plot_title <- paste("Radar Chart for Cluster",i, sep = " ")
  radial.plot(kmc_summary[,i], labels = rownames(kmc_summary),
              radial.lim = c(-2,2), rp.type = "p", main = plot_title,
              line.col = "red", lwd = 2, show.grid.labels = 0)
}

#Q5: Comparison of Clusters
kmc_cluster1 <- College_F[College_kmc$cluster == 1,]
kmc_cluster2 <- College_F[College_kmc$cluster == 2,]
kmc_cluster3 <- College_F[College_kmc$cluster == 3,]
kmc_t_result <- data.frame()

for(i in 1:17){
  kmc_t_result[i,1] <- t.test(kmc_cluster1[,i], kmc_cluster2[,i], alternative = "two.sided")$p.value
  kmc_t_result[i,2] <- t.test(kmc_cluster1[,i], kmc_cluster3[,i], alternative = "two.sided")$p.value
  kmc_t_result[i,3] <- t.test(kmc_cluster2[,i], kmc_cluster3[,i], alternative = "two.sided")$p.value
}
rownames(kmc_t_result) <- colnames(College_F)
colnames(kmc_t_result) <- c("cluster1/2","cluster1/3","cluster2/3")
kmc_t_result

##Hierarchical Clustering
#Data Preparation
College_H <- College[,-1]
College_H <- College_H[,-1]
College_H_scaled <- as.data.frame(scale(College_H, center = TRUE, scale = TRUE))

#Q6 - dendrogram
cor_Mat <- cor(t(College_H_scaled), method = "spearman")
dist_College <- as.dist(1-cor_Mat)

hr_s <- hclust(dist_College, method = "single", members = NULL)
plot(hr_s, hang = -1)

hr_c <- hclust(dist_College, method = "complete", members = NULL)
plot(hr_c, hang = -1)


#Q7 - K = 3 Hierarchical Clustering Radar Chart
#single linkage
mycl <- cutree(hr_s, k = 3)
mycl
plot(hr_s, hang = -1)
rect.hclust(hr_s, k=3, border = "red")

College_hs <- data.frame(College_H_scaled, clusterID = as.factor(mycl))
hs_summary <- data.frame()
for (i in 1:(ncol(College_hs)-1)){
  hs_summary = rbind(hs_summary, tapply(College_hs[,i],
                                        College_hs$clusterID, mean))
}
colnames(hs_summary) <- paste("cluster",c(1:3))
rownames(hs_summary) <- colnames(College_H)
hs_summary

par(mfrow = c(1,3))
for(i in 1:3){
  plot_title <- paste("Radar Chart for Singl Linkage Cluster", i, sep = " ")
  radial.plot(hs_summary[,i], labels = rownames(hs_summary),
              radial.lim = c(-2,2), rp.type = "p", main = plot_title,
              line.col = "red", lwd = 3, show.grid.labels = 0)
}

#complete linkage
mycl2 <- cutree(hr_c, k = 3)
mycl2
plot(hr_c, hang = -1)
rect.hclust(hr_c, k=3, border = "red")

College_hc <- data.frame(College_H_scaled, clusterID = as.factor(mycl2))
hc_summary <- data.frame()
for (i in 1:(ncol(College_hc)-1)){
  hc_summary = rbind(hc_summary, tapply(College_hc[,i],
                                        College_hc$clusterID, mean))
}
colnames(hc_summary) <- paste("cluster",c(1:3))
rownames(hc_summary) <- colnames(College_H)
hc_summary

par(mfrow = c(1,3))
for(i in 1:3){
  plot_title <- paste("Radar Chart for Complete Linkage Cluster", i, sep = " ")
  radial.plot(hc_summary[,i], labels = rownames(hc_summary),
              radial.lim = c(-2,2), rp.type = "p", main = plot_title,
              line.col = "red", lwd = 3, show.grid.labels = 0)
}


#Compare K-Means with single and complete linkage
kmc_hs <- kmc_summary - hs_summary
kmc_hc <- kmc_summary - hc_summary

kmc_hs
kmc_hc
mean(colMeans(kmc_hs))
mean(colMeans(kmc_hc))


##DBSCAN
#Data Preparation
College_D <- College[,-1]
College_D <- College_D[,-1]
College_D_scaled <- as.data.frame(scale(College_D, center = TRUE, scale = TRUE))


#Q9 - DBSCAN
options <- list(eps = c(0.1,0.3,0.5,0.7,1,1.3,1.5,1.7,2,2.3,2.5,3,5),
                minPts = c(20,50,100,150,200)) %>%
  cross_df()

for(i in 1:nrow(options)){
  DBSCAN_tmp <- dbscan(College_D_scaled, eps = options$eps[i], MinPts = options$minPts[i])
  
  print(DBSCAN_tmp)
}

#Q11 - Visualization
#DBSCAN
College.pca <- prcomp(College_D, center = TRUE, scale. = TRUE)
print(College.pca)
par(mfrow = c(1,1))
plot(College.pca, type = "l")
summary(College.pca)

DBSCAN_1$clusterID<-as.factor(DBSCAN_1$clusterID)
autoplot(College.pca, data = cluster_db, colour = 'clusterID')

#kmeans
College_k.pca <- prcomp(College_F, center = TRUE, scale. = TRUE)
print(College_k.pca)
par(mfrow = c(1,1))
plot(College_k.pca, type = "l")
summary(College_k.pca)

College_kmc$clusterID<-as.factor(College_kmc$clusterID)
autoplot(College_k.pca, data = cluster_kmc, colour = 'clusterID')

#hierarchical complete
College_hc.pca <- prcomp(College_H, center = TRUE, scale. = TRUE)
print(College_hc.pca)
par(mfrow = c(1,1))
plot(College_hc.pca, type = "l")
summary(College_hc.pca)

College_hc$clusterID<-as.factor(College_hc$clusterID)
autoplot(College_hc.pca, data = College_hc, colour = 'clusterID')


#hierarchical single
College_hs.pca <- prcomp(College_H, center = TRUE, scale. = TRUE)
print(College_hs.pca)
par(mfrow = c(1,1))
plot(College_hs.pca, type = "l")
summary(College_hs.pca)

College_hs$clusterID<-as.factor(College_hs$clusterID)
autoplot(College_hs.pca, data = College_hs, colour = 'clusterID')






