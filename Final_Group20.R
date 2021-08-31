##Stat 331 Final Project by Matthew Hartz

rm(list=ls())

#Set my working directory

setwd("/Users/matthewhartz/Documents/r_practice/stat331/stat331_final")

# Import Libraries
library(caret)
library(cluster)
library(factoextra)
library(DescTools)
library(e1071)



#Load in CSV

df <- read.csv("WQMarketing.csv")


#Get details on Df
str(df)
summary(df)
Abstract(df)

#Plot each variable to further understand it's distribution
Desc(df)


#Check if a certain color wine is more likely to be high quality
white <- df[ which(df$color == 'white'),]
red <- df[which(df$color == 'red'),]

red <- table(red$quality_level)
white <- table(white$quality_level)

plot(red)
plot(white)

#Create shortcut for variables
nums <- c("fixed_acidity", "volatile_acidity", "citric_acid","residual_sugar","chlorides",
          "free_sulfur_dioxide","total_sulfur_dioxide", "density", "pH","sulphates", "alcohol")

vars <- c("fixed_acidity", "volatile_acidity", "citric_acid","residual_sugar","chlorides",
          "free_sulfur_dioxide","total_sulfur_dioxide", "density", "pH","sulphates", "alcohol",
          "color","quality_level")

df$color <- factor(x = df$color,
                   levels = c("red", "white"),
                   ordered = FALSE)

#Create factor for target variable
df$quality_level <- factor(x = df$quality_level,
                    levels = c("High", "Low"),
                    ordered = TRUE)

#Plot Target Variable to Understand its distribution
plot(x = df$quality_level, main = "Quality Distribution")


#Create Correlation Matrix for Variables to check for multicollinearity 
cor(df[,nums])


#Normalize Numerical Data
df_pp <- preProcess(x = df, method = c('center', 'scale'))
df_sc <- predict(object = df_pp, newdata = df)

#Set seed and create Training and Testing subsets
set.seed(205)

sub <- createDataPartition(y = df_sc$quality_level, p = 0.80, list = FALSE)

train <- df_sc[sub, ]

test <- df_sc[-sub, ] 


#Cluster Analysis using HCA

#Claculate Eucleadian distance
df_dist <- dist(x = df_sc[ ,nums])


ward_clust <- hclust(d = df_dist, 
               method = "ward.D2")


# Create the Dendrogram
plot(ward_clust, 
     xlab = NA, sub = NA, 
     main = "Ward's Method")

# Overlay boxes identifying clusters for a k = 6 clustering solution
rect.hclust(tree = ward_clust, 
            k = 6, 
            border = hcl.colors(6))

ward_clust_cutree <- cutree(tree = ward_clust,
                         k = 6)


fviz_cluster(object = list(data = df_sc[ ,nums], 
                           cluster = ward_clust_cutree))


clus_means_HCA <- aggregate(x = df_sc[ ,nums],
                            by = list(ward_clust_cutree),
                            FUN = mean)


matplot(t(clus_means_HCA[ ,-1]), 
        type = "l", 
        ylab = "", 
        xlim = c(0, 11), 
        xaxt = "n", 
        col = 1:6, 
        lty = 1:6, 
        main = "Cluster Centers") 

axis(side = 1, 
     at = 1:11, 
     labels = nums, 
     las = 2) 

#Create legend
legend("left", 
       legend = 1:6, 
       col = 1:6, 
       lty = 1:6, 
       cex = 0.6) 

#Aggregate Clusters to see target variable frequency
aggregate(x = df[,13],
          by = list(ward_clust_cutree),
          FUN = table)

#-----------------------------------------------------------------------------

#Classification of Low and High quality wine using Naive Bayes


#Naive Bayes needs variables to be normally distributed so using centered and scaled dataframe
#We can use boxcox for pp if numerics are all positive


#Create NB model

nb_model <- naiveBayes(x = train[ ,vars],
                     y = train$quality_level,
                     laplace = 1)
#Check Model Output
nb_model


#Crate predicitions for training
nb.train <- predict(object = nb_model, 
                    newdata = train[ ,vars], 
                    type = "class")


train_conf <- confusionMatrix(data = nb.train, 
                              reference = train$quality_level, 
                              positive = "High",
                              mode = "everything")

#Print out confusion matrix for training data
train_conf

nb.test <- predict(object = nb_model, 
                   newdata = test[ ,vars], 
                   type = "class")

test_conf <- confusionMatrix(data = nb.test, 
                             reference = test$quality_level, 
                             positive = "High",
                             mode = "everything")

#print out confusion matrix for Test data
test_conf

#Overall performance
test_conf$overall[c("Accuracy", "Kappa")]

cbind(Training = train_conf$overall,
      Testing = test_conf$overall)

#Class Level Performance

cbind(Training = train_conf$byClass,
      Testing = test_conf$byClass)


#Save R data
save.image(file="/Users/matthewhartz/Documents/r_practice/stat331/stat331_final/Final_Group20.RData")
