rm(list=ls())

#Date: Oct 8th, 2018
#Purpose: Analysis of Online News Popularity
library(caTools)
library(caret) # Accuracy
library(ggplot2)
library(rpart)
library(rpart.plot)
library(rattle)
library(ROCR)
library(pROC)
library(olsrr)
library(e1071)

#Read data
#Downloading dataset and importing
library(data.table)
temp <- tempfile()
download.file('http://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip', temp)
data  <- data.table(read.table(unzip(zipfile = temp, 
                                     files = 'OnlineNewsPopularity/OnlineNewsPopularity.csv', 
                                     junkpaths = TRUE), header = TRUE, sep = ',', stringsAsFactors = FALSE))

#Using Local Dataset location
data <- read.csv(file="C:/Users/Shivam Pandit/Desktop/Data Science/Individual Project/bank-additional-full.csv", header=TRUE, sep=",")


#summary before cleaning
summary(data)

#############################################
# Exploratory Data Analysis & Cleaning Data:
#############################################
# Check for any missing values:
sum(is.na(data))

#Removing outlier
data=data[!data$n_non_stop_words==1042,]
summary(data)

#Removing non predictive variables 
data <- subset( data, select = -c(url ,is_weekend ) )

#Combining Plots for EDA for visual analysis
par(mfrow=c(2,2))
for(i in 2:length(data)){hist(data[,i],
                              xlab=names(data)[i] , main = paste("[" , i , "]" ,
                                                                 "Histogram of", names(data)[i])  )}

#Converting categorical values from numeric to factor - Weekdays
for (i in 31:37){
  data[,i] <- factor(data[,i])
  
}

#Converting categorical values from numeric to factor - News subjects
for (i in 13:18){
  data[,i] <- factor(data[,i])
}

#check classes of data after transformation
sapply(data, class)

#Checking importance of news subjects(categorical) on shares
for (i in 13:18){
  
  boxplot(log(data$shares) ~ (data[,i]), xlab=names(data)[i] , ylab="shares")
}

#Checking importance of weekdays on news shares
for (i in 31:37){
  
  boxplot(log(data$shares) ~ (data[,i]), xlab=names(data)[i] , ylab="shares")
}


#########################################################
# Sampling the dataset into training data and test data:
#############################################################
splitdata<- sample.split(data,SplitRatio = 0.7)
train_data <- subset(data, splitdata == TRUE)
test_data <- subset(data, splitdata == FALSE)



#Using Linear Model
###################
fit_lm <- lm(shares ~ ., data = train_data)
#plot(fit_lm)

summary(fit_lm)

#Forward stepwise regression to select variables
#################################################
#fit_lmstep <- ols_step_forward_p(fit_lm)

#using stepwise regression
#fit_lmstep <- step(fit_lm)

#Taking important variables
###########################

d2 <- subset( data, select = c(n_tokens_title,timedelta, kw_avg_avg, self_reference_min_shares,
                             kw_min_avg, num_hrefs, kw_max_max, avg_negative_polarity,
                             data_channel_is_entertainment, weekday_is_monday, 
                             LDA_02, kw_min_max, average_token_length, global_subjectivity,
                             kw_max_min, global_rate_positive_words, 
                             n_tokens_content, n_non_stop_unique_tokens,
                             min_positive_polarity, weekday_is_saturday,
                             data_channel_is_lifestyle, kw_avg_max,
                             kw_avg_min, title_sentiment_polarity,
                             num_self_hrefs, self_reference_max_shares,
                             n_tokens_title, LDA_01, kw_min_min, shares) )
summary(d2$shares)
dim(d2)

#########################################################
# Sampling the dataset based on best variables
#############################################################
splitdata<- sample.split(d2,SplitRatio = 0.7)
traindata <- subset(d2, splitdata == TRUE)
testdata <- subset(d2, splitdata == FALSE)


# Now, we fit a model with all the variables;
fit_lmbest <- lm(shares ~ ., data = traindata)
#plot(fit_lm)
summary(fit_lmbest)
layout(matrix(1:4, 2, 2))
plot(fit_lmbest)



#taking log shares to optimize model
####################################
d2$shares <- log(d2$shares)
summary(d2$shares)
fit_lmlog <- lm(shares ~ ., data = traindata)
summary(fit_lmlog)

sum(is.na(d2))

# Define articles with shares larger than 7.244 (median) as popular article
#######################################################################
d2$shares <- as.factor(ifelse(d2$shares > 7.244,1,0))
hist(log(data$shares), col = c("black", "gray") 
     , xlab="Shares", main="Shares Frequency" )

#########################################################
# Sampling the dataset based on best variables
#############################################################
splitdata<- sample.split(d2,SplitRatio = 0.7)
traindata <- subset(d2, splitdata == TRUE)
testdata <- subset(d2, splitdata == FALSE)

##############################
#Implementing Nayive bayes 
###############################
nav.mod <- naiveBayes(shares~.,traindata)
#pred <- predict(nav.mod,testdata)
newsnb.pred<-predict(nav.mod,testdata,type="class" )
newsnb.prob<-predict(nav.mod,testdata,type="raw" )

confusionMatrix(newsnb.pred,testdata$shares)

#ROC Curve for Nayive Bayes
newsnb.roc <- roc(testdata$shares,newsnb.prob[,2])
plot(newsnb.roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="grey", print.thres=TRUE , main="ROC for NB")

############################################
#Implementing KNN
###########################################
kNN3 <- train(shares~., data = traindata, method = "knn", 
              maximize = TRUE,
              trControl = trainControl(method = "cv", number = 10),
              preProcess=c("center", "scale"))
ggplot(kNN3) + geom_line() + geom_smooth() + theme_light()

predictedkNN3 <- predict(kNN3, newdata = testdata)
confusionMatrix(predictedkNN3, testdata$shares)


##########################################################
###       Implementing CART
#########################################################

# Classification and Regression Trees
news.cart<-rpart(shares ~.,traindata,method='class')

par(mfrow=c(1,1))
fancyRpartPlot(news.cart, digits=2, palettes = c("Purples", "Oranges"))

#predict
cart_pred<-predict( news.cart,testdata ,type="class")
cart_prob<-predict( news.cart,testdata ,type="prob")

# Confusion matrix
confusionMatrix(cart_pred, testdata$shares)


#####################
#Correlation matrix
######################
cm <- d2
dim(cm)
for (i in 1:30){
  cm[,i] <- as.numeric(cm[,i])
}

sapply(cm, class)
library(Hmisc)

cormat <- cor(cm,use = "everything",
              method = c("pearson","kendall","spearman"))
plot(cormat)

#visualizing Correllation matrix
library(corrplot)
corrplot(cormat, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)


#########################################
#Implememnting SVM(Support Vector Machine)
#########################################
set.seed(3233)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
svm_Linear <- train(shares ~., data = traindata, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)
svm_pred <- predict(svm_Linear, newdata = testdata)

plot(svm_Linear_Grid)
confusionMatrix(svm_pred, test_data$shares )


#########################
#Implementing C5.0
###########################
#install.packages("C50")
library(C50)
newsc50<-C5.0(shares ~.,traindata,trials=20)
newsc50.pred<-predict(newsc50,testdata,type="class" )
newsc50.prob<-predict(newsc50,testdata,type="prob" )
# Confusion matrix
confusionMatrix(newsc50.pred, testdata$shares)
plot(newsc50)

#ROC Curve for C5.0
newsc50.roc <- roc(testdata$shares,newsc50.prob[,2])
plot(newsc50.roc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="grey", print.thres=TRUE, main="ROC for C5.0")

