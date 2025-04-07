##############################################################
# this source code shows three machine learning techniques to predict houses prices
# Machine learning techniques: linear regression (LR), k Nearest Neighborhood (kNN), and supported vector regression (SVR)
##############################################################

#set the current working directory
setwd("d:/UiO-EnergyInformatics-EI/Teaching2017/WindEnergyForecasting-Assignment2")  

#read CSV file
MyData <- read.csv("HousePriceData.csv", sep=";", header=TRUE) 

#scattplot points
plot(Price ~ Area, data = MyData)  

#linear regression function, taking the form y ~ x, 
#which should be read something like "y as a function of x"

lmOut = lm(Price ~ Area, data = MyData)
lmOut

#plotting the regression line on an existing scatterplot 
abline(lmOut, col='red')  


#################################################
#kNN model
################################################
library (FNN)

AreaTest = seq (70, 200, by=5)

# Look at kNN regression 
knnmodel1 = knn.reg(train=matrix(MyData$Area,ncol=1),test=matrix(AreaTest,ncol=1),y=MyData$Price, k=1)
knnmodel2 = knn.reg(train=matrix(MyData$Area,ncol=1),test=matrix(AreaTest,ncol=1),y=MyData$Price, k=2)

windows()

par(mfrow=c(1,2))
plot(MyData$Area, MyData$Price, main=paste("k =",1), lwd=2, xlab="Area", ylab="Price")
lines(AreaTest, knnmodel1$pred,col="red",lwd=2)

plot(MyData$Area, MyData$Price, main=paste("k =",2), lwd=2, xlab="Area", ylab="Price")
lines(AreaTest,knnmodel2$pred,col="green4",lwd=2)

#################################################
#SVR model
################################################

library(e1071)

#Support vector machine/regression
AreaTest = seq (70, 200, by=5)
df <- data.frame(x = MyData$Area, y = MyData$Price)

#SVR model
svrmodel <- svm(y ~ x, data = df)
predictedPrice = predict(svrmodel, newdata = data.frame(x = AreaTest))

windows()
#scattplot points
plot(Price ~ Area, data = MyData, lwd=4, xlab="Area", ylab="Price")

# Add points for fitted svrmodel
lines(AreaTest, predictedPrice , col = "red", lwd=2 )
