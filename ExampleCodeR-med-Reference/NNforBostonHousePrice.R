# Clear workspace
rm(list=ls())

library(neuralnet)

normalize<-function(x)
{return ((x-min(x))/(max(x)-min(x)))}



# Function that returns Root Mean Squared Error
rmse <- function(error)
{
  sqrt(mean(error^2))
}


#set the current working directory
setwd("d:/UiO-EnergyInformatics-EI/Teaching2017/CodinginRExamples")  

#read CSV file

boston <- read.csv('Boston_House.csv', stringsAsFactors = F)

#boston <- boston[-1]

normalize <- function(x) 
{
  return ((x - min(x)) / (max(x) - min(x)))
}

#normalize the data in the range from 0 to 1
boston <- as.data.frame(lapply(boston, normalize))

boston <- as.data.frame(boston)

#75% of all data are training data; the rest data are test data
maxIndex <- length(boston$MEDV)
sliceIndex <- round(maxIndex * 0.75)
trainingData <- boston[1:sliceIndex,]
testData <- boston[sliceIndex:maxIndex,]

#calling neuralnet function to build the model. The function will reurn a neural network objec that can make predictions

bostonModel <-
  neuralnet(MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + LSTAT
            , data = trainingData, hidden = c(5,4,3))

#generate the prediction on the test data
modelResults <- compute(bostonModel, testData[1:12])

#calculate the correlation between the predicted concrete strength and the true value
cor(modelResults$net.result, testData$MEDV)

# Calculate rmse
errorNN <- modelResults$net.result - testData$MEDV
rmse(errorNN)


# plot the neural networks
plot(bostonModel)
