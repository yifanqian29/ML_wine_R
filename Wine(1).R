#Read dataset
wine_df= read.csv('C:/STSCI5740/wine-quality-white-and-red.csv')
summary(wine_df)
white=wine_df[wine_df$type=='white',c(-1)]
red=wine_df[wine_df$type=='red',c(-1)]
#library the package
library(class)
library(caret)
library(caTools)
library(e1071)
library(MASS)
library(dplyr)
library(tidyr)
library(randomForest)
library(kknn)
library(kernlab)
library(tree)
library(leaps)
library(nnet)
library(corrplot)
library(RCurl)
library(psych)
library(tidyverse)
library(usethis)
library(lattice)
library(devtools)
library(ggfortify)
library(ggplot2)
library(pROC)
library(naivebayes)

#split dataset into train and test
split.white = sample.split(white$quality, SplitRatio = 0.75)
split.red = sample.split(red$quality, SplitRatio = 0.75)
white.train = subset(white, split.white == TRUE)
white.test = subset(white, split.white == FALSE)
red.train = subset(red, split.red == TRUE)
red.test = subset(red, split.red == FALSE)

#Histogram
hist(red$quality,main="Red Wine Quality Distribution",
     xlab="Quality",
     xlim=c(1,10),
     col="darkmagenta",
     breaks=c(1,2,3,4,5,6,7,8,9,10)
)
hist(white$quality,main="White Wine Quality Distribution",
     xlab="Quality",
     xlim=c(1,10),
     breaks=c(1,2,3,4,5,6,7,8,9,10)
)
cor(wine_df[c(-1)])
cor(red)
cor(white)
summary(red)
summary(white)
pairs(red[,c(-1)])

#variable selection using BIC
regfit.full=regsubsets(quality~.,data=red[-1],nvmax=11) #all subset selection with max # of variables=19
reg.summary2=summary(regfit.full)

a<-which.min(reg.summary2$bic)
plot(reg.summary2$bic,xlab="Number of Variables",ylab="BIC",type='l')
points(a,reg.summary2$bic[a],col="red",cex=2,pch=20)

lm1 <- lm(quality ~ . , data = red[-1])
step(lm1,direction='backward')

#white wine BIC method 
regfit.full2=regsubsets(quality~.,data=white[-1],nvmax=11)
reg.summary3=summary(regfit.full2)

b<-which.min(reg.summary3$bic)
plot(reg.summary3$bic,xlab="Number of Variables",ylab="BIC",type='l')
points(b,reg.summary3$bic[b],col="red",cex=2,pch=20)

lm2 <- lm(quality ~ . , data = white[-1])
step(lm2,direction='backward')

#correlation matrix for red wine and white separately 
par(mfrow = c(1,1))
cor.red <- cor(red[-1])
corrplot(cor.red, method = 'number')
cor.white<-cor(white[-1])
corrplot(cor.white, method = 'number')

#Logistic Regression
#Set wine with qulity 3-5 to be not good and greater than 6 be good wine
set.seed(100)
white["quality"] = 1 * (white["quality"] >= 6)
GoodQuality = white['quality'] == 1
NotGoodQuality = white['quality'] == 0

red["quality"] = 1 * (red["quality"] >= 6)
GoodQuality = red['quality'] == 1
NotGoodQuality = red['quality'] == 0

split.white = sample.split(white$quality, SplitRatio = 0.75)
split.red = sample.split(red$quality, SplitRatio = 0.75)
white.train = subset(white, split.white == TRUE)
white.test = subset(white, split.white == FALSE)
red.train = subset(red, split.red == TRUE)
red.test = subset(red, split.red == FALSE)

#fit the logistic regression
set.seed = (100)
glm.fit.white = glm(formula = quality~., data = white.train, family = binomial)
summary(glm.fit)
glm.fit.red = glm(formula = quality~., data = red.train, family = binomial)
summary(glm.fit)

#derive the MSE
glm.probs.red = predict(glm.fit.red,red.test,type = "response")
glm.pred.red = rep("No",length(glm.probs.red))
glm.pred.red[glm.probs.red > 0.05] = 1 
table(glm.pred.red,red.test$quality)
mean(glm.pred.red != red.test$quality)

glm.probs.white = predict(glm.fit.white,white.test,type = "response")
glm.pred.white = rep("No",length(glm.probs.white))
glm.pred.white[glm.probs.white > 0.05] = 1 
table(glm.pred.white,white.test$quality)
mean(glm.pred.white != white.test$quality)

#reload the dataset since the logistic regression change quality to dummy variable
wine_df= read.csv('C:/STSCI5740/wine-quality-white-and-red.csv')
summary(wine_df)
white=wine_df[wine_df$type=='white',c(-1)]
red=wine_df[wine_df$type=='red',c(-1)]

split.white = sample.split(white$quality, SplitRatio = 0.75)
split.red = sample.split(red$quality, SplitRatio = 0.75)
white.train = subset(white, split.white == TRUE)
white.test = subset(white, split.white == FALSE)
red.train = subset(red, split.red == TRUE)
red.test = subset(red, split.red == FALSE)
#fit a multi-nomial logistic regression
set.seed(100)
  multilogis.red = multinom(quality~.,data = red.train)
  summary(multilogis.red)
  multilogis.white = multinom(quality~.,data = white.train)
  summary(multilogis.white)
  
#derive the confusionmatrix of the multinomial logistic regression
#Red wine data set
winetrain.red = red.train
winetest.red = red.test
winetrain.red$pred_quality = predict(multilogis.red,newdata = winetrain.red)
winetest.red$pre_quality = predict(multilogis.red,newdata = winetest.red)

winetrain.red$pred_quality11 = as.factor(winetrain.red$pred_quality)
winetrain.red$quality11 = as.factor(winetrain.red$quality)

confusionMatrix(winetrain.red$pred_quality11,winetrain.red$quality11)

#White wine dataset
winetrain.white = white.train
winetest.white = white.test
winetrain.white$pred_quality = predict(multilogis.white,newdata = winetrain.white)
winetest.white$pre_quality = predict(multilogis.white,newdata = winetest.white)

winetrain.white$pred_quality11 = as.factor(winetrain.white$pred_quality)
winetrain.white$quality11 = as.factor(winetrain.white$quality)

confusionMatrix(winetrain.white$pred_quality11,winetrain.white$quality11)

#KNN For Red wine dataset
#Standardize the variables
set.seed(100)
PreprocVaules = preProcess(red.train,method = c("center","scale"))
trainsc = predict(PreprocVaules,red.train)
testsc  = predict(PreprocVaules,red.test)
#check if standardize, find out if standard deviation = 1
sd(trainsc$fixed.acidity)

#Process the data
winetrain = trainsc[,-12]
winetest = testsc[,-12]
winetrainlabel = trainsc[,12,drop = TRUE]
winetestlabel = testsc[,12,drop = TRUE]
#derive the model accuracy by confusionMatrix
wineknns = knn(winetrain,winetest,winetrainlabel,19)
winetestlabel1 = as.factor(winetestlabel)
wineknns1 = as.factor(wineknns)
confusionMatrix(winetestlabel1,wineknns1)

#KNN For White wine dataset
#Standardize the variables
set.seed(100)
PreprocVaules = preProcess(red.train,method = c("center","scale"))
trainsc = predict(PreprocVaules,white.train)
testsc  = predict(PreprocVaules,white.test)
#check if standardize, find out if standard deviation = 1
sd(trainsc$fixed.acidity)

#Process the data
winetrain = trainsc[,-12]
winetest = testsc[,-12]
winetrainlabel = trainsc[,12,drop = TRUE]
winetestlabel = testsc[,12,drop = TRUE]
#derive the model accuracy by confusionMatrix
wineknns = knn(winetrain,winetest,winetrainlabel,69)
winetestlabel1 = as.factor(winetestlabel)
wineknns1 = as.factor(wineknns)
confusionMatrix(winetestlabel1,wineknns1)


plot(wineknn)
#SVM
tune.out.white.ln = tune(svm, quality ~ ., data = white.train, kernel = "linear",ranges = list(cost = c(0.001,0.1, 1, 10), tolerance=c(0.01,0.25,1), gama=1.14),scale = TRUE)
summary(tune.out.white.ln)
ypred.white.ln=predict(tune.out.white.ln$best.model, white.test)

error_best_mod= abs(white.test$quality - ypred.white.ln)
a=floor(ypred.white.ln)==white.test$quality
b=ceiling(ypred.white.ln)==white.test$quality
c=a|b
test.df=data.frame(floor= c(floor(data.frame(ypred.white.ln)$ypred.white.ln)),ceil=c(ceiling(data.frame(ypred.white.ln)$ypred.white.ln)),true=c(white.test$quality),match=c(c))
summary(test.df)

tune.out.red.ln = tune(svm, quality ~ ., data = red.train, kernel = "linear",ranges = list(cost = c(0.001,0.1, 1, 10), tolerance=c(0.01,0.25,1), gama=1.14),scale = TRUE)
summary(tune.out.red.ln)
ypred.red.ln=predict(tune.out.red.ln$best.model, red.test)
error_best_mod_red= abs(red.test$quality - ypred.red.ln)
a.red=floor(ypred.red.ln)==red.test$quality
b.red=ceiling(ypred.red.ln)==red.test$quality
c.red=a.red|b.red
test.df.red=data.frame(floor= c(floor(data.frame(ypred.red.ln)$ypred.red.ln)),ceil=c(ceiling(data.frame(ypred.red.ln)$ypred.red.ln)),true=c(red.test$quality),match=c(c.red))

summary(test.df.red)

red$quality=as.factor(red$quality)
white$quality=as.factor(white$quality)
set.seed(100)
split.white = sample.split(white$quality, SplitRatio = 0.75)
split.red = sample.split(red$quality, SplitRatio = 0.75)

white.train = subset(white, split.white == TRUE)
white.test = subset(white, split.white == FALSE)
red.train = subset(red, split.red == TRUE)
red.test = subset(red, split.red == FALSE)

svmfit.white = svm(quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates, data = white.train, kernel = "radial", gamma = 1, cost = 10, tolerance=0.25,scale = TRUE)
#summary(svmfit.white)
confusionMatrix(predict(svmfit.white, white.test), white.test$quality)

tune.out.white.no.t = tune(svm, quality ~ ., data = white.train, kernel = "radial",ranges = list(cost=c(5^(-2:2)), gamma = seq(0.1, 2, length = 10)),scale = TRUE)
summary(tune.out.white.no.t)

tune.out.white = tune(svm, quality ~ ., data = white.train, kernel = "radial",ranges = list(cost=c(5^(-2:2)), gamma = seq(0.1, 2, length =10), tolerance=c(0.01,0.25,0.5)),scale = TRUE)
summary(tune.out.white)

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated 10 times
  repeats = 10)
svm.grid <- expand.grid(C = c(5^(-2:2)), sigma = seq(0.1, 2, length = 10))
svm.train <- train(quality ~ ., data = white.train, method = "svmRadial",
                   trControl = fitControl,
                   tuneGrid = svm.grid,
                   preProcess = c("center", "scale"))
plot(svm.train)

svm.train

bestmod.white=tune.out.white$best.model
summary(bestmod.white)
ypred.white=predict(tune.out.white$best.model, white.test)
#table(predict=ypred.white, truth=white.test$quality)
confusionMatrix(ypred.white, white.test$quality)

svmfit.red = svm(quality ~ fixed.acidity + volatile.acidity + citric.acid + residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + density + pH + sulphates, data = red.train, kernel = "radial", gamma = 1, cost = 1, scale = FALSE)
summary(svmfit.red)

svm.grid <- expand.grid(C = c(5^(-2:2)), sigma = seq(0.1, 2, length = 10))
svm.train.red <- train(quality ~ ., data = red.train, method = "svmRadial",
                       trControl = fitControl,
                       tuneGrid = svm.grid,
                       preProcess = c("center", "scale"))
plot(svm.train.red)

tune.out.red = tune(svm, quality ~ ., data = red.train, kernel = "radial",ranges = list(cost=c(5^(-2:2)), gamma = seq(0.1, 2, length = 10), tolerance=c(0.01,0.25,0.5)),scale = TRUE)
summary(tune.out.red)

ypred.red=predict(tune.out.red$best.model, red.test)
confusionMatrix(ypred.red, red.test$quality)

#Random forest
#red wine subset
tree.redwine <- tree(quality ~., data =red.train)
summary(tree.redwine)
plot(tree.redwine)
title(main='unpruned decesion tree for red wine')
text(tree.redwine,pretty = 0)
yhat.tree<-predict(tree.redwine,newdata=red.test)
mean((yhat.tree-red.test$quality)^2) #MSE using decesion tree 
table(red.test[,13],round(yhat.tree)) #confusion matrix for decision tree 
mean(red.test[,13]==round(yhat.tree))  

cv.redwine<-cv.tree(tree.redwine)
plot(cv.redwine$size, cv.redwine$dev, type = "b")
tree.min <- which.min(cv.redwine$dev)
points(tree.min, cv.redwine$dev[tree.min], col = "red", cex = 2, pch = 20)
prune.redwine<-prune.tree(tree.redwine,best=7)

plot(prune.redwine)
title(main='pruned decesion tree for red wine')
text(prune.redwine,pretty=0)
yhat.prunetree<-predict(prune.redwine,newdata=red.test)
mean((yhat.prunetree-red.test$quality)^2)
table(red.test[,13],round(yhat.prunetree))
mean(red.test[,13]==round(yhat.prunetree))

bag.redwine<- randomForest(quality ~ ., data = red.train, mtry = 12, ntree = 500, importance = TRUE)
yhat.bag <- predict(bag.redwine, newdata = red.test)
mean((yhat.bag - red.test$quality)^2) # MSE using bagging 
table(red.test[,13],round(yhat.bag))  #confusion matrix for bagging
mean(red.test[,13]==round(yhat.bag)) # accuracy 
importance(bag.redwine)
varImpPlot(bag.redwine) # visualization 

#trying using different mtry
rf.redwine <- randomForest(quality ~ ., data = red.train, mtry =1 , ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.redwine, newdata = red.test)
mean((yhat.rf - red.test$quality)^2)#MSE using random forest
table(red.test[,13],round(yhat.rf))  #confusion matrix for random forest
mean(red.test[,13]==round(yhat.rf))
importance(rf.redwine)
varImpPlot(rf.redwine)

rf.redwine <- randomForest(quality ~ ., data = red.train, mtry =2 , ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.redwine, newdata = red.test)
mean((yhat.rf - red.test$quality)^2)#MSE using random forest
table(red.test[,13],round(yhat.rf))  #confusion matrix for random forest
mean(red.test[,13]==round(yhat.rf))
importance(rf.redwine)
varImpPlot(rf.redwine)

rf.redwine <- randomForest(quality ~ ., data = red.train, mtry =3 , ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.redwine, newdata = red.test)
mean((yhat.rf - red.test$quality)^2)#MSE using random forest
table(red.test[,13],round(yhat.rf))  #confusion matrix for random forest
mean(red.test[,13]==round(yhat.rf))
importance(rf.redwine)
varImpPlot(rf.redwine)

rf.redwine <- randomForest(quality ~ ., data = red.train, mtry =4 , ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.redwine, newdata = red.test)
mean((yhat.rf - red.test$quality)^2)#MSE using random forest
table(red.test[,13],round(yhat.rf))  #confusion matrix for random forest
mean(red.test[,13]==round(yhat.rf))
importance(rf.redwine)
varImpPlot(rf.redwine)
confusionMatrix(as.factor(round(yhat.rf)),as.factor(redwine.test$quality))

rf.redwine <- randomForest(quality ~ ., data = red.train, mtry =5 , ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.redwine, newdata = red.test)
mean((yhat.rf - red.test$quality)^2)#MSE using random forest
table(red.test[,13],round(yhat.rf))  #confusion matrix for random forest
mean(red.test[,13]==round(yhat.rf))
importance(rf.redwine)
varImpPlot(rf.redwine)

rf.redwine <- randomForest(quality ~ ., data = red.train, mtry =6 , ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.redwine, newdata = red.test)
mean((yhat.rf - red.test$quality)^2)#MSE using random forest
table(red.test[,13],round(yhat.rf))  #confusion matrix for random forest
mean(red.test[,13]==round(yhat.rf))
importance(rf.redwine)
varImpPlot(rf.redwine)


rf.redwine <- randomForest(quality ~ ., data = red.train, mtry =11 , ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.redwine, newdata = red.test)
mean((yhat.rf - red.test$quality)^2)#MSE using random forest
table(red.test[,13],round(yhat.rf))  #confusion matrix for random forest
mean(red.test[,13]==round(yhat.rf))
importance(rf.redwine)
varImpPlot(rf.redwine)

rf.redwine <- randomForest(quality ~ ., data = red.train, mtry =12 , ntree = 500, importance = TRUE)
yhat.rf <- predict(rf.redwine, newdata = red.test)
mean((yhat.rf - red.test$quality)^2)#MSE using random forest
table(red.test[,13],round(yhat.rf))  #confusion matrix for random forest
mean(red.test[,13]==round(yhat.rf))
importance(rf.redwine)
varImpPlot(rf.redwine)

#white wine subset
tree.whitewine <- tree(quality ~., data =white.train)
summary(tree.whitewine)
plot(tree.whitewine)
title(main='orignial decesion tree for white wine')
text(tree.whitewine,pretty = 0)
yhat.tree2<-predict(tree.whitewine,newdata=white.test)
mean((yhat.tree2-white.test$quality)^2) #MSE using decesion tree 
table(white.test[,13],round(yhat.tree2))  #confusion matrix for decesion tree 
mean(white.test[,13]==round(yhat.tree2))  


cv.whitewine<-cv.tree(tree.whitewine)
plot(cv.whitewine$size, cv.whitewine$dev, type = "b")
tree.min <- which.min(cv.whitewine$dev)
points(tree.min, cv.whitewine$dev[tree.min], col = "red", cex = 2, pch = 20)

bag.whitewine<- randomForest(quality ~ ., data = white.train, mtry = 12, ntree = 500, importance = TRUE)
yhat.bag2 <- predict(bag.whitewine, newdata = white.test)
mean((yhat.bag2 - white.test$quality)^2) # MSE using bagging 
table(white.test[,13],round(yhat.bag2))  #confusion matrix for bagging
mean(white.test[,13]==round(yhat.bag2)) # accuracy 
importance(bag.whitewine)
varImpPlot(bag.whitewine)

#trying using different mtry
rf.whitewine <- randomForest(quality ~ ., data = white.train, mtry = 1, ntree = 500, importance = TRUE)
yhat.rf2 <- predict(rf.whitewine, newdata = white.test)
mean((yhat.rf2 - white.test$quality)^2)#MSE using random forest
table(white.test[,13],round(yhat.rf2))  #confusion matrix for random forest
mean(white.test[,13]==round(yhat.rf2))
importance(rf.whitewine)
varImpPlot(rf.whitewine)

rf.whitewine <- randomForest(quality ~ ., data = white.train, mtry = 2, ntree = 500, importance = TRUE)
yhat.rf2 <- predict(rf.whitewine, newdata = white.test)
mean((yhat.rf2 - white.test$quality)^2)#MSE using random forest
table(white.test[,13],round(yhat.rf2))  #confusion matrix for random forest
mean(white.test[,13]==round(yhat.rf2))
importance(rf.whitewine)
varImpPlot(rf.whitewine)
confusionMatrix(as.factor(round(yhat.rf2)),as.factor(whitewine.test$quality))


rf.whitewine <- randomForest(quality ~ ., data = white.train, mtry = 3, ntree = 500, importance = TRUE)
yhat.rf2 <- predict(rf.whitewine, newdata = white.test)
mean((yhat.rf2 - white.test$quality)^2)#MSE using random forest
table(white.test[,13],round(yhat.rf2))  #confusion matrix for random forest
mean(white.test[,13]==round(yhat.rf2))
importance(rf.whitewine)
varImpPlot(rf.whitewine)

rf.whitewine <- randomForest(quality ~ ., data = white.train, mtry = 4, ntree = 500, importance = TRUE)
yhat.rf2 <- predict(rf.whitewine, newdata = white.test)
mean((yhat.rf2 - white.test$quality)^2)#MSE using random forest
table(white.test[,13],round(yhat.rf2))  #confusion matrix for random forest
mean(white.test[,13]==round(yhat.rf2))
importance(rf.whitewine)
varImpPlot(rf.whitewine)

rf.whitewine <- randomForest(quality ~ ., data = white.train, mtry = 5, ntree = 500, importance = TRUE)
yhat.rf2 <- predict(rf.whitewine, newdata = white.test)
mean((yhat.rf2 - white.test$quality)^2)#MSE using random forest
table(white.test[,13],round(yhat.rf2))  #confusion matrix for random forest
mean(white.test[,13]==round(yhat.rf2))
importance(rf.whitewine)
varImpPlot(rf.whitewine)

rf.whitewine <- randomForest(quality ~ ., data = white.train, mtry = 6, ntree = 500, importance = TRUE)
yhat.rf2 <- predict(rf.whitewine, newdata = white.test)
mean((yhat.rf2 - white.test$quality)^2)#MSE using random forest
table(white.test[,13],round(yhat.rf2))  #confusion matrix for random forest
mean(white.test[,13]==round(yhat.rf2))
importance(rf.whitewine)
varImpPlot(rf.whitewine)

rf.whitewine <- randomForest(quality ~ ., data = white.train, mtry = 11, ntree = 500, importance = TRUE)
yhat.rf2 <- predict(rf.whitewine, newdata = white.test)
mean((yhat.rf2 - white.test$quality)^2)#MSE using random forest
table(white.test[,13],round(yhat.rf2))  #confusion matrix for random forest
mean(white.test[,13]==round(yhat.rf2))
importance(rf.whitewine)
varImpPlot(rf.whitewine)

rf.whitewine <- randomForest(quality ~ ., data = white.train, mtry = 12, ntree = 500, importance = TRUE)
yhat.rf2 <- predict(rf.whitewine, newdata = white.test)
mean((yhat.rf2 - white.test$quality)^2)#MSE using random forest
table(white.test[,13],round(yhat.rf2))  #confusion matrix for random forest
mean(white.test[,13]==round(yhat.rf2))
importance(rf.whitewine)
varImpPlot(rf.whitewine)

###Third part
eval = function(pred, true, plot = F, title = "") {
  rmse = sqrt(mean((pred - true)^2))
  mae = mean(abs(pred - true))
  cor = cor(pred, true)
  if (plot == TRUE) {
    par(mfrow = c(1,2), oma = c(0, 0, 2, 0))
    diff = pred - true
    plot(jitter(true, factor = 1), 
         jitter(pred, factor = 0.5),
         pch = 3, asp = 1,
         xlab = "Truth", ylab = "Predicted") 
    abline(0,1, lty = 2)
    hist(diff, breaks = 20, main = NULL)
    mtext(paste0(title, " predicted vs. true using test set"), outer = TRUE)
    par(mfrow = c(1,1))}
  return(list(rmse = rmse,
              mae = mae,
              cor = cor))
}

eval_class = function(prob, true, plot = F, title = "") {
  # find cutoff with the best kappa
  cuts = seq(0.01, 0.99, by=0.01)
  kappa = c()
  for (cut in cuts){
    cat = as.factor(ifelse(prob >= cut, 1, 0))
    cm = confusionMatrix(cat, true, positive = "1")
    kappa = c(kappa, cm$overall[["Kappa"]])
  }
  opt.cut = cuts[which.max(kappa)]
  
  pred = as.factor(ifelse(prob >= opt.cut, 1, 0))
  confM = confusionMatrix(pred, true, positive = "1")
  
}
install.packages("RCurl")
install.packages("psych")
install.packages("tidyverse")
install.packages("usethis")
install.packages("dplyr")
install.packages("lattice")
install.packages("caret")
install.packages("devtools")
install.packages("ggfortify")
install.packages("ggplot2")
install.packages("pROC")
install.packages("naivebayes")
library(RCurl)
library(psych)
library(tidyverse)
library(dplyr)
library(usethis)
library(lattice)
library(caret)
library(devtools)
library(ggfortify)
library(ggplot2)
library(pROC)
library(naivebayes)

#white
myfile = getURL('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')
white = read.csv(textConnection(myfile), header = T, sep = ";")
n = nrow(white); p = ncol(white); dim(white)               

head(white)
str(white)
summary(white)
pairs.panels(white)

set.seed(100)
idx = sample(n, 0.75*n)
white.train = white[idx,]; dim(white.train)

white.test = white[-idx,]; dim(white.test)

normalize_train = function(x) (x - min(x))/(max(x) - min(x))
white.train.norm = data.frame(apply(white.train[,-p], 2, normalize_train), 
                        quality = white.train[,p])
summary(white.train.norm)

# normalize test set using the values from train set to make prediction comparable
white.train.min = apply(white.train[,-p], 2, min)
white.train.max = apply(white.train[,-p], 2, max)
white.test.norm = data.frame(sweep(white.test, 2, c(white.train.min, 0)) %>% 
                         sweep(2, c(white.train.max-white.train.min, 1), FUN = "/"))
summary(white.test.norm) 


tr.lm.interract = lm(quality~ .^2, data = white.train.norm)
summary(tr.lm.interract)


#Linear Regression

hist(white$quality)  
shapiro.test(white$quality)

tr.lm = lm(quality~., data = white.train.norm)
summary(tr.lm)

tr.lm.pred = predict(tr.lm, white.test.norm[,-p])
tr.lm.eval = eval(tr.lm.pred, white.test.norm$quality, plot = T, title = "lm: "); unlist(tr.lm.eval)

#10-fold Validation
set.seed(100)
trctrl <- trainControl(method = "cv", number = 10, savePredictions=TRUE)
nb_fit <- train(quality~., white, method = "lm", trControl=trctrl, tuneLength = 0)

pre.y = predict(nb_fit,white.test[,1:11])
for( i in 1:length(pre.y)){
  if((pre.y[i] - trunc(pre.y[i]))<0.5){
    pre.y[i] = trunc(pre.y[i])
  }else{
    pre.y[i] = trunc(pre.y[i])+1
  }
}
accur_rate<-sum(pre.y == white.test[,12])/length(white.test[,12]);accur_rate
confusionMatrix(factor(pre.y, levels = c(3,4,5,6,7,8,9)), factor(white.test$quality))

#Linear Regression after Variable Selection

hist(white$quality)  
shapiro.test(white$quality)

tr.lm = lm(quality ~ (fixed.acidity + free.sulfur.dioxide + sulphates + pH + alcohol + density + residual.sugar + volatile.acidity), data = white.train.norm)
summary(tr.lm)

tr.lm.pred = predict(tr.lm, white.test.norm[,-p])
tr.lm.eval = eval(tr.lm.pred, white.test.norm$quality, plot = T, title = "lm: "); unlist(tr.lm.eval)

#10-fold Validation after Variable Selection
set.seed(100)
trctrl <- trainControl(method = "cv", number = 10, savePredictions=TRUE)
nb_fit <- train((quality) ~ (fixed.acidity + free.sulfur.dioxide + sulphates + pH + alcohol + density + residual.sugar + volatile.acidity), white, method = "lm", trControl=trctrl, tuneLength = 0)


train_temp = white[,c('fixed.acidity','free.sulfur.dioxide','sulphates','pH','alcohol','density','residual.sugar','volatile.acidity','quality')]
nb_fit_8 <- train(quality ~ ., train_temp, method = "lm", trControl=trctrl, tuneLength = 0)
test_temp = white.test[,c('fixed.acidity','free.sulfur.dioxide','sulphates','pH','alcohol','density','residual.sugar','volatile.acidity','quality')]
pre.yy = predict(nb_fit_8,test_temp[,1:8])
for( i in 1:length(pre.yy)){
  if((pre.yy[i] - trunc(pre.yy[i]))<0.5){
    pre.yy[i] = trunc(pre.yy[i])
  }else{
    pre.yy[i] = trunc(pre.yy[i])+1
  }
}
accur_rate<-sum(pre.yy == test_temp[,9])/length(test_temp[,8]);accur_rate
confusionMatrix(factor(pre.yy, levels = c(3,4,5,6,7,8,9)), factor(test_temp$quality))

#Polynomial Regression
tr.qm = lm(quality~ poly(fixed.acidity, 2) + 
             poly(volatile.acidity,2) + 
             poly(citric.acid,2) + 
             poly(residual.sugar,2) +  
             poly(chlorides,2) + 
             poly(free.sulfur.dioxide,2) +
             poly(total.sulfur.dioxide,2) + 
             poly(pH,2) + 
             poly(sulphates,2) + 
             poly(alcohol,2) +
             poly(free.sulfur.dioxide * total.sulfur.dioxide) ,
           data = white.train.norm)
summary(tr.qm)

tr.qm.pred = predict(tr.qm, white.test.norm[,-p])
tr.qm.eval = eval(tr.qm.pred, white.test.norm$quality, plot=T, title="quadratic model: ");unlist(tr.qm.eval)

for( i in 1:length(tr.qm.pred)){
  if((tr.qm.pred[i] - trunc(tr.qm.pred[i]))<0.5){
    tr.qm.pred[i] = trunc(tr.qm.pred[i])
  }else{
    tr.qm.pred[i] = trunc(tr.qm.pred[i])+1
  }
}
accur_rate<-sum(tr.qm.pred == white.test[,12])/length(white.test[,12]);accur_rate
confusionMatrix(factor(tr.qm.pred, levels = c(3,4,5,6,7,8,9)), factor(white.test$quality))


#red
myfile = getURL('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv')
red = read.csv(textConnection(myfile), header = T, sep = ";")
n = nrow(red); p = ncol(red); dim(red)

head(red)
str(red)
summary(red)
pairs.panels(red)

set.seed(100)
idx = sample(n, 0.75*n)
red.train = red[idx,]; dim(red.train)

red.test = red[-idx,]; dim(red.test)

normalize_train = function(x) (x - min(x))/(max(x) - min(x))
red.train.norm = data.frame(apply(red.train[,-p], 2, normalize_train), 
                        quality = red.train[,p])
summary(red.train.norm)

# normalize test set using the values from train set to make prediction comparable
red.train.min = apply(red.train[,-p], 2, min)
red.train.max = apply(red.train[,-p], 2, max)
red.test.norm = data.frame(sweep(red.test, 2, c(red.train.min, 0)) %>% 
                         sweep(2, c(red.train.max-red.train.min, 1), FUN = "/"))
summary(red.test.norm) 


tr.lm.interract = lm(quality~ .^2, data = red.train.norm)
summary(tr.lm.interract)


#Linear Regression

hist(red$quality)  
shapiro.test(red$quality)

tr.lm = lm(quality~., data = red.train.norm)
summary(tr.lm)

tr.lm.pred = predict(tr.lm, red.test.norm[,-p])
tr.lm.eval = eval(tr.lm.pred, red.test.norm$quality, plot = T, title = "lm: "); unlist(tr.lm.eval)

#10-fold Validation
set.seed(100)
trctrl <- trainControl(method = "cv", number = 10, savePredictions=TRUE, verboseIter = TRUE)
nb_fit <- train(quality ~ ., red, method = "lm", trControl=trctrl, tuneLength = 0)

pre.y = predict(nb_fit,red.test[,1:11])
for( i in 1:length(pre.y)){
  if((pre.y[i] - trunc(pre.y[i]))<0.5){
    pre.y[i] = trunc(pre.y[i])
  }else{
    pre.y[i] = trunc(pre.y[i])+1
  }
}
accur_rate<-sum(pre.y == red.test[,12])/length(red.test[,12]);accur_rate
confusionMatrix(factor(pre.y, levels = c(4,5,6,7,8)), factor(red.test$quality))

#Linear Regression after Variable Selection

hist(red$quality)  
shapiro.test(red$quality)

tr.lm = lm(quality ~ (free.sulfur.dioxide + pH + total.sulfur.dioxide + chlorides + sulphates + volatile.acidity + alcohol), data = red.train.norm)
summary(tr.lm)

tr.lm.pred = predict(tr.lm, red.test.norm[,-p])
tr.lm.eval = eval(tr.lm.pred, red.test.norm$quality, plot = T, title = "lm: "); unlist(tr.lm.eval)


#10-fold Validation after Variable Selection
set.seed(100)
trctrl <- trainControl(method = "cv", number = 10, savePredictions=TRUE)
train_temp = red[,c('free.sulfur.dioxide','pH','total.sulfur.dioxide','chlorides','sulphates','volatile.acidity','alcohol','quality')]
nb_fit_7 <- train(quality ~ ., train_temp, method = "lm", trControl=trctrl, tuneLength = 0)
test_temp = red.test[,c('free.sulfur.dioxide','pH','total.sulfur.dioxide','chlorides','sulphates','volatile.acidity','alcohol','quality')]
pre.yy = predict(nb_fit_7,test_temp[,1:7])
for( i in 1:length(pre.yy)){
  if((pre.yy[i] - trunc(pre.yy[i]))<0.5){
    pre.yy[i] = trunc(pre.yy[i])
  }else{
    pre.yy[i] = trunc(pre.yy[i])+1
  }
}
accur_rate<-sum(pre.yy == test_temp[,8])/length(test_temp[,8]);accur_rate
confusionMatrix(factor(pre.yy, levels = c(4,5,6,7,8)), factor(test_temp$quality))

#Polynomial Regression
tr.qm = lm(quality~ poly(fixed.acidity, 2) + 
             poly(volatile.acidity,2) + 
             poly(citric.acid,2) + 
             poly(residual.sugar,2) +  
             poly(chlorides,2) + 
             poly(free.sulfur.dioxide,2) +
             poly(total.sulfur.dioxide,2) + 
             poly(density,2) + 
             poly(pH,2) + 
             poly(sulphates,2) + 
             poly(alcohol,2) + 
             poly(pH * density) +
             poly(free.sulfur.dioxide * total.sulfur.dioxide), 
           data = red.train.norm)
summary(tr.qm)

tr.qm.pred = predict(tr.qm, red.test.norm[,-p])
tr.qm.eval = eval(tr.qm.pred, red.test.norm$quality, plot=T, title="quadratic model: ");unlist(tr.qm.eval)
for( i in 1:length(tr.qm.pred)){
  if((tr.qm.pred[i] - trunc(tr.qm.pred[i]))<0.5){
    tr.qm.pred[i] = trunc(tr.qm.pred[i])
  }else{
    tr.qm.pred[i] = trunc(tr.qm.pred[i])+1
  }
}
accur_rate<-sum(tr.qm.pred == red.test[,12])/length(red.test[,12]);accur_rate
confusionMatrix(factor(tr.qm.pred, levels = c(4,5,6,7,8)), factor(red.test$quality))

