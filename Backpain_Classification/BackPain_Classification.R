load("backpain.RData")
set.seed(1000)
library(rpart)
library(nnet)
attach(dat)
#
str(dat)
# checking for missing values
dat[!complete.cases(dat),]
# No missing values are found

# Splitting dataset into test train and validation 25% 50% 25% respectively
N<-nrow(dat)
trainInd<-sample(1:N,size = 1*N,replace = TRUE)
trainInd<-sort(trainInd)
ind<-setdiff(1:N,trainInd)
validInd <- sample(setdiff(1:N,trainInd),size=0.20*length(ind))
validInd <- sort(validInd)
testInd <- setdiff(1:N,union(trainInd,validInd))


# Classification Tree
fit.r<-rpart(PainDiagnosis~.,data = dat,subset = trainInd)
pred.r <- predict(fit.r,type="class",newdata=dat)
tab.r <- table(PainDiagnosis[validInd],pred.r[validInd])
acc.r<-sum(diag(tab.r))/sum(tab.r)

#boxplot(PainDiagnosis,SurityRating)
#dat<-dat[-which(dat$SurityRating %in% boxplot.stats(SurityRating)$out),]
#dat<-dat[-354,]

# Logistic
fit.l <- multinom(PainDiagnosis~.,data = dat,subset = trainInd)
pred.l <- predict(fit.l,type="class",newdata=dat)
tab.l <- table(PainDiagnosis[validInd],pred.l[validInd])
acc.l <- sum(diag(tab.l))/sum(tab.l)

# Random forest
library('randomForest')
library('gclus')
fit.rf<-randomForest(PainDiagnosis~.,data=dat[trainInd,])
pred.rf<-predict(fit.rf,type = "response",newdata=dat)
tab.rf <- table(PainDiagnosis[validInd],pred.rf[validInd])
acc.rf <- sum(diag(tab.rf))/sum(tab.rf)

# Bagging
library('adabag')
fit.bag<-bagging(PainDiagnosis~.,data=dat[trainInd,])
pred.bag<-predict(fit.bag,type="class",newdata = dat)
tab.bag<-table(PainDiagnosis[validInd],pred.fit$class[validInd])
acc.bag<-1-sum(diag(tab.bag))/sum(tab.bag)

# SVM
library('kernlab')
fit.svm<-ksvm(PainDiagnosis~.,data=dat[trainInd,])
pred.svm<-predict(fit.svm,type = "response",newdata=dat)
tab.svm <- table(PainDiagnosis[validInd],pred.svm[validInd])
acc.svm <- sum(diag(tab.svm))/sum(tab.svm)

#Boosting
fit.boost <- boosting(PainDiagnosis~.,data=dat[trainInd,],boos=FALSE,coeflearn="Breiman",mfinal = 30)
pred.boost<- predict(fit.boost,newdata=dat)
tab.boost<-table(PainDiagnosis[validInd],pred.boost$class[validInd])
acc.boost<-1-sum(diag(tab.boost))/sum(tab.svm)



# Maximum accuracy obtained is for Random Forrest and SVM.

# applying test subset to the model and checking accuracy
tab.rf <- table(PainDiagnosis[trainInd],pred.rf[trainInd])
acc.rf <- sum(diag(tab.rf))/sum(tab.rf)

tab.svm <- table(PainDiagnosis[trainInd],pred.svm[trainInd])
acc.svm <- sum(diag(tab.svm))/sum(tab.svm)

# Random forest performed well on train subset.

# The variable importance plot for both

varImpPlot(fit.rf)
varImp(fit.rf)

# Iterating 100 times


res<-matrix(NA,100,8)
iterlim <- 100
for (iter in 1:iterlim)
{
  # Sample 50% of the data as training data
  # Sample 25% of the data as validation
  # Let the remaining 25% data be test data
  
  N<-nrow(dat)
  trainInd<-sample(1:N,size = 1*N,replace = TRUE)
  trainInd<-sort(trainInd)
  ind<-setdiff(1:N,trainInd)
  validInd <- sample(setdiff(1:N,trainInd),size=0.20*length(ind))
  validInd <- sort(validInd)
  testInd <- setdiff(1:N,union(trainInd,validInd))
  
  # Fit a classifier to only the training data
  
  fit.r<-rpart(PainDiagnosis~.,data = dat,subset = trainInd)
  fit.l <- multinom(PainDiagnosis~.,data = dat,subset = trainInd)
  #fit.glm<-glm(PainDiagnosis~.,data=newdat,family="binomial")
  fit.rf<-randomForest(PainDiagnosis~.,data=dat[trainInd,])
  fit.svm<-ksvm(PainDiagnosis~.,data=dat[trainInd,])
  fit.boost <- boosting(PainDiagnosis~.,data=dat[trainInd,],boos=FALSE,coeflearn="Breiman",mfinal = 30)
  fit.bag<-bagging(PainDiagnosis~.,data=dat[trainInd,],mfinal = 30)
  
  pred.r <- predict(fit.r,type="class",newdata=dat)
  pred.l <- predict(fit.l,type="class",newdata=dat)
  #pred.glm<-predict(fit.glm,type = "response",newdata=newdat)
  pred.rf<-predict(fit.rf,type = "response",newdata=dat)
  pred.svm<-predict(fit.svm,type = "response",newdata=dat)
  pred.boost<- predict(fit.boost,newdata=dat)
  pred.bag<-predict(fit.bag,type="class",newdata = dat)
  
  # Accuracy on validation
  acc<-c()
  tab.r <- table(PainDiagnosis[validInd],pred.r[validInd])
  acc[1]<-sum(diag(tab.r))/sum(tab.r)
  
  tab.l <- table(PainDiagnosis[validInd],pred.l[validInd])
  acc[2] <- sum(diag(tab.l))/sum(tab.l)
  
  #tab.glm <- table(PainDiagnosis[validInd],round(pred.glm[validInd]))
  #acc[3] <- sum(diag(tab.glm))/sum(tab.glm)
  
  tab.rf <- table(PainDiagnosis[validInd],pred.rf[validInd])
  acc[3] <- sum(diag(tab.rf))/sum(tab.rf)
  
  tab.svm <- table(PainDiagnosis[validInd],pred.svm[validInd])
  acc[4] <- sum(diag(tab.svm))/sum(tab.svm)
  
  tab.boost<-table(PainDiagnosis[validInd],pred.boost$class[validInd])
  acc[5]<-1-sum(diag(tab.boost))/sum(tab.svm)
  
  tab.bag<-table(PainDiagnosis[validInd],pred.bag$class[validInd])
  acc[6]=1-sum(diag(tab.bag))/sum(tab.bag)
  
  res[iter,1] <- acc[1]
  res[iter,2] <- acc[2]
  res[iter,3] <- acc[3]
  res[iter,4] <- acc[4]
  res[iter,5] <- acc[5]
  res[iter,6] <- acc[6]
  s=which.max(acc)
  switch (s,
    {
      tab <- table(PainDiagnosis[testInd],pred.r[testInd])
      acc <- sum(diag(tab))/sum(tab)
      res[iter,7] <- 1
      res[iter,8] <- acc 
    },
    {
      tab <- table(PainDiagnosis[testInd],pred.l[testInd])
      acc <- sum(diag(tab))/sum(tab)
      res[iter,7] <- 2
      res[iter,8] <- acc
    },
    {
      tab <- table(PainDiagnosis[testInd],pred.rf[testInd])
      acc <- sum(diag(tab))/sum(tab)
      res[iter,7] <- 3
      res[iter,8] <- acc
    },
    {
      tab <- table(PainDiagnosis[testInd],pred.svm[testInd])
      acc <- sum(diag(tab))/sum(tab)
      res[iter,7] <- 4
      res[iter,8] <- acc
    },
    {
      tab <- table(PainDiagnosis[testInd],pred.boost$class[testInd])
      acc <- 1-sum(diag(tab))/sum(tab)
      res[iter,7] <- 5
      res[iter,8] <- acc
      },
    {
      tab <- table(PainDiagnosis[testInd],pred.bag$class[testInd])
      acc <- 1-sum(diag(tab))/sum(tab)
      res[iter,7] <- 6
      res[iter,8] <- acc
    }
  )
  
}

colnames(res)<-c("valid.r","valid.l","valid.rf","valid.svm","valid.boosting","valid Bagging","chosen","test")

apply(res[,-7],2,summary)
table(res[,7])
