#################   		Installing Packages  			####################

library(lattice)
library(caret)
library(rpart)
library(gbm)
library(Matrix)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

########			 Data Input 				 #########

train<-read.csv("train.csv")
test<-read.csv("test.csv")

summary(train)
summary(test)



###################### 				Data Exploration 			##########################

#####    			
head(train$Gender)
as.numeric(rbinom(1, 1.0, 0.5))

Loan_Status1=matrix(-1,nrow(train),1)
Loan_Status1[which(train$Loan_Status=='Y')]=1
Loan_Status1[which(train$Loan_Status=='N')]=0
Loan_Status1<-as.matrix(Loan_Status1)
train$Loan_Status<-Loan_Status1

for (i in 1:nrow(test))
{
    if(is.na(test$Credit_History[i]))
    {
        test$Credit_History[i]=as.numeric(rbinom(1, 1.0, 0.5))
    }
}

plot(train$LoanAmount,train$ApplicantIncome)
pairs(~.,train)
boxplot(train$LoanAmount)

train[which(train$Loan_Status=='N' & train$Education=='Not Graduate' & train$LoanAmount<200),]

loanamountNA<-train[which(is.na(train$LoanAmount)),]

linear_reg<-lm(train$ApplicantIncome~train$LoanAmount, train[which(train$LoanAmount<400),])

loanamountNA$LoanAmount<-(loanamountNA$ApplicantIncome-(1.036e+02))/(7.927e-03)

train$LoanAmount[which(is.na(train$LoanAmount))]=loanamountNA$LoanAmount

##########################################			GBM 	Method			#########################


train_gbm<-train[,!names(train) %in% c("Loan_ID","Married","Gender","Dependents","Property_Area","Self_Employed","Education")]
test_gbm<-test[,!names(test) %in% c("Loan_ID","Married","Gender","Dependents","Property_Area","Self_Employed","Education")]
formula_gbm<-as.formula(paste("Loan_Status~",paste( names(train)[!names(train) %in% c("Loan_ID","Married","Gender", "Education","Dependents","Property_Area","Self_Employed","Loan_Status") ], collapse="+")))

model.gbm<-gbm(formula_gbm, train_gbm, shrinkage = 0.01, interaction.depth=5, n.trees=177,distribution="bernoulli", verbose=TRUE,train.fraction = 0.9)

###### best till now
###########model.gbm<-gbm(formula_gbm, train, shrinkage = 0.01, interaction.depth=5, n.trees=489,distribution="bernoulli", verbose=TRUE,train.fraction = 0.5)

gbm.perf(model.gbm)

summary(model.gbm)

pred_gbm_train<-predict(model.gbm, train_gbm, type="response")
pred_gbm<-predict(model.gbm,test_gbm, type="response")

pred_gbm_r_ch<-matrix(-1,367,1)

pred_gbm_r_ch[which(pred_gbm>0.7)]='Y'
pred_gbm_r_ch[which(pred_gbm<0.7)]='N'

write.csv(pred_gbm_r_ch,"output_gbm.csv")


################################################## 			Xgboost  Method			 #################################

train_xgb2<-train[,!names(train) %in% c("Loan_Status","Loan_ID","Married","Gender","Dependents","Property_Area","Self_Employed","Education")]
test_xgb2<-test[,!names(test) %in% c("Loan_ID","Married","Gender","Dependents","Property_Area","Self_Employed","Education")]
train_xgb2<-as.matrix(train_xgb2)
test_xgb2<-data.matrix(test_xgb2)

dtrain <- xgb.DMatrix(train_xgb2, label = Loan_Status1, missing=NA)

xgboost.model <- xgboost(dtrain, max.depth = 10,eta = 0.01, nround =1000, verbose=1, objective = "binary:logistic",train.fraction=0.7,early_stopping=TRUE)

xgb_pred<-predict(xgboost.model, dtrain, missing=NA)

setinfo(dtrain, "base_margin", pred_gbm_train)

xgboost.model <- xgboost(dtrain, max.depth = 10,eta = 0.01, nround =1000, verbose=1, objective = "binary:logistic",train.fraction=0.5,early_stopping=TRUE)


########################################## 		Cross Validation  			########################################## 

cv.model<-xgb.cv(data=train_xgb2, label = Loan_Status1,max.depth = 5,eta = 0.001, nround =1000, verbose=1, missing=NA,objective = "binary:logistic", nfold = 10)

summary(cv.model)

##################################

xgb_pred1<-predict(xgboost.model, test_xgb2, missing=NA)

#summary.gbm(xgboost.model1)
xgb_pred2=matrix(data=-1, 367, 1)

xgb_pred2[which(xgb_pred1>0.5)]='Y'
xgb_pred2[which(xgb_pred1<0.5)]='N'

write.csv(xgb_pred2, "output_XGB.csv")

#############################  		Regression Tress 		########################################## 

formula_reg<-as.formula(paste("Loan_Status~",paste( names(train)[!names(train) %in% c("Loan_Status","Loan_ID") ], collapse="+")))

model.reg<-rpart(formula_reg,train, cp=0.000005)

pred.reg<-predict(model.reg,test)

pred_reg<-round(pred.reg)

pred_reg_r_ch<-matrix(-1,367,1)

pred_reg_r_ch[which(pred_reg==1)]='Y'
pred_reg_r_ch[which(pred_reg==0)]='N'

write.csv(pred_reg_r_ch, "output_reg.csv")

############### 				Ensemble     Modelling 		#####################

##   To be inplemented

