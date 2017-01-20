#########     Installing Packages   ########
install.packages("readr")
library(caret)
library(rpart)
library(gbm)
library(Matrix)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(plyr)

######## 		Data Input		###########

train<-read.csv("train.csv")
test<-read.csv("test.csv")
summary(test)
summary(train)
str(train)

#################### Replacing with MODE and label encoding in train/ filling NA values ##############

###########  MODE for train   #######################

uniqv2<-unique(train$Product_Category_2[!is.na(train$Product_Category_2)])
uniqv3<-unique(train$Product_Category_3[!is.na(train$Product_Category_3)])
mode_categ2<-uniqv2[which.max(tabulate(match(train$Product_Category_2, uniqv2)))]
mode_categ3<-uniqv3[which.max(tabulate(match(train$Product_Category_3, uniqv3)))]

train$Product_Category_3[is.na(train$Product_Category_3)]=mode_categ3
train$Product_Category_2[is.na(train$Product_Category_2)]=mode_categ2

###########  MODE for test   #######################

uniqv21<-unique(test$Product_Category_2[!is.na(test$Product_Category_2)])
uniqv31<-unique(test$Product_Category_3[!is.na(test$Product_Category_3)])
mode_categ21<-uniqv21[which.max(tabulate(match(test$Product_Category_2, uniqv21)))]
mode_categ31<-uniqv31[which.max(tabulate(match(test$Product_Category_3, uniqv31)))]

test$Product_Category_3[is.na(test$Product_Category_3)]=mode_categ31
test$Product_Category_2[is.na(test$Product_Category_2)]=mode_categ21

"###########  Label Encoding for Train  ####################

gender_encode=matrix(0,nrow(train),1)
city_encode=matrix(-1,nrow(train),1)
product_encode=matrix(-1,nrow(train),1)
usr_encode=matrix(-1,nrow(train),1)
stay_In_Current_City_Years_encode=matrix(-1,nrow(train),1)
age_encode=matrix(-1,nrow(train),1)

gender_encode[which(train$Gender=='F')]=1
gender_encode[which(train$Gender=='M')]=0

city_encode[which(train$City_Category=='A')]=1
city_encode[which(train$City_Category=='B')]=2
city_encode[which(train$City_Category=='C')]=3

uniqage<-unique(train$Age[!is.na(train$Age)])
k=1
for (i in uniqage)
{
  age_encode[which(train$Age==i)]=k
  k=k+1
}

uniqstay<-unique(train$Stay_In_Current_City_Years[!is.na(train$Stay_In_Current_City_Years)])
k=1
for (i in uniqstay)
{
  stay_In_Current_City_Years_encode[which(train$Stay_In_Current_City_Years==i)]=k
  k=k+1
}

uniqprod<-unique(train$Product_ID[!is.na(train$Product_ID)])
k=1
for (i in uniqprod)
{
  product_encode[which(train$Product_ID==i)]=k
  k=k+1
}


train_new<-train
train_new$Gender<-gender_encode
train_new$City_Category<-city_encode
train_new$Product_ID<-product_encode
train_new$Stay_In_Current_City_Years<-stay_In_Current_City_Years_encode
train_new$Age<-age_encode

train_xgb<-train_new[,names(train_new) %in% c("Product_ID", "User_ID", "Product_Category_1","Gender")]
train_xgb<-as.matrix(train_xgb)
label_train<-matrix(train$Purchase, nrow(train), 1)

summary(train_xgb)

#str(train_new)

###########  Label Encoding For Test  ##################

gender_encode_test=matrix(0,nrow(test),1)
city_encode_test=matrix(-1,nrow(test),1)
product_encode_test=matrix(-1,nrow(test),1)
usr_encode_test=matrix(-1,nrow(test),1)
stay_In_Current_City_Years_encode_test=matrix(-1,nrow(test),1)
age_encode_test=matrix(-1,nrow(test),1)

gender_encode_test[which(test$Gender=='F')]=1
gender_encode_test[which(test$Gender=='M')]=0

city_encode_test[which(test$City_Category=='A')]=1
city_encode_test[which(test$City_Category=='B')]=2
city_encode_test[which(test$City_Category=='C')]=3

uniqage<-unique(test$Age[!is.na(test$Age)])
k=1
for (i in uniqage)
{
  age_encode_test[which(test$Age==i)]=k
  k=k+1
}

uniqstay<-unique(test$Stay_In_Current_City_Years[!is.na(test$Stay_In_Current_City_Years)])
k=1
for (i in uniqstay)
{
  stay_In_Current_City_Years_encode_test[which(test$Stay_In_Current_City_Years==i)]=k
  k=k+1
}

uniqprod<-unique(test$Product_ID[!is.na(test$Product_ID)])
k=1
for (i in uniqprod)
{
  product_encode_test[which(test$Product_ID==i)]=k
  k=k+1
}


test_new<-test
test_new$Gender<-gender_encode_test
test_new$City_Category<-city_encode_test
test_new$Product_ID<-product_encode_test
test_new$Stay_In_Current_City_Years<-stay_In_Current_City_Years_encode_test
test_new$Age<-age_encode_test

test_new<-test_new[,names(test_new) %in% c("Product_ID", "User_ID", "Product_Category_1","Gender")]
test_new<-as.matrix(test_new)
summary(test_new)
#str(test_new)"

################# ##		One-Hot Encoding for Train		  ################

for(names in c("Gender","City_Category","Stay_In_Current_City_Years","Age"))
{
  uniq<-unique(train[names][!is.na(train[names])])
  for(feature in uniq)
  {
    train[[toString(feature)]]<- 0
  
    train[[toString(feature)]][which(train[[names]]==feature)]=1
   
  }
  train[[names]]<-NULL
}

train$Product_Category_1<-train$Product_Category_1/max(train$Product_Category_1)
train$Product_Category_2<-train$Product_Category_2/max(train$Product_Category_2)
train$Product_Category_3<-train$Product_Category_3/max(train$Product_Category_3)
train$User_ID<-NULL
train$Product_ID<-NULL
head(train)
summary(train)

################# 		One-Hot Encoding for Test 		 ################

for(names in c("Gender","Occupation","City_Category","Stay_In_Current_City_Years","Age"))
{
  uniq<-unique(test[names][!is.na(test[names])])
  for(feature in uniq)
  {
    test[[toString(feature)]]<- 0
    
    test[[toString(feature)]][which(test[[names]]==feature)]=1
    
  }
  test[[names]]<-NULL
}

test$User_ID<-NULL
test$Product_ID<-NULL
head(test)
summary(test)
str(train[[10]])
names(train)


##############################  			Regression Trees  Method		#############################

formula<-as.formula(paste("Purchase~",paste( names(train)[!names(train) %in% c("Purchase")], collapse="+")))

train.model<-rpart(formula,train_new, cp=0.00005)

##############################			GBM Method			####################################

formula_gbm<-as.formula(paste("Purchase~",paste( names(train)[names(train) %in% c("Occupation","Marital_Status","Product_Category_2","Product_Category_3","F","M","A","B","C")], collapse="+")))

train_new<-train[,names(train)[names(train) %in% c("Occupation","Purchase","Marital_Status","Product_Category_2","Product_Category_3","F","M","A","B","C")]]

gbm.model<-gbm(formula_gbm, train_new, shrinkage = 0.01, interaction.depth=2, n.trees=100,distribution="gaussian", verbose=TRUE)

summary.gbm(gbm.model)

plot.gbm(gbm.model)

pred_gbm<- predict.gbm(gbm.model,test,n.trees=1000)



######################### 		Xgboost Method 				################################


xgboost.model <- xgboost(data = train_xgb, label = label_train, max.depth = 10,eta = 0.09, nround =700, verbose=1,objective = "reg:linear")

xgb_pred<-predict(xgboost.model, test_new)


################### 			Witing output to file  			 ##############

write.csv(test,"out3.csv")

write.csv(pred,"output_rpart.csv")

write.csv(pred_gbm,"output_gbm.csv")

write.csv(xgb_pred, "output_xgb.csv")
