########################### 			Load Data   		###################

install.packages("png")
library(png)

############ 			Train data 			##########################

list_file<-list.files("\Path", pattern = "*.png")    ##		Creating list of PNG files
trainx<-matrix(-1,10000, 784)
k=1
list_file1<-head(list_file, 10000)
setwd("\Path")
for (i in list_file1)
{
  trainx[k,1:784]<-matrix(readPNG(i), 1, 784)
  k=k+1
}
setwd("\Path")
trainy<-as.matrix(read.csv("train.csv"))
trainy<-head(trainy,10000)

############# 			Test data 		###########################

list_file_test<-list.files("\Path", pattern = "*.png")

testx<-matrix(-1,21000, 784)
k=1
#list_file_test<-head(list_file_test, 1000)
setwd("\Path")
for (i in list_file_test)
{
  testx[k,1:784]<-matrix(readPNG(i), 1, 784)
  k=k+1
}
nrow(trainx)
setwd("\Path")


##########################  		Show images  			##################

image(matrix(trainx[2,], 28)[,28:1], col=gray(12:1/12))
image(matrix(testx[2,], 28)[,28:1], col=gray(12:1/12))


########### sigmoid function  #########

g<-function(n)
{
  return(1.0/(1.0+exp(-n)))
}

########### sigmoid gradient  #########

GS<-function(n)
{
  return(g(n)*(1-g(n)))
}

########### parameters ############

# Theta1 has dimension of 25 X (28*28 +1)
# Theta2 has dimension of 10 X 26

####### initialization of Thetas ############

epsilon_init=0.05
Theta00=matrix(runif(800*(28*28+1), 0, 1)*2*epsilon_init-epsilon_init, 800, 28*28+1)
Theta2=matrix(runif(10*(801), 0, 1)*2*epsilon_init-epsilon_init, 10, 801)

####### optimization loops #########

#######  cost function  #########

m=0.5
alpha=0.005
lamdba=0.01
maxIter=15
J=matrix(0,1,maxIter)
error=matrix(0, 10, 1)
out_delta=matrix(0,10,1)
hidd_delta2=matrix(0,801,1)

for(i in 1:maxIter)
{
  if(i%%25==0)
  {
    m=alpha*0.3
  }
  for(k in 1:10000)
  {
    
    ########## forward propagation ##########
    
    a1=cbind(t(trainx[k,]), 1)
    a1=t(a1)
    z2=Theta00%*%a1
    a2=g(z2)
    a2=rbind(a2,1)
    z5=Theta2%*%a2
    a5=g(z5)
    #cat("\n", a3, "\n")
    
    ########## backpropagation ##############
    
    y_out=matrix(0, 10,1)
    y_out[match(trainy[k,2],c(1,2,3,4,5,6,7,8,9,10))]=1
    
    J[1,i]=J[1,i]+m*((-1)*t(y_out)%*%log(a5)-t(1-y_out)%*%log(1-a5))
    
    ########## output layer ####################
    
    error=a5-y_out
    out_delta=error*GS(z5)
    
    ########## hidden layer 3 ##################
    
    hidd_delta2=(head(t(Theta2)%*%out_delta,800))*GS(z2)
    
    ########## delta calculations  #############
    
    #delta=delta+dim(hidd_delta0%*%a1
    
    ########## weight updation ###############
    
    Theta00=Theta00-alpha*(m*(hidd_delta2%*%t(a1))+lambda*Theta00)    #########    With Regularization
    
    Theta2=Theta2-alpha*(m*(out_delta%*%t(a2))+lamdba*Theta2)
    
    cat("\n","i=", i,"\t", "k=", k)
  }
  cat("\n", i)
}

########### plot cost function ################

plot(1:maxIter,J, 'l')

########################## Output and prediction  #################
sink("test_out1.csv")
inp2=21000
count=0
for(i in 1:inp2)
{
  a1=cbind(t(testx[i,]), 1)
  a1=t(a1)
  z2=Theta00%*%a1
  a2=g(z2)
  a2=rbind(a2,1)
  z5=Theta2%*%a2
  a5=g(z5)
  for(k in 1:10)
  {
    if(a5[k,1]==max(a5))
    {
      index=k;  
    }
  }
  cat(list_file_test[i],",", index,"\n")
}
sink()

#################### percentage success ##########

eff=(count/inp2)*100

print(eff)
