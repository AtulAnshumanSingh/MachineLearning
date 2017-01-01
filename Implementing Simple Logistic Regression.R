################ Logistic Regression ##################

################# Data Input
x <- read.table("ex4x.dat")
y <- read.table("ex4y.dat")
x<-matrix(as.numeric(unlist(x)),nrow=nrow(x))
y<-matrix(as.numeric(unlist(y)),nrow=nrow(y))
x<-cbind(matrix(1, nrow(x),1), x)
m=nrow(x)

################# plotting of data  ################ 

plot(x[1:40, 2], x[1:40, 3], pch = 3, xlab = "Score1", ylab = "Score2") 
points(x[41:80, 2], x[41:80, 3], pch = 1, xlab = "Score1", ylab = "Score2")

#################  sigmoid function   ################ 
g<-function(n)
{
    return(1.0/(1.0+exp(-n)))
}

#################   features matrix  ################ 
theta<-matrix(0, 1, 3)

#################   Cost function  ################ 
J=matrix(0,1,50)

###################################
hypoth=matrix(11, nrow(x), 1)
hypoth1=t(theta%*%t(x))
for(i in 1: nrow(x))
{
    hypoth[i,1]=g(hypoth1[i,1])
}

###################################
#optimization loops
for(i in 1:50)
{    
    J[,1]<-(1/m)*(t(y)%*%log(hypoth)-t(1-y)%*%log(1-hypoth))
    hess=(1/m)*t(t(hypoth)%*%(1-hypoth)%*%t(x))%*%x              ### Hessian matrix computation
    theta=theta-(1/2m)*(t(theta*t(x)-y))
}









