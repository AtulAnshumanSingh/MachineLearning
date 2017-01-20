################ Logistic Regression ##################

## Set Directory

m=47;

###############   Data Input ###############

x <- read.table("ex3x.dat")
y <- read.table("ex3y.dat")
x<-matrix(as.numeric(unlist(x)),nrow=nrow(x))
y<-matrix(as.numeric(unlist(y)),nrow=nrow(y))
x<-cbind(matrix(1, 47,1),x)

##############     Feature scaling  ###############

meanx2=mean(x[,2]);
meanx3=mean(x[,3]);
stdx2=sd(x[,2]);
stdx3=sd(x[,3]);
x[,2]=(x[,2]-meanx2)/stdx2;
x[,3]=(x[,3]-meanx3)/stdx3;

##############  Coeff matrix  ###############

theta<-matrix(0, 1, 3)

##############   Learning Rate  ###############

alpha= 1;

##############  Cost function  ###############

J=matrix(0,1,50);

##############  Optimization loops  ###############

for(i in 1:50)
{
    J[,i]<-(1/2*m)*(t(t(theta%*%t(x))-y))%*%(t(theta%*%t(x))-y)
    hypoth=theta%*%t(x)
    hypoth<-t(hypoth)
    theta=theta-alpha*(1/m)*(t(hypoth-y)%*%x)
}

#############    Plot   ###############

plot(1:50,J, "l")
theta
theta%*%(c(1,(1650-meanx2)/stdx2,(3-meanx3)/stdx3))
plot(x[,3],y)










