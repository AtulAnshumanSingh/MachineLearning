#####################  Data Input
m=47;
x <- read.table("ex3x.dat")
y <- read.table("ex3y.dat")
x<-matrix(as.numeric(unlist(x)),nrow=nrow(x))
y<-matrix(as.numeric(unlist(y)),nrow=nrow(y))

######################   Feature scaling

meanx1=mean(x[,1]);
meanx2=mean(x[,2]);
stdx1=sd(x[,1]);
stdx2=sd(x[,2]);
x[,1]=(x[,1]-meanx1)/stdx1;
x[,2]=(x[,2]-meanx2)/stdx2;

######################  Features matrix

theta<-matrix(0, 1, 2)
theta

######################  Learning Rate

alpha= 0.5;

######################  Cost function

J=matrix(0,1,50);
J_old=Inf;

######################   Optimization loops

for(i in 1:50)
{
    J[,1]<-(1/2*m)*(t(t(theta%*%t(x))-y))%*%(t(theta%*%t(x))-y)
    theta=theta-(1/2m)*(t(theta*t(x)-y))
}

