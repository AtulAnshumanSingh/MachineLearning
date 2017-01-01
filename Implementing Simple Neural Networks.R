################   Neural Networks  ############

######################## Data input ########################

load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('train-images.idx3-ubyte')
  test <<- load_image_file('t10k-images.idx3-ubyte')
  train$y <<- load_label_file('train-labels.idx1-ubyte')
  test$y <<- load_label_file('t10k-labels.idx1-ubyte')  
}


##########################  Show images  #################

image(matrix(train$x[1,], 28)[,28:1], col=gray(12:1/12))


#########################  Subsetting the data  ###########


inp=5000
inp2=1000

trainy<-matrix(as.numeric(unlist(train$y)), 60000)
trainy<-head(trainy,inp)
trainx<-matrix(as.numeric(unlist(train$x)), 60000, 784)
trainx<-head(trainx,inp)

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

epsilon_init=1.5
Theta00=matrix(runif(200*(28*28+1), 0, 1)*2*epsilon_init-epsilon_init, 200, 28*28+1)
Theta0=matrix(runif(100*(201), 0, 1)*2*epsilon_init-epsilon_init, 100, 201)
Theta1=matrix(runif(25*(101), 0, 1)*2*epsilon_init-epsilon_init, 25, 101)
Theta2=matrix(runif(10*(26), 0, 1)*2*epsilon_init-epsilon_init, 10, 26)

####### optimization loops #########

#######  cost function  #########

m=9
maxIter=1.5
J=matrix(0,1,maxIter)
error=matrix(0, 10, 1)
out_delta=matrix(0,10,1)
hidd_delta0=matrix(0,201,1)
hidd_delta1=matrix(0,101,1)
hidd_delta2=matrix(0,26,1)

for(i in 1:maxIter)
{
    for(k in 1:inp2)
    {
        
        ########## forward propagation ##########
        
        a1=cbind(t(trainx[k,]), 1)
        a1=t(a1)
        z2=Theta00%*%a1
        a2=g(z2)
        a2=rbind(a2,1)
        z3=Theta0%*%a2
        a3=g(z3)
        a3=rbind(a3,1)
        z4=Theta1%*%a3
        a4=g(z4)
        a4=rbind(a4,1)
        z5=Theta2%*%a4
        a5=g(z5)
        #cat("\n", a3, "\n")
        
        ########## backpropagation ##############
        
        y_out=matrix(0, 10,1)
        y_out[match(trainy[k,1],c(1,2,3,4,5,6,7,8,9,10))]=1
        
        J[1,i]=J[1,i]+(1/m)*((-1)*t(y_out)%*%log(a5)-t(1-y_out)%*%log(1-a5))
        
        ########## output layer ####################
        
        error=a5-y_out
        out_delta=error*GS(z5)
        
        ########## hidden layer 3 ##################
        
        hidd_delta2=(head(t(Theta2)%*%out_delta,25))*GS(z4)
        
        ########## hidden layer 2 ##################
        
        hidd_delta1=(head(t(Theta1)%*%hidd_delta2,100))*GS(z3)
        
        ########## hidden layer 1 ##################
        
        hidd_delta0=(head(t(Theta0)%*%hidd_delta1,200))*GS(z2)
        
        ########## delta calculations  #############
        
        #delta=delta+dim(hidd_delta0%*%a1
        
        ########## weight updation ###############
        
        Theta00=Theta00-(1/m)*(hidd_delta0%*%t(a1))
        
        Theta0=Theta0-(1/m)*(hidd_delta1%*%t(a2))
        
        Theta1=Theta1-(1/m)*(hidd_delta2%*%t(a3))
        
        Theta2=Theta2-(1/m)*(out_delta%*%t(a4))
    }
  cat("\n", i)
}

########### plot cost function ################

plot(1:maxIter,J, 'l')

########################## Output and prediction  #################

count=0
for(i in 1:inp2)
{
  a1=cbind(t(trainx[i,]), 1)
  a1=t(a1)
  z2=Theta00%*%a1
  a2=g(z2)
  a2=rbind(a2,1)
  z3=Theta0%*%a2
  a3=g(z3)
  a3=rbind(a3,1)
  z4=Theta1%*%a3
  a4=g(z4)
  a4=rbind(a4,1)
  z5=Theta2%*%a4
  a5=g(z5)
  for(k in 1:10)
  {
    if(a5[k,1]==max(a5))
    {
      index=k;  
    }
  }
  if(index==trainy[i,1])
  {
    print(index)
    count=count+1
  }
}


#################### percentage success ##########

eff=(count/inp2)*100

print(eff)







