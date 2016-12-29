setwd("C:/Users/ACER/Desktop/Study Material/Machine Learning")

x<-read.table("ex4x.dat")
x<-matrix(as.numeric(unlist(x)), nrow(x))
y<-read.table("ex4y.dat")
y<-matrix(as.numeric(unlist(x)), nrow(y))
