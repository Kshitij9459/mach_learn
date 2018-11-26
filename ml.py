import numpy as np
from mnist import MNIST
from mlxtend.data import loadlocal_mnist
import random
import math


def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1/(1 + math.exp(gamma))
  else:
    return 1/(1 + math.exp(-gamma))
    
    
X,Ytemp=loadlocal_mnist(images_path='/home/garg/Documents/Machine Learning/machine-learning-ex3/ex3/sample/train-images-idx3-ubyte', labels_path='/home/garg/Documents/Machine Learning/machine-learning-ex3/ex3/sample/train-labels-idx1-ubyte')

n=X.shape[1]
m=X.shape[0]
Y=np.zeros((m,10))
Y[:,0]=Ytemp
htheta=np.zeros(m)

X=np.insert(X,0,1,axis=1)
theta=np.ones((10,n+1))

for i in range(m):
	temp= Y[i][0]
	temp=(int)(temp)
	Y[i][temp]=1
	Y[i][0]=0
	
lamb=10000
alpha=0.01

for j in range(5):

	print(j)
	
	for k in range(10):

		product=np.matmul(X,theta[k])
		
	
		for i in range(m):
			htheta[i]=sigmoid(product[i])
			
	
		change_theta=(np.matmul(X.transpose(),htheta-Y[:,k]))/m
		change_theta=change_theta+(lamb*theta[k])/m
		change_theta[0]=change_theta[0]-(lamb*theta[k][0])/m
		theta[k]=theta[k]-alpha*change_theta
	

Xtest,Ytest=loadlocal_mnist(images_path='/home/garg/Documents/Machine Learning/machine-learning-ex3/ex3/sample/t10k-images-idx3-ubyte', labels_path='/home/garg/Documents/Machine Learning/machine-learning-ex3/ex3/sample/t10k-labels-idx1-ubyte')

m=Xtest.shape[0]
count=0
Xtest=np.insert(Xtest,0,1,axis=1)

for k in range(10):
	
	product=np.matmul(Xtest,theta[k])
	
	for i in range(m):
		
		htheta[i]=round(sigmoid(product[i]),0)
		if(htheta[i]==1):
			if(Ytest[i]==k):
				count=count+1
				
				
print(count*100/m)				
		

