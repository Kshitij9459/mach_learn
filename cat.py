import numpy as np
import h5py
import tkinter
import matplotlib.pyplot as mpl
import scipy
import PIL as Image
from scipy import ndimage
from lr_utils import load_dataset



def sigmoid(z):
	
	s=1/(1+np.exp(-z))
	
	return s
	
def initialize(dim):

	w=np.zeros((dim,1))
	b=0
	
	assert(w.shape==(dim,1))
	assert(isinstance(b,float) or isinstance(b,int))
	
	return w,b
	
def propogate(w,b,X,Y,lambd):

	#Compute the  cost function and its gradient
	
	m=X.shape[1]
	
	A=sigmoid(np.matmul(w.T,X)+b)
	
	cost = -1*(np.sum(np.multiply(Y,np.log(A))) + np.sum(np.multiply(1-Y,np.log(1-A))))/m + lambd*np.sum(w*w)/(2*m) 
	
	grad_w=(1/m)*np.matmul(X,(A-Y).T)+lambd*w/m
	grad_b=(1/m)*(np.sum(A-Y))
	
	#Checking 
	
	assert(grad_w.shape==w.shape)
	assert(grad_b.dtype==float)
	
	return grad_w,grad_b,cost
	
def optimize(w,b,X,Y,no_of_iterations,alpha,lambd,print_cost=False):
	
		#optimizing w and b parameters
	costs=[]
	
	for i in range(no_of_iterations):
		
		grad_w,grad_b,cost=propogate(w,b,X,Y,lambd)
		
		w=w-alpha*grad_w
		b=b-alpha*grad_b
		
		if i%10 == 0:
			costs.append(cost)
			
		if print_cost and i%100==0:
			print("Cost after %i th iteration:%f\n"%(i,cost))
			
	return w,b,costs
	
def predict(w,b,X):

	#Predict the value of y
	
	m=X.shape[1]
	Y_prediction=np.zeros((1,m))
	w=w.reshape(X.shape[0],1)
	
	A=sigmoid(np.matmul(w.T,X) + b)
	
	for i in range(m):
		
		if A[0][i]> 0.5:
			Y_prediction[0][i]=1
			
	assert(Y_prediction.shape==(1,m))
	
	return Y_prediction
	
								
			

train_set_x_org,train_set_y,test_set_x_org,test_set_y,classes=load_dataset()
n_train=train_set_x_org.shape[0]
n_test=test_set_x_org.shape[0]
num_px=train_set_x_org.shape[1]

train_set_x_flatten=train_set_x_org.reshape(train_set_x_org.shape[0],-1).T
test_set_x_flatten=test_set_x_org.reshape(test_set_x_org.shape[0],-1).T

train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

w,b=initialize(train_set_x.shape[0])

alpha=0.001
no_of_iteration=10000
lambd=100

w,b,costs=optimize(w,b,train_set_x,train_set_y,no_of_iteration,alpha,lambd,True)

Y_prediction_train=predict(w,b,train_set_x)
Y_prediction_test=predict(w,b,test_set_x)

print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

mpl.plot(costs)
mpl.ylabel('cost')
mpl.xlabel('iterations')
mpl.title("Learning rate =" +str(alpha))
mpl.show()

my_image = "IMG-20181130-WA0022.jpg"

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T

my_predicted_image = predict(w, b, my_image)

mpl.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
