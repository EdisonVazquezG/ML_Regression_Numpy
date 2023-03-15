import matplotlib.pyplot as plt
import torch
import numpy as np 
from numpy.random import random
from mpl_toolkits.mplot3d import Axes3D



x = random((30,2)) ## generate random values in an array of dimensions 30x2

## generate labels corresponding to input data x 
y = np.dot(x,[2.,-3.]) + 1 ## multiplying the vector x with the two values 2 and -3 with respect to each column of x and then summing 1 
## generating the initial values for w and b
w_source = np.array([2.,-3.])
b_source = np.array([1.])



# Creation of class for gradient process
class gradient_learning:
    def __init__(self,x,y,b,epoch,lr,w=[]):
        self.w = w
        self.x = x
        self.y = y
        self.b = b
        self.epoch = epoch
        self.learning_rate = lr

    def forward_pass(self):
        return np.dot(self.x,self.w) + self.b
    
    def f_loss(self,y_pred):
        return (y_pred - self.y)**2

    def gradient(self):
        return 2*(np.dot(self.x,self.w) + self.b - self.y)*(self.x).T, 2*(np.dot(self.x,self.w) + self.b - self.y) ## this is tow function 

    def training(self):
        w_updated = np.array([0.,0.])
        
        for e in range(self.epoch):
            y_pred = self.forward_pass()
            loss = self.f_loss(y_pred)
            grad = self.gradient()
            w_updated[0] = sum(grad[0][0])
            w_updated[1] = sum(grad[0][1])
            b_grad = sum(grad[1])
            self.w = self.w - self.learning_rate*w_updated
            self.b = self.b - self.learning_rate*b_grad

            print("Progress - Epoch: " + str(e) + " | " + " Loss = " + str(sum(loss)) + " b_grad: " + str(b_grad) + " w_grad: " + str(w_updated))
            print("Parameters estimated: " + "w1 = " + str(self.w[0]) + " w2 = " + str(self.w[1]) + " b = " + str(self.b))
    
               
epoch_training = gradient_learning(x,y,0.10105847,100,0.01,w=[0.75873186,0.90213242])
epoch_training.training()


### Functions to plot the points in the space with the plane produced by the W and b values
def plot_figs(fig_num,elev,azim,x,y,weights,bias):
    fig = plt.figure(fig_num,figsize=(4,3))
    plt.clf()
    ax = Axes3D(fig,elev=elev,azim=azim)
    ax.scatter(x[:,0],x[:,1],y)
    ax.plot_surface(np.array([[0,0],[1,1]]),
    np.array([[0,1],[0,1]]),
    (np.dot(np.array([[0,0,1,1],[0,1,0,1]]).T,weights)+bias).reshape((2,2)),
    alpha=0.5)
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_zlabel("y")

def plot_views(x,y,w,b):
    elev = 43.5
    azim = -110
    plot_figs(1,elev,azim,x,y,w,b[0])
    plt.show()


plot_views(x,y,w_source,b_source)

## Testing with the best W and b values to fit the linear regression to the points
w_source_gradient = np.array([2.0237815725133887,-2.962509624488648])
b_source_gradient = np.array([-0.035727341257828456])

plot_views(x,y,w_source_gradient,b_source_gradient)


