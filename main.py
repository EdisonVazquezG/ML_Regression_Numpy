import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def ml_regression(X,y,iter):

    ## Initialize values for m and b
    m = np.random.randn() ## normal standar distribution
    b = np.random.randn()
    N = len(X)
    log, mse = [], [] # lists to store learning process

    ## Initialize the value for learning rate
    learning_rate = 1e-6

    for t in range(iter):
        ## evaluation function 
        y_pred = m*X + b
        
        ## cost function
        loss = (np.square(y_pred - y).sum())/N
                
        # Backprop to compute gradients of m,b with respect to loss
        # Update weights
        b -= learning_rate*((y_pred - y).sum())/N
        m -= learning_rate*(((y_pred - y)*x).sum())/N

        log.append((m, b))
        mse.append(mean_squared_error(y, (m*X + b)))        
    
    return m, b, log, mse

## Create random input and output data
x = np.linspace(0, 100, 100)
delta = np.random.uniform(2,10, size=(100,))
y = .4 * x +3 + delta
plt.plot(x, y, 'ro', label='y=m*X + b')
plt.show() 

## Results
iter = 1000
m, b, log, mse = ml_regression(x,y,iter)

### MSE graph
epoch_ = np.linspace(0, 1000, 1000)
plt.plot(epoch_, mse, '-r', label='MSE')
plt.show() 


### straight line in the cloud points
y_ = m*x + b
plt.plot(x, y, 'ro', label='y = m*X + b')
plt.plot(x,y_, '-r', color = "blue")
plt.show() 


