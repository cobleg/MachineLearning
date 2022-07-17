# Overview
Logistic function:

$$ g(z)=\frac{1}{1+e^{-z}} $$
where $0<g(z)<1$ and $z$ is an index that maps one or more input variables, typically as a linear combination where the linear parameters act as weights. That is:

$$ z= \boldsymbol{w} \cdot \boldsymbol{x} + b $$

There will be a threshold value of $g(z)$ that delineates the difference between the choices (e.g. 0.5). The logistic function belongs to a family of functions called [sigmoid functions](https://en.wikipedia.org/wiki/Sigmoid_function), chacterised by a `s` shaped function.

The output of $g(z)$ is typically interpreted as a probability of a given choice. A restriction on the output is that the list of output probabilities must sum to exactly 1.

That is $P(g(z)=0)+P(g(z)=1)=1$ 

# Loss function
The  [[loss function]] for the [[logistic model]] is specified as:

$$ L(f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)}), y^{i}) = \begin{cases} -log(f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)})) \ \ \ \ \ \ \ \ \text{       if } y^{(i)} = 1 \\
-log(1-f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)})) \ \text{     if }y^{(i)} = 0 \end{cases} $$
where
$\boldsymbol{\vec{w}}$ is a vector of weights to be estimated
$\boldsymbol{\vec{x}}^{(i)}$ is a vector of input variables corresponding to sample $i$
$b$ is the bias
$y^{(i)}$ is the target (actual observation) variable corresponding to sample $i$

The logistic loss curves for binary target labels are different for each outcome to ensure that the loss function is always convex as shown in this diagram:
![[Pasted image 20220710094802.png]]
Source: [Loss Function (Part II): Logistic Regression](https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11)

# Cost function
The cost function contains the logistic loss function within it:

$$ J(\boldsymbol{\vec{w}},b) = \frac{1}{m}\sum_{i=0}^{m-1}[-y^{(i)}log(f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)}))-(1-y^{i})log(1-f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)}))] $$

where
$m$ is the number of training examples in the training data set
$\boldsymbol{\vec{x}}$ is a vector of input data for training
$\boldsymbol{\vec{w}}$ is a vector of weights
$b$ is the bias parameter

# Gradient descent for logistic regression
The [[gradient descent]] algorithm is used to calculate the weights in logistic regression.


# Code
## Logistic function
```python
import numpy as np

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g
    
# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# Plot z vs sigmoid(z)
%matplotlib widget
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
```

## Cost function

```python
import numpy as np
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost

```

# Logistic regression

```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

%matplotlib inline

# load dataset
X_train, y_train = load_data("data/ex2data1.txt")

print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))

print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))

print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
    g = 1/(1+np.exp(-z))
    
    return g

print ("sigmoid(0) = " + str(sigmoid(0)))

print ("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))

# UNIT TESTS
from public_tests import *
sigmoid_test(sigmoid)

def compute_cost(X, y, w, b, lambda_= 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost 
    """

    m, n = X.shape
  
    loss_sum = 0.0
    for i in range(m):
        z_wb = 0
        # loop over each feature
        for j in range(n):
            z_wb_ij = np.dot(X[i][j],w[j])
            z_wb += z_wb_ij
            
        # add the bias term
        z_wb += b
        f_wb_i = sigmoid(z_wb)
        loss_sum +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    total_cost = loss_sum / m

    return total_cost

m, n = X_train.shape

# Compute and display cost with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w (zeros): {:.3f}'.format(cost))

# Compute and display cost with non-zero w
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at test w,b: {:.3f}'.format(cost))


# UNIT TESTS
compute_cost_test(compute_cost)
```

## Gradient for logistic regression

```python
def compute_gradient(X, y, w, b, lambda_=None): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) variable such as house size 
      y : (array_like Shape (m,1)) actual value 
      w : (array_like Shape (n,1)) values of parameters of the model      
      b : (scalar)                 value of parameter of the model 
      lambda_: unused placeholder.
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w. 
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ### 
    for i in range(m):
        z_wb = 0
        
        for j in range(n): 
            z_wb += np.dot(X[i][j],w[j]) #X[i, j] * w[j]
        z_wb += b
        f_wb = sigmoid(z_wb) 
        err_i  = f_wb  - y[i]                       #scalar
        
        dj_db_i = err_i 
        dj_db += dj_db_i
        
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
            
    dj_dw = dj_dw/m 
    dj_db = dj_db/m
    ### END CODE HERE ###

        
    return dj_db, dj_dw
    
# Compute and display gradient with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w (zeros):{dj_db}' )
print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}' )

# Compute and display cost and gradient with non-zero w
test_w = np.array([ 0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)

print('dj_db at test_w:', dj_db)
print('dj_dw at test_w:', dj_dw.tolist())

# UNIT TESTS    
compute_gradient_test(compute_gradient)
# dj_db at test_w: -0.5999999999991071
# dj_dw at test_w: [-44.831353617873795, -44.37384124953978]

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant
      
    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing


np.random.seed(1)
intial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
initial_b = -8


# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
    ### START CODE HERE ### 
    # Loop over each example
    for i in range(m):   
        z_wb = 0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb += np.dot(X[i][j],w[j])
        
        # Add bias term 
        z_wb += b 
        
        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb) 

        # Apply the threshold
        p[i] = f_wb>=0.5
        
    ### END CODE HERE ### 
    return p
# Test your predict code
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5

tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')

# UNIT TESTS        
predict_test(predict)
#Output of predict: shape (4,), value [0. 1. 1. 1.]

#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))
# Train Accuracy: 92.000000
```