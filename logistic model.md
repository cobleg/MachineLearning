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

    m, n = X.shape[0]
    
    loss_sum = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        loss_sum +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    total_cost = loss_sum / m

    return total_cost

m, n = X_train.shape

# Compute and display cost with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w (zeros): {:.3f}'.format(cost))

```