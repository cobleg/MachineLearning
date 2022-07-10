# Overview
Given machine learning algorithms typically rely on optimisation algorithms to fit parameters, it is necessary to define  cost functions. A cost function is defined as a function that measures the aggregate difference between the observed measures of a target variable and the machine learning algorithm's output (i.e. the predicted values) over a given training set. 

There are various types of cost functions, such as the:
- [[Squared Error Cost]]
- [[Logistic Cost]]

## Cost function versus loss function
The machine learning literature refers to cost functions and loss functions. A [[loss function]] is defined as a measure of difference of a single example (i.e. a specimen of the target variable) and the predicted value. By contrast, a cost function measures the aggregate (i.e. average) deviation across the entire training set.

A given cost function will be a function of a loss function. For example, define a loss function as the absolute deviation between predicted and target values for a given target value $i$:

$$ L(f_{\boldsymbol{\vec{w}},b}, y^{(i)})= g(\hat{y}^{(i)}, y^{(i)} )$$
where

$$ \hat{y}^{(i)} = f(\boldsymbol{\vec{w}}(x^{(i)}), b) $$
and
$\hat{y}^{(i)}$ is predicted value based on a function describing the relationship between input variable $x^{(i)}$ and target variable $y^{(i)}$ for sample $i$. 
$\boldsymbol{\vec{w}}$ is a vector of weights
$b$ is the bias parameter

The cost function will be defined as a function of the loss function:
$$ J(\boldsymbol{\vec{w}},b)=h(g(f_{\boldsymbol{\vec{w}},b}, y^{(i)}) )$$

## Stochastic gradient descent
> **Stochastic gradient descent** (often abbreviated **SGD**) is an iterative method for optimizing an [objective function](https://en.wikipedia.org/wiki/Objective_function "Objective function") with suitable [smoothness](https://en.wikipedia.org/wiki/Smoothness "Smoothness") properties (e.g. [differentiable](https://en.wikipedia.org/wiki/Differentiable_function "Differentiable function") or [subdifferentiable](https://en.wikipedia.org/wiki/Subgradient_method "Subgradient method")).
[Wikipedia](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

```python
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

`lr` is the learning rate 

# Application example
This code block provides an example

```Python
import copy, math
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# construct training data
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

def compute_cost(X, y, w, b): 
    """
    compute cost
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
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing
    
# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.

# some gradient descent settings
alpha = 5.0e-7
iterations = 1000

# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
    
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()
    
```