# Overview
[Gradient descent](https://builtin.com/data-science/gradient-descent) is an optimisation algorithm for finding a local minimum of a differentiable, [convex function](https://en.wikipedia.org/wiki/Convex_function). Gradient descent is simply used in machine learning to find the values of a function's parameters (coefficients) that minimize a [[cost function]] as far as possible.

Note that gradient descent is efficiently implemented as [[batch gradient descent]]. 

# Gradient descent equations
The equations are:

$$ w_{s+1} = w_s-\alpha \times \frac{\partial}{\partial w}J(w,b)$$

$$ b_{s+1} = b_s-\alpha \times \frac{\partial}{\partial b}J(w,b) $$
where:
$w_s$ is the weight (or slope) parameter at step $s$
$\alpha >0$ is the learning rate
$J(w,b)$ is the [[cost function]] (or cost function) , which is a function of the weight ($w$) and bias parameters ($b$). 

Gradient descent will take progressively smaller steps as the machine learning algorithm approaches a minimum (eighter local or global) since the derivate will be smaller in each step. 

The learning rate ($\alpha$) has a big impact on the efficiency of the machine learning algorithm. If too small, achieving convergence for a machine learning model will take a long time (i.e. many #epochs).  If too large, then convergence in the macine learning algorithm may not converge at all.

If stuck at a local minimum (as distinct to a global minimum), it will be necessary to jump, possibly by changing the parameters.

## Gradient descent for a linear model

Assuming the loss function:

$$ J(w,b) = \frac{1}{2}\Sigma_{i=1}^{m}(f(_{w,b}(x^{(i)})-y^{(i)})^2 $$

where 
$f(_{w,b}(x^{(i)})=wx^{(i)}+b$

The gradient descent equations are by the [chain rule](https://en.wikipedia.org/wiki/Chain_rule):

$$ w_{s+1} = w_s-\alpha \times \frac{1}{m}\Sigma_{i=1}^{m}x^{(i)}(wx^{(i)}+b-y^{(i)}) $$
$$ b_{s+1} = b_s-\alpha \times \frac{1}{m}\Sigma_{i=1}^{m}(wx^{(i)}+b-y^{(i)}) $$
## Gradient descent for a logistic model
The [[loss function]] for logistic descent is different and can be reviewed in the note on the [[logistic model]]. 

The first derivative of the [[logistic model]] is:

$$ f(_{w,b}(x^{(i)}))=\frac{1}{1+ e^{-\boldsymbol{\vec{w}} \cdot \boldsymbol{\vec{x}}^{(i)}+b}} $$

# Gradient descent algorithm
With the equations defined, the gradient descent algorithm is:

repeat until convergence {
$$ w_{s+1} = w_s-\alpha \times \frac{1}{m}\Sigma_{i=1}^{m}x^{(i)}(wx^{(i)}+b-y^{(i)}) $$
$$ b_{s+1} = b_s-\alpha \times \frac{1}{m}\Sigma_{i=1}^{m}(wx^{(i)}+b-y^{(i)}) $$} stop when change in $w$ and $b$ are close to zero (e.g. less than 0.001)



# Code
There is custom code below to show the details of implementation. However, in practice the [[Python]] library [sci-kit learn](https://scikit-learn.org/stable/index.html) is used. 

## Gradient linear descent
Custom code:
```Python
#Function to calculate the cost
def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i # cumulative sum
        dj_dw += dj_dw_i # cumulative sum
    dj_dw = dj_dw / m # calculate the average 
    dj_db = dj_db / m # calculate the average
        
    return dj_dw, dj_db
```

## Gradient logistic descent
```python
import copy, math

def compute_gradient_logistic(X, y, w, b): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar
        
    return dj_db, dj_dw  

def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         #return final w,b and J history for graphing

```
## Stochastic gradient descent
Custom code:
```python
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

## torch.nograd
[torch.no-grad](https://pytorch.org/docs/stable/generated/torch.no_grad.html?highlight=torch%20no_grad#torch.no_grad) is one of several mechanisms that can enable or disable gradients locally see [Locally disabling gradient computation](https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc) for more information on how they compare.

# Scikit-learn functions
Relevant [scikit-learn](https://scikit-learn.org/stable/about.html) functions are:
- [`sklearn.linear_model.SGDRegressor`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model "sklearn.linear_model")
- 