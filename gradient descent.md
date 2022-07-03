# Overview
[Gradient descent](https://builtin.com/data-science/gradient-descent) is an optimisation algorithm for finding a local minimum of a differentiable, [convex function](https://en.wikipedia.org/wiki/Convex_function). Gradient descent is simply used in machine learning to find the values of a function's parameters (coefficients) that minimize a [[loss function]] as far as possible.

# Gradient descent equations
The equations are:

$$ w_{s+1} = w_s-\alpha \times \frac{\partial}{\partial w}J(w,b)$$

$$ b_{s+1} = b_s-\alpha \times \frac{\partial}{\partial b}J(w,b) $$
where:
$w_s$ is the weight (or slope) parameter at step $s$
$\alpha >0$ is the learning rate
$J(w,b)$ is the [[loss function]] (or cost function) , which is a function of the weight ($w$) and bias parameters ($b$). 

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

# Gradient descent algorithm
With the equations defined, the gradient descent algorithm is:

repeat until convergence {
$$ w_{s+1} = w_s-\alpha \times \frac{1}{m}\Sigma_{i=1}^{m}x^{(i)}(wx^{(i)}+b-y^{(i)}) $$
$$ b_{s+1} = b_s-\alpha \times \frac{1}{m}\Sigma_{i=1}^{m}(wx^{(i)}+b-y^{(i)}) $$} stop when change in $w$ and $b$ are close to zero (e.g. less than 0.001)

# Code

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
        dj_db += dj_db_i # cumultive sum
        dj_dw += dj_dw_i # cumulative sum
    dj_dw = dj_dw / m # calculate the average 
    dj_db = dj_db / m # calculate the average
        
    return dj_dw, dj_db
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