# Overview
Regularisation in the [[logistic model]] is implemented by adding a penalty to the logistic model inside the [[cost function]].

That is, define the [[cost function]] as:

$$ J(\boldsymbol{\vec{w}},b) = \frac{1}{m}\sum_{i=0}^{m-1}[-y^{(i)}log(f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)}))-(1-y^{i})log(1-f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)}))] $$
where:
$\boldsymbol{\vec{w}}$ is a vector of weights
$b$ is a bias term
$m$ is the number of data examples or samples
$i$ is the $i$-th sample, i.e. a specific example corresponding to a row in a tabular data set
$f_{\boldsymbol{\vec{w}}, b}(\boldsymbol{\vec{x}}^{(i)})=\text{sigmoid}(\boldsymbol{\vec{w}} \cdot \boldsymbol{\vec{x}}^{(i)} + b)$
$\boldsymbol{\vec{x}}$ is a vector of input variables from a training data set
$y$ is the target variable from the training data set

The regularised cost function simply adds this term to the cost function:

$$ \frac{\lambda}{2m} \sum_{j=0}^{n-1}w_j^2 $$
Hence, the regularised cost function is:

 $$ J(\boldsymbol{\vec{w}},b) = \frac{1}{m}\sum_{i=0}^{m-1}[-y^{(i)}log(f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)}))-(1-y^{i})log(1-f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)}))] + \frac{\lambda}{2m} \sum_{j=0}^{n-1}w_j^2 $$
# Code
The regularised cost function can be implemented like this:

```python
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar
             
    cost = cost/m                                                      #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar
```

A test example:
```python
import numpy as np
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)
```


## Worked example

```python


```