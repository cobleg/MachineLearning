# Overview
Given DLNNs rely on optimisation algorithms to fit parameters, it is necessary to define loss (aka cost) functions.

One common loss function is the Squared Error Loss function:

$$ J(w,b) = \frac{1}{2}\Sigma_{i=1}^{m}(f(_{w,b}(x^{(i)})-y^{(i)})^2 $$
where
$m$ is the number of observations
$f_{w,b}(x)$ is a function describing the relationship between input variable $x$ and output variable $y$. 
$w$ is the slope parameter
$b$ is the bias parameter
$y$ is the output variable

Note that this cost function only has a single global minimum and no local minima. 

# Code
## Squared loss function
The squared difference between the predicted and actual. This is used for fitting linear regression models.
```python
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

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

# cosntruct bias and weights
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

# construct a linear regression model
def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b     
    return p    



```