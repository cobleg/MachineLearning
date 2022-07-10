# Overview
The Squared Error Cost function is defined as:

$$ J(\boldsymbol{\vec{w}},b) = \frac{1}{2}\Sigma_{i=1}^{m}(f(_{\boldsymbol{\vec{w}},b}(x^{(i)})-y^{(i)})^2 $$
where
$m$ is the number of observations
$f_{w,b}(x)$ is a function describing the relationship between input variable $x$ and output variable $y$. 

Note that this cost function only has a single global minimum and no local minima. 

# Code
## Squared Error Cost
The squared difference between the predicted and actual. This is used for fitting linear regression models.
```python
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

