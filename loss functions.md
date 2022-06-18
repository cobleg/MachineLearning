# Overview
Given DLNNs rely on optimisation algorithms to fit parameters, it is necessary to define loss (aka cost) functions.

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
