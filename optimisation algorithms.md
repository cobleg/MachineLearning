# Overview
DLNNs fit parameters using optimisation algorithms.

## Gradient descent
The equations are:

$$ w_{s+1} = w_s-\alpha \times \frac{\delta}{\delta w}J(w,b)$$

$$ b_{s+1} = b_s-\alpha \times \frac{\delta}{\delta b}J(w,b) $$
where:
$w_s$ is the weight (or slope) parameter at step $s$
$\alpha >0$ is the learning rate
$J(w,b)$ is the [[loss function]] (or cost function) , which is a function of the weight ($w$) and bias parameters ($b$). 

The learning rate ($\alpha$) has a big impact on the efficiency of the machine learning algorithm. If too small, achieving convergence for a machine learning model will take a long time (i.e. many #epochs).  If too large, then convergence in the macine learning algorithm may not converge at all.


# Code
## Stochastic gradient descent

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