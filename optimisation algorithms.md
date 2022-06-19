# Overview
DLNNs fit parameters using optimisation algorithms.

## Gradient descent
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

Assuming the loss function:

$$ J(w,b) = \frac{1}{2}\Sigma_{i=1}^{m}(f(_{w,b}(x^{(i)})-y^{(i)})^2 $$

where 
$f(_{w,b}(x^{(i)})=wx^{(i)}+b$

The gradient descent equations are by the [chain rule](https://en.wikipedia.org/wiki/Chain_rule):

$$ w_{s+1} = w_s-\alpha \times \frac{1}{m}\Sigmax^{(i)}(wx^{(i)}+b-y^{(i)}) $$


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