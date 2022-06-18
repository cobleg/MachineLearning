# Overview
Linear algebra (or matrix operations) are routinely required in DLNN models. 

# Code
```python
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return torch.matmul(X, w) + b
```

## torch.matmul
[Matrix product of two tensors](https://pytorch.org/docs/stable/generated/torch.matmul.html)

