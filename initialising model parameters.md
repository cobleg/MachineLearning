# Overview
DLNNs require parameter initialisation. This note contains instructions on how to implement this.

# Code

```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

## torch.normal
[torch.normal](https://pytorch.org/docs/stable/generated/torch.normal.html)

Returns a tensor of random numbers drawn from separate normal distributions whose mean and standard deviation are given.

