# Overview
An activation function amplifies or compresses numeric input depending on whether it is greater than, or less than, a given input.

For example, using a threshold of 0.5, the activation function could change the value 0.49 to 0.01 while the value 0.5 is changed to 1.

# Activation functions
## ReLU
### Formula
Rectified linear unit (ReLU):

$$ReLU(x)=max(x,0)$$

![[Pasted image 20220606135159.png]]
Source: [Dive into Deep Learning; multilayer perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)

### Code
```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

## Sigmoid
### Formula
The sigmoid function creates a non-linear `s` shape. For example:

$$sigmoid(x)=\frac{1}{1+exp(-x)}$$
![[Pasted image 20220606135340.png]]
Source: [Dive into Deep Learning; multilayer perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)

### Code
```python
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

## TANH
### Formula
The tanh function squashes its inputs to the interval -1, 1.

$$tanh(x)=\frac{1-exp(-2x)}{1+exp(-2x)}$$

### Code
```python
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

```

