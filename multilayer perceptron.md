# Overview
The multilayer perceptron (MLP) is a neural network that contains multiple layers as in:

![[Pasted image 20220606131723.png]]

Source: [Dive into Deep Learning, multilayer perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html)

Note:
1. layers that require computation are included in the count, so this MLP has 2 layers, not 3.
2. the MLP layers are `fully connected`.
3. every input influences every neuron in the hidden layer and each of these influences every neuron in the output layer.

# Formula
The MLP is a system of equations:

$$ \boldsymbol{H} = \sigma(\boldsymbol{XW^{(1)}}+\boldsymbol{b^{(1)}})$$
$$ \boldsymbol{O}=\boldsymbol{HW^{(2)}} + \boldsymbol{b^{(2)}} $$

where:
$\boldsymbol{H}$ is the hidden layer variable matrix
$\sigma$ is a nonlinear activation function
$\boldsymbol{X}$ is the input variable matrix
$\boldsymbol{W}$ is the weights matrix
$\boldsymbol{b}$ is the bias matrix
$\boldsymbol{O}$ is the output matrix

# Code
```python
# load libraries and initialise parameters
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# set up the activation function
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# construct the model
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # Here '@' stands for matrix multiplication
    return (H@W2 + b2)

# set up the loss function
loss = nn.CrossEntropyLoss(reduction='none')

# train the model
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# create prediction and compare to actuals
d2l.predict_ch3(net, test_iter)

```

## Concise code

```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

```
