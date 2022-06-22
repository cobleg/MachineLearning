# Overview
> Linear neural networks predict the output as a linear function of the inputs.  Every node doesn't do anything fancier than `Sum(W*x)`. This sum is passed to the next layer.  Very simple, very intuitive. 

> Non linear, as the name suggest, break the linearity with the help of a bunch of activation functions.  A neuron does calculate `Sum(W_x)` but passes `Activation(Sum(W_x))` to break linearity.

[Difference between linear and nonlinear neural networks?](https://www.kaggle.com/general/196486)

Linear neural networks are similar to linear statistical regression. This is used to predict continuous quantities, e.g. how much, how many of something.

One layer in a linear neural network can be thought of as a linear model:

$$ f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot{} \vec{x}+b $$
where
$$ \vec{w} = [w_1,w_2,w_3,...,w_n]$$ $$ \vec{x} = [x_1,x_2,x_3,...,x_n]$$

# Code
## Pytorch

```python
## Set up linear neural network
# load required libraries and initialise parameters
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# this section creates a data set
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
# end of section

# read the dataset
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays) # the asterisk indicates that `data_arrays` is a parameter
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# `nn` is an abbreviation for neural networks
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))

# initialise parameters in `net`
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# define the loss function
loss = nn.MSELoss() # mean squared error loss function

# define the trainer
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

## end of linear neural network set up

## Now train the model
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# review the resulting estimated parameters
w = net[0].weight.data
print('error in estimating w:', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)

```

## Standard linear function
Set up the required Python libraries
```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

```

Define the sample size `m` and create a data sample

```Python
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")

# x_train is the input variable 
# y_train is the target 
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

m = x_train.shape[0] # shape[0] reports the length of the tensor
print(f"Number of training examples is: {m}")
```

Plot the data points

```Python
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
```

Define a linear function

```Python


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb
```

Compute the model and plot the results

```Python
w = 100
b = 450
print(f"w: {w}")
print(f"b: {b}")

# x_train is the input variable 
# y_train is the target 
x_train = np.array([1.0, 2.0])

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```

