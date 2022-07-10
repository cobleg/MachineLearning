# Overview
> **PyTorch** is an [open source](https://en.wikipedia.org/wiki/Open_source "Open source") [machine learning](https://en.wikipedia.org/wiki/Machine_learning "Machine learning") [framework](https://en.wikipedia.org/wiki/Software_framework "Software framework") based on the [Torch](https://en.wikipedia.org/wiki/Torch_(machine_learning) "Torch (machine learning)") library,(https://en.wikipedia.org/wiki/PyTorch#cite_note-3)[[4]](https://en.wikipedia.org/wiki/PyTorch#cite_note-4)[[5]](https://en.wikipedia.org/wiki/PyTorch#cite_note-5) used for applications such as [computer vision](https://en.wikipedia.org/wiki/Computer_vision "Computer vision") and [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing "Natural language processing"),(https://en.wikipedia.org/wiki/PyTorch#cite_note-6) primarily developed by [Meta AI](https://en.wikipedia.org/wiki/Meta_AI "Meta AI").
>[PyTorch](https://en.wikipedia.org/wiki/PyTorch)

# PyTorch functions
There are specific PyTorch functions that are required to implement and train a neural network.

`Flatten`: converts a matrix to a vector

`Linear`: applies a linear transformation to input data

`ReLU`: applies the rectified linear unit function

`Sequential`: adds modules to a container in the order that they are passed in the constructor.

`torch.mm`: performs matrix multiplication

`torch.rand`: creates a tensor filled with random numbers

## Linear neural network
`nn.Linear`: is a class (i.e. a model or blueprint) that creates objects referred to as linear models or [[linear neural networks]]. `CLASS torch.nn.Linear(in_features, out_features, bias=True)`

Applies a linear transformation to the incoming data: $\boldsymbol{Y} = \boldsymbol{x}W' + \boldsymbol{b}$ 

Parameters:

-   **in_features** – size of each input sample (i.e. size of x)
-   **out_features** – size of each output sample (i.e. size of y)
-   **bias** – If set to False, the layer will not learn an additive bias. Default: True

[LINEAR](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=nn%20linear#torch.nn.Linear)

See concrete example here: [What is the class definition of nn.Linear in PyTorch?](https://stackoverflow.com/questions/54916135/what-is-the-class-definition-of-nn-linear-in-pytorch)

## Loss function
Define a [[cost function]]
```python
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

### Notes
`nn.MSELoss()`: Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input *x* and target *y*. [MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)

`train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)` 

`.shape`: provides information about the number of dimensions of an array [Python NumPy shape with examples](https://pythonguides.com/python-numpy-shape/) 

## Logarithm of the root mean square error
Calculates the prediction error given the true values and a [[cost function]].
```python
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()
```

### Notes
`torch.clamp`: rescales elements of a numerical array to fit between given minimum and maximum values [Torch.Clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp)
`torch.log`: returns a new tensor with th natural logarithm of the elements of input. [Torch.Log](https://pytorch.org/docs/stable/generated/torch.log.html) 
`torch.sqrt`: rturns a new tensor with the square-root of the elements of the input. [torch.sqrt](https://pytorch.org/docs/stable/generated/torch.sqrt.html)  

## Model training function
This function is useful for [[training]] a neural network model.
```python
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

### Notes
`torch.optim.Adam`: Adam is a stochastic optimiser, with its name derived from `adaptive moment estimation`. Some of Adam’s advantages are that the magnitudes of parameter updates are invariant to rescaling of the gradient, its stepsizes are approximately bounded by the stepsize hyperparameter, it does not require a stationary objective, it works with sparse gradients, and it naturally performs a form of step size annealing. [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf) 
