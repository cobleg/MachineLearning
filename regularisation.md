# Overview
Regularization is a technique used to reduce the errors by fitting the function appropriately on the given training set and avoid overfitting.  The commonly used regularization techniques are:  
1.  L1 regularization
2.  L2 regularization (aka weight decay)
3.  Dropout regularization

[Regularization in Machine Learning](https://www.geeksforgeeks.org/regularization-in-machine-learning/)

## $L_2$ regularisation
The $L_2$ is a type of *norm* where a *norm* is a measure of magnitude and is similar to Pythagorus theorem:

$$||x||_2 = \sqrt{x_1^2+x_2^2+x_3^2+...+x_n^2}$$

[Norm](https://mathworld.wolfram.com/Norm.html)

The $L_2$ norm is applied to [[cost function]] to act as protection against overfitting. Technically, there is a penalty term added to the loss function that increases as the number of weights increases.

That is, given a loss function:

$$L(\boldsymbol{w}, b)$$
The $L_2$ is added:
$$L(\boldsymbol{w}, b) + \frac{\lambda}{2}|| \boldsymbol(w) ||^2$$
# Code
There is a norm function in PyTorch:
```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

Implementation in code:
```python
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # The L2 norm penalty term has been added, and broadcasting
            # makes `l2_penalty(w)` a vector whose length is `batch_size`
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', torch.norm(w).item())

# training without regularisation
train(lambd=0)

# training with regularisation
train(lambd=3)
```
