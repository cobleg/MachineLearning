# Overview
Dropout refers to setting some neurons in a DLNN to zero. This is done to prevent overfitting.

![[Pasted image 20220607060405.png]]
Source: [Dropout in Practice](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html)

# Code
There is a dropout function available in PyTorch:
```python
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

```