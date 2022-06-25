# Overview
Once the code and input data are prepared for a DLNN, a new model can be trained.

# Training steps
Execute the following loop:

-   Initialize parameters $(𝐰,𝑏)$ 
-   Repeat until done
    -   Compute gradient $𝐠←∂(𝐰,𝑏)\frac{1}{|B|}\Sigma_{𝑖∈}l(𝐱^{𝑖},𝑦^{𝑖},𝐰,𝑏)$ 
    -   Update parameters $(𝐰,𝑏)←(𝐰,𝑏)−𝜂𝐠$

# Code

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

`lr` is the learning rate
`epoch` An epoch means training the neural network with all the training data for one cycle. In an epoch, we use all of the data exactly once. A forward pass and a backward pass together are counted as one pass. [Epoch in neural networks](https://www.baeldung.com/cs/epoch-neural-networks)
