# Objective
This notebook provides a guide in using [[PyTorch]] for creating and fitting deep learning neural networks.

# Coding
Deep Learning Neural Networks (DLNN) require set up of the data and scripts to construct and run.

## Set up steps
Key steps are:
1. Set up the enivronment and subdirectories/folders
2. Prepare input data sets
	1. [[Create or obtain data]] 
	2. [[load data]] 
	3. [[normalising data]] 
3. Write script files
4. Execute and write output files to disk

## Interactive build process & recipes
Jupyter notebooks can be set up to contain notes and code blocks. This allows for code to be prepared in an interactive manner.

### Useful code references
Given that everything needs to be coded first, here are some handy recipes for the build process:
1. [[visual analytics]]
2. [[initialising model parameters]] 
3. [[matrix operations]] 
4. [[optimisation algorithms]] 
5. [[loss functions]] 
6. [[regularisation]] 
7. [[dropout]] 
8. [[distribution shift]] 

Other useful Python functions:
1. [[slicer]] 
2. [[zip]]
3. 

# Process
The DLNN process occurs in several steps:

1. [[training]] 
2. validating
3. testing

# Architectures
There are various architectures used in defining a neural network:

1. [[linear neural networks]] 
2. [[softmax regression]] 
3. [[multilayer perceptron]] 

# References
[Dive into deep learning](https://d2l.ai/index.html)

[[Python]] 

[Pytorch](https://pytorch.org/)

# Glossary of terms
There are many specific names given to various parts of the DLNN architecture and process as follows:

`activation`: is the act of switching neuron on as if it is a switch with on/off states. An `activation function` contains a threshold value that determines whether the numeric input is amplified to a much larger number or compressed. For example, using a threshold value of 0.5, an [[activation functions]] could convert the value 0.49 to zero (or close to it) while the value of 0.51 is converted to 1. This creates a binary switching between states: 0,1.

`back propagation`: Backpropagation, short for _backward propagation of errors is a method for calculating derivatives inside feedforward [neural networks](https://deepai.org/machine-learning-glossary-and-terms/neural-network). In [fitting a neural network](https://en.wikipedia.org/wiki/Artificial_neural_network#Learning "Artificial neural network"), backpropagation computes the [gradient](https://en.wikipedia.org/wiki/Gradient "Gradient") of the [loss function](https://en.wikipedia.org/wiki/Loss_function "Loss function") with respect to the [weights](https://en.wikipedia.org/wiki/Glossary_of_graph_theory_terms#weight "Glossary of graph theory terms") of the network.

`epoch`: An epoch means training the neural network with all the training data for one cycle. In an epoch, we use all of the data exactly once. A forward pass and a backward pass together are counted as one pass. [Epoch in neural networks](https://www.baeldung.com/cs/epoch-neural-networks)

`features`:  a **feature** is an individual measurable property or characteristic of a phenomenon, a numeric feature can be conveniently described by a feature vector. [feature (machine learning)](https://en.wikipedia.org/wiki/Feature_(machine_learning)

`feedforward neural network`: A **feedforward neural network (FNN)** is an [artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network "Artificial neural network") wherein connections between the nodes do _not_ form a cycle. [Wikipedia](https://en.wikipedia.org/wiki/Feedforward_neural_network) 

`flatten`: convert a tensor into a single dimension (i.e. a vector). For example, a tensor with dimensions `2 x 3 x 4` has a total of 24 elements, which can be flattened to a single vector of `1 x 24` which still has 24 elements. [flatten a PyTorch tensor](https://www.aiworkbox.com/lessons/flatten-a-pytorch-tensor) 

`fully connected`: The layer is said to be _fully-connected_ when each of its inputs is connected to each of its outputs by means of matrix-vector multiplication.

`learning rate`: The learning rate affects how big a step you update your parameters to move towards the minimum point in your loss function. If your step is too big, you might miss the minimum point on your loss function. If your step is too small, it might take too long to converge to the minimum point. [what is learning rate in neural network?](https://datascience.stackexchange.com/questions/69917/what-is-learning-rate-in-neural-network)

`loss function`: The loss function in a neural network quantifies the difference between the expected outcome and the outcome produced by the machine learning model. From the loss function, we can derive the gradients which are used to update the weights. The average over all losses constitutes the cost. [An Introduction to Neural Network Loss Functions](https://programmathically.com/an-introduction-to-neural-network-loss-functions/) 

`overfitting`: occurs when a model fails to distinguish between a systematic pattern in the underlying data and noisy, irregular or random variation in the data. If the model fails to ignore random noise then it will be a poor predictor when using new data that wasn't used to train the model. This is analogous to rote learning.

`underfitting`: occurs when a model fails to fully identify the systematic pattern in the underlying data. This could be caused by an insufficiently flexible model (e.g. fitting a linear model to a second-order polynomial relationship between two variables.)




