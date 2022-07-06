# Overview
Classification refers to machine learning algorithms that make choices between mutually exclusive classes (aka categories). The simplest form is binary classification (i.e. {0, 1}, {yes, no}, {true, false} choices).

As indicated with binary choices, the classes are contained within a list of items. More generally, this is described as discrete choice.

# Function to fit to discrete choices
A function that fits to binary choice data is the logistic function:

$$ g(z)=\frac{1}{1+e^{-z}} $$
where $0<g(z)<1$ and $z$ is an index that maps one or more input variables, typically as a linear combination where the linear parameters act as weights. That is:

$$ z= \boldsymbol{w} \cdot \boldsymbol{x} + b $$

There will be a threshold value of $g(z)$ that delineates the difference between the choices (e.g. 0.5). The logistic function belongs to a family of functions called sigmoid functions, chacterised by a `s` shaped function.

The output of $g(z)$ is typically interpreted as a probability of a given choice. A restriction on the output is that the list of output probabilities must sum to exactly 1.

That is $P(g(z)=0)+P(g(z)=1)=1$ 








