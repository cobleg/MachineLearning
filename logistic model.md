# Overview
Logistic function:

$$ g(z)=\frac{1}{1+e^{-z}} $$
where $0<g(z)<1$ and $z$ is an index that maps one or more input variables, typically as a linear combination where the linear parameters act as weights. That is:

$$ z= \boldsymbol{w} \cdot \boldsymbol{x} + b $$

There will be a threshold value of $g(z)$ that delineates the difference between the choices (e.g. 0.5). The logistic function belongs to a family of functions called [sigmoid functions](https://en.wikipedia.org/wiki/Sigmoid_function), chacterised by a `s` shaped function.

The output of $g(z)$ is typically interpreted as a probability of a given choice. A restriction on the output is that the list of output probabilities must sum to exactly 1.

That is $P(g(z)=0)+P(g(z)=1)=1$ 

The  [[loss function]] for the [[logistic model]] is specified as:

$$ L(f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)}), y^{i}) = \begin{cases} -log(f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)})) \ \ \ \ \ \ \ \ \text{       if } y^{(i)} = 1 \\
-log(1-f_{\boldsymbol{\vec{w}},b}(\boldsymbol{\vec{x}}^{(i)})) \ \text{     if }y^{(i)} = 0 \end{cases} $$
where
$\boldsymbol{\vec{w}}$ is a vector of weights to be estimated
$\boldsymbol{\vec{x}}^{(i)}$ is a vector of input variables corresponding to sample $i$
$b$ is the bias
$y^{i}$ is the target (actual observation) variable corresponding to sample $i$

The logistic loss curves for binary target labels are different for each outcome to ensure that the loss function is always convex as shown in this diagram:
![[Pasted image 20220710094802.png]]
Source: [Loss Function (Part II): Logistic Regression](https://towardsdatascience.com/optimization-loss-function-under-the-hood-part-ii-d20a239cde11)
# Code

```python
import numpy as np

print("Input to exp:", input_val)
print("Output of exp:", exp_val)

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g
    
# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

# Plot z vs sigmoid(z)
%matplotlib widget
import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
```

