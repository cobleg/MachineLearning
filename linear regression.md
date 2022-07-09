# Overview
[Linear regression](https://machinelearningmastery.com/linear-regression-for-machine-learni) is an algorithm that involves using input data to fit a [linear function](https://en.wikipedia.org/wiki/Linear_function):

$$ f_{\boldsymbol{\vec{w}}, b}(\boldsymbol{\vec{x}})=\boldsymbol{\vec{w}} \cdot \boldsymbol{\vec{x}} + b $$
where
$\boldsymbol{\vec{w}}$ is a vector of weights to be estimated
$\boldsymbol{\vec{x}}$ is a vector of input variables
$b$ is the bias

# Cost function for linear regression
Fitting a linear model requires an efficient way of calculating the vector of weights $\boldsymbol{\vec{w}}$. One efficient method is to use a cost function (aka [[loss function]]) which is named the: squared error cost function:

$$ J(\boldsymbol{\vec{w}}, b) = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{2}(f_{\boldsymbol{w},b}(\boldsymbol{\vec{x}}^{(i)})-y^{(i)})^2$$
where

$m$ is the total number of sample batchs
$\boldsymbol{\vec{x}}^{(i)}$ is a vector of input variables corresponding to sample batch $i$ 
$y^{(i)}$ is the target (or output) variable corresponding to sample batch $i$

When using linear regression, this cost function ($J(\boldsymbol{\vec{w}}, b$)) is [convex](https://en.wikipedia.org/wiki/Convex_function) with respect to $\boldsymbol{\vec{w}}$ and $b$.


