# Overview
A decision boundary marks the delineation between different clusters of input data corresponding to the classes in classification models.

Assuming data are distinctly clustered, it should be possible to cleanly delineate between the classes of data. For example, the values of $z$ should be clustered along the number line and not evenly distributed across classes. See this article for more information: [Decision Boundary in Machine Learning](https://thecleverprogrammer.com/2020/07/29/decision-boundary-in-machine-learning/)

Decision boundaries can be linear or non-linear and can sometimes be more easily identified using combinations of input variables.

# Logistic model
When logistic model is used for classification, the decision boundary can be plotted in the input space.

# Code

```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt

# create training data
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 

# plot the data
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()

# plot the decision boundary

# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()

```
