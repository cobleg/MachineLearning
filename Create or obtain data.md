# Overview
Machine learning algorithms require data, which can either be created or obtained.

# Creating data
Data can be created by constructing data objects within a machine learning environment.

# Code
## Python
Within the Python platform, data can be constructed using NumPy. The code block below constructs the input data set `x`  as a two dimensional(rows, columns) object ( matrix) consisting of three rows by four #features per row. A dependent variable `y` is constructed as a one dimensional object (vector) consisting of three rows. 

```Python
import numpy as np
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
```