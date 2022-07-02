# Introduction
Data normalisation can speed up the model training process. There are various normalisation methods used, such as:
- Feature scaling
- Mean normalisation
- Z-score normalisation
# Feature scaling
Features are variables that are used as input data to train a model. Often the features have different scales, that the data values of each feature lie on different parts of the real number line. For example, Feature 1 ($1,000<x_1<100,000$) and Feature 2 ($-100<x_2<-50$). 

Feature values can be rescaled by calculating the maximum and minimum values of each feature set and dividing all of the observations in the feature set by the difference between the maximum and minimum values.

$$ x_{scaled} = \frac{x-Min(x)}{(Max(x)-Min(x))}$$

This resulting rescaled feature will lie in the $(0,1)$ range. 

# Mean normalisation
Mean normalisation requires calculating the mean of each feature set and dividing each feature by its mean.

$$ x_{norm}=\frac{x}{\bar{x}}$$

The mean-normalised feature values will be centred around 0 on the real number line.

# Z-score normalisation
The feature is rescaled by deducting the mean from each observation and divide by the standard deviation.


$$ x_{z} = \frac{x-\bar{x}}{\sigma} $$


The result will be values that are centred around zero and range between -1 and 1.

## Code

```python

import numpy as np

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
 
#check our work
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)
```
