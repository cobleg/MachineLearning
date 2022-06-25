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




