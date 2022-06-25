# Introduction
Data normalisation can speed up the model training process. There are various normalisation methods used, such as:
- Feature scaling
- Mean normalisation
# Feature scaling
Features are variables that are used as input data to train a model. Often the features have different scales, that the data values of each feature lie on different parts of the real number line. For example, Feature 1 ($1,000<x_1<100,000$) and Feature 2 ($-100<x_2<-50$). 

Feature values can be rescaled by calculating the maximum value of each feature and dividing all of the observations in the feature set by the maximum.

$$ x_{1,scaled} = \frac{x_1}{(Max(x_1)-Min(x_1))}$$

This resulting rescaled feature will lie in the $(0,1) in a range i which the feature observation values does not exceed 1. 