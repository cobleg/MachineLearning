# Overview
Distribution shift refers to systematic changes to datasets that can lead to errononeous and poorly performing machine learning models. 

Distribution shift can occur between the training, validation and test datasets. It can also occur over time as the model encounters new data in the real world.

# Types of distribution shift
The main types of distribution shift are:
1. covariate shift
2. label shift
3. concept shift

## Covariate shift
This is the distribution (i.e. relative frequency of occurence) of labels changing over time. For example, the relative proportions of categories of items in datasets. That is, a class of item starts out as a rare occurrence in training data but is actually common in the real world dataset.

### Treatment
Reweighting the occurences of specific classes of training data set can help improve the prediction accuracy. For a well defined population, the known relative occurence of each class can be used instead of the relative frequency in the sample of data used for training. 

## Label shift
Occurs when the relationship between an input variable and an output variable changes. This is likely to happen when there is a one-to-many relationship between input and output variables.

For example, the relationship between a `symptom` and the diagnosed `disease`. A `cough` is an example of a `symptom`, but can be caused by multiple diseases. An unbalanced training data set that does not include enough examples of all of the underlying `diseases` that cause a `cough` can mean the trained model overpredicts one `disease` relative to the possible other diseases.

## Treatment
Use bias-corrected temperature scaling (BCTS) as explained in this reference:
[Battling label distribution shift in a dynamic world](https://towardsdatascience.com/battling-label-distribution-shift-in-a-dynamic-world-bc1f4c4d2f92)

## Concept shift
Occurs when there are hidden variables that are not explicitly contained in the training data that causes predictions to become increasingly inaccurate over time. 

For example, the quantity of ice cream cones sold where the training data set contains only summer sales and no winter sales. The concept of `season` is only implicitly contained in the data and there is no `explicit` feature or control variable contained in the training data. This causes an over-prediction of ice cream cone sales in winter.

## Treatment
Periodically re-train the model to capture any concept shift. This will result in changed model weights that should lead to improved predictions.