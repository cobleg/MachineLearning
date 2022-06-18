# Overview
Softmax regression is useful for classifying items, i.e. allocating an item to a given list. the simplest list is a binary Yes or No, {0,1}.

Softmax works by calculating the conditional probability of a given specimen being a member of one of the items in the given list. For example, an image of an animal being a dog.

# Formula
The softmax formula is:

$$ \boldsymbol{\hat{y} } = softmax(\boldsymbol{o})$$ where $\hat{y_j}=\frac{exp(o_j)}{\Sigma_k exp(o_k)}$  

The $o_j$ are called the logits and are defined as:

$$o_j=x_j \times w_{j,k} + x_{j+1} \times w_{j,k+1} + x_{j+2} \times w_{j,k+2} + b_j$$

In matrix form:

$$O=XW'+b$$

The softmax uses the log-likelihood loss function, also known as cross-entropy.



