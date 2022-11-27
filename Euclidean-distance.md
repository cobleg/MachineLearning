## Definition
The [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between two points in Euclidean space is the length of a line segment between the two points.

## Application to calculating graph weights
The Euclidean distance between two nodes in a graph is calculated according to:
$$
W_{ij}=exp(-\frac{\Sigma_{d=1}^d(x_i^d-x_j^d)^2}{\sigma^2})
$$
where $W_{ij}$ is the edge weight, $x$ is the node coordinates in one dimension for nodes $i$  and $j$ and $\sigma$ is a hyperparameter.

[@chenGraphBasedLabelPropagation2021]

## References

Chen, Long, Venkatesh Ravichandran, and Andreas Stolcke. 2021. “Graph-Based Label Propagation for Semi-Supervised Speaker Identification.” In _Interspeech 2021_, 4588–92. ISCA. [https://doi.org/10.21437/Interspeech.2021-1209](https://doi.org/10.21437/Interspeech.2021-1209).
