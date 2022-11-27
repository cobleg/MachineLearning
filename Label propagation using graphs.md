Label propagation is a semi-supervised machine learning algorithm that begins with a small sample of labelled data and a larger sample of unlabelled data. The algorithm relies on clustering theory where close proximity of data elements implies similarity. In other words, the further apart data elements are, the less likely that they are similar.

## Benefits of label propagation
The primary benefit is the ability to label data quickly and accurately. 

## Use cases
Common use cases are:
- Labelling of large data sets
- Fraud detection where rare discoveries of fraud can be used to search for undiscovered cases of fraud.

## Graphs for label propagation
A graph algorithm can be added to guide label propagation. This requires several steps:
1. Construct a graph adjacency matrix  to indicate which nodes are connected via  edges 

[@zhouLearningLocalGlobal2003]

## References
Zhou, Dengyong, Olivier Bousquet, Thomas Lal, Jason Weston, and Bernhard Schölkopf. 2003. “Learning with Local and Global Consistency.” In _Advances in Neural Information Processing Systems_. Vol. 16. MIT Press. [https://proceedings.neurips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html](https://proceedings.neurips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html).