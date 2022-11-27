Label propagation is a semi-supervised machine learning algorithm that begins with a small sample of labelled data and a larger sample of unlabelled data. The algorithm relies on clustering theory where close proximity of data elements implies similarity. In other words, the further apart data elements are, the less likely that they are similar.

## Benefits of label propagation
The primary benefit is the ability to label data quickly and accurately. 

## Use cases
Common use cases are:
- Labelling of large data sets
- Fraud detection where rare discoveries of fraud can be used to search for undiscovered cases of fraud.

## Graphs for label propagation
A graph algorithm can be added to guide label propagation. This requires several steps:
1. Construct a graph adjacency matrix  [@AdjacencyMatrix2022] to indicate which nodes are connected via  edges 
2. Calculate the affinity matrix $W$ defined as $W_{ij}=exp(-||x_i-x_j||^2/2\sigma^2)$  if $i \ne j$ and $W_{ii}=0$ 
3. Iterate $F(t+1)=\alpha SF(t)+(1-\alpha)Y$ until convergence, where $\alpha$ is a parameter in $(0,1)$.
4. Let $F^{*}$ denote the limit of the sequence $\{F(t)\}$. Label each point $x_i$ as a label $y_i=arg max_{j \le c} F_{ij}^{*}$  

####  Notes about the algorithm
The parameter $\alpha$ enables a weighted average of information across time. When $\alpha=1$ only the $SF(t)$ part is active, ignoring information in $Y$. The opposite occurs when $\alpha=0$. 
$F^{*}$ is the steady state value and this determines the label value. 

[@zhouLearningLocalGlobal2003]

## References
“Adjacency Matrix.” 2022. In _Wikipedia_. [https://en.wikipedia.org/w/index.php?title=Adjacency_matrix&oldid=1119322733](https://en.wikipedia.org/w/index.php?title=Adjacency_matrix&oldid=1119322733).

Zhou, Dengyong, Olivier Bousquet, Thomas Lal, Jason Weston, and Bernhard Schölkopf. 2003. “Learning with Local and Global Consistency.” In _Advances in Neural Information Processing Systems_. Vol. 16. MIT Press. [https://proceedings.neurips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html](https://proceedings.neurips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html).