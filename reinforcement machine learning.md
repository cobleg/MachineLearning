# Overview
> Reinforcement learning (RL) is a major branch of machine learning that is concerned with how to learn control laws and policies to interact with a complex environment from experience.
>  [Data Driven Science & Engineering, *Machine Learning, Dynamical Systems, and Control*](https://faculty.washington.edu/sbrunton/databookRL.pdf) 

In reinforcement learning, an agent senses the state of its environment and learns to take appropriate actions to optimise future rewards. The ultimate goal in reinforcement learning is to learn an effective control strategy or set of actions through positive or negative reinforcement.

The term `reinforcement` refers to rewards used to reinforce desirable actions. A reinforcement learning agent senses the state of its environment and learns to take appropriate actions to achieve optimal immediate or delayed rewards. 

The agent arrives at a sequence of different states $\boldsymbol{s}_k \in S$ (set $S$ is the set of all possible states) by performing actions $\boldsymbol{a}_k \in A$ (set $A$ is the set of all possible actions) leading to positive or negative rewards $\boldsymbol{r}_k$. 

Conceptually, the reinforcement learning process is:
- unstructured exploration
- trial-and-error used to learn rules
- exploitation where a strategy is chosen and optimised within the learned rules

# Components of reinforcement learning
There are several components to the architecture of reinforcement learning:
- Agent
- Environment
- Reward
- Action
- Observed state

## Policy
The agent is typically a neural network that takes the state of the environment aas the input and produces a policy as an output. The policy is typically framed as a set of pseudo-probabilities within a [[classification]] model. 

## Environment
The environment is constructed as a stochastic, nonlinear dynamical system that evolves as a Markov Decision Process.


# References

Steven L. Brunton and J. Nathan Kutz (2021). [Data Driven Science & Engineering, *Machine Learning, Dynamical Systems, and Control*](https://faculty.washington.edu/sbrunton/databookRL.pdf) 

