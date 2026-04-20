
# Motivation

Direct policy search like we did with evolutionary methods

Problem - what if objective is not just one number

Can be intrinsically multiobjective or we can introduce helper objectives
Helper objectives - help find good behavior (for another objective) - e.g. if reward is sparse, it's hard to get it by random exploration

Another motivation - if we have many tasks?  We can solve them one by one 

We have a set of solutions and not just one -> "innovization"


# Multi-objective problem

We have an objective space, policy parameters are in decision space

How to compare solutions in objective space? Pareto dominance relation, for a given point  it separates objective space in regions that are better / worse / incomparable to this point (and thus relations of dominance) - relation of partial order

We are searching for Pareto front - set of solutions that are not dominated

Trade-off between convergence and diversity of the found front

We obtain thus multiple solutions

# Evolutionary methods for MO

Black box optimization, no need for derivatives

In general, an individual can be represented as binary vectors (although usually vectors of floats), we can then measure their fitness, select them, do cross-over, mutate, do off-spring

Note: mutation and crossover are independent of fitness

# For TP: DEAP

`toolbox.register()` to declare mutation, cross-over, selection etc operators, they are then accessed by algos

# Collapse to a single objective

We can do a weighted sum of objectives...
Drawbacks: doesn't allow to access the whole Pareto front
Why? because weighted sum forms a hyperplane level curve on the objective space
**Note that it's thus impossible to find intermediate solutions with weighted sum of objectives for non-convex pareto front**

Another solution: weighted Tchebychev - max of distances to optimal point - allows to discover the full Pareto front

# Evolutionary Multi-objective

Nice since allows black box

How to select individuals - rank based fitness

SPEA2 - Strength Pareto algo #todo 


## NSGA2 
more efficient, Non Dominated Sorting Genetic Algo - elitist, diversity preservation

one population - multiple solution fronts (take a population, find Pareto front, then find Pareto front if you don't account for the first one, etc - single Pareto front in O(MN^2) - a lot but better than O(N^3) if no trick is used)

Crowding distance for diversity preservation - we prioritize individuals with higher crowing distance

Modification to algos to make them multi-objectiove is basically in selectors

Bitwise vs polynomial mutation

polynomial mutation is nice to handle cases with limits on parameter values

Simulated binary crossover, different spread factor

NSGA2 is not limited to numbers! we can optimize graphs etc. (as long as we have mutation, cross-over operators) (contrary to CMA-ES which is limited to real valued vectors)

Epsilon-dominance - simplifies Pareto front

## Indicator based selection

Hypervolume represented by Pareto front - a way to decide on which Pareto front is better

# Many objective optimization

NSGA2 works nice only for 2-3 objectives, *many* is more than 3

Why? More objectives - less chances that one is better and another (in Pareto sense)

Pareto relation doesn't work anymore

So problem is not just in computation required!

Hard to get enough samples to represent Pareto front

How to visualize 3+ dims? - scatter plots - dimension by dimension, by still too messy
better - radar chart and parallel plot #lookup - one line by solution

How to model it?  Decompose the problem into N subproblems, for each subroblem optimize it with respect to an optimal point -> MOEA/D

NSGA3, we select points closest to reference lines #todo  notebook

# Applications

Helper objectives

Crossing reality gap - we can add an objective maximizing transferability (simulation vs real world), another approach - maximize mutual information #todo between what and what?

Moral of the story:

Weighted sum of objectives is often bad....

It can help to have many solutions in robotics








