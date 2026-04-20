Make evolve structures like programs (trees, ...), neural networks, etc
# Evolution of morphologies

Can model morphology with finite-state state machines

# Genetic programming GP

## Symbolic regression

Not just fit a function, but find its symbolic expression!

Genotype = equation

Representation as an **abstract syntax tree**

We define which terminals (variables and constants) and primitives (operations) are allowed 

Maximum depth of this tree is a parameter

Crossover: subtree exchange

Mutation: regenerate a random subtree

Selection: tournament

Main problem: bloat - huge tree ≈ overfitting, we can:
- limit depth
- multi-objective fitness-parsimony
- double tournament: fitness, parsimony

But not only trees are possible to model with GP:
- linear GP - linear (sequential)  programs
- cartesian GP - schemas with logic gates?
- grammatical evolution

# Neuroevolution

Evolution is super easy to parallelize

We can search for an architecture directly! - we can do it before doing conventional gradient-based NN training

## Cellular encoding 

Genetic programming for neuroevolution

Primitive operators define rules to generate networks

A tree with those operators -> network

# Specific representations

SANE: encode connections and weights in a sequence ( linear program? )
ESP:  #todo ?

## Graph directly

challenge: crossovers of graphs

NEAT: neuroevolution of augmenting topologies

Genome: list of nodes and of connections 

Crossover: based on innovation number (that is affected when the node is created) #todo 

Mutation: add connection / node, but no drastic changes (otherwise we approach just a uniform exploration)

on slides DIS = disabled

can do recurrence

speciation: define niches with a threshold (computed from number of shared / disjoint neurons / connections)
fitness sharing: normalize a fitness of an individual by number of similar individuals (in the same niche)? 

because of this pecularity in selection, we cannot use algos like NSGA-II directly

it's like favor BFS over DFS

## Compositional pattern-producing networks (CPPN)

Can use other fancy functions in nodes?

HyperNEAT

PicBreeder - they generate images

Skull example - to get the skull, we pass through stages that don't look at all like a skull

# Novelty Search

Premature convergence - problem coming from the fact that fitness can define a goal nicely, but not the way to get there

So instead of looking for optima solution, we can look for novel ones - favor exploration with novelty-based fitness - maximize mean distance to nearest neighbors

For some tasks it converges faster than those with goal-oriented fitness

We can use it as a helper process - combine goal-oriented and novelty-oriented fitnesses and do e.g. NSGA-II

There is no optimal solution by design

DNN: simplified NEAT

When we want to model memory, agents often find ways to solve memory-requiring tasks without it (e.g. T-maze)

We can favor the system to become more modular if we favor shorter connections?

# Neuroevolution + Deep Learning

DeepNEAT - we can define the whole NN modules with nodes and train it with gradient-based methods

CoDeepNEAT - #lookup 

NSGA-Net

# Custom representations

Recommndations:
- crossover must preserve characteristics of parents more or less (modularity of genome must be reflected in phenotypic modularity), otherwise ca sert a rien
- #todo 
- mutation must be limited
- mapping genotype - phenotype, it's nice to have locality, synonymity - similar genotypes produce similar phenotypes


