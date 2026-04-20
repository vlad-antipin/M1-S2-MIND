
Clonal approach - same params

Heterogenous approach - individuals are in competition

Swarming = collective robotics with no central control, only local, usually homogenous agents
# Cooperative setups

## Clonal approach - homogenous team

In classical evolutionary robotics we did off-line design method

In collective robotics, we take a candidate solution and transform it into behavior of some robots (genotype -> phenotype), e.g. copy the same neural network in each robot

Easy to optimize

Can decode it into a behavior so that all robots are very similar - cloning

NB: population of candidate solutions ≠ population ("essaim"fr, team) of robots obtained from a candidate solutions

Example: learn robots to go over a pit by attaching to one another

Robots can learn to coordinate their actions

Homogenous team is capable of behavior specialization! - e.g. as for ants, it's more efficient when one ant drops leaves and another cuts them on the ground, so did the robots in simulation! And it's possible even with cloning, since robot's behavior is influenced not only by its genome by environmental circumstances as well

## Heterogenous teams

Genotype is not the same, e.g. take different candidate solutions for each robot -> harder scaling and evaluation

Robots are thus in competition

Different level of selection - individual vs team. Combined with team type (homogenous vs heterogenous) - 4 combinations:

When tasks require cooperation:
- Homogenous team with team selection - safe choice
- Homogenous team with individual selection - redundant with the first one
- Heterogenous team with team selection - inefficient
	- credit assignment issue when team has one very good individual and others are catastrophic, so it will dilute performance of good individuals and add noise
- Heterogenous team with individual selection 
	- cooperation can appear if it's beneficial for the cooperating individual

# Biological modelling

2 approaches - mathematical models (e.g. from game theory) vs in vivo studies (e.g. bacteria)

Studies on cooperation: with direct (e.g. share food) / indirect (e.g. for relatives) benefit

Stag hunt problem from game theory #todo 

We can introduce mutation operators: network duplication with random change, during evaluation each robot choses a network - symmetry is broken -> leader-follower behavior - from coordination game we pass to individual optimization then

mechanistic constraints can change a lot

individual-level adaptation leads to collective behaviors

Evolvability is crucial for adaptation to changing environment

# CCL

Clonal approaches are a simple and good default choice













