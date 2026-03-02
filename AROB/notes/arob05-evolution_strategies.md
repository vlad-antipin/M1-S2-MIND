# Evolution Strategies


Algorithms based on meta-heuristics inspired from biological evolution, aiming to optimize problems defined in $\mathbb{R}^n$ (like NNs) 

Genetic and artificial evolution algorithms, other as well (ants, collectives? ...)

We will optimize NN weights here

X - search space, find there $x^{\star}$ optimizing a function
candidate solution a:= (x, f(x)) 
f(x) is some score / performance, outputs a scalar

we will do black-box stochastic optimization, which is:
- derivative free (so we don't and can't do gradient descent)
	- so we are not limited by using nice derivable functions
- population-based (try multiple times)
- iterative 
- stochastic

$x^{\star}$ is rarely reached, but we can approach it
f(x) is fitness, like in biology, opposite of loss basically

Evolutionary strategies:
- initiate randomly
- get population of N candidate solutions
- evaluate with f (black box)
- selection (of parents)
- variation (to get descendants, as if you did mutations)
- replacement (can keep selected parents as well)
- repeat
- terminate if quality threshold is reached  or if calculation budget is reached
- return best solution

2 principles: selection of the fittest, blind variation

$(\mu,\lambda)$ notation (nb parents, nb children)

$(\mu + \lambda)$ - keep best parents and add parents
$(\mu,\lambda)$ - keep only children
population size = $\mu + \lambda$
NOTE: $\mu <$ population size

and always less or equal of parents than children

it's important to save best solutions to not forget them

problem with neighborhood exploration / variation: as well curse of dimensionality is relevant, when we apply variation, we end up in a smaller dimensional subspace :(

# (1+1)-ES

keep 1 parent, generate 1 child, population size is always 2

```
init: x in R^n
init: sigma > 0

for i in range(evaluations):
	x_new = x + sigma*N(0, I) 
	if fitness(x_new) >= fitness(x):
		x = x_new
```

does a random walk actually / brownian motion

parent x, child x_new (parent is kept implicitely)

note that it's important to update even if fitness is the same to not be blocked in neutrality plateau

can add lookup table

problem - when we reach optimum, we will oscillate, solution - dicrease noise when fitness is close to optimum, adaptative approach

It's adapted only to convex / unimodal problems, need to try multiple initiations sometimes

we can add 1/5 rule, assuming that we have spherical problem - convex and quadratic
#todo why 1/5? - adaptive convergence / autoadaptive algorithm

```
... same ...
	if fitness(x_new) >= fitness(x):
		x = x_new
		sigma = 2*sigma
	else:
		sigma = 2**(-1/4) * sigma 
```

#todo show why 1/5, linked to sphere geometry

problem - neutrality plateau problem is even worse

#todo simulated recuit (annealing?) does inverse - increases exploration when fitness doesn't change, nice for multimodal case

it's sensitive as well, need to do more evaluations

some types of fitness landscapes:
- multimodal
- with plateaus
- noisy
- discontinuous

# (1, n)-ES 

generalisation of previous, multiple children 

population is like a moving hypersphere (because one parent is kept, otherwise populations could "diverge")

for more advanced approaches, 

we can use it to establish the gradient

nice with multimodal landscapes, but exploits single best parent

NOTE: interest to use other correlation matrix than identity, often we need to couple dimensions

##  CEM-ES
Cross entropy method

$(\mu / \mu, \lambda)$-ES 
Move centroid instead, by weighting children with weights derived from fitness

we choose a subgroup of best mu out of lambda children , recompute centroid and std (weighting those samples), but increase sigma a bit to avoid "vanishing noise"

#todo  bad for non-separable cases

again the problem that we explore the same dimensions in the same way, we can fix it by estimating the whole covariance matrix (not necessarily diagonal)

NB: sometimes we don't weight best children, just take best children
NB: value of score is not obvious, so sometimes we prefer using ranking instead, score is rather a differential indicator - order of scores matters more than absolute difference

## CMA-ES

state-of-art algorithm

idea of gradient estimation
#todo rank-mu apdate
rank-1-update - integrate along the time correlations between populations??

very auto-adaptive! see slide 73

 NB: it's important that exploration is big along the direction symmetrically, it's not really a waste to explore the opposite direction, like this we can go back easier if this step was bad?
 
 #todo slide 75

# Applications to Robotics

#lookup actor critic - SARSA or Q-learning in spirit (I think 2)

ES in RL can allow to use less neurons in NNs since no need to do gradient descent and thus need to do less dimensions??? #lookup why

#lookup Direct policy search, use reward directly as fitness













