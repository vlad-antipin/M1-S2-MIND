Reuse past experience to learn more efficiently - two approaches - learn model or use replay buffer (experience replay), they turn out to be similar

Model free RL is very slow and sample inefficient

Eligibility traces, Dyna, replay buffer, ...

Agent can learn model during exploration and then propagate reqrds using this model, problem - if reward is found too fast, agent can be locked

# Eligibility traces

Not yet model based, but agent back propagates the reward towards recent places, but in. smart way, avoiding high memory consumption
so we use eligibility e(s) - when was the last time state s was visited? slide 7, basically scale delta so that recently visited states are updated more, so needed memory is fixed to state space

so we get TD(lambda), SARSA(lambda), Q...

if lambda=0 we can algorithms we already saw

lambda=1 turns out to be the same as Monte carlo #todo

# Dyna

Model based, learn T and r

learn where do you end up and whar reward you get when you do an action in a state

with incremental self-supervised

so it's like doing Bellman backup "in your head"

Two approaches: learn model and policy simultaneously and not

## Model acquisition first

How to collect data? From a random policy? Or we already have a dataset

What model to learn? deterministic or stochastic T

How to cover all s x a space? when to stop learning?

How to learn policy then? - Using dynamic programming (GPI), Q-learning, SARSA... when to stop?

Q-learning is more efficient here in real life, since we avoid interacting with real world environment

## Simultaneous learning

We use current policy to collect the data, we learn until policy doesn't get better

What to learn faster - model or policy?
Problem - if policy is stuck, model will not improve

Used by DYNA

### Dyna-AHC

Adaptive heristic critic

We can switch between model based and model free, Sutton shown that using model is more efficient

## Dyna-PI, Dyna-Q, Dyna-AC

Sometimes do random transitions "in the model" and apply backups

We can do a better propagation (prioritized sweeping)

We can do model generalization: 
- anticipatory learning classifier systems: 
	- represent state as a set of binary variables, gives compact models
	- problem: space explodes
- Factored MDP
	- state variables as decision trees

# Replay buffer / experience replay

eligibility traces: need a lot of memory

we can keep previous transitions instead in replay buffer

looks like model based, but we don't do a model, just replay transitions to 

like for episodic memory, replay buffer = hyppocampus

sars is enough for Q-learning, need sarsa for SARSA

**it turns out that it's equivalent to doing a model!** since model itself is based on past transitions

RB improves stability since we can decorrelate samples in randomlt drawn mini-batches - make them iid!
helps with divergence issue (deadly triad)

When size of RB is fixed, we do FIFO!

we can reuse samples - sample efficiency is higher

in deterministic T case, it's enough to have 1 transition for each s,a 

in stochastic T case it's more complicated since need to store transitions in good proportions 

RB was one of keys to success of DQN, but was introduced before

Types of RB:
- dictionary
- FIFO
- episode by episode
- ...

## Successor Representation

#todo 

## Prioritized sweeping vs Prioritized exp replay

Instead of randomly sampling transitions from RB, do smarter:

### Prioritized sweeping

From model based

Most useful transition are the ones highest TD-errors, recent past 

Sort them and replay

turns out to be almost equivalent to eligibility traces

Queue-DYNA - replay first transiitons with higher TD (same, largest first) + replay the samples most liekly visiter first (forward approach, focused dyna) - combines replaing past and planning future 

Animals seem to do this

### Prioritized exp replay

In context of DQN - sort RB in TD-errors, but still sample randomly, just give them more proba - liek largest first, just probabilistic

## CCL:

RB and MBRL - basically the same thing #todo slide 33
it's not really the case when we use FA since it does generalization outside what was already seen

# Biological counterpart

"Cognitive map"

Behaviorism (reactive behavior, no internal cognitive states, we were not able reach them experimentally long ago, close to MFRL) vs cognitive sciences (now we can access "internal states" - neurosciences, like MBRL)

A rat can learn a maze - builds its model

Place cells in hcp - like RBF???

Grid cells, border cells ...

When rat is sleeping, it does backward replay

#lookup trajectory stitching

#todo paper of Mattar & Daw

Can an animal "replay" a situation that it has never seen? (in favor of MBRL with FA) slide 45. But in general in neuro papers it's mostly about exp replay :(

Probably, animals use both MBRL and MFRL











