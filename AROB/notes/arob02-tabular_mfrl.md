Tabular - actions and states are discrete

In deep RL we have state (and eventually action) space that are continuous

Connection to psychology and computational neuroscience

Supervised learning - correct answer is given during training
Cost-sensitive learning - only reward is given / value of action 
RL - value signal is a scalar

Need exploration, since if you have one reward you don't know whether it's good or bad (even if it's negative), need to compare

Exploration / exploitation trade-off

we decrease exploration during exploration, but problem is that you never know whether your policy is actually optimal

$\epsilon$ -greedy is common one

Sequentiality: 
Bandits - one state, you are simply looking for the best actions
Sequential problems - action changes state

# MDP

Agent, environment
Agent does actions, environments gives rewards and next states

Transition function SxA -> Pr(S)
Reward function SxA -> R (gives immediate rewards)

Deterministic transition function is a special case of stochastic

Immediate rewards: for entering in a state, for doing a specific action in a state

Policy can be deterministic or stochastic as well

For any MDP there is an optimal deterministic policy, great since looking for a deterministic policy is easier

Return = aggregation of rewards

We could have summed rewards to compute the return - problematic for infinite horizon

We can average over a window as well - uglier for convergence proofs?

So we do discounting with a factor gamma
$V^\pi(s_{t_{0}}) = \sum_{t=t_{0}}^{\infty}\gamma ^t r(s_{t}, \pi(s_{t}))$
$\gamma \in [0,1]$
High gamma - future rewards are important
Low gamma - agent prefers close / immediate rewards 

Connection to neuroscience - NT levels

MDP must satisfy Markov property

next state and reward depend only on current state and action, not the past
Reactive agents (having no memory of the past) can be thus optimal

Markov property doesn't hold when:
- observations of the state are not informative enough (partial observability)
- when several agents take actions
- transitions depend on time

we can add variables to space, but it can explose state space, and time is not like other variables...

# Dynamic Programming

Assumes that MDP model (T and r functions) is completely known

For a given policy, we can define two functions:

stat-value function $V^{\pi}(s)$
- expected return given that you start in state s and follow the policy

action-value function $Q^{\pi}(s,a)$ (called "quality fn" previously)
- expected return given that you start in state s, do the action a and follow the policy

for DP we can use both V and Q

### Bellman equations

deterministic transition functon and policy:
- from definition of discounted return and value function can deduce:
	$V^{\pi}(s_{t}) = r_{t+1} + \gamma V^{\pi}(s_{t+1})$

stochastic transition functon:
- deterministic policy
	- #todo 
- stochastic policy
	- #todo 

in general: 
$V^{\pi}(s) = \sum_{a}\pi(a|s)\sum_{s', r} p(s',r|s,a) (r + \gamma V^{\pi}(s'))$

slide 25
$V^{\pi}(s) = \sum_{a} \pi(a|s)Q^{\pi}(s,a)$
#todo do the opposite

## Bellman operators

To guarantee convergence, we need a contractive operator

Bellman optimality operator:
we do the best action and update value function based on value estimates of "neighboring" states (bootstrapping)
-> **value iteration**

It gives the value of the optimal policy, we can get it back with argmax over actions r + gamma V

NB: when there are ties, np.argmax takes the first arg

Bellman operator #todo  formula
-> evaluate policy
but we can improve it by making it greedy wrt to this value (policy improving), reevaluate, make greedy, ... when it doesn't change no more we have a policy that is greedy wrt to its own value function - it is optimal
-> **policy iteration**

GPI - can vary k

# Temporal Difference (TD) mechanisms

Now we don't know the MDP model (T and r)

Either we can estimate the model - model-based approaches

Or we can do model-free - Actor Critic is a special case

The general idea of incremental estimation:
- $E_{k+1}(s) = E_{k}(s) + \frac{1}{k+1}[r_{k+1}-E_{k}(s)]$
- $E_{k+1}(s) = E_{k}(s) + \alpha[r_{k+1}-E_{k}(s)]$ 
- i.e. $E_{k+1}(s) = (1-\alpha)E_{k}(s) + \alpha r_{k+1}$ 
- new estimation = old estimation + scaled error for a new observation
- can use a small constant $\alpha$ instead of storing $\frac{1}{k+1}$
- $\alpha$ - learning rate

TD error = estimation using Bellman eq with new reward - current estimation:
$\delta_{t} = r_{t+1} + \gamma V(s_{t+1}) - V(s_{t})$ 
measures a "surprise" - positive or negative

increment V with $\alpha\delta_{t}$

## TD(0)

Convergence proven provided $\epsilon$-greedy exploration policy

Other exploration policies:
 - $\epsilon$-greedy - choose often the best action (argmax Q), otherwise choose uniformly another action
	 random walk is good for convergence proof
- roulette wheel, proba proportional to abs - too much of exploration
- softmax
	- can play with temperature - high -> uniform, low -> max

Problem: TD(0) can only evaluate a given policy, 
we need Q(s,a) to know which action to take

# RL algorithms

## Sarsa

#todo formulas

Like TD(0), but uses Q value instead and is based on s,a,r,s,a

#todo  in TME, issue of taking action a' slide 51

Problem: we need to know in advance the next action -> on-policy approach

## Q-learning 

Similar, but do the best next action instead, better

NB: slide 53, (1-terminated) to handle the terminal state

x(1-terminated) since need to do update if episode ended not due truncation

# Actor-critic

delta is computed from critic and used to update both critic and actor (update its probab but need to normalize so that proba stays valid)
Note that learning rate for actor and for critic can be different
Lr is larger for critic

AC resembles more policy iteration than Sarsa / q-learning

NB: connection btw Q-learning and AC, can represent argmax Q as actor
Can update actor smartly - change it only if you update Q for this state 










