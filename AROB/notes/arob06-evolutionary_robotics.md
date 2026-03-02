Use ES to train robots - **direct policy search** (instead of using values etc., we directly connect policy to reward), in contrast with gradient policy search

$\theta$ - params , $f(\theta)$ - fitness
we agglomerate objectives in one (but how - will know later)

So we have a parametrized policy which can be more complicated that just NNs, e.g. moving skeleton where we can change the morphology of the muscle attachments

Genotype (weight space) -to- phenotype (action space) mapping

slide 19 - J(\theta) is return (aggregated reward) - here we are liberated from needing to use instantaneous reward - why?

However if a lot of dimensions, it's harder to do CMA-ES for example since covariance matrix updates are O(N^2), so we contrain ourselves to using diagonal covariance matrix - it's faster although degraded

Optimize structure (topology) of NN - constructivist approach vs pruning

# Simple ANNs

ANN controller for navigation task

sensorimotor loop

slide 27: fitness is an integral over time of three terms: favorise translational speed, defavorizes rotations and sensations detecting walls

 Don't put "generations" on plots, plot evaluations instead! #question why?  (generations ar artefact of algo) -  nb evolutions = nb generation times population size = reflects real computation bias

Analyze trajectories in fitness space

Perceptual aliasing - ambiguous perception (we cannot discriminate two states), e.g. monocular vision of robot - cannot discriminate small ans close vs big and far objects - poses problems in terms of markovian property (memorylessness)

Somehow robot solves this problem without using memory (e.g. recurrence)

How come? Robots have learned to avoid places with perceptual aliasing

Richer environments are closer to markovians - plus de reperes

Memory - internal states - not just universal estimator, can approximate Turing machine?
like in e.g. Echo state network (type of RNN?) #lookup liquid state machine and echo state network (ESN learns only the last layer picking activation from some of internal units - dynamic selection)

# Levels of learning

Full general vs plastic general based on how it adapts to other tasks

#todo teacher-motor architecture, combining learning and evolution
(intuitively teacher doesn't know where in parameter space the motor should change,  but it learns the direction?)

Baldwin's effect, learning guides evolution -> genetic assimilation 

Lamarckian evolution - learnt params are reintegrated in the genome

# Reality gap

We use simulation during training, but there might be problems to integrate it in real env

What to do? Add noise to simulation to maker it more realistic, or vice versa start with decomposed simple problems and then do more complex simulation

We can as well adapt the simulation during learning in real world?

Physics induced NN - apply soft constraints with diff equations known from physics, adapt them to reality

Open issues: graph programming going beyond search in R^n

