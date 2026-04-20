
Back to RL formulation

Exploration issues: example of robot with objects on the table - naive exploration doesn't work very well... but novelty driven exploration (increasing novelty of perception)

Example with grasping task with robots - initially used only images for training? funny - success rate of (top-down constrained) random movements is 10-30%

#todo idea for pmind - do constrained uniform exploration
#todo pmind: do novelty driven exploration!

# NS-QD

So novelty driven approach will require two steps - exploration and then explotation of acquired knowledge

Example from Picbreeder of skull with interactive learning (last CM), problem is that intermediate steps to get the skull don't look like a skull at all

So instead we can search for a set of very diverse examples - it avoids being blocked in a local extremum

Pros:
- we can explore the space of reachable /  valid states
- as well as the optimal zone
- helps with reality gap since we can consider multiple optimal solutions
## Quality-Diversity (QD) algos

Inspired from biological evolution

So instead of finding one politic, we search all "interesting" politics (thetas) covering the outcome (?) space

Note that exploration of thetas (policy space), even uniform, doesn't guarantee the coverage of the state (?) space (example of the robot exploring the room)

So theory behind it:

Trajectory space is huge, so instead we project in in **behavior space** (e.g. final position for navigation task)

QD produces solutions that are both diverse and performing

So fitness becomes local and it's better than frustrating fitness-shaping

Evaluation function gives now not only fitness, and behavior as well

#todo exact algo

https://quality-diversity.github.io

How to quantify performance in a container? diversity (archive size ...) combined with fitness (max, min, ...) => gather them in QD score and optimize, get Pareto front

Two families of algos: 

### Novelty search
-  very simple - use novelty as fitness
- novelty is relative to what we have already explored
- novelty is an estimator of density in behavior space in **pop + archive**
- can use with EA algos, e.g. NEAT
- example of a maze - explore until the solution is found
- just uniform exploration of politics - very very poor coverage of b. space, whereas NS covers well
- NS seems to converge to uniform exploration of behavior space
- in NS individuals are replacedmuch faster than for EA based on fitness, distance to parents is rather big (so prone to instabilities)
QD algo with NS: NSLC - do NSGA-II with novelty objective and local competition (compared to closest neighbors)

We can do behavior shaping - play with ways to come up with behavior space or condition it
### MAP Elite

most used now

very trivial - define cells (boxes) in behavior space and keep the best one (in terms of fitness) in each box - addition mechanism

if individual lands in empty cell, it's kept

population size increases (contrary to most other EA algos)

hyperparameter - number / size of cells, tricky to find the good one, since hard to work if too many cells - smarter way than a grid will be Voronoi tesselation

founder effect - individuals with larger evolvability reach further?

#lookup def of evolvability

so we can use the obtain behavior-performance map - we can explore it with Bayesian optimization (episode based) and adapt the robot e.g. for reality gap or when conditions change a bit (e.g. robot damage recovery)

#todo gripper example

##

From model free to model based: 
Dynamics aware QD

From replay buffer build a dynamics model and do some imagine rollouts in addition to real rollouts

Autonomous definition of behavior space => We can use latent space of autoencoder and behavior space! (?)

 Sparse rewards => #todo SERENE algo

Action redescription: Generative adversarial policy network (to "interpolate" the solutions found with QD)

We can do another algo on top of QD found instances - meta-learning, to find most-adaptable 

Can combine QD + LLM

#lookup VLA - vision-language-action

Moral of the story - very easy and robust 













