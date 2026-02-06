
Recall perceptron: weighted sum and bias, linear decision boundary, w is perpendicular to it

SGD is better for noisy data (why?), minibatch - good compromise

For perceptron, two unicity problems: good decision boundary is not unique and for the same desicion boundary I can have infinity of different hyperplanes just by scaling w

Another  problem: decision boundary can be only linear
How to solve it? Combine perceptrons (see IDLE for neural networks) - but introduces a lot of new parameetrs, or project data ourselves (feature engineering with expert knowledge) or be smarter with kernels

So wee keep the same linear algo, just on projected features $\phi(x)$, e.g. for RBF using projection points as basis

Problem - overfitting, how to handle noise

# SVM

Let's tackle first overfitting and noise robustness problem

SVM is very powerful and elegant and still used for classification where there is not enough data for NN (e.g. medicine)

#todo show that $d = \frac{w^Tx + b}{||w||}$  - functional distance between point x and the decision boundary

hard to constraint directly length of w, so instead we fix the margin - distance to points closest to decision boundary must be = 1, NB that it is symmetric for both sides of decision boundary
So the margin contains no point in case of perfect separability

Maximizing margin (euclidean distance) is equivalent to minimizing L2 norm of w - regularization on w

Recall RBF version analogous kNN, by minimizing w norm, we set some of weights small or even zero, so as if we have calibrated number of projection points in the kNN-like algo 

And if separation is not perfect? We can allow for errors  with slack variables and include them in objective function (multiplied by coeff defining what is more important - margin or errors). If you rearrange the equation a bit #todo you get hinge loss (like perceptron, just shifted by one)!! 

K is a hyperparameter to tune

# Optimization

Objective function turns out to be convex, but another problem is that we do it with inequality constraint - how to optimize it?

Inequality constraint is a bit simpler than equality contraint - we can try to do GD in the contraint region, but it's problematic if optimum is on the constraint boundary

Smarter solution: Lagrangian, we can visualize it by isolevels / isocurves (level curves), gradient is perpendicular to the tangent, take a point on a contraint, if gradients of objective function and contraint are not collinear, then we can still move on contraint and ameliorate a bit the objective function, **when two gradients align** - the tangent for contraint and for objective fn is the same, so **moving along the constraint will not change the value of objective function** - we have found a local optimum #todo  draw

Need to check after with Hessian whether it's max, min or sth else...

#todo  how inequalities are handled - dual problem #todo why can 

For  inequality contraint , you can just do GD and if you stumble upon the boundary, can go along it - #lookup inactive contraints 

KKT optimality constraints #todo 

It turns out that optimal w is a linear combination of some training points - those that are on the margin

What we get in the end is that sum of alphas is contraint to sum to a contraint - we can do coordinate descent, descent in one direction, then compensate #lookup 

# Kernel Trick

NB: kernels from 

In all formulation of problem resolution, there is always an inner product

K(x1, x2) = inner product of projections phi of x 1 and x2

we can see in the formulas we got with optimization that K is enough, never need to get phi directly!!

so we need to find an efficient way to compute inner product directly instead of doing expensive projection and inner product on projected features

Gaussian kernel #lookup link to RBF?? #todo  show why it's equivalent to projection in infinite dimension space of polynomial basis. Note that here we do not use a fixed basis of function  - is it still kNN-like in spirit? yes, but point importance will be learned (alpha)

Admissible kernel - #todo properties, positive semi definite fn ... 

Kernel is a similarity measure between two objects!

Can work not only in $\mathbb{R}^d$, can do similarity on graph, text etc #lookup  connection in spirit to MDS in unsupervised

SVM - good since has the margin, so it works nice when there are few datapoints

Kernel trick works everywhere with hinge cost, #lookup can we adapt it for logistic regression?












