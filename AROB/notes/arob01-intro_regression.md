# Intro

RL vs Evolutionary methods

Soft Actor Critic - most advanced before 2018?

Tabular RL - discrete actions and states

After you code a TME, you have a week to write a scientific-like report in LaTeX (2 reports: 1 on regression, 1 on tabular RL)

Course mark = average TME note + 2 reports

# Regression

Why? Function approximation is useful for:
Robot models:
-  geometric
	- e.g. arm-robot - we want to take an object in x,y,z by manipulating angles between robot parts
- cinematic
	- angular velocity linked to velocity
- dynamic
	- dynamic systems (with laws of mechanics...)

Deep RL:
- function approximation of e.g. Q-function

Regression = Function approximation

Set of related measurements input-output (x,y) 
Find latent function
Liner or non-linear
Can transform features yourself to linearize the problem or delegate it to ANN (≈ universal function approximator)

Batch of points in **design matrix** NxD, observations by lines

How to find latent function, avoid overfitting?

Model $\hat f$ for latent function (ground truth?) $f$
Impossible to know the error since $f$ is unknown,  but we can estimate it

Generalization capacity of the model - measure the error for unseen points

#question classification vs regression - what about ordered discrete variables? like a score?

# Least Squares

$min(\mathbf{y} - \hat f(\mathbf X ))^2$ 

offset trick - add ones column to the design matrix, intercept becomes one of weights

extended X is called $\bar{\mathbf X}$ here,  and $\mathbf \theta$ for weights

set gradient to zero, obtain $\mathbf \theta^\star = (\bar{\mathbf X}^T\bar{\mathbf X})^{-1}\bar{\mathbf X}\mathbf y$
problem 1 - matrix inversion is expensive, many dimensions - very expensive - solution: GD

problem 2 - potential signularities of X^T X can result in unstable big theta, solution: regularization

for L2 regularization (=Ridge = Tikhonov):

$\mathbf \theta^\star = (\lambda \mathbf I + \bar{\mathbf X}^T\bar{\mathbf X})^{-1}\bar{\mathbf X}\mathbf y$

#question  is it pseudoinverse?

## Non-linear models

Two solutions:
- multiple local linear models(e.g. piecewise?, function is approximately linear if we zoom)
- project in space where relation becomes linear
There are unified approaches as well!

### 1st way: Locally Weighted Regression (LWR)

Individual errors are weighted by regions, weight = value Gaussian centered on this region

relative importance
we fit as many linear models as there are receptive fields

slide 26 #todo pseudoinverse

LWPR - ... LW Projection R #todo using PLS
XCSF - position Gaussians randomly and then move them to increase the covering

Gaussian Mixture Regression (GMR)
good for active learning - where to look for new points?

#todo summary slide 32


### 2nd way: RBFN and ANN

Instead $\hat f (\mathbf X) = \mathbf w \mathbf X$
$\hat f (\mathbf X) = \langle \mathbf w, \phi_\theta(\mathbf X) \rangle$ extended with bias #todo 

find a feature function hoping that latent function will become linear int this new space

#todo slide 36

same idea as receptive fields in neuro

it's equivalent to approximation of a function hypersurface with linear combinations of Gaussians

#question are prameters of Gaussians learned?

Kernel Ridge Regression (KRR)
Kernel function #todo slide  39 !!!!!!
#todo conditions for being kernel function
is matrix K a similarity matrix?
#lookup Gram matrix

problem - kernel expansion problem since K is NxN

RBFN is better, since E basis functions and not N like for KRR
NB: here matrix (G) is not symmetric

# LWR vs RBFNs

#todo slide 42, one is a special case of another !!
But does it mean that LWR is better by definition since more expressive? Why not (he said algos behind are different)?


# Gradient Descent

Matrix inversion is O(D^3)... So we do GD

Batch GD (using all points per update of parameters)

Problem - local vs global minimum

slide 51 #todo  batch vs iterative vs incremental

SGD #lookup Adam

# NN

Note that we can look at it as RBF (see upper left picture slide 58)
But difference is that parameters for feature projection are learned and not just predefined!
We let the model find the best projection so that problem becomes linear













