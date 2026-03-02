
# Regression and kernels

Kernels allow working with any type of objects (text, graph)

Kernel - inner product in some projection space

Projection $\phi$ , projected X is $\Phi$ 

Linear regression problem can be as well represented in form of inner products! and thus we can use kernels

$\Phi ^T \Phi$ is a Gram matrix K - matrix of inner products

so we can do gradient descent and inference with using just kernel! - we can avoid using phi and use only k, so no need to do explicitly the projection

# Gaussian Process 

Deterministic process with gaussian (white) noise

Hypothesis of Gaussian noise - like in MAPSI, independent $\epsilon_{i}$ (same noise for all x_i) ~ N(0, sigma^2)

we can use independence of examples and do MLE

why adding constraining hypothesis of gaussian noise instead of just using MSE? - **we can estimate the noise**  (and thus **uncertainty**) in addition to prediction, which is not the case of MSE

# Multivariate Gaussian

Naive Bayse - independent coordinates - diagonal $\Sigma$ 

Affine transformation with invertible T and mu: 
$Cov(TX + \mu) = TT^{\top}$  #todo proofs on slide 13

## Gaussian magic :)

Everything is gaussian, marginal projections (conditioning) of gaussians are guassian, sums of gaussians are gaussian...

How to visualize high dimensional gaussian? #todo  slide 22: sample points from this gaussian and plot in 1D and observe relations between coordinates (correlations results in sinchronized signals)

# Gaussian Process

we don't work in feature space, our points are the dimensions?

GP construct similarities

Instead of finding the best model, we can integrate across all possible models (w) 

so we can use our examples as dimensions and estimate how they are correlated! - covariance matrix is a kernel!! #todo 

so we can just fabricate the notion of similarity between the examples

#todo : definition formelle + slide 29 + Bayesian optimization

#todo : watch videos / lectures on this subject slide 30

O(N^3) - espensive

Kernel choice 

Midterm: everything except today's lecture