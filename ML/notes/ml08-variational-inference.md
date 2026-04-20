Classical approach - estimate parameters, problem - can we trust the learned model?
We can take a model and refine it, try adversarial examples, OOD...

Terminology NB: model <=> parameters

Bayesian inference - analyze a posterior distribution $p(w|D)$ of parameters and then prediction with predictive distribution $p(y|x,D) = \int p(y|x,w)p(w|D)dw$ - expected value over all possible models 
#todo what hypotheses? why not $p(y|x,D) = \int p(y|x,w, D)p(w|x,D)dw$ ? - $y \bot D | w$ and $w \bot x$ ?

Good since it: allows to quantify uncertainty, avoids overfitting

So instead of learning a model, we learn a distribution of models

Bayes rule: problematic term is evidence $p(D) = \int p(D|w)p(w)dw$ - very often is not computable

How to evaluate uncertainty? 
- p.ex. empirical - learn multiple models and look how predictions are distributed #todo not exactly what we want with Bayesian methods? (e.g. convex problems will give always the same response)
- Bayesian models: analyze posterior parameter distribution NB: on those fancy plots they put MAP as a e.g. decision boundary, for "confidence intervals" sometimes there is an analytical form, however sometimes we need to sample models

So instead of MLE / risk minimization, we do MAP: minimize -log(posterior)

Predictive distribution $p(y|x,D) = \int p(y|x,w)p(w|D)dw$ is an expected value over models, can be estimated by sampling

Usually there is no analytical form for posterior and predictive distribution computation :(

Conjugate prior - distribution so that posterior  is of the same form as prior

Conjugate prior for Gaussian distribution is Gaussian

Bayesian Linear Regression - 

we can add noise to our linear regression model - residuals follow a normal distribution, we evaluate noise as well - empirical variance of residuals

we can integrate Gaussian prior on weights

On logistic regression it will not be so nicely computable :(  -> Monte Carlo

MAP with Gaussian prior (same $\sigma$) = Ridge regularization 
NB: since $\sigma$ is the same, weights and thus the data is supposed to be on the same scale

Problem - we have only MAP but we want the whole posterior

So we can approximate posterior by a Gaussian centered on MAP and covariance fitted on Hessian

And then we can use Monte Carlo to estimate predictive distribution

Variational inference - another approach to approximate posterior - gradient descent of the loss on distributions -  KL divergence

ELBO - evidence lower bound #lookup 

Mean field hypothesis - parameters are independent (strong hypothesis since for e.g. decision trees parameters are fitted sequentially so not independent, or even NN)
we can do coordinate gradient descent



