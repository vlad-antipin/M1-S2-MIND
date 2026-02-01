
#todo toml file and uv for TME

# Tasks and cost functions

Previously we talked about MLP in the context of classification, but it's. just because it's easy to present

MLP for regression task - easiest non-linearity we can introduce

We can modify:
- structure of layers (e.g. like CNN)
- loss function to adapt to task (classification, regression)

## Cost functions

Local error: for un example $\Delta(y_{i}, \hat{y_{i}})$
Global error: $\sum_{i} \Delta(y_{i}, \hat{y_{i}})$

**MSE**
- $\Delta(y_{i}, \hat{y_{i}}) = (\hat{y_{i}} - y_{i})^2$ 
- we can normalize it by $n$, but it doesn't change optimization
- may need to normalize if you want to compare MSE between different models

Motivation for square:
- MLE
- derivative is not flat like for absolute value
	- it corrects more errors that are more flagrant
	- (the further you are from 0, the more the tangent to parabole is steep, the more it will contribute to update)

For classification, MSE can function as well ( #todo how?) but it's bad 
NB: for decision boundary, $w^Tx=0$ is arbitrary, if you change it before training, it will be just absorbed in a bias term
but once the model is trained, we can play around with the boundary (play with sensitivity specificity, ROC)

$f_x{w} \geq c$ forms a margin of arbitrary shape #todo 

**BCE** (Binary Cross Entropy)
- we model binary label proba - benoulli trick - NLL equivalent to cross entropy!

For multiclass case we could have labeled them as 0,1,2,3..., but since we want them to be *equidistant*, so we use one-hot encoding which corresponds to a special (deterministic) case of multinomial distribution 

Why do we consider all $k$ classes and not $k-1$? Because we need to have all $k$ neurons for final normalizations

So we obtain some non-normalized predictions in the output, we could have used $max$ as activation for the final layer - problem: it's not derivable everywhere....
so we do $softmax$ instead - behaves like $max$ but is derivable

$\text{softmax}(z)_{i} = \frac{\exp(z_{i})}{\sum_{j} \exp(z_{j})}$ 
exponent - contrast is pretty high, small values collapse almost to zero, big almost to 1, if there are similar values (model is not sure) model will not be forced to give 1, so at the very beginning of training the model is not forced to choose THE class

problem - numeric explosion - mitigated by taking log for loss (in pytorch it's even in the same block - log softmax)

NB: btw with binary classification the similar issue - decision is a threshold ($sign$ function), so we use sigmoid or tanh 

About the loss, the same thing as for BCE, we use cross entropy or equivalently KL divergence

#todo multiclass (seen here) vs multilabel (multiple binary clasifiers??)

#todo and if classes have an order? (e.g. discreet ordinal variable)

NB: in this simple case, errors for all classes are equivalent, but we can penalize misclassification of some classes more 

So we made some tricks so that our loss is derivable, but our final goal is to maximize number of well classified points (that we measure with score -accuracy, precision...) and we have to check it since we don't have a guarantee with out ticks

We want to optimize the score (can't do GD), but can only optimize the loss (with GD since it's derivable)... - a problem in ML

In ML we want to generalize i.e. want model to work on random samples from out data, to generalize on test data - need validation / test

Optimization and learning are different...

Another problem - when to stop GD? When few meaningful variables - easy to motivate, when a lot of uninterpretable variables 

## Feature representation

#todo  slides 15-38

MLP = dense layers = fully connected layers

Autoencoders - like PCA but non-linear, do a representation of examples in few dimensions, allowing to reconstruct the examples
**encoder - decoder architecture** with **latent space** in between
we want $f^-1(f(x)) \approx x$ (self-supervised)
once encoder - decoder is trained, we can use encoder to get representations

NB: if activation function is linear, it's equivalent to PCA!!! yes, we can do it with GD

Nice to create features, problem for classification - because of MSE ( #todo ?), the image that we want to classify becomes blurred #todo 

#lookup how is it used for generative models?
