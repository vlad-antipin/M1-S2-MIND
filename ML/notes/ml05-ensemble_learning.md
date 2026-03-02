# Learning theory

Convergence theory, that our algo converges to optimal classifier provided enough examples

Sets of data E, function F, cost function is a prior basically

Risk is the expected cost, can't compute it directly (don't know joint distribution), can estimate with empirical risk - average cost, give more weight for combs X,Y that have more examples

Bayesian risk - minimal possible risk for this family of functions

We want to find a function approaching the Bayesian risk

"Consistance" of algorithm - if with n-> inf emperical risk approaches optimal risk
#todo universally consistent?

Different types of convergences: linear, logarithmic, exponential, we want algorithms that converge the fastest

No free lunch: if you compare two models, there will always exist a data distribution that will make one algorithm converge faster than another and vice versa! - there is no one model which is the best for all cases

Intuition: imagine you have an algo that had 100% accuracy on test, if there is a dataset with inverted test labels, algo will have 0% accuracy... so can always find a case where it sucks

so universally consistent algo doesn't exist? so have to choose models depending on use case

can't minimize real risk, only empirical risk -> overfitting 

how to control it - restrain F

Structural minimization #todo  (more in the context of decision trees - one model can be included in another)

slide 9: error decomposition #todo !!
approximation error = bias = introduced by us by choosing a specific family of functions

estimation error = variance = does the learnt function differ a lot depending on data?

think in terms of edge cases: very expressive functions - green space is big, very unexpressive functions - green space is super small

Example: finite search space #todo 
d binary descriptors and 1 binary outcome - $2^{2^d}$ possible functions - cf LRC

#todo borne PAC

Number of functions in F reflects expressivity

Expressivity ex: a family of functions shatters the set if there exists a distribution of points s.t. for every distribution of labels we can separate them? (e.g. for linear SVM everything breaks at hyperplane dimesion + 1)

VC-dim - nb of points that can be shattered #todo 

#lookup algorithmic bias - bias about how solution is found - Newton, GD, ... depending on how we scan the function space

# Ensemble learning

Joker game: who do you trust more - one expert or a large group of independent people? second is usually better

#todo how to define independence in this case? deterministic model doesn't give independent results... solutions
- bagging: force independence with introducing randomness
- boosting: independent by construction

## Bagging
### Decision trees 

from logical point of view - disjunction of conjunctions 
algorithm top-down, greedy, split nodes, maximizing entropy

we do conditional entropy - weighted average of children entropies, have balance between children

we maximize Information gain = initial entropy - conditional entropy

bunch of stop criteria

no stochastic in this algo

### Random Forest

Bagging: take multiple trees and work with the forst

2 sources of randomness: randomly choose samples and dimensions

it limits overfitting

doing forest of very shallow trees - each tree becomes a mini exprect of a region of data

## Boosting

Correct classifiers one by one, how to avoid overfitting - take "stupid" classifiers, additionally they are easy to learn

### AdaBoost

We can change the weight of samples (weight their losses)! Give more weight for mispredicted samples!

Off-top: error is > 0.5 (otherwise we need to inverse the prediction), and thus alpha is always positive #todo why alpha has this formula? it depends on error, interpreted as mixture coefficients for models slide 26
we normalize weights so that the form a distribution
formulation is iterative! a given weight implicitly depends on all previous classifiers

what stupid model to use? e.g. stump - tree of depth 1

## Gradient Boosting

what we can do instead is to fit residuals

Bagging - most popular models
Boosting - amazing for tabular data


# Evaluation

#todo 

# Multi-class classification

Multi-class mono-label (here), multi-label (when one example can belong to more then one class)

By nature multi-class: kNN, trees, bayesian classifiers

Naive strategies to adapt binary classifiers:
- one-vs-one
	- obtain a symmetric matrix, we look at marginal sums - and only one column will make sense with classifiers train on the real class of our examples, other will output some bullshit, we hope that they cancel out each other and that 
	- k(k-1)/2 - polynomial in nb classes 
	- redundancy, finer comparisons, but a lot of useless classifiers
- one-vs-all
	- biggest problem - class imbalance
	- as well no redundancy
	- but faster - k

multi-class SVMs

NN with softmax

what if a lot (>1000) classes? sparse problem?

2 approaches:
- flat - embed in smaller space
- hierarchical - in class tree

### Error Correcting Output Code (ECOC)

From information theory

We embed our data so that we get smaller number of classes #todo 

it's like a binary encoding of classes, in O(log k)? 

#todo table: K' by K, each of K' classifiers predicts -1, 1 for each of K classes, on the rows we detect the majority class for each classifier, do sum by columns (multiplied by majority class of each classifier) and the highest score gives predicted class, so each classifier is specialized on distinction of certain groups of samples

### Hierarchical Approach

Builds a class partition tree

prediction - a path

no redundancy

O(log k ) as well


Otherwise, neural networks handle nice this case provided a lot of data











