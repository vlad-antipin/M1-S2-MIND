Slides and annals are on the MIND site
ML = Statistical learning

Application examples:
- smap detection
- digit recognition 
- ...

How to get data:
- captchas
- captchas in porn
- exploit people from poor countries

Here, it's data-driven, we want to avoid integration of expert knowledge - to avoid biases and make an algorithm more auto-sufficient

NB: lingo example = observation = individual

Supervised learning - based on the type of label
- classification - discrete class
- regression  - continuous variable(s)
- forecasting - n last values to predict next values of time series
- data completion - e.g. for missing data from sensors
- ranking  - rank items
- recommendation - couples user-item

Unsupervised learning
- clustering
	- problem: clusters can be meaningless...
- representation learning (dictionary learning) #lookup - self-learning? - to create embeddings #lookup random projection
- sequence analysis
- hierarchic representation
- anomaly detection

Reinforcement Learning 
- learn to act / interact with environment

# Bayesian Learning with Density estimation 

Classification like was done in MAPSI, but we were dealing with discrete r.v.

Log-transform features if it helps to make arithmetic differences comparable

Dummy classifier: predict majority class

Risk - expected error

When descriptors are continuous, we cannot model joint distribution entry by entry... 
we consider probability densities then

use Bayes rule, then choose the most probable class $f(\mathbf{x}) = \arg \max_y p(y|\mathbf{x})$   

Risk = 1 - proba of predicted class

Bayesian classifier - the classifier with the smallest risk

How to estimate the density?
- assume independence between dimensions
- discretize ....
- others to follow

Do a histogram if a variable is naturally discrete, same in 2D

With continuous variables - discretize, but how?
- sometimes it's natural (e.g. by day o by week)
- if not enough bins - underfitting, if too many - overfitting

Go from PDF to PMF - integrate, for constant proba need to scale by volume, same for inverse path - normalize by bin "volume"
$P(A) = \int_x p_{const}dx = p_{const} V(A)$ 

Problem with histogram - sensitive to "effet de bord" - how we position the grid
Solution - consider a hypercube window centered on points of interest
It's a Bernoulli for each hypercube $V(A) = r^d$ 

We can do better than hypercube - various kernels (Gaussian, Laplassian, ...) - KDE #lookup connections to convolution? 

How to know which density estimation is good? It's hyperparameter, but how to tune it? Create a test set and compute its likelihood with the density estimated from train
NB: attention to comparing likelihoods, more points - less likelihood!

Conclusion:
Histogram vs Kernel
Kernel is distance-based, no model learned contrary to histogram
Furthermore kernels are bad with high-dimensional data because of curse of dimensionality

Estimator of Nadaraya-Watson
Use density estimation for classification, then get p(y|x) and do a difference p(y=+1|x) - p(y=-1|x) and analyse its sign in different regions. It's similar to kNN, but instead an averaging over local  neighborhood (avg weighted by ditance)
Perzen? window #lookup 








