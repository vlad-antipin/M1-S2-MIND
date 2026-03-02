
Need labels for classification, usually do captchas or exploit people in poor countries. Human labels are expensive and not always consistent.

Unsupervised learning is very subjective. Confirmation bias is often the case

If we want, we can always find a way to find clusters even in pure noise especially when we have many dimensions

Need to define similarity metric

2 approaches: partitioning / modelling
Clustering: hard / soft (proba of belonging)

Very domain-specific

Formalisation:

we want to assign k clusters
look at between- and within- cluster distance
problem: if high variance (noise) - can cluster after dimension reduction

# Clustering

## K-means

pi = partiotioning into D1, ..., Dk
A - assignements

$\arg\min_{\pi=(D_{1},\dots,D_{k})\sum_{i=1}^K}\sum_{x_{i}\in D_{i}} d(x_{j}, \mu_{i})$  - minimize within-cluster distance?

mu_i - centroids (baricenter)

Can be nice for anomaly detection

Converges to local minimum, need to do several initializations and take the one with the smallest objective

Can be used for image segmentation / compression

Usually converges, but for very specific cases can oscillate, so put timepoint 

K-means needs distance #lookup is it possible to do k-means only with distance matrix? no I guess

K-medioids - distance / similarity metric is enough, "centroid" belongs to dataset! - so can use distance metrics that don't verify triangular inequality, so can use e.g. cosine similarity etc

with cosine dist - radial partitionning

Voronoi cells, Voronoi partitioning

What is a good initialization - cover well the space, distant one from another, close to the data

Heuristic: sample initializations iteratively based on densities of points and already proposed (proba is bigger closer to data and further from already initiated centroids) -> **k-means++** (but k still must be fixed in advanced)

K-means - no hierarchy in clusters, if we lower k, we don't merge already existent clusters

## Hierarchical clustering

Hierarchical approaches: top-down  (divide) and bottom-up (agglomerative)

Agglomerative approaches 

It's enogh to have similarity metric

Distance between clusters? often not really a distance (linkage) - min (single linkage), max (complete linkage) or average 

E.g. for complete linkage: merge clusters if their farthest points are the closest compared to other clusters

We obtain dendrogram - can cut it at different levels to obtain clusters - #lookup Wald clustering?

Useful in NLP: see hierarchy in semantics!

and Byte pair encoding for tokenization ? #lookup 

it looks greedy - problem? #lookup 

again in NLP: make a sliding window, count concurrencies, normalize and it gives a similarity metric, then agglomerate

## Density approaches

k-means results in convex partitions #todo why?
we want to have more complex structures

we can assume that clusters are islands of high densities separated by regions of low density

DBSCAN - robust to noise and outliers (not the case for k-means since mean is sensitive to outliers)

#lookup Application in dbscAn

consider epsilon-neighborhood of a point and cluster only if there are enough (m) points - if neighborhood is m-dense, otherwise consider this point as noise point

too small epsilon: everything is noise
too large epsilon: everything is the same cluster

#question border / frontier points? - not alone, but not m-dense neithe 

how to choose epsilon? - knn ?? #lookup 

DBSCAN - no assumption on the form of cluster

Problem - if density is not the same for different clusters - bad if there are groups of different densities - OPTICS, HDBSCAN? are better

## Spectral clustering

Project data on a relation graph! Then cut edges between dissimilar points

Laplacian matrix? - eigen-decompose it

#todo theory behind cuts

In general, how to choose nb of clusters:
- are clusters compact? 
- Elbow method based on ratio within-cluster distances / total variance
- silhouette per point - mean of distances to points of closest cluster - mean of ... for the same cluster - we want a higher silhouette

# Dimension reduction

Why? - good for interpretability

Data often lives in a subspace or manifold of smaller dimension

Curse of dimensionality - almost all points become equidistant, clustering becomes tricky

We either select dimensions or create new ones

PCA - find new lower dimensional basis (each basis being a linear combination of original dimensions) so that most of variability is preserved

Eigen-decomposition of covariance matrix

Scree plot - elbow method

Correlation circle - attention, arrows close to origin are poorly represented by PCs

## MDS

distance matrix is enough
Compute gram matrix!! G = (X-m)(X-m).T
we need to eigen-decompose G and take eigenvectors

#question but Gram is similarity, not distance matrix

## Isomap

MDS with geodesic distance - nice to unfold manifolds 

How to get geodesic distance - from neighborhood / connectivity graph, do Dijkstra on it (shortest weighted (by distance) path)

## t-SNE

Latent space is very high-dimensions for NN, so methods before still yeild a lot of dimensions

So we do a projection trying to preseve neighborhood of each point - preserve densities / distributions

Compare distributions - KL-distance

we look for $f_{\theta}$ - a projection parametered by theta
$f_{\theta} \in \mathbb{R}^{d'}$ d' << d
$\theta = \arg\min_{\theta} KL(P(X)|P(f_{\theta}(X)))$ - check the formula

what does perplexity corresponds to?

ATTENTION: easy to cheat - get false clusters even in uniform data! UMAP is a bit better but still can cheat... 

## Dictionary learning

PCA learns orthogonal basis, we can create a dictionary of redundant "primitives" instead and linearly combine them - slide 47

Sparsity constraint with zero norm - hard to optimize

## NMF

Non-negative matrix factorization #lookup 

## Auto-Encoders

Encoder - bottleneck - Decoder

Loss is reconstruction error - self supervised

How to interpret the result?

