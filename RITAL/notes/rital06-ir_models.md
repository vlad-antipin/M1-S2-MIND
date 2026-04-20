Simple hypothesis - more common words -> better match

# Boolean models

Documents are indexed by keywords 

Nice for interpretability e.g. for lawyers 

But how to make intersections of doment sets (based on whether they have a term) fast? We can, when we index documents, order indicies of documents! - enough to have two pointers going through documents lists corresponding to terms:
e.g. t_i -> d1, d7, d53, ...  t_j -> d2, d10, d53, ...

Problem - no notion of importance of a word or an answer

# Vector models

Represent documents in term or latent space

two approaches:
- word2vec + mean 
	- d≈1024
	- less discriminative due to averaging
- TF-IDF
	- d=|V|
	- more precise
	- faster since we can use optimizations of algos for sparse matrices
	- not semantic

in practice we prefer TF-IDF in IR, since we value precision a lot

About similarities - dot product vs cosine, cosine is preferred since we are more interested by semantic similarity which is encoded by direction (relative composition) and not the number of terms #todo 

# Probabilistic model

### Binary independent model (BIM)

We model the proba to have D=d, Q=q given that they are relevant R
we get P(R|q,d), if we want to order by it, it will come down to ordering by P(d|R,q) / P(d|not R,q)

then we make a hypothesis of independence like in Naive Bayes

problem (slide 11) - product across terms that are not in the documents - a lot of stupid computing, we do a trick so that we compute the product across all terms which is a constant

So with some tricks and 3 hypotheses, we get something looking like IDF...

(no TF there since it's binary)

so it's a kind of theoretical justification for IDF part of TF-IDF

### BM25

was THE model before neural nets

adds notion of TF

is a term elite - representative of a document

term frequencies are modeled with a mixture of 2 Poisson distributions - elite (higher lambda) and non-elite

being elite is just a binary variable and modeling its proba is kinda implicit BIM

a lot of theory... and then heuristic with tf / (tf + const)

so it's like TF-IDF but a bit better - there is saturation phenomenon 

#lookup is it applied in NLP? why not, especially for binary classification?

in the end, we have a way to compute weights used to evaluate relevance of a document

# Language models

Like Dirichlet etc..

General idea - use generative models to evaluate the probability of a question / document being generated from this doc / q

We can use the hypothesis of independence of terms 

# Query reformulation

We can enrich queries (e.g. add synonyms) 

Relevance feedback based on user feedback - we make query resemble d+, and further from d-
Pseudo -//- based on analysis of top returned documents















