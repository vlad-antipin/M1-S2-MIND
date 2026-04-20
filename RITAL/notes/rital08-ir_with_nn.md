
Learning To Rank

How to combine results of models like BM25 (depends on query), PageRank (independent of query)

we can do classic ML (like logistic regression) to predict a probability of relevance, based on features Ф that are represented by a bunch of scores like BM25 on title, on url etc...
Ф can be learned itself, but superslow :(

we can do in two steps re-ranking - first find top-k documents with the simpler method (maximize the recall) and second - reorder and maximize more elaborate IR metrics

problem with metrics - they are not derivable, problem is thus relaxed to have a nice derivable loss -> #todo ordinal regression, rankSVM etc, we can compute losses by pair #todo pointwise vs pairwise vs listwise

pointwise - ignore the order, just look at he relevance 
- PCE (pointwise cross entropy)
pairwise - relevant documents have to have better score that irrelevant
- max-margin 
listwise - ?, like softrank, make something like NDCG continuous and derivable
- NCE #todo 
- SoftRank - inspired by NDCG, but rank is soft - instead of 0-1 proba that a document is ranked  as r, there is a proba btw 0 and 1, d(r) - discount
	- model the score as a normal distriubutions with the same variance sigma
Where imbalance is the problem - over / under sampling #todo  for PCE oversampling of +?
# The sources of data

How do we know that d is in Dq+? if it was clicked by enough of users, or manual labelling, or use hyperlink text as a source of semantic of the doc

Problem with D- - harder to obtain, but we can suppose if it wasn't labelled Dq+ that it's Dq-

Another problem - a lot of very irrelevant ("orthogonal") documents and we can overfit them, so instead a lifehack would be to take Dq- from the e.g. top-1000 docs obtained with BM25
-> we have a problem of false negatives because of it
we can filter for FN with a very performant models like transformers, or do a distillation to speed it up? 
distillation works very good when data is noisy (a lot of FN), for distillation we can chain models of the same or different architecture - train one model on the original labels, train the second one on the predicted labels etc 

**distillation is super efficient**

problem of NN models - high overfitting, to be tested with BEIR dataset

representaion learning - match not just vocabulary, but whether it matches semantically

Architectures: pass Q and D separately through attention blocks and then take a dot product of their CLS tokens - rather weak but fast, 
strong but slow - concatenate Q and D and take CLS to predict score

intermediate approach - ColBERT, get a interaction matrix query-doc and treat it as an image - pass through attention blocks and get CLS
- based on similarity matrix
- how to get it? e.g. cos similarity between word2vec or representations from transformers
- takes MaxSim - sum over query and document terms of maximumal cosine similarities?

generative models RankGPT didn't work that well since super expensive, but can be used as teacher

Symmetric (same embedding for queries and doc) is more practical then asymmetric

transformer >> CNN >> RNN 
those models work fast enough to be used instead of BM25 #todo ah ouais???

Dense indices - smart way to index dense representations

# Parsimonious models

Imitate BM25 with dense representation