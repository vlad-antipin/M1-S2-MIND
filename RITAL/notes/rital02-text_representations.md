
# Preprocessing

Problem during preprocessing -encoding (e.g. accents in french, arab letters...)

Given preprocessed text, get vector representations

Ways to represent the text: 
- series of characters 
	- bag of characters can work! some first models did characer by character
	- advantage: vocabulary is guaranteed to be completely known and is small
	- #lookup examples of tasks where it works
- set of words (BoW)
- N-gram dictionary
	- sliding window taking N-grams 
	- can be at the level of words or letters

Token - raw inputs, can be based characters, words, N-grams, now it's rather subwords, good for for example formation of new words
#lookup byte pair encoding (BPE)

Normalization of words -stemming, especially helpful when it comes to plurals etc, we can also query knowledge bases beforehand  - sometimes it's smarter than just relying on the learning algorithm

Lemmatization: e.g. lions -> lion, are -> be #lookup what is lemma, linguistics based
Stemming: based on statistics, crop the word

Can use regex e.g. expanding contractions like I'll to I will

#lookup visualisations of BoW - heatmaps, clouds, ...

Zipf's law: kind of power law for word frequencies that are inversely proportional to its frequency rank

Heap's law: when you add words in the collection, we add words to vocabulary, but it slowly saturates, proportional to log  

# Representations

Let's not take order in consideration - BoW

One-hot representation (Binary BoW) -  word is present / absent #lookup for naive bayes, is it enough? need counts normally

BoW - use frequency instead, as a number of instances of this word
But can as well give higher weights for words that are discriminative enough across the documents. Imagine that you have a document having a lot of "the", well all other documents have a lot of "the" as well, so smaller weights need to be given to "the".

Term frequency: $\forall i, \sum_{k} d_{ik}^{(tf)} =1$
Term frequency - inverse document frequency $d_{ik}^{(tfidf)} = d_{ik}^{(tf)}\log \frac{|C|}{|\{ d: t_{k} \in d \}|} = d_{ik}^{tf}\log \frac{1}{ df_{k}}$ for corpus C, document d_i is i-th document, k is kth word in vocabulary

why log - to saturate inverse df

Whatever it was binary, frequency (counts) or tf-idf we have vectors and variables to work with

Packages `nltk`, `sklearn`

BoW 
- pros: easy, fast
- cons: 
	- order and sentence structure is lost, so some tasks are not even doable, like text generation 
	- semantic gap: car, automobile etc are completely unrelated, unless you do it yourself with knowledge bases, or smarter representations like LSA
- extensions:
	- N-grams add a bit of semantics
		- problem: explosion of dimensionality -> curse of dimensionality, problems with euclidean distances - all points are almost equidistant
			- can do KL divergence instead
			- cos? #todo 

# Usupervised approaches for representations

LSA (Latent Semantic Analysis), K-means, Probabiilistic LSA
Latent Dirichlet Allocation (LDA) the most popular, direct ancestor of NN approaches
We worked with embedding far before NN

LSA - does TruncatedSVD, idea similar to PCA #todo difference - just centering?  #question  why not center

#todo understand the picture slide 32 and 33, what are matrices on the margins

so LSA extracts semantic info based on proximity of documents, new dimensions are "latent themes / subject" (thématiques latentes)

we can do t-SNE on LSA space, so two steps of dimension reduction #question why not to use t-SNE directly? is sparsity crucial?

Quality assessment with purity coefficient

No one good way to do, can play around e.g. apply k-means on LSA result

Graphical models:

PLSA - probabilistic LSA,  one word can belong to several themes? - learned with EM algo! #todo 

LDA - extension of PLSA by integrating the prior, learned with Gibbs sampling

Theme has to be homogenous, i.e. words are exclusively associated for a given theme with a certain proba, good theme is the one that has rather uniform distribution for the words that compose this theme, i.e. maximal proba has to be the smallest - sums of all probas is perplexity?? note that for a theme prbas don't sum to one since not all terms are included??  #todo  If there is a peak in proba, then theme is not well chosen

























