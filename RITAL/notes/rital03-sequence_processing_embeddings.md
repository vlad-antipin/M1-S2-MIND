
# Sequence processing

Markov chains (MM)

CRFs are discriminative - used a lot today

## Markov Model

like in MAPSI

params: lambda = {pi, A}

## HMM

hidden states and observations

when hidden states are not hidden, easy to learn - just count

Useful for PoS (Part of Speech) tagging

e.g. after "the" there is often a noun

For decoding - Viterbi (see MAPSI)

Can analyse transition matrix as heatmap

## CRF

Discriminative - oppose one class to others, need negative examples

We will reuse this idea in neural networks

basically applies softmax to representations, but how to find them?

CRF - each states considers all observations

#todo HMM vs CRF, HMM can be used for generation, CRF not?

HMM is count based, estimate probas, was used a lot before, now almost not used, HMM is more used in social networks 

CRF - positive vs negative #lookup does it need order

We can add CRF to transformer embeddings and it works good! #lookup 

We have to get features for words (word, digit, is capitalized, ...)

#todo slide 18: NB -> HMM vs LogReg -> CRF

# Word embeddings

Sequence is accounted for when latent representation is learned! 

#lookup different ways to combine sequential models and embeddings

Embeddings - vectors of real number, corresponding to latent representation

First embeddings, based on BoW - LSA, ... 

Based on NNs:
2003 - first embedding on text
2014 - word2vec 

We want to be able to compute semantic similarity of words based on the similarity of their embeddings

One-hot repreentations - dot product is zero #todo really? don't think so tbh

Text embeddings 

Distributional hypothesis: words that appear in similar contexts (based on neighboring words), have similar meanings
-> co-occurence matrix -> SVD 

Word2vec - use NNs and not SVD to learn representation
how to know inputs / outputs - ask people how each word is close to one another? - bad
So we use auto-supervised approach, input from sliding window, based on context predict middle word, context size is a hyperparameter (we usually do 5)

Note that preprocessing is rather optional here

2 NN architechtures - SkipGram (middle word -> context), CBOW (context -> middle word)

for inputs / outputs - one hot representations - indication matrices? at the end - softmax
#todo hierarchical softmax

embeddings are in hidden layers - W matrix is embedding matrix! #question  To get embedding of a given word, furnish a vector 0,..,0,1,0,...,0 and look at activations or not ?? There are 2 matrices W (one for w's of first layer, another for last)- one for middle word, one for context

for NLP we usually take W corresponding to middle word,  for IR sometimes we consider both

CBoW is better for frequent words, SkipGram for rare - why?

problem - softmax is expensive when vocabulary is huge (doing sum in denominator), so instead just sample negative examples
increase proba of positive samples and decrease this of negative  ones? #question  connection to NN? cosine similiarity?

Then can reduce dimensions with tSNE, PCA etc...

slide 33 - poke around

those embeddings can be used in classical ML models

NOTE: it's not task dependent

In IR it doesn't work

 Glove - based on co-occurence matrix, mixes different approaches #todo 

We know how to get word embeddings, but **how to represent documents??** 
- at the beginning we did means of embeddings.... then give it to e.g. classifier, very bad for large documents
- or build embeddings of documents directly - doc2vec: just add index of the document - surprisingly it works!

Document size is often an issue

Measure similarity: cos, PMI

Problem - what if there is an unknown word - one strategy to do UNK word, not ideal...

Recent extensions: 
- FastText 
	- word2vec on subwords instead (BPE - byte pair encoding) - ok for "unknown" words
- ELMo 
	- nice for polysemic words
	- contextual embeddings - first step towards transformers
- BERT
	- transformers on text

# Knowledge bases

So we can handle semantics automatically, however good knowledge bases are more reliable, especially in scientific domains / medicine

WordNet #lookup & metrics on knowledge graph are available in NLTK
Google KG #lookup 












