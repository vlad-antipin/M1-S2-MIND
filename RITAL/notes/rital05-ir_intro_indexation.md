# Information retrieval 

Contrary to SQL like cases, we handle non-structured data

#lookup RAG applications

Evaluation campaigns: TREC e.g. Clinical Trials Track, CLEF

3 steps:
- indexation
- pairing / matching - assign a score to a pair question - document
- reformulation

System relevance (user-agnostic) vs user relevance (a bit harder)

Trade-off between recall and precision in IR - send a few of high-quality docs or a lot but maybe lower quality, in IR Recall@10 - for first 10 documents?

#lookup NDSG

# Preprocessing

Index the documents -> BoW, TF-IDF, embeddings .. -> index and inverted index

# Retrieval

TAAT - term at a time

We can eliminate bad matches efficiently since we know the lower bound of sum of scores

DAAT - document at a time

First vs second stage

Dense (from NN) vs sparse (usual) representations





