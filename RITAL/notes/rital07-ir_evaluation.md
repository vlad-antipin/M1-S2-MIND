
User evaluation - a real person scores whether the document was useful

#lookup test corpora for IR

Rank and relevance related metrics

Recall and precision as for binary classification (relevant - not relevant, retrieved - not retrieved): high recall - returns all relevant docs, high precision - returns only relevant docs

P@k - precision at top k docs returned

Precision-Recall curve

Probabilictic interpretation of ROC-AUC and AP #todo 

# Rank oriented metrics

MRR = mean reciprocal rank (over relevant documents, if document is not there , reciprocal rank is 0, as if rank was infinite)

DCG = discounted cumulative gain, if the ground truth is itself ranked - sum of gradual relevances of returned documents, discounted by sth depending of rank in the answer  (e.g. 1/log2(i)), can normalize it by a maximum possible value so that is btw 0 and 1 (max DCG for perfectly ranked case) -> NDCG

# Comparison of IR systems

E.g.  compare means of scores with t-student (paired since two systems are tested on the same queries)

an individual - a query

don't forget Bonferroni correction for multiple comparisons


# Other protocols

Use user logs - analyze what user has clicked, whether they reformulated the question

A/B testing - just deploy and compare
user click = relevant document

can do with interleaving - instead of proposing options A and B separately, propose a mix - better since it becomes paired, and if one system is really bad, there will not be a user that will have only horrible results


