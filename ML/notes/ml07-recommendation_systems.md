
Problem that e.g. ratings on movies are subjective, not as classic classification

User and products are symmetric

User info, product info -> affinity prediction / ranked lists of items

Problem - a lot of items noted as "good", how to order them? We need to diversify the propositions as well, which is not handled by affinity score

Utility function

Approaches: 
## CB - content based 
	- tabular or textual description
	- very scalable
	- a lot of feature engineering -> kNN graph
	- rather item centered than user centered - not personalized
	- no authority score #lookup PageRank has it or doesn't?
## CF - collaborative filtering 

Based only on interactions user-item (linked to behavior modeling), compare preference history between users, instance based or rather models like SVD

Example from logic - frequent item set, not very useful

We want to produce affinity score

Neighborhood-based approaches 

Cosine similarity is preferred to euclidean distance #lookup is it more robust to imputation of missing values?

Slope based alternatives

Bi-partite graph analysis with e.g. random walk

Problem with missing values: MCAR hypothesis is usually false (users don't rate movies they never seen or that they don't like)

We can decompose user-item matrix - do SVD on it 
$X=U\Sigma V^T$  (user x item) ≈ (user x latent user) . (latent user x latent item) . (item x latent item)
#todo link to PCA and whether how do we integrate new data

Intuition: users usually prefer not just separate movies, but genres, and symmetrically, there are categories of users

NMF (Non-negative Matrix Factorization) - constrain SVD to make it more interpretable, like this values can be better interpreted as scores since we know that it cannot be counterbalanced by a negative value, nice for source separation (came from physics)

AUC is a bad metric since we are interested only by a first part (top items)
#todo why?

slide 49 - I, U - one hot encoded matrices

#todo NN











