When we have more than words - sentences, documents, ... we can do smarter than doc2vec

Approaches: CNN, RNN, Transformers
# CNN

local feature extraction, can do on text as well!

so temporal dimension is like in 1D image, and for each word we lookup the corresponding embedding (e.g. learned of from word2vec)  

Note:
1. In text CNNs, the sequence length is the spatial axis and embedding dimensions act like channels; kernels slide across words but span the full embedding depth.
2. We do not convolve across embedding dimensions because they are feature coordinates without spatial structure, not neighboring locations.

```python
self.conv = nn.Conv1d(
	in_channels=embedding_dim,   # d
	out_channels=num_filters,    # F
	kernel_size=kernel_size      # k (over sequence)
)
```

# RNN

cf IDLE, we accumulate information as we progress in text, LSTM and GRU handle the issue of vanishing gradient and thus of long term memory

LASCIATEMI MANGIARE

Many to many parallel
- nice for POS tagging

Many to one
- e.g. classify text (use the last hidden state which cumulated info from the past)
- for VisualQA (in case where answer is kinda a label), note that CNNs and LSTMS before final FC layers can be pre-trained and used either just as initializations or even fix them and don't update during backprop

Many to many
- like chat gpt
- translation - speech2text
One to many 
- generate text from seed (e.g. generate story from an image)

We can train it in self-supervised manner like for word2vec

Once RNN is trained we can use it for task other than it as trained for!

How to evaluate stuff like VQA? Well we can compute accuracies, analyse in which cases it makes errors, look into the representations / attention - "open the black box", analyze why it doesn't work, for which examples

e.g. a way to verify the quality of fine-tuned representations is to apply them to different tasks (or those it was originally trained on)

BiLSTM - in both directions, more global knowledge

## ElMo

embedding layer -> BiLSTM -> output layer

# Attention Model & Transformers

LSTM was kinda attention, we retain info that is important

For transformers, instead of of considering a word and sequence behind it, we consider a context window and build attention map (NxN)

self attention:

We use dictionary idea - representation for query and key to compute the value, attention score (value vs attention score) for a word

CLS token - global token - combination of all other models in sentence

We can integrate info about position, whether it's part of question / answer as embedding vectors

#todo multihead attention, a lot of parallel branches of self-attention

Bidirectional Encoder Representations from Transformer (BERT)

Generative Pre-trained Transformer - basically classification task

Prompt (in context) learning allows few-shot 










