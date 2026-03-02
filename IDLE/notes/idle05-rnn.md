
With CNNs, constitute first foundation models

Motivation: 
- how to treat temporal aspect - causality / order (compare to CNN, can compare in all directions, locally)
	- e.g. in audio, text (even if we can treat in both directions)
- it solves as well inputs of variable size
	- e.g.MLP or CNN initially need fixed size inputs
	- life-hack as for CNN will not work? #lookup 
- in CNN we had a notion of receptive field, in RNN it's notions of handling long and short term dependencies 

Size of hidden state is a hyper-parameter, $h_t$ is updated each time new input arrives, note that weight matrices are the same at each t at inference 

if we don't update hidden state a lot, we don't memorize, if we update too much, we forget....

$h_t$ theoretically depends on the whole history, but in practice on only recent information

$h_{t} = \sigma(W_{ih} + W_{hh}h_{{t-1}} + b_{h})$
$y_{t} = W_{hy}h_{t} + b_{y}$ ?

We can unfold this recurrence and obtain just a usual computation graph (BPTT - backprop though time)

#todo how exactly weight sharing affects backprop #todo simple example

BPTT - when develop a chan rule - very long multiplication chain back in time, so issue of vanishing gradient even with one hidden layer

So in practice we just limit T (fix the size of time window)

W's and thus number of parameters doesn't depend on the dataset size (like for CNN)

We can stack layers

Problem of RNN - successive handling, and we cannot parallelize it

What to use for prediction? (and thus for loss) - depends on task:
- take only the last time-point: $y_{T}$ - equivalent to take $h_{T}$ and do MLP
- or each $y_{t}$ - but need to tag each element of sequence (like POS)

Generation: #todo check it
- one-to-many
	- text geenration
	- all weights are still shared
	- seed (random or sth like prompt) , $h_{0}$-> $y_{1}$, $h_{1}$ ->  $y_{2}$, $h_{2}$ -> ... use outputs as inputs
	- trained by masking parts of sentence
- many-to-one
	- e.g. sentiment analysis
	- similar, but at first uses $x_{1}, x_{2}, x_{3}, \dots$ (prompt) as inputs and then like one to
- many-to-many
	- e.g. for translation, résumé

Conceptually, we can do prompt engineering and utilisation d'un LLM en few shot - which consists in modifying not weights but hidden states 

In practice, learning is almost not possible beyond a small window of recent terms because of vanishing gradient, of gradient exploses exponentially if norm of W is larger than 1, we can just clip it? #todo 

Another problem - hidden state is of limited size

Solutions: LSTM, GRU,... or learn context with attention layers, subtility of transofrmers is that context window is fixed by design (but now it's pretty big, and at least we are honest about context window)

## LSTM

We choose what we store and what we forget - derivable dictionary, creates an additive path in during backprop - similar in spirit to solution with skip connections

Forget, input and output gates - CRUD logic 

if $f_{t}$ is zero, we forget
if $i_{t}$ is zero, we don't update with input (so we skip updates)
if $o_{t}$ is zero, we don't output

we can see the cell $c_{t}$ as long-term memory, and $h_{t}$ as short term memory

why do we call it derivable dictionary #lookup 

another interpretation - similar to skip connections of ResNet

## Bi-RNN / Bi-LSTM

We can look in both directions, it works only we when have access to the full sequence 

Similar for transformers if we center the context window on current word

Fundamental distinction: e.g. GPT (only past, for generation) vs BERT (both directions, for global comprehension)

## Applications to text

How to encode text - one-hot-encoding, we get high-dimensional (vocab size) sparse vector at the very beginning, the very first layer is implemented as representation learning - embeddings, so we learn a huge embedding matrix - but instead of doing multiplication of sparse vector of 0 and one 1, we just choose the line of embedding matrix! (we can imagine it as a dictionary)

In computation graph, it's always one of line of W considered

Can use pre-trained embeddings from e.g. word2vec (self-supervised), #todo fine-tuning would be not fixing embeddings but only initiate them

## Foundation models & Fine Tuning

#todo !!

think of using chat gpt for classification:
zero-shot - no examples given
few-shot - give some examples

#lookup we can fine-tune prompts?





