NB: model is just a form, it can be completely wrong
# Linear models

Limitations: non-linear separation, XOR-like class disposition (disconnected regions)

Another problem: outliers, solution: GD instead of perfect separation...

Always relevant since we can try to make the problem linear 

Why XOR is popular counterexample? Historically people wanted to mimic logical 
operator with a function, perceptron is not capable of it

## Perceptron -> MLP

Hyperplane $w^Tx = \braket{ w,x} = \sum_i{x_iw_i} = 0$ 
NB: vector of weights is perpendicular to it

Perceptron - same thing, $x_i$ are entries (input neurons), $w_i$ - synaptic weights, $\sum_i{x_iw_i}$ as output (output neurons)

For binary classification - sign of inner product = predicted class (-1, 1)

How to introduce non-linearity? Add intermediary layers (MLP)?
$w^{\text{(layer)}}_{\text{next n. prev. n.}}$ - layer parameters, giving relative importance of each entry in this layer

Imagine 2 layers + output
second layer:
$y_i = \sum_{j}{w_{ij}^{(1)}x_j}$ 
output:
$z = \sum_{j}w_j^{(2)}y_i = \sum_{j}w_j (\sum_{k}{w_{ik}^{(1)}x_k})$ 
is still linear!!

NB: in practice we use affine models (with bias), abus de language to call them linear

#todo indices!!!!!!!!!!!! and matrix form

$z = W^{(2)} W^{(1)} x$  - can be represented with just one layer $W = W^{(2)}W^{(1)}$

So adding layers doesn't introduce non-linearity
We use then activation functions (often noted with h, g)
$z = h( W^{(2)} h(W^{(1)})) x$

which function? 
e.g. sign - bad for GD, since descent on the flat surface is not great...... and not derivable in 0
better options:
- tanh (+1?)  and sigmoid - similar form - good for final classification, gradient is small for a well classified point (flat region), larger for those close to decision boundary
- ReLU (Rectified Linear Unit) = $(x)_+$ - nice for intermediary layers? non-derivable at zero, but not a big issue

After all, those functions allow to approximate any non-linearity, no need to se prendre la tete

Forward pass - inference, as opposed to backward needed for learning

Loss function for learning, e.g. MSE
- local error $(\hat y_i - y_i)^2$ 
- global error $\sum_{i\in \text{train exaples}}(\hat y_i - y_i)^2 = ||\hat y - y||_2^2$

How to compute gradient of this loss wrt  parameters of each layer? Backpropagation layer by layer
Implementation of chain rule for compositions of functions,  when you compute partial derivatives for gradient
$\frac{ \partial L }{\partial {w_{i1}}} = \frac{ \partial L }{\partial {a_1}} \frac{ \partial a_1 }{\partial {w_{i1}}} = ...$

For a parameter of given layer, you need to consider thus a path from loss up to the corresponding input 

$F(x) = b_n \circ b_{n-1} \circ \dots \circ b_2 \circ b_1 (x) = b_n( b_{n-1}( ... b_2(b_1(x))))$ 
$b_k$ - one layer
$(b\circ g)^\prime = b ^\prime \circ g \ g^\prime$ 

NB: since you know these functions, you can programmatically derive them just following few rules
You can take a semantic approach, applying derivation rules e.g. $Pow(\text{"x"},2)$ and define derivative for it (e.g. a method for a class)  convenient to automatically compute gradients - like you derived yourself with rules in a table

NB: nb of nodes in intermediary neurons is a hyperparameters

Multiclass - one hot encoding $(0,\dots, 0, 1, 0,\dots, 0)$ but network outputs a distribution (and not necessarily normalized)

NB: activation function application to a hidden layer output can be considered as a separate
layer without parameters

Here, it was dense / strongly connected layers 

Theoretically, one hidden layer is enough to approximate any function!!! Just need enough nodes... But it turns out that making more layers is more optimal for learning

Problem with flat regions of activation function - blocks learning...
Solution: other architectures and strategies (ReLU, CNN, ...) #lookup 

PyTorch vs TensorFlow: PyTorch is used more
autograd - explicit derivation rules, simplifies backpropagation
tensors can remember  forward path and can go backwards to compute gradient

#todo practice working with tensors in pytorch
