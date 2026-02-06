# Computational graph

 keep track of history of operations done on tensor
Inputs: network params and examples
Outputs: loss (we derivate it), scores (don't derive)

NB: exactly the same idea for blocks - just consider all it's parameters - need to implement forward and backward for the modules

Nest functions and concentrate on operators (do backward for all common operators) - better implementation, easier to implement models from articles

Note that gradients sum over incoming branches ("complete" chain rule) - this is why pytorch accumulates the gradient and we need to manually reset it to zero for new epoch, in practice before each call to `loss.backward()` 

`requires_grad=True` whether you would want to compute gradient for it, if False (by default) you treat it like constant / don't care to derivate wrt it

Comp. graph must by acyclic!! So you can do no circular dependencies!!!

However, in python such thing doesn't do circular dependencies, since each time new node is created instead:
```python
w = w - alpha * w.grad # off place? comp. graph creates new node
```
however there is a long chain in the graph that we don't need - very bad!! backward() doesn't know what to do...

one solution:
```python
with torch.no_grad(): # don't follow the gradient for those operations
	w -= alpha * grad_L # in place
w.grad.zero_() # reset gradient to zero
```

another solutionL
```python
with torch.no_grad(): 
	w = w - alpha * grad_L # off place
w.requires_grad_() # creates new graph (cut this chain), w being a leaf
```

For DL, the convention is to treat parameters of the same block together (put them in the same tensor?)

Could put all parameters in one tensor - ugly for big models

we can get parameters from the graph using `.parameters()`

sometines we want to update each parameters differently - e.g. update some parameters but not others (using a kind of prior, like for CNN) - common in foundation models #lookup 

Another strategy: (not very common, maybe doesn't work) - quit pytorch world
```python
w.data = w.data - alpha* w.grad.data # manipulate value directly
w.grad.data.zero_()
```

another one, quit the graph with `clone()` #todo

# MLP in pytorch

`forward()` - essential! 

`backward()` - optional, usually don't need to personalize unless pytorch doesn't know how to do backward or if you want to compute the gradient differently

MLP layer = dense layer = fully connected (fc)

```python
class MLP(nn.module):
	
	# architecture:
	def __init__(self):
		super.__init__()
		self.fc1 = nn.linear(..., ...) # ... - nb of input and output neurons
		self.fc2 = nn.linear(..., ...) 
		self.fc3 = nn.linear(..., ...) 
	
	# inference:
	def forward(self, x):
		x = x.view(x.size(0), -1) # flatten
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x
		
model = MLP()
model(x) # inference, calls forward
```

NB: on practice, final non-linearity such as  softmax with CE loss are usually combined together to avoid numeric explosion of exponent ( `nn.CrossEntropyLoss()` contains softmax directly as well, otherwise need NLL (neg log likelihood))

`F.relu()` from functional, `nn.Relu()` as a module and can use `nn.sequential()` #lookup 

now training:

```python
critetion = nn.CrossEntropyLoss() # contains softmax directly as well!
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam has inertia?

for epoch in range(10):
	model.train() # not necessary here
	# but need it in case the behavior
	# is differet in train compared to inference
	
	for batch_x, batch_y in train_loader:
		optimizer.zero_grad() # reinitialize gradients
		outputs = model(batch_x)
		loss = criterion(outputs, batch_y)
		loss.backward()
		optimizer.step() # update params with GD
```

compute the score:
```python
model.eval() # set it to inference

with torch.no_grad():
	# do by batch just for memory reasons - to avoid loading everything in RAM
	for batch_x, batch_y in test_loader: 
		outputs = model(batch_x)
		_, predicted = torch.argmax(outputs, 1)
		
		total += batch_y.size(0)
		# item to quit pytorch universe 
		# (otherwise there might be issue if work of GPU?):
		correct += (predicted == batch_y).sum().item() 
score = correct / total
```

NB: `torch.argmax(outputs, 1)` is  along 1st axis, since axis 0 is batch axis

# Other frameworks

We love pytorch in labs (you have more freedom when it comes to gradient...), but in production we use more tensorflow

Calculation graph is everywhere

Google: 
- TensorFlow 1: horrible, static graph #lookup computation and construction of comp. graph are separated (not convenient for debugging)
- TensorFlow 2: closer to pytorch, computation and construction of comp. graph at the same time - dynamic graph (eager) + compilation #lookup 
- Keras - very high-level abstraction on tensorflow, very easy but limited for small prototypes, prettier outputs
Meta: PyTorch

JAX - less common, compilation JIT (just in time) - more efficient #lookup 

## Tensorflow 1

```python
# declaration and initialization:
x = tf.placeholder(tf.float32, name="x")
w = tf.get_variable("w", initializer=tf.constant(10,0))
y = tf.pow(x, 2)

# compute:
with tf.Session() as sess:
	sess.run()
	# TODO: finish
```

#todo , but anyway you'll not use it, it's too bad for debugging, but internal graph management is simpler though, a bit more efficient
cannot do python loops there, need to use tf functions....

## Tensorflow 2

Better! Has immediate execution
NB: inverse logic compared to pytorch, by default no grad, to keep track of operations in graph need explicitely  `with tf.GradientTape() as tape:`
#todo  syntax

`tf.keras` - can use pieces of tf 

## JAX

define python functions using jnp (jax numpy) operations and manipulate callables with jax instead

when it's interpreted, can build something close to comp. graph having access to parser info - syntax tree

another approach -  from bytecode 

#lookup how is it implemented?

just in time (JIT) more efficient compilation with `@jax.jit`

# PyTorch in practice

```python
model = MLP()
critetion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) 

for epoch in range(10):
	model.train() 
	epoch_loss = 0.0
	
	for batch_idx, (batch_x, batch_y) in train_loader:
		optimizer.zero_grad() 
		outputs = model(batch_x)
		loss = criterion(outputs, batch_y)
		loss.backward()
		optimizer.step() 
		epoch_loss += loss().item() #?
```

What is lacking here? 
Keep track of values! 
Loss, validation scores - plot the curves to analyse the training behavior!!! 
Save in vector form - pdf or svg, NOT PNG and save the values behind those plots
Don't forget to give unique names
ML Flow are good for saving all that
Save model itself regularly, e.g. each 10-100 epochs
Especially when it comes to overfitting, we are not interested in the final model, but rather the one in the middle corresponding to the highest validation score

How to avoid copy-pasting code?
PyTorch Lightning - define training step, allowing to do fit(), not mature yet

Tensorboard - a bit more primitive version for, we can log (evolving) values and check gradients (check if it vanishes), good for debugging

ML FLow is more adapted to store final values, hyperparameters...

Dataloaders - better memory management, avoid putting everything in RAM, handles multiprocessing, dynamic data augmentation? #lookup 
```python
loader = DataLoader(dataset, batch_size=32, shuffle=True, ...) #TODO
```

transform?? #lookup 

Custom dataset:

```python
class CustomDataset(Dataset):
	def __init__(self, ...):
		... # e.g. load the data from file
	def __len__(self):
		...
	def __getitem__(self, idx):
		...
```

all other smart stuff like parallelization, batch handling is done by pytorch 

better save (hyper)parameters than the model itself torch.save_...? 

Interoperable formats - ONNX - good to share models with others













