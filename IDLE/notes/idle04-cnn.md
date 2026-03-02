Note that these ideas can be applied to other signals than just  images - e.g. 1D convolutions on audio, 2D convolutions of time-frequency representation of the signal, 3D convolutions on video or 3D images

Why MLP is not enough?

Need to fit too many parameters, since dense structure, especially since we want **invariance** to geometric transformations (and stuff like occlusions etc...), we would need even bigger MLP

Before deep learning we were engineering features (see BIMA)

We want **equivariance** as well - if input is transformed, we want output to be transformed similarly

Convolution is exactly that! - linear time invariant system (see BIMA) - it's translation invariant but what about rotation (e.g. have one horizontal filter vs hor.+ vert. filter? are two filters like this are enough to detect all rotations) #lookup 

Convolution can be seen as a small perceptron (kernel) that we slide on the image, or as a big perceptron but with **shared weights**, and a lot of zeros set to 0 - sparse interactions

back-propagation will treat a lot of weights as the same

for each convolutional layer there can be multiple input channels (e.g. 3 for RGB, or outputs of previous layers), and we decide on how many kernels we want to apply and thus define the number of output channels - but don't get overwhelmed with number of channels, complexity is rather expressed in depth of the CNN (think BIMA, several filters are usually enough)

For convolution layers:
Make bigger kernel - no really need for it since we can make a deeper net instead, 3-5 kernel is enough

Padding is pretty bad as well, since we invent poor data

Output channels are the most important hyperparam (try 8, 16, 32...)

Stride is kept at 1 usually, since we delegate this function of size dicreasing to pooling layers

For pooling  - we can use max pooling, can be seen as applying e.g. 2x2 filter with stride 2, but notice that we don't necessarily do pooling systematically since it will limit the maximal depth of the network

Receptive field - the deeper the activation map is, the more input pixels contribute to it's values, so small kernel with a lot of layers works good

**VGG** is a classical CNN architecture: Convolution layers (+ReLU), pooling is done each several layers, flatten the result MLP ( i.e. dense / fully connected + ReLU) in the end

Explore architecture in 2 ways:  simplify to see if it performs as good, add complexity to see if it performs better

Interestingly, most of parameters are still in final MLP!

**ResNet** - another architecture of CNN, problem that with more layers gradient vanishes and learning becomes bad (so it's rather underfitting than overfitting!), so we can add "identity layers", that do nothing - **residual layers** / skip connections, so we avoid vanishing gradient problem. It allowed for very deep architectures, but less parameters since final MLP can be simpler!!

Skip connections is a common lifehack used by recurrent NNs and transformers.

ImageNet - reference dataset, 1M images with 1000 classes, images 224x224

But what to do with real life bigger images? Can slide the model with for loop... very expensive but we get a position, not just detection

But sliding detector looks like a convolution - can be indeed expressed as a 1x1 kernel after final MLP of this big CNN and it preserves spatial information #todo what does it do exactly and where? - **completely convolutional NN**

#todo can be seen as bagging (take results on small regions of image)

We can analyse what's hapenning in the model, by e.g. analysing what images activate each layer the most - do GD on image inputs! https://distill.pub/2017/feature-visualization/ 

https://en.wikipedia.org/wiki/DeepDream

amazing site! https://distill.pub

Conseptually, convolution layers do image representation / automatic feature engineering, MLP for classification

The fact that convolution is a kind of restrained weights architecture, it can be seen as a kind of prior / regularization

Fine tuning, can deactivate gradient for some of layers

#todo CLIP and image encoders

CNN were at foundation of foundation models - pretrained models capable of working e.g. at zero shot ... #lookup Unet for medical imaging



























