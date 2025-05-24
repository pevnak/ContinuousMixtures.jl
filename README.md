# ContinuousMixtures.jl

This is a small library implementing *"Continuous mixtures of tractable probabilistic models.", Correira et al., Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 6. 2023.*.

The repository addressed curiousity of author, who wanted to know if writing a small custom slow CUDA kernel can help to reduce memory issues of the original implementation. Therefore while the repository works with CPU, it was designed to be used with CUDA. Since the author has found the package useful for fitting simple mixtures, which are ocasionally useful, a simple training loop has been added. The main limitation of the model is it works only with Categorical distribution, which needs to be represented by `Integers` from `1,...,n`. For Normal distribution, you either need to add a kernel, or you can try to file an issue and write the main author.

For those not familiar with Continous Mixture models (CMM), reading the original publication is strongly recommended. For others, here is a short recap. CMMs maximize likelihood of a mixture model `\sum_{i}p(x|zᵢ)` on the training set, where `zᵢ` are samples of normal distribution `N(0,I).` The key innovation is that latent variables `zᵢ` are sampled fresh at each optimization step, while the parameters being optimized belong to a generator network `p(x|zᵢ)` - typically a feed-forward neural network.
This means mixture component parameters are optimized implicitly through the generator, allowing the model to represent an effectively infinite mixture with shared structure. After the generator `p(x|zᵢ)` is fitted, the latents can be generated and finetuned by optimizing `\sum_{i}p(x|zᵢ)` with respect to `zᵢ.`

Let's demonstrate the library on a small example on useless data
```julia
using ContinuousMixtures
using ContinuousMixtures.CUDA
using ContinuousMixtures.Flux
using Random
using Serialization

data = rand(1:5, 17, 1023)

n_categories = 5
n_dimension = size(data, 1)
n_components = 32
encoder_dim = 16

model = Chain(Dense(encoder_dim,64,leakyrelu),
	Dense(64, 128, leakyrelu),
	Dense(128, n_categories * n_dimension, leakyrelu),
	Base.Fix2(reshape, (n_categories, n_dimension, :)),
	x -> logsoftmax(x, dims = 1),
	)

model, zᵢ, stats = train_mixture_model(model, data; n_categories, n_components, encoder_dim)
```
To obtain the logits of compoments, we need to project the latent, i.e. `model(zᵢ)`. The returned tensor has the format [logits, dimension, component],
where `logits` are logits of categorical distribution, `dimention` is the dimention of the component, and `component` is the component.  `train_mixture_model` by default finetune the latents. See the help for options.


The library exports two main functions:
* `train_mixture_model` which is used in the above example to optimize the complete mixture.
* `finetune_mixture_latents` which optimizes the latent variables which defines the mixture.
* `create_model` creates the model with correct dimentions (the above example creates the model manually).

Note that according to the original publication, there are not weights on component, they are considered uniform.
