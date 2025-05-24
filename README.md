# SmoothMixture.jl

This is a small library implementing *"Continuous mixtures of tractable probabilistic models.", Correira et al., Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 6. 2023.*. The repository was a created as the author was curious, if writing a small custom CUDA kernel can help to reduce memory issues of the original implementation. Therefore the repository is mainly designed to train with CUDA. It is also limited to model only Categorical distribution, which needs to be represented by `Integers` from `1,...,n`. For Normal distribution, you either need to add a kernel, or you can try to file an issue and write the main author.

What is the idea behind continuous mixture? The idea is to maximize likelihood of a mixture model `\sum_{i}p(x|zᵢ)` on the training, but the trick is that `zᵢ` are sampled in each minibatch and we are optimizing parameters of the generator `p(x|zᵢ),` which is modelled by a feed-forward neural network. For details, see the original publication. It is interesting model.

Let's demonstrate the library on a small example, where we generate random data.

```julia
using SmoothMixtures
using SmoothMixtures.CUDA
using SmoothMixtures.Flux
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

model, centers, stats = train_model(model, data; n_categories, n_components, encoder_dim)
```
