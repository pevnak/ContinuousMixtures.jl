# ContinuousMixtures.jl

This is a small library implementing *"Continuous mixtures of tractable probabilistic models.", Correira et al., Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 6. 2023.*.

The repository addressed curiousity of author, who wanted to know if writing a small custom slow CUDA kernel can help to reduce memory issues of the original implementation. Therefore while the repository works with CPU, it was designed to be used with CUDA. Since the author has found the package useful for fitting simple mixtures, which are ocasionally useful, a simple training loop has been added. The main limitation of the model is it works only with Categorical distribution, which needs to be represented by `Integers` from `1,...,n`. For Normal distribution, you either need to add a kernel, or you can try to file an issue and write the main author.

For those not familiar with Continous Mixture models (CMM), reading the original publication is strongly recommended. For others, here is a short recap. CMMs maximize likelihood of a mixture model `\sum_{i}p(x|zᵢ)` on the training set, where `zᵢ` are samples of normal distribution `N(0,I).` The key innovation is that latent variables `zᵢ` are sampled fresh at each optimization step, while the parameters being optimized belong to a generator network `p(x|zᵢ)` - typically a feed-forward neural network.
This means mixture component parameters are optimized implicitly through the generator, allowing the model to represent an effectively infinite mixture with shared structure. After the generator `p(x|zᵢ)` is fitted, the latents can be generated and finetuned by optimizing `\sum_{i}p(x|zᵢ)` with respect to `zᵢ.`

Let's demonstrate the library on a small example of fitting flower data. We first demonstrate the library on categorical distribution, therefore we will quantize flower to 40 levels
```julia
using Test
using ContinuousMixtures
using ContinuousMixtures.CUDA
using ContinuousMixtures.Flux
using Random
using Serialization
using CairoMakie

function flower(n;npetals = 8)
    theta = 1:npetals
    n = div(n, length(theta))
    data = mapreduce(hcat, theta * (2pi/npetals)) do t
        ct = cos(t)
        st = sin(t)

        x0 = tanh.(randn(n) .- 1) .+ 4.0 .+ 0.05.* randn(n)
        y0 = randn(n) .* 0.3

        x = x0 * ct .- y0 * st
        y = x0 * st .+ y0 * ct
        Float32.([x y]')
    end
    clamp.(data ./ 5, -1f0, 1f0)
end

categorize(x::AbstractArray) = 1 .+ round.(Int, (1 .+ x) / 0.05)
data = flower(10_000)
data_i = categorize(data)

n_dimension = size(data, 1)
n_components = 1024
encoder_dim = 16
n_categories = maximum(data_i)

model = Chain(Dense(encoder_dim, 64, leakyrelu),
    BatchNorm(64),
    Dense(64, 128, leakyrelu),
    BatchNorm(128),
    Dense(128, n_categories * n_dimension, leakyrelu),
    BatchNorm(n_categories * n_dimension),
    Dense(n_categories * n_dimension, n_categories * n_dimension),
    Base.Fix2(reshape, (n_categories, n_dimension, :)),
    CategoricalMixture,
)

model, zᵢ, stats = train_mixture_model(model, data_i; n_components, encoder_dim, max_epochs=2000, finetune_epochs = 0)    
gmm = model(zᵢ)
f(gmm, x, y) = ContinuousMixtures.sumlogsumexp_logprob(gmm, categorize(reshape(([x, y]),:,1)))
heatmap(minimum(data[1,:]):0.1:maximum(data[1,:]), minimum(data[2,:]):0.1:maximum(data[2,:]), (x,y) -> exp(f(gmm, x, y)))
```

To fit the data using Gaussian mixture, we switch from `CategoricalMixture` to `GaussianMixture.`

```julia

data = flower(10_000)

n_dimension = size(data, 1)
n_components = 128
encoder_dim = 16

model = Chain(Dense(encoder_dim, 64, leakyrelu),
    BatchNorm(64),
    Dense(64, 128, leakyrelu),
    BatchNorm(128),
    Dense(128, 2 * n_dimension, leakyrelu),
    BatchNorm(2 * n_dimension),
    Dense(2 * n_dimension, 2 * n_dimension),
    Base.Fix2(reshape, (2, n_dimension, :)),
    GaussianMixture,
)

model, zᵢ, stats = train_mixture_model(model, data; n_components, encoder_dim, max_epochs=2000, finetune_epochs = 100)    
gmm = model(zᵢ)
f(gmm, i, j) = ContinuousMixtures.sumlogsumexp_logprob(gmm, reshape(Float32.([i,j]),:,1))
heatmap(minimum(data[1,:]):0.01:maximum(data[1,:]), minimum(data[2,:]):0.01:maximum(data[2,:]), (x,y) -> exp(f(gmm, x, y)))
```



The library exports two main functions:
* `train_mixture_model` which is used in the above example to optimize the complete mixture.
* `finetune_mixture_latents` which optimizes the latent variables which defines the mixture.
* `create_model` creates the model with correct dimentions (the above example creates the model manually).

Note that according to the original publication, there are not weights on component, they are considered uniform.
