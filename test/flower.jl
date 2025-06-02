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
