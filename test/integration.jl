using Test
using ContinuousMixtures
using ContinuousMixtures.CUDA
using ContinuousMixtures.Flux
using Random
using Serialization

@testset "naive integration test with CategoricalMixture" begin
    data = rand(1:5, 17, 1023)

    n_categories = 5
    n_dimension = size(data, 1)
    n_components = 32
    encoder_dim = 16

    model = Chain(Dense(encoder_dim, 64, leakyrelu),
        Dense(64, 128, leakyrelu),
        Dense(128, n_categories * n_dimension, leakyrelu),
        Base.Fix2(reshape, (n_categories, n_dimension, :)),
        CategoricalMixture,
    )

    model, zᵢ, stats = train_categorical_mixture_model(model, data; n_categories, n_components, encoder_dim)
    @test model(zᵢ) isa CategoricalMixture
end

@testset "naive integration test with GaussianMixture" begin
    data = rand(Float32, 17, 1023)

    n_dimension = size(data, 1)
    n_components = 32
    encoder_dim = 16

    model = Chain(Dense(encoder_dim, 64, leakyrelu),
        Dense(64, 128, leakyrelu),
        Dense(128, 2 * n_dimension, leakyrelu),
        Base.Fix2(reshape, (2, n_dimension, :)),
        x -> GaussianMixture(x[1,:,:], 1f-6 .* softplus.(x[2,:,:])),
    )

    model, zᵢ, stats = train_mixture_model(model, data; n_components, encoder_dim)
    @test model(zᵢ) isa GaussianMixture
end