module ContinuousMixtures
using CUDA
using ChainRulesCore
using Flux
using MLUtils

include("categorical/categorical.jl")
export CategoricalMixture
include("normal/normal.jl")
export GaussianMixture

include("train.jl")
export train_mixture_model, create_model, finetune_mixture_latents, train_categorical_mixture_model

end
