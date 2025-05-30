module ContinuousMixtures
using CUDA
using ChainRulesCore
using Flux
using MLUtils

include("categorical/categorical.jl")
include("normal/normal.jl")

include("train.jl")
export train_mixture_model, create_model, finetune_mixture_latents

end
