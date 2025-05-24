module SmoothMixtures
using CUDA
using ChainRulesCore
using Flux
using MLUtils

include("categorical/categorical.jl")

include("train.jl")
export train_model, create_model, finetune_centers

end # module SmoothMixtures
