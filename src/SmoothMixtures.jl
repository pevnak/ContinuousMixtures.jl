module SmoothMixtures
using CUDA
using ChainRulesCore
using Flux

include("categorical/categorical.jl")
greet() = print("Hello World!")

end # module SmoothMixtures
