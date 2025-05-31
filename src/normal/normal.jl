
"""
	struct GaussianMixture{M<:AbstractMatrix,S<:AbstractMatrix}
		μ::M
		Σ::S
	end

	GaussianMixture with uniform distribution of components consisting of normally distributed components with diagonal variance.

"""
struct GaussianMixture{M<:AbstractMatrix,S<:AbstractMatrix}
	μ::M
	Σ::S
	function GaussianMixture(μ::M, Σ::S) where {M<:AbstractMatrix,S<:AbstractMatrix}
		size(μ) == size(Σ) || error("dimension of mean and variance should be the same")
		new{M,S}(μ, Σ)
	end
end

Base.show(io::IO, m::GaussianMixture) = print(io, "GaussianMixture (dim = $(size(m.μ,1)) comps = $(size(m.Σ, 2))")


include("cpu.jl")
include("cuda.jl")

function ChainRulesCore.rrule(::typeof(sumlogsumexp_logprob), m::GaussianMixture, x)
    # The gradient is `softmax`, but both compute `tmp` so it's worth saving.
    logp, max_ = logprob(m, x)
    max_ = max_'

    # this should be fused to the second kernel
    tmp = exp.(logp .- max_)
    sum_tmp = sum(tmp; dims = 1)
    lkl = sum(max_ .+ log.(sum_tmp))

    function sumlogsumexp_pullback(dy)
    	∇logp = unthunk(dy) .* tmp ./ sum_tmp
    	∇m, ∇x = ∇logprob(∇logp, m, x)
    	(NoTangent(), ∇m, ∇x)
    end
    return lkl, sumlogsumexp_pullback
end