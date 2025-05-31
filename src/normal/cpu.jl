"""
    logprob_normal(μ::Matrix{<:Real}, x::Matrix{<:Real})

Compute the log probability of observations `x` given the categorical distribution parameters `logits`.

# Arguments
- `μ::Matrix{<:Real}`: A matrix (n_dimension, n_components) containing means of normal distributions
  * n_dimension: the number of dimensions in each observation
  * n_components: the number of components in the mixture
- `x::Matrix{<:Real}`: A matrix of shape (n_dimension, n_observations) containing real observation

# Returns
- `log_probs::Matrix{<:Real}`: A matrix of shape (n_components, n_observations) containing the log probability of each observation for each component.
- `mx::Vector{<:Real}`: A vector of maximum log probabilities for each observation, used for numerical stability in subsequent calculations.

# Implementation Details
This function iterates through all components, observations, and dimensions to compute
the log probability of each observation for each component. It then finds the maximum
log probability for each observation for numerical stability in subsequent calculations.
"""
function logprob(m::GaussianMixture{Matrix{T},Matrix{T}}, x::Matrix{T}) where {T<:Real}
	μ, Σ = m.μ, m.Σ
	n_observations = size(x,2)
	n_components = size(μ, 2)
	n_dimension = size(μ, 1)
	n_dimension == size(x,1) || error("dimension of observation do not match dimension of components")
	log_probs = zeros(T, n_components, n_observations)
	@inbounds for i in axes(μ,2)
		for j in axes(x,2)
			for k in 1:n_dimension
				log_probs[i,j] -= (μ[k,i] - x[k, j])^2/(2*Σ[k,i]) + log(2*π*Σ[k,i])/2
			end
		end
	end
	mx = vec(maximum(log_probs, dims = 1))
	log_probs, mx
end


function ∇logprob(∇logp::AbstractMatrix, m::GM, x::AbstractMatrix{T}) where {T<:Real, GM<:GaussianMixture{Matrix{T},Matrix{T}}}
	μ, Σ = m.μ, m.Σ
	n_observations = size(x,2)
	n_components = size(μ, 2)
	n_dimension = size(μ, 1)
	n_dimension == size(x,1) || error("dimension of observation do not match dimension of components")

	∇μ = zeros(T, size(μ))
	∇Σ = zeros(T, size(Σ))
	∇x = zeros(T, size(x))
	for i in axes(μ,2)
		for j in axes(x,2)
			for k in 1:n_dimension
				∇μ[k,i] -= (μ[k,i] - x[k, j])*∇logp[i,j]/Σ[k,i]
				∇x[k,j] += (μ[k,i] - x[k, j])*∇logp[i,j]/Σ[k,i]
				∇Σ[k,i] += ((μ[k,i] - x[k, j])^2/(2*Σ[k,i]^2) - 1/(2*Σ[k,i])) * ∇logp[i,j]
			end
		end
	end
	return(Tangent{GM}(;μ = ∇μ, Σ = ∇Σ), ∇x)
end


function ∇logprob_fused(∇y, m::GM, x::AbstractMatrix{T}, mx, logprobs, sumexp) where {T<:Real, GM<:GaussianMixture{Matrix{T},Matrix{T}}}
	μ, Σ = m.μ, m.Σ
	∇μ = zeros(T, size(μ))
	∇x = zeros(T, size(x))
	∇Σ = zeros(T, size(Σ))
	@inbounds for i in axes(μ,2) # index of the component
	    for k in axes(x,1)
			for j in axes(x,2)
				o = ∇y * exp(logprobs[i, j] - mx[j]) / sumexp[j]
				∇μ[k,i] -= (μ[k,i] - x[k, j]) * o / Σ[k,i]
				∇x[k,j] += (μ[k,i] - x[k, j]) * o / Σ[k,i]
				∇Σ[k,i] += ((μ[k,i] - x[k, j])^2/(2*Σ[k,i]^2) - 1/(2*Σ[k,i])) * o
			end
		end
	end
	return(Tangent{GM}(;μ = ∇μ, Σ = ∇Σ), ∇x)
end
