function logprob(logits::Array{<:Real, 3}, x::Matrix{<:Integer})
	n_components = size(logits, 3)
	n_observations = size(x,2)
	n_dimension = size(logits, 2)
	n_dimension == size(x,1) || error("dimension does not match")

	log_probs = zeros(Float32, n_components, n_observations)
	@inbounds for i in 1:n_components
		for j in 1:n_observations
			for k in 1:n_dimension
				log_probs[i,j] += logits[x[k, j], k, i]
			end
		end
	end
	mx = vec(maximum(log_probs, dims = 1))
	log_probs, mx
end

function ∇logprob(∇logprobs::Matrix{<:Real}, logits::Array{<:Real, 3}, x::Matrix{<:Integer})
	n_components = size(logits, 3)
	n_observations = size(x,2)
	n_dimension = size(logits, 2)
	n_dimension == size(x,1) || error("dimension does not match")

	∇logits = zeros(Float32, size(logits,1), n_dimension, n_components)
	@inbounds for i in 1:n_components
		for j in 1:n_observations
			for k in 1:n_dimension
				∇logits[x[k, j], k, i] += ∇logprobs[i,j]
			end
		end
	end
	∇logits
end

function sumlogsumexp(logprobs::AbstractMatrix, mx::AbstractVector)
    o = zero(eltype(logprobs))
    sumexp = similar(mx)
    @inbounds for j in axes(logprobs,2)
        oⱼ = zero(eltype(logprobs))
        for i in axes(logprobs,1)
            oⱼ += exp(logprobs[i,j] - mx[j])
        end
        sumexp[j] = oⱼ
        o += log(oⱼ) + mx[j]
    end
    o, sumexp
end

"""
Compute the gradient of the log probability of a categorical distribution.

Arguments:
- `∇y`: gradient of the log probability of the categorical distribution.
- `logits`: logits of the categorical distribution.
- `x`: indices of the categorical distribution.
- `mx`: maximum of the logits.
- `logprobs`: log probabilities of the categorical distribution.
- `sumexp`: sum of the exponentials of the logits.

Returns:
- `∇logits`: gradient of the logits of the categorical distribution.
"""
function ∇logprob_fused(∇y, logits::Array{T, 3}, x::Array{<:Integer,2}, mx, logprobs, sumexp) where {T}
	∇logits = similar(logits)
	∇logits .= 0
	@inbounds for i in axes(logits,3) # index of the component
	    for kᵢ in axes(logits,2)
			for j in axes(x,2)
				o = ∇y * exp(logprobs[i, j] - mx[j]) / sumexp[j]
				xⱼ = x[kᵢ, j]
				∇logits[xⱼ, kᵢ, i] += o
			end
		end
	end
	return(∇logits)
end
