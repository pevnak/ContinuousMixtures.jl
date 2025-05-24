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

