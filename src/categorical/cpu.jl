"""
    logprob(logits::Array{<:Real, 3}, x::Matrix{<:Integer})

Compute the log probability of observations `x` given the categorical distribution parameters `logits`.

# Arguments
- `logits::Array{<:Real, 3}`: A 3D array of shape (n_categories, n_dimension, n_components) containing
  the log probabilities of each category for each dimension and component.
  * n_categories: the number of possible values for each dimension
  * n_dimension: the number of dimensions in each observation
  * n_components: the number of components in the mixture
- `x::Matrix{<:Integer}`: A matrix of shape (n_dimension, n_observations) containing the 
  categorical observations. Each entry must be an integer in the range 1:n_categories.

# Returns
- `log_probs::Matrix{<:Real}`: A matrix of shape (n_components, n_observations) containing 
  the log probability of each observation for each component.
- `mx::Vector{<:Real}`: A vector of maximum log probabilities for each observation, used for
  numerical stability in subsequent calculations.

# Implementation Details
This function iterates through all components, observations, and dimensions to compute
the log probability of each observation for each component. It then finds the maximum
log probability for each observation for numerical stability in subsequent calculations.
"""
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

"""
    ∇logprob(∇logprobs::Matrix{<:Real}, logits::Array{<:Real, 3}, x::Matrix{<:Integer})

Compute the gradient of log probabilities with respect to logits.

# Arguments
- `∇logprobs::Matrix{<:Real}`: Gradient of the loss with respect to log probabilities,
  with shape (n_components, n_observations).
- `logits::Array{<:Real, 3}`: A 3D array of shape (n_categories, n_dimension, n_components)
  containing the log probabilities.
- `x::Matrix{<:Integer}`: A matrix of shape (n_dimension, n_observations) containing
  the categorical observations.

# Returns
- `∇logits::Array{<:Real, 3}`: Gradient of the loss with respect to logits,
  with the same shape as `logits`.

# Implementation Details
This function computes the gradient of log probabilities with respect to the model parameters (logits)
during backpropagation. The process works as follows:

1. Initialize a zero gradient tensor with the same shape as the logits
2. For each component, observation, and dimension:
   - Determine which category was observed in the data (`x[k, j]`)
   - Accumulate the gradient for that specific category, dimension, and component
   - The gradient is added to the corresponding position in the gradient tensor

The function efficiently handles the sparse nature of categorical gradients:
- Only the positions corresponding to observed categories receive non-zero gradients
- All other positions remain zero
- This matches the typical pattern in categorical cross-entropy gradients

This gradient computation is used during model training with gradient-based optimizers
like Adam or SGD. The function is designed to work with automatic differentiation
systems and custom adjoints/pullbacks.
"""
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

"""
    sumlogsumexp(logprobs::AbstractMatrix, mx::AbstractVector)

Compute the log-sum-exp of log probabilities in a numerically stable way.

# Arguments
- `logprobs::AbstractMatrix`: Matrix of log probabilities with shape (n_components, n_observations).
- `mx::AbstractVector`: Vector of maximum log probabilities for each observation, used for
  numerical stability.

# Returns
- `o::Real`: The total log likelihood, computed as the sum of log-sum-exp across all observations.
- `sumexp::Vector{<:Real}`: Vector of summed exponentials for each observation, used in gradient calculations.
"""
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
    ∇logprob_fused(∇y, logits::Array{T, 3}, x::Array{<:Integer,2}, mx, logprobs, sumexp) where {T}

Compute the gradient of the log probability of a categorical distribution using a fused operation
for improved efficiency.

# Arguments
- `∇y`: Scalar gradient of the loss with respect to the output log probability.
- `logits`: 3D array of shape (n_categories, n_dimension, n_components) containing the 
  log probabilities of the categorical distribution.
- `x`: Matrix of shape (n_dimension, n_observations) containing the categorical observations.
- `mx`: Vector of maximum log probabilities for each observation, used for numerical stability.
- `logprobs`: Matrix of shape (n_components, n_observations) containing log probabilities.
- `sumexp`: Vector of summed exponentials for each observation from the forward pass.

# Returns
- `∇logits`: Gradient of the loss with respect to logits, with the same shape as `logits`.

# Implementation Details
This function is an optimized version of gradient computation that fuses multiple operations
to improve performance. The implementation:

1. Pre-allocates the gradient tensor with the same shape as logits
2. Performs a 3-level nested loop over components, dimensions, and observations
3. For each combination:
   - Computes the softmax-like term using pre-computed values from the forward pass
   - Multiplies by the upstream gradient `∇y`
   - Adds the result to the corresponding position in the gradient tensor

Key performance optimizations include:
- Reusing intermediate results (logprobs, mx, sumexp) from the forward pass
- Fusing the softmax computation with the gradient accumulation
- Using in-place operations to avoid temporary allocations
- Applying bounds checking optimization with `@inbounds`

This fused implementation is critical for efficient training of large mixture models,
especially when used with automatic differentiation frameworks.
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
