struct CategoricalMixture{M<:AbstractArray{<:Real,3}}
    logits::M
    function CategoricalMixture(logits::M) where {M<:AbstractArray{<:Real,3}}
        new{M}(logsoftmax(logits, dims = 1))
    end
end

Base.show(io::IO, m::CategoricalMixture) = print(io, "CategoricalMixture (dim = $(size(m.logits,1)) comps = $(size(m.logits, 2))")

include("cpu.jl")
include("cuda.jl")

"""
    sumlogsumexp_logprob(logits, x)

Compute the log likelihood of observations `x` given the categorical distribution parameters `logits`,
where logits defines a mixture of categorical distributions with uniform weight.

# Arguments
- `logits::AbstractArray{<:Real, 3}`: A 3D array of shape (n_categories, n_dimension, n_components) containing
  the log probabilities of each category for each dimension and component.
  * n_categories: the number of possible values for each dimension
  * n_dimension: the number of dimensions in each observation
  * n_components: the number of components in the mixture
- `x::AbstractMatrix{<:Integer}`: A matrix of shape (n_dimension, n_observations) containing the 
  categorical observations. Each entry must be an integer in the range 1:n_categories.

# Returns
- `lkl::Real`: The total log likelihood of all observations under the mixture model.

# Implementation Details
This function first computes the log probabilities for each component and observation,
then applies the log-sum-exp trick for numerical stability when summing across components.
The non-fused version separates these operations, which may be less efficient but more readable.

The computation occurs in two main steps:
1. Call `logprob` to compute log probabilities for each component and observation
2. Sum these probabilities using the log-sum-exp trick to prevent numerical overflow/underflow:
   - Subtract the maximum log probability from each value before exponentiation
   - Sum the resulting exponentials
   - Take the logarithm of this sum
   - Add back the maximum value

This approach allows working with very small or large probabilities without hitting
floating-point limitations. The function is used primarily during model training to
evaluate how well the model fits the data.

# See Also
- `sumlogsumexp_logprob_fused`: A more efficient implementation that fuses operations
- `logprob`: Computes the component-wise log probabilities
"""
function sumlogsumexp_logprob(m, x)
    logp, max_ = logprob(m, x)
    sum(max_' .+ log.(sum(exp.(logp .- max_'); dims = 1)))
end

function ChainRulesCore.rrule(::typeof(sumlogsumexp_logprob), m, x)
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

"""
    sumlogsumexp_logprob_fused(logits, x)

Efficiently compute the log probability of observations `x` given the categorical distribution parameters `logits`.
This is a fused operation that combines the computation of log probabilities and their aggregation to minimize
memory allocations and improve performance.

# Arguments
- `logits::AbstractArray{<:Real, 3}`: A 3D array of shape (n_categories, n_dimension, n_components) containing
  the log probabilities of each category for each dimension and component.
  * n_categories: the number of possible values for each dimension
  * n_dimension: the number of dimensions in each observation
  * n_components: the number of components in the mixture
- `x::AbstractMatrix{<:Integer}`: A matrix of shape (n_dimension, n_observations) containing the 
  categorical observations. Each entry must be an integer in the range 1:n_categories.

# Returns
- `lkl::Real`: The total log likelihood of all observations under the mixture model.

# Implementation Details
This function combines the log probability computation and the log-sum-exp operation into
a single fused workflow to improve performance. It works in three steps:

1. Call `logprob` to compute log probabilities and max values
2. Call `sumlogsumexp` to efficiently aggregate these probabilities
3. Return the final log likelihood

The fused implementation is particularly important for:
- Reducing memory allocations during training
- Enabling efficient gradient computation via ChainRulesCore.rrule
- Optimizing performance on both CPU and GPU hardware

During training, this function is typically used in the loss calculation, where
minimizing the negative log likelihood is the optimization objective.

# Performance Considerations
- On GPU, this implementation uses specialized CUDA kernels for maximum efficiency
- Memory usage is optimized by reusing intermediate results across operations
- The gradient computation is similarly fused for improved backpropagation performance

# Notes
- This function has optimized implementations for both CPU and GPU (CUDA)
- Used in training mixture models to compute the objective function
- For gradient-based optimization, this function is paired with its gradient
  computation via ChainRulesCore.rrule
"""
function sumlogsumexp_logprob_fused(m, x)
	log_probs, max_ = logprob(m, x)
	lkl, sumexp = sumlogsumexp(log_probs, max_)
	return(lkl)
end

function ChainRulesCore.rrule(::typeof(sumlogsumexp_logprob_fused), m, x)
    # The gradient is `softmax`, but both compute `tmp` so it's worth saving.
    log_probs, max_ = logprob(m, x)
    lkl, sumexp = sumlogsumexp(log_probs, max_)
    function sumlogsumexp_pullback(dy)
    	∇m, ∇x = ∇logprob_fused(dy, m, x, max_, log_probs, sumexp)
    	(NoTangent(), ∇m, ∇x)
    end
    return lkl, sumlogsumexp_pullback
end
