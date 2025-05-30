
"""
    logprob_kernel!(log_probs, mx, logits, x, n_components, n_observations, n_dimension)

CUDA kernel for computing the log probability of observations for each component.

# Arguments
- `log_probs`: Output buffer for log probabilities (n_components, n_observations)
- `mx`: Output buffer for maximum log probabilities per observation
- `logits`: Categorical distribution parameters (n_categories, n_dimension, n_components)
- `x`: Categorical observations (n_dimension, n_observations)
- `n_components`: Number of mixture components
- `n_observations`: Number of observations in the batch
- `n_dimension`: Number of dimensions in each observation

# Implementation Details
This CUDA kernel uses a parallel execution model where:
- Each block corresponds to one component (blockIdx.x)
- Each thread corresponds to one observation (threadIdx.x)

The kernel:
1. Initializes the maximum log probability for each observation to a minimum value
2. Computes the log probability for each component-observation pair
3. Atomically updates the maximum log probability for each observation
4. Stores the computed log probabilities in the output buffer

The kernel uses thread synchronization to ensure proper initialization of the maximum
values before parallel computation begins.
"""
function logprob_kernel!(log_probs, mx, logits, x, n_components, n_observations, n_dimension)
    i = blockIdx().x # index of the component
    j = threadIdx().x # index of the observation (minibatch)
    if i == 1
        mx[j] = typemin(Float32)
    end
    CUDA.sync_threads()

    # check bounds such that we do not have to care about number of allocated threads
    i > n_components && return (nothing)
    j > n_observations && return (nothing)

    o = 0.0f0
    @inbounds for k in 1:n_dimension
        o += logits[x[k, j], k, i]
    end
    log_probs[i, j] = o
    CUDA.@atomic mx[j] = max(mx[j], o)
    return (nothing)
end

function logprob(logits::CuArray{<:Real,3}, x::CuArray{<:Integer,2})
    n_components = size(logits, 3)
    n_observations = size(x, 2)
    n_dimension = size(logits, 2)
    n_dimension == size(x, 1) || error("dimension does not match")

    log_probs = similar(logits, size(logits, 3), size(x, 2))
    max_ = similar(logits, n_observations)

    @cuda threads = n_observations blocks = n_components logprob_kernel!(log_probs, max_, logits, x, n_components, n_observations, n_dimension)

    return (log_probs, max_)
end


"""
    sumlogsumexp_kernel!(cb, scratch_space, log_probs, sumexp, mx, n_components, n_observations, Δ, n_chunks)

CUDA kernel for computing the log-sum-exp operation in a numerically stable way.

# Arguments
- `cb`: Output buffer for the final log likelihood value
- `scratch_space`: Temporary buffer for parallel reduction
- `log_probs`: Log probabilities matrix (n_components, n_observations)
- `sumexp`: Output buffer for summed exponentials
- `mx`: Maximum log probabilities per observation
- `n_components`: Number of mixture components
- `n_observations`: Number of observations
- `Δ`: Chunk size for parallel reduction
- `n_chunks`: Number of chunks for parallel reduction

# Implementation Details
This kernel implements a parallel reduction algorithm to compute the log-sum-exp:
1. Each thread computes exponentials for a chunk of components
2. The results are stored in shared memory (scratch_space)
3. A parallel reduction sums these values efficiently
4. The first thread in each block computes the final log-sum-exp value
5. The result is atomically added to a global counter

The kernel uses a two-phase approach:
- First phase: Compute exponentials and store in scratch space
- Second phase: Perform parallel reduction to sum the exponentials

This approach is highly efficient for large numbers of components, avoiding
sequential bottlenecks that would occur in a naive implementation.
"""
function sumlogsumexp_kernel!(cb, scratch_space, log_probs, sumexp, mx, n_components, n_observations, Δ, n_chunks)
    i = threadIdx().x  # component
    j = blockIdx().x   # observation

    # Let's first compute the exponents and reduce chunks
    v = 0.0f0
    range_k = ((i-1)*Δ+1):(i*Δ)
    @inbounds for kᵢ in range_k
        kᵢ > n_components && continue
        v += exp(log_probs[kᵢ, j] - mx[j])
    end
    scratch_space[i, j] = v

    # then perform the reduction
    d = 1
    while d < n_chunks
        sync_threads()
        row = 2 * d * (i - 1) + 1
        @inbounds if row + d <= n_chunks
            scratch_space[row, j] = scratch_space[row, j] + scratch_space[row+d, j]
        end
        d *= 2
    end

    sync_threads() # this might not be needed
    if i == 1
        sumexp[j] = scratch_space[i, j]# here is the reduced version
        CUDA.atomic_add!(pointer(cb), log(sumexp[j]) + mx[j])
    end

    return (nothing)
end

"""
	sumlogsumexp(log_probs::CuMatrix{<:Real}, mx::CuVector)

	Compute the log-sum-exp of the log-likelihood of each observation for each component.

	parameters:
	log_probs: (n_components, n_observations), log-likelihood of each observation for each component
				`log_probs` can be computed by `logprob` function

	mx: (n_observations), maximum of the log-likelihood across components.
				This is used to make the computation of log-sum-exp numerically stable.
				`mx` is returned as a second argument from `logprob`.

	return: (likelihood, sumexp)
		likelihood is the log-sum-exp of all data
		sumexp is the sum of the likelihood across components which is used in the gradient
"""
function sumlogsumexp(log_probs::CuMatrix{<:Real}, mx::CuVector) 
    max_threads = 64
    n_components = size(log_probs, 1)
    n_observations = size(log_probs, 2)

    Δ = cld(n_components, max_threads)
    n_chunks = cld(n_components, Δ)

    scratch_space = similar(log_probs, n_chunks, n_observations)
    sumexp = similar(mx)
    cb = CUDA.zeros(1)
    @cuda threads = n_chunks blocks = n_observations sumlogsumexp_kernel!(cb, scratch_space, log_probs, sumexp, mx, n_components, n_observations, Δ, n_chunks)
    Vector(cb)[1], sumexp
end


"""

	The kernel has different order of iterations from the forward one `logprob_kernel` to
	avoid race-condition to the memory.

	Each backward kernel computes one component, dimension, and the innter loop is over observations.
	The forward kernel computes one component, and observation, and the innter loop is over dimensions.
	Moreover, since the `n_dims` can be larger than number of threads, the thread has to operate over small range
	denoted as Δ.
"""
function ∇logprob_kernel!(∇logits, ∇logprobs, x, n_components, n_observations, n_dimension, n_categories, Δ)
    i = blockIdx().x # index of the component
    k = threadIdx().x # index of the dimension of the logit

    # check bounds such that we do not have to care about number of allocated threads
    i > n_components && return (nothing)

    range_k = ((k-1)*Δ+1):(k*Δ)
    # let's zero everything out
    @inbounds for kᵢ in range_k
        kᵢ > n_dimension && continue
        for z in 1:n_categories
            ∇logits[z, kᵢ, i] = 0
        end
    end
    # now we increment the gradient
    @inbounds for kᵢ in range_k
        kᵢ > n_dimension && continue
        for j in 1:n_observations
            o = ∇logprobs[i, j]
            ∇logits[x[kᵢ, j], kᵢ, i] += o
        end
    end
    return (nothing)
end


"""
	∇logprob(∇logprobs::CuMatrix{T}, logits::CuArray{T, 3}, x::CuArray{<:Integer,2}) where {T}

	the gradient of the `logprob` with respect to the logits
"""
function ∇logprob(∇logprobs::CuMatrix{T}, logits::CuArray{T,3}, x::CuArray{<:Integer,2}) where {T}
    n_components = size(logits, 3)
    n_observations = size(x, 2)
    n_dimension = size(logits, 2)
    n_categories = size(logits, 1)
    size(∇logprobs, 2) == n_observations || error("dimension does not match")
    n_dimension == size(x, 1) || error("n_dimensions does not match")
    Δ = cld(n_dimension, 1024)

    ∇logits = similar(logits, n_categories, n_dimension, n_components)

    @cuda threads = cld(n_dimension, Δ) blocks = n_components ∇logprob_kernel!(∇logits, ∇logprobs, x, n_components, n_observations, n_dimension, n_categories, Δ)

    return (∇logits, NoTangent())
end

"""

	The kernel has different order of iterations from the forward one `logprob_kernel` to
	avoid race-condition to the memory.

	Each backward kernel computes one component, dimension, and the innter loop is over observations.
	The forward kernel computes one component, and observation, and the innter loop is over dimensions.
	Moreover, since the `n_dims` can be larger than number of threads, the thread has to operate over small range
	denoted as Δ.
"""
function ∇logprob_fused_kernel!(∇logits, ∇y, x, log_probs, mx, sumexp, n_components, n_observations, n_dimension, ::Val{n_categories}, ::Val{max_threads}) where {n_categories,max_threads}
    i = blockIdx().x # index of the component
    k = threadIdx().x # index of the dimension of the logit
    Δ = cld(n_dimension, max_threads)
    local_storage = CuStaticSharedArray(Float32, max_threads * n_categories)
    offset = (k - 1) * n_categories

    # check bounds such that we do not have to care about number of allocated threads
    i > n_components && return (nothing)

    range_k = ((k-1)*Δ+1):(k*Δ)
    for kᵢ in range_k
        kᵢ > n_dimension && continue
        for o in 1:n_categories
            local_storage[offset+o] = 0
        end
        @inbounds for j in 1:n_observations
            o = ∇y * exp(log_probs[i, j] - mx[j]) / sumexp[j]
            xⱼ = x[kᵢ, j]
            local_storage[offset+xⱼ] += o
        end

        for o in 1:n_categories
            ∇logits[o, kᵢ, i] = local_storage[offset+o]
        end
    end
    return (nothing)
end

function ∇logprob_fused(∇y, logits::CuArray{T,3}, x::CuArray{<:Integer,2}, mx, log_probs, sumexp) where {T}
    max_threads = 64
    n_components = size(logits, 3)
    n_observations = size(x, 2)
    n_dimension = size(logits, 2)
    n_categories = size(logits, 1)
    n_dimension == size(x, 1) || error("n_dimensions does not match")
    Δ = cld(n_dimension, max_threads)
    ∇logits = similar(logits, n_categories, n_dimension, n_components)

    @cuda threads = cld(n_dimension, Δ) blocks = n_components ∇logprob_fused_kernel!(∇logits, ∇y, x, log_probs, mx, sumexp, n_components, n_observations, n_dimension, Val(n_categories), Val(max_threads))

    return (∇logits, NoTangent())
end
