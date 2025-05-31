CuGMM{T} = GaussianMixture{<:CuArray{T,2}}

function logprob_normal_kernel!(logp, mx, μ, Σ, x, n_components, n_observations, n_dimension)
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
        o -= (μ[k,i] - x[k, j])^2/(2*Σ[k,i]) + log(2*π*Σ[k,i])/2
    end
    logp[i, j] = o
    CUDA.@atomic mx[j] = max(mx[j], o)
    return (nothing)
end

function logprob(m::CuGMM{T}, x::CuMatrix{T}) where {T}
    μ, Σ = m.μ, m.Σ
    n_components = size(μ, 2)
    n_observations = size(x, 2)
    n_dimension = size(μ, 1)
    n_dimension == size(x, 1) || error("dimension does not match")

    logp = similar(μ, n_components, n_observations)
    max_ = similar(μ, n_observations)

    @cuda threads = n_observations blocks = n_components logprob_normal_kernel!(logp, max_, μ, Σ, x, n_components, n_observations, n_dimension)

    return (logp, max_)
end


"""

	The kernel has different order of iterations from the forward one `logprob_kernel` to
	avoid race-condition to the memory.

	Each backward kernel computes one component, dimension, and the innter loop is over observations.
	The forward kernel computes one component, and observation, and the innter loop is over dimensions.
	Moreover, since the `n_dims` can be larger than number of threads, the thread has to operate over small range
	denoted as Δ.
"""
function ∇logprob_normal_kernel!(∇μ, ∇Σ, ∇logp, μ, Σ, x, n_components, n_observations, n_dimension, Δ)
    i = blockIdx().x # index of the component
    k = threadIdx().x # index of the dimension of the logit

    # check bounds such that we do not have to care about number of allocated threads
    i > n_components && return (nothing)

    range_k = ((k-1)*Δ+1):(k*Δ)
    # now we increment the gradient
    @inbounds for kᵢ in range_k
        kᵢ > n_dimension && continue
        ∇μᵢ = 0f0
        ∇Σᵢ = 0f0
        for j in 1:n_observations
            ∇μᵢ -= (μ[kᵢ,i] - x[kᵢ, j])*∇logp[i,j] / Σ[kᵢ,i]
            ∇Σᵢ += ((μ[kᵢ,i] - x[kᵢ, j])^2/(2*Σ[kᵢ,i]^2) - 1/(2*Σ[kᵢ,i])) * ∇logp[i,j]
            # ∇x[k,j] += (μ[k,i] - x[k, j])*∇logp[i,j]/Σ[k,i]
        end
        ∇μ[kᵢ,i] = ∇μᵢ
        ∇Σ[kᵢ,i] = ∇Σᵢ
    end
    return (nothing)
end


"""
	∇logprob(∇logprobs::CuMatrix{T}, logits::CuArray{T, 3}, x::CuArray{<:Integer,2}) where {T}

	the gradient of the `logprob` with respect to the logits
"""
function ∇logprob(∇logp::CuMatrix{T}, m::CuGMM{T}, x::CuMatrix{T}) where {T<:Real}
    μ, Σ = m.μ, m.Σ
    n_components = size(μ, 2)
    n_observations = size(x, 2)
    n_dimension = size(μ, 1)
    n_dimension == size(x, 1) || error("dimension does not match")
    size(∇logprobs, 2) == n_observations || error("dimension does not match")
    Δ = cld(n_dimension, 1024)

    ∇μ, ∇Σ = similar(μ), similar(Σ)
    @cuda threads = cld(n_dimension, Δ) blocks = n_components ∇logprob_normal_kernel!(∇μ, ∇Σ, ∇logp, μ, Σ, x, n_components, n_observations, n_dimension, Δ)

    return (Tangent{CuGMM{T}}(μ = ∇μ, Σ = ∇Σ), NoTangent())
end

"""

	The kernel has different order of iterations from the forward one `logprob_kernel` to
	avoid race-condition to the memory.

	Each backward kernel computes one component, dimension, and the innter loop is over observations.
	The forward kernel computes one component, and observation, and the innter loop is over dimensions.
	Moreover, since the `n_dims` can be larger than number of threads, the thread has to operate over small range
	denoted as Δ.
"""
function ∇logprob_fused_kernel!(∇μ, ∇Σ, ∇y, μ, Σ, x, logp, mx, sumexp, n_components, n_observations, n_dimension, ::Val{max_threads}) where {max_threads}
    i = blockIdx().x # index of the component
    k = threadIdx().x # index of the dimension of the logit
    Δ = cld(n_dimension, max_threads)

    # check bounds such that we do not have to care about number of allocated threads
    i > n_components && return (nothing)

    range_k = ((k-1)*Δ+1):(k*Δ)
    for kᵢ in range_k
        kᵢ > n_dimension && continue
        ∇μᵢ = 0f0
        ∇Σᵢ = 0f0
        for j in 1:n_observations
            o = ∇y * exp(logp[i, j] - mx[j]) / sumexp[j]
            ∇μᵢ -= (μ[kᵢ,i] - x[kᵢ, j])*o / Σ[kᵢ,i]
            ∇Σᵢ += ((μ[kᵢ,i] - x[kᵢ, j])^2/(2*Σ[kᵢ,i]^2) - 1/(2*Σ[kᵢ,i])) * o
        end
        ∇μ[kᵢ,i] = ∇μᵢ
        ∇Σ[kᵢ,i] = ∇Σᵢ
    end
    return (nothing)
end

function ∇logprob_fused(∇y, m::CuGMM{T}, x::CuMatrix{T}, mx, logp, sumexp)  where {T<:Real}
    μ, Σ = m.μ, m.Σ
    max_threads = 64
    n_components = size(μ, 2)
    n_observations = size(x, 2)
    n_dimension = size(μ, 1)
    n_dimension == size(x, 1) || error("dimension does not match")
    size(∇logprobs, 2) == n_observations || error("dimension does not match")
    Δ = cld(n_dimension, max_threads)
    ∇μ, ∇Σ = similar(μ), similar(Σ)

    @cuda threads = cld(n_dimension, Δ) blocks = n_components ∇logprob_fused_kernel!(∇μ, ∇Σ, ∇y, μ, Σ, x, logp, mx, sumexp, n_components, n_observations, n_dimension, Val(max_threads))

    return (Tangent{CuGMM{T}}(μ = ∇μ, Σ = ∇Σ), NoTangent())
end
