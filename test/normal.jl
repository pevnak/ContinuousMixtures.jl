using ContinuousMixtures
using ContinuousMixtures.CUDA
using ContinuousMixtures.Flux
using Test
using Random
using FiniteDifferences
using Distributions
using Distributions: PDiagMat

using ContinuousMixtures: GaussianMixture, logprob, ∇logprob, sumlogsumexp_logprob, sumlogsumexp_logprob_fused


"""
	Reference and naive implementation of sumlogsumexp_logprob
"""
function logprob_reference(m::GaussianMixture, x)
	n_observations = size(x,2)
	n_components = size(m.μ, 2)
	n_dimension = size(m.μ, 1)
	n_dimension == size(x,1) || error("dimension of observation do not match dimension of components")

	logp = zeros(eltype(x), n_components, n_observations)
	for i in 1:n_components
		for j in 1:n_observations
			logp[i,j] = logpdf(DiagNormal(m.μ[:,i], PDiagMat(m.Σ[:,i])), x[:,j])
		end
	end
	logp
end

"""
	Reference and naive implementation of sumlogsumexp_logprob
"""
function sumlogsumexp_logprob_reference(m, x)
	logp, max_ = logprob(m, x)
	sum(logsumexp(logp, dims = 1))
end

@testset "Normal distribution" begin
	@testset "CPU version" begin
		n_components = Int(2^3)
		n_observations = 50
		n_dimension = 7

		# Let's first verify that the cpu version of logprob is correct
		m = GaussianMixture(randn(Float64, n_dimension, n_components), softplus.(randn(Float64, n_dimension, n_components)))
		x = rand(Float64, n_dimension, n_observations);

		@testset "Distribution implementation equals ContinuousMixtures" begin
			@test logprob(m, x)[1] ≈ logprob_reference(m, x)
			@test ContinuousMixtures.sumlogsumexp(logprob(m, x)...)[1] ≈ sumlogsumexp_logprob_reference(m, x)
			@test sumlogsumexp_logprob(m, x) ≈ sumlogsumexp_logprob_reference(m, x)
		end

		@testset "nonfused gradient is correct" begin
			w = randn(n_components, n_observations);
			@test grad(central_fdm(5, 1), μ -> sum(w .* logprob(GaussianMixture(μ, m.Σ), x)[1]), m.μ)[1] ≈ ∇logprob(w, m, x)[1].μ
			@test grad(central_fdm(5, 1), Σ -> sum(w .* logprob(GaussianMixture(m.μ, Σ), x)[1]), m.Σ)[1] ≈ ∇logprob(w, m, x)[1].Σ
			@test grad(central_fdm(5, 1), x -> sum(w .* logprob(m, x)[1]), x)[1] ≈ ∇logprob(w, m, x)[2]
		end

		@testset "Finite diff gradient of Distributions matches that of ContinuousMixtures" begin
			@test grad(central_fdm(5, 1), μ -> sumlogsumexp_logprob(GaussianMixture(μ, m.Σ), x), m.μ)[1] ≈ grad(central_fdm(5, 1), μ -> sumlogsumexp_logprob_reference(GaussianMixture(μ, m.Σ), x), m.μ)[1]
			@test grad(central_fdm(5, 1), Σ -> sumlogsumexp_logprob(GaussianMixture(m.μ, Σ), x), m.Σ)[1] ≈ grad(central_fdm(5, 1), Σ -> sumlogsumexp_logprob_reference(GaussianMixture(m.μ, Σ), x), m.Σ)[1]
			@test grad(central_fdm(5, 1), x -> sumlogsumexp_logprob(m, x), x)[1] ≈ grad(central_fdm(5, 1), x -> sumlogsumexp_logprob_reference(m, x), x)[1]
		end

		@testset "Automatic gradient of nonfused version" begin
			fval, ∇m = Flux.Zygote.withgradient(m -> sumlogsumexp_logprob(m, x), m)
			∇m = only(∇m)
			@test fval ≈ sumlogsumexp_logprob_reference(m, x)
			@test grad(central_fdm(5, 1), μ -> sumlogsumexp_logprob(GaussianMixture(μ, m.Σ), x), m.μ)[1] ≈ ∇m.μ
			@test grad(central_fdm(5, 1), Σ -> sumlogsumexp_logprob(GaussianMixture(m.μ, Σ), x), m.Σ)[1] ≈ ∇m.Σ

			fval, ∇x = Flux.Zygote.withgradient(x -> sumlogsumexp_logprob(m, x), x)
			∇x = only(∇x)
			@test fval ≈ sumlogsumexp_logprob_reference(m, x)
			@test grad(central_fdm(5, 1), x -> sumlogsumexp_logprob(m, x), x)[1] ≈ ∇x
		end

		@testset "Automatic gradient of fused version" begin
			fval, ∇m = Flux.Zygote.withgradient(m -> sumlogsumexp_logprob_fused(m, x), m)
			∇m = only(∇m)
			@test fval ≈ sumlogsumexp_logprob_reference(m, x)
			@test grad(central_fdm(5, 1), μ -> sumlogsumexp_logprob_fused(GaussianMixture(μ, m.Σ), x), m.μ)[1] ≈ ∇m.μ
			@test grad(central_fdm(5, 1), Σ -> sumlogsumexp_logprob_fused(GaussianMixture(m.μ, Σ), x), m.Σ)[1] ≈ ∇m.Σ

			fval, ∇x = Flux.Zygote.withgradient(x -> sumlogsumexp_logprob_fused(m, x), x)
			∇x = only(∇x)
			@test fval ≈ sumlogsumexp_logprob_reference(m, x)
			@test grad(central_fdm(5, 1), x -> sumlogsumexp_logprob_fused(m, x), x)[1] ≈ ∇x
		end
	end

	@testset "GPU version of logprob" begin
		@testset "iteration $(i)" for i in 1:10
			Random.seed!(i)
			n_components = Int(2^3)
			n_observations = 50
			n_dimension = 7

			# Let's first verify that the cpu version of logprob is correct
			m = GaussianMixture(randn(Float64, n_dimension, n_components), softplus.(randn(Float64, n_dimension, n_components)))
			x = rand(Float64, n_dimension, n_observations);

			cu_x = cu(x)
			cu_m = GaussianMixture(cu(m.μ), cu(m.Σ))

			@testset "logprob forward pass" begin
				log_probs, mx = logprob(m, x)
				cu_logprobs, cu_mx = logprob(cu_m, cu_x)
				@test Matrix(cu_logprobs) ≈ log_probs
				@test mx ≈ vec(maximum(log_probs, dims = 1))
				@test Vector(cu_mx) ≈ mx

				@test sumlogsumexp_logprob_reference(m, x) ≈ ContinuousMixtures.sumlogsumexp_logprob(m, x)
				@test ContinuousMixtures.sumlogsumexp_logprob(cu_m, cu_x) ≈ ContinuousMixtures.sumlogsumexp_logprob(m, x)
			end

			@testset "sumlogsumexp forward pass" begin
				log_probs, mx = logprob(m, x)
				cu_logprobs, cu_mx = logprob(cu_m, cu_x)
				cu_lkl, cu_sumexp = ContinuousMixtures.sumlogsumexp(cu_logprobs, cu_mx)
			    log_probs, mx = logprob(m, x)
			    sumexp = sum(exp.(log_probs .- mx'); dims = 1)

			    @test cu_lkl ≈ sum(log.(cu_sumexp) .+ cu_mx)
			    @test cu_lkl ≈ sum(log.(sumexp) .+ mx')
			    @test Vector(cu_sumexp) ≈ vec(sumexp)
			end

			@testset "backward pass" begin
				∇logprobs = randn(Float32, n_components, n_observations)
				cu_∇logprobs = cu(∇logprobs)

				∇m, ∇x = ∇logprob(∇logprobs, m, x)

				cu_∇m, cu_∇x = ∇logprob(cu_∇logprobs, cu_m, cu_x)
				@test cu_∇x == NoTangent()
				@test Array(cu_∇m.μ) ≈ ∇m.μ
				@test Array(cu_∇m.Σ) ≈ ∇m.Σ

				fval, ∇m = Flux.Zygote.withgradient(m -> ContinuousMixtures.sumlogsumexp_logprob(m, x), m)
				cu_fval, cu_∇m = Flux.Zygote.withgradient(m -> ContinuousMixtures.sumlogsumexp_logprob(m, cu_x), cu_m)
				@test fval ≈ cu_fval
				∇m, cu_∇m = only(∇m), only(cu_∇m)
				@test Array(cu_∇m.μ) ≈ ∇m.μ
				@test Array(cu_∇m.Σ) ≈ ∇m.Σ

				cu_fval, cu_∇m = Flux.Zygote.withgradient(m -> ContinuousMixtures.sumlogsumexp_logprob_fused(m, cu_x), cu_m)
				@test fval ≈ cu_fval
				cu_∇m = only(cu_∇m)
				@test Array(cu_∇m.μ) ≈ ∇m.μ
				@test Array(cu_∇m.Σ) ≈ ∇m.Σ
			end
		end
	end
end
