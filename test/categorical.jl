using ContinuousMixtures
using ContinuousMixtures.CUDA
using ContinuousMixtures.Flux
using ContinuousMixtures.ChainRulesCore
using Test
using Random
using FiniteDifferences
using Distributions

using ContinuousMixtures: CategoricalMixture, logprob, ∇logprob, sumlogsumexp_logprob, sumlogsumexp_logprob_fused


function logprob_reference(m::CategoricalMixture, x)
	n_observations = size(x,2)
	n_components = size(m.logits, 3)
	n_dimension = size(m.logits, 2)
	n_dimension == size(x,1) || error("dimension of observation do not match dimension of components")

	logp = zeros(eltype(m.logits), n_components, n_observations)
	for i in 1:n_components
		for j in 1:n_observations
			for k in axes(x,1)
				logp[i,j] += logpdf(Categorical(softmax(m.logits[:,k,i])), x[k,j])
			end
		end
	end
	logp
end

"""
	Reference and naive implementation of sumlogsumexp_logprob
"""
function sumlogsumexp_logprob_reference(m::CategoricalMixture, x)
	logp, max_ = logprob(m::CategoricalMixture, x)
	sum(logsumexp(logp, dims = 1))
end

@testset "Categorical distribution" begin
	@testset "CPU version" begin
		n_categories = 10
		n_dimension = 7
		n_components = Int(2^3)
		n_observations = 50

		# Let's first verify that the cpu version of logprob is correct
		m = CategoricalMixture(randn(Float64, n_categories, n_dimension, n_components));
		x = rand(UInt8(1):UInt8(n_categories), n_dimension, n_observations);

		@testset "Distribution implementation equals ContinuousMixtures" begin
			@test logprob(m, x)[1] ≈ logprob_reference(m, x)
			@test ContinuousMixtures.sumlogsumexp(logprob(m, x)...)[1] ≈ sumlogsumexp_logprob_reference(m, x)
			@test sumlogsumexp_logprob(m, x) ≈ sumlogsumexp_logprob_reference(m, x)
		end

		@testset "Finite diff gradient of Distributions matches that of ContinuousMixtures" begin
			@test grad(central_fdm(5, 1), logits -> sumlogsumexp_logprob(CategoricalMixture(logits), x), m.logits)[1] ≈ grad(central_fdm(5, 1), logits -> sumlogsumexp_logprob_reference(CategoricalMixture(logits), x), m.logits)[1]
		end

		@testset "Automatic gradient of nonfused version" begin
			fval, ∇logits = Flux.Zygote.withgradient(logits -> sumlogsumexp_logprob(CategoricalMixture(logits), x), m.logits)
			∇logits = only(∇logits)
			@test fval ≈ sumlogsumexp_logprob_reference(m, x)
			@test grad(central_fdm(5, 1), logits -> sumlogsumexp_logprob(CategoricalMixture(logits), x), m.logits)[1] ≈ ∇logits

			fval, ∇x = Flux.Zygote.withgradient(x -> sumlogsumexp_logprob(m, x), x)
			@test only(∇x) === nothing
		end

		@testset "Automatic gradient of fused version" begin
			fval, ∇logits = Flux.Zygote.withgradient(logits -> sumlogsumexp_logprob_fused(CategoricalMixture(logits), x), m.logits)
			∇logits = only(∇logits)
			@test fval ≈ sumlogsumexp_logprob_reference(m, x)
			@test grad(central_fdm(5, 1), logits -> sumlogsumexp_logprob_fused(CategoricalMixture(logits), x), m.logits)[1] ≈ ∇logits

			fval, ∇x = Flux.Zygote.withgradient(x -> sumlogsumexp_logprob_fused(m, x), x)
			@test only(∇x) === nothing
		end
	end

	# assuming CPU version is correct
	@testset "GPU version of logprob" begin
		@testset "iteration $(i)" for i in 1:10
			Random.seed!(i)
			n_categories = rand(1:20)
			n_dimension = rand(1:2^10)
			n_components = Int(2^rand(0:12))
			n_observations = rand(1:50)

			x = rand(UInt8(1):UInt8(n_categories), n_dimension, n_observations);
			m = CategoricalMixture(randn(Float32, n_categories, n_dimension, n_components))

			cu_x = cu(x)
			cu_m = CategoricalMixture(cu(m.logits))

			@testset "forward pass" begin
				log_probs, mx = logprob(m, x)
				cu_logprobs, cu_mx = logprob(cu_m, cu_x)
				@test Matrix(cu_logprobs) ≈ log_probs
				@test mx ≈ vec(maximum(log_probs, dims = 1))
				@test Vector(cu_mx) ≈ mx

				@test sumlogsumexp_logprob_reference(m, x) == ContinuousMixtures.sumlogsumexp_logprob(m, x)
				@test ContinuousMixtures.sumlogsumexp_logprob(cu_m, cu_x) ≈ ContinuousMixtures.sumlogsumexp_logprob(m, x)
			end

			@testset "second kernel" begin
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

				∇logits, ∇x = ∇logprob(∇logprobs, m, x)

				cu_∇logits, cu_∇x = ∇logprob(cu_∇logprobs, cu_m, cu_x)
				@test cu_∇x == NoTangent()
				@test Array(cu_∇logits.logits) ≈ ∇logits.logits

				fval, ∇m = Flux.Zygote.withgradient(m -> ContinuousMixtures.sumlogsumexp_logprob(m, x), m)
				cu_fval, cu_∇m = Flux.Zygote.withgradient(m -> ContinuousMixtures.sumlogsumexp_logprob(m, cu_x), cu_m)
				@test fval ≈ cu_fval
				@test Array(∇m[1].logits) ≈ Array(cu_∇m[1].logits)

				cu_fval, cu_∇m = Flux.Zygote.withgradient(m -> ContinuousMixtures.sumlogsumexp_logprob_fused(m, cu_x), cu_m)
				@test fval ≈ cu_fval
				@test Array(∇m[1].logits) ≈ Array(cu_∇m[1].logits)
			end
		end
	end

end
