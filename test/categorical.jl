using SmoothMixtures
using SmoothMixtures.CUDA
using Test

using FiniteDifferences, Flux

using SmoothMixtures: logprob, logsumexp, sumlogsumexp_logprob, sumlogsumexp_logprob_reference


function sumlogsumexp_logprob_reference(logits, x)
	log_probs, max_ = logprob(logits, x)
	sum(logsumexp(log_probs, dims = 1))
end

@testset "Categorical distribution" begin 

	@testset "CPU version is correct" begin
		n_categories = 10
		n_dimension = 7
		n_components = Int(2^3)
		n_observations = 50

		# Let's first verify that the cpu version of logprob is correct
		logits = randn(Float32, n_categories, n_dimension, n_components);
		x = rand(UInt8(1):UInt8(n_categories), n_dimension, n_observations);
		log_probs, mx = logprob(logits, x)
		@test grad(central_fdm(5, 1), logits -> sum(logprob(logits, x)[1]), logits)[1] ≈ ∇logprob(ones(Float32, n_components, n_observations), logits, x)

		@test sumlogsumexp_logprob(logits, x) ≈ sumlogsumexp_logprob_reference(logits, x)
		@test grad(central_fdm(5, 1), logits -> sumlogsumexp_logprob(logits, x), logits)[1] ≈ grad(central_fdm(5, 1), logits -> sumlogsumexp_logprob_reference(logits, x), logits)[1]

		fval, ∇logits = Flux.Zygote.withgradient(logits -> sumlogsumexp_logprob(logits, x), logits)
		@test fval ≈ sumlogsumexp_logprob_reference(logits, x)
		@test ∇logits[1] ≈ grad(central_fdm(5, 1), logits -> sumlogsumexp_logprob(logits, x), logits)[1]
	end

	# This is setting more aking to ContinousMixtures
	n_categories = 256
	n_dimension = 768
	n_components = Int(2^14)
	n_observations = 50

	# assuming CPU version is correct
	@testset "GPU version of logprob" begin
		@testset "iteration $(i)" for i in 1:10
			Random.seed!(i)
			n_categories = rand(1:20)
			n_dimension = rand(1:2^10)
			n_components = Int(2^rand(0:12))
			n_observations = rand(1:50)

			logits = randn(Float32, n_categories, n_dimension, n_components);
			x = rand(UInt8(1):UInt8(n_categories), n_dimension, n_observations);
			
			cu_logits = CuArray(logits)
			cu_x = CuArray(x)

			@testset "forward pass" begin
				log_probs, mx = logprob(logits, x)
				cu_logprobs, cu_mx = logprob(cu_logits, cu_x)
				@test Matrix(cu_logprobs) ≈ log_probs
				@test mx ≈ vec(maximum(log_probs, dims = 1))
				@test Vector(cu_mx) ≈ mx

				@test sumlogsumexp_logprob_reference(logits, x) == sumlogsumexp_logprob(logits, x)
				@test sumlogsumexp_logprob(cu_logits, cu_x) ≈ sumlogsumexp_logprob(logits, x)
			end

			@testset "second kernel" begin
				log_probs, mx = logprob(logits, x)
				cu_logprobs, cu_mx = logprob(cu_logits, cu_x)
				cu_lkl, cu_sumexp = sumlogsumexp(cu_logprobs, cu_mx)
			    log_probs, mx = logprob(logits, x)
			    sumexp = sum(exp.(log_probs .- mx'); dims = 1)

			    @test cu_lkl ≈ sum(log.(cu_sumexp) .+ cu_mx)
			    @test cu_lkl ≈ sum(log.(sumexp) .+ mx')
			    @test Vector(cu_sumexp) ≈ vec(sumexp)
			end

			@testset "backward pass" begin
				∇logprobs = ones(Float32, n_components, n_observations)
				cu_∇logprobs = CuArray(∇logprobs)

				∇logits = ∇logprob(∇logprobs, logits, x)

				cu_∇logits = ∇logprob(cu_∇logprobs, cu_logits, cu_x)
				@test Array(cu_∇logits) ≈ ∇logits

				fval, ∇logits = Flux.Zygote.withgradient(logits -> sumlogsumexp_logprob(logits, x), logits)
				cu_fval, cu_∇logits = Flux.Zygote.withgradient(logits -> sumlogsumexp_logprob(logits, cu_x), cu_logits)
				@test fval ≈ cu_fval
				@test Array(∇logits[1]) ≈ Array(cu_∇logits[1])

				cu_fval, cu_∇logits = Flux.Zygote.withgradient(logits -> sumlogsumexp_logprob_fused(logits, cu_x), cu_logits)
				@test fval ≈ cu_fval
				@test Array(∇logits[1]) ≈ Array(cu_∇logits[1])
			end
		end
	end

end
